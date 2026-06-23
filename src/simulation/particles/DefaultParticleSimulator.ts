import type { ParticleLODLevel, ParticleSystem } from "../../particles";
import {
	PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
	PARTICLE_TRANSIENT_BATCHES_KEY,
	type FrameContext,
} from "../../pipeline/types";
import { ParticleBatchBuilder } from "./ParticleBatchBuilder";
import type { IParticleSimulator } from "./IParticleSimulator";
import { ParticleSimulationCore } from "./ParticleSimulationCore";
import { CameraType } from "../../cameras/Camera";
import { MAX_STEP_SECONDS } from "./constants";
import type { SystemRuntimeState } from "./types";

const FULL_QUALITY_LOD: ParticleLODLevel = Object.freeze({
	distance: Number.POSITIVE_INFINITY,
	projectedSize: 0,
	simulationIntervalFrames: 1,
	spawnScale: 1,
	maxParticlesScale: 1,
	renderSortRatio: 1,
});

interface DefaultParticleSimulatorOptions {
	backendTag?: string;
	strictFailure?: boolean;
	maxParticlesPerSystem?: number;
}

export class DefaultParticleSimulator implements IParticleSimulator {
	private _runtimeBySystemId = new Map<string, SystemRuntimeState>();
	private _core = new ParticleSimulationCore();
	private _batchBuilder = new ParticleBatchBuilder();
	private _backendTag: string;
	private _strictFailure: boolean;
	private _maxParticlesPerSystem: number;

	constructor(options: DefaultParticleSimulatorOptions = {}) {
		this._backendTag = options.backendTag ?? "default";
		this._strictFailure = options.strictFailure ?? true;
		this._maxParticlesPerSystem = Math.max(
			1,
			Math.floor(options.maxParticlesPerSystem ?? Number.POSITIVE_INFINITY)
		);
	}

	public beginFrame(context: FrameContext): void {
		const particleSystems = context.scene.particleSystems ?? [];
		if (particleSystems.length === 0) {
			this._runtimeBySystemId.clear();
			return;
		}
		for (const system of particleSystems) {
			system.updateWorldMatrix(system.parent?.worldMatrix);
		}

		const activeIds = new Set(particleSystems.map((system) => system.id));
		for (const systemId of this._runtimeBySystemId.keys()) {
			if (!activeIds.has(systemId)) {
				this._runtimeBySystemId.delete(systemId);
			}
		}
	}

	public simulate(context: FrameContext, deltaTimeSeconds: number): void {
		const particleSystems = context.scene.particleSystems ?? [];
		if (particleSystems.length === 0) {
			context.transient.set(PARTICLE_TRANSIENT_BATCHES_KEY, []);
			context.transient.set(PARTICLE_MESH_TRANSIENT_BATCHES_KEY, []);
			return;
		}

		const totalDt = Math.max(0, deltaTimeSeconds);
		for (const system of particleSystems) {
			system.updateWorldMatrix(system.parent?.worldMatrix);
			const runtime = this._getSystemRuntime(system);
			runtime.frameIndex++;

			const activeLODLevel = this._resolveLODLevel(system, runtime, context);
			runtime.activeLODLevel = activeLODLevel;
			const effectiveMaxParticles = this._resolveEffectiveMaxParticles(
				system,
				activeLODLevel
			);
			this._core.trimToMaxParticles(runtime, effectiveMaxParticles);

			runtime.pendingSimulationTime += totalDt;
			const intervalFrames = Math.max(
				1,
				Math.floor(activeLODLevel.simulationIntervalFrames)
			);
			if (runtime.frameIndex % intervalFrames !== 0) {
				continue;
			}

			let stepBudget = runtime.pendingSimulationTime;
			runtime.pendingSimulationTime = 0;
			while (stepBudget > 0) {
				const stepDt = Math.min(MAX_STEP_SECONDS, stepBudget);
				this._core.stepSystem(system, runtime, stepDt, {
					spawnScale: Math.max(0, activeLODLevel.spawnScale),
					maxParticles: effectiveMaxParticles,
				});
				stepBudget -= stepDt;
			}
		}
	}

	public emitRenderBatches(context: FrameContext): void {
		const particleSystems = context.scene.particleSystems ?? [];
		if (particleSystems.length === 0) {
			context.transient.set(PARTICLE_TRANSIENT_BATCHES_KEY, []);
			context.transient.set(PARTICLE_MESH_TRANSIENT_BATCHES_KEY, []);
			return;
		}

		const batches = [];
		const meshBatches = [];
		for (const system of particleSystems) {
			if (!system.visible) continue;
			const runtime = this._runtimeBySystemId.get(system.id);
			if (!runtime) continue;
			const sortRatio = runtime.activeLODLevel?.renderSortRatio ?? 1;
			const built = this._batchBuilder.buildBatches(
				system,
				runtime,
				context,
				sortRatio
			);
			batches.push(...built.billboardBatches);
			meshBatches.push(...built.meshBatches);
		}

		context.transient.set(PARTICLE_TRANSIENT_BATCHES_KEY, batches);
		context.transient.set(PARTICLE_MESH_TRANSIENT_BATCHES_KEY, meshBatches);
	}

	public endFrame(): void {}

	private _resolveLODLevel(
		system: ParticleSystem,
		runtime: SystemRuntimeState,
		context: FrameContext
	): ParticleLODLevel {
		const lod = system.lod;
		if (!lod?.enabled || !lod.levels || lod.levels.length === 0) {
			runtime.lodLevelIndex = 0;
			runtime.lodCandidateLevelIndex = -1;
			runtime.lodCandidateFrameCount = 0;
			return FULL_QUALITY_LOD;
		}

		const levels = this._normalizeLODLevels(system);
		const distance = this._computeDistanceToCamera(system, context);
		const projectedSize = this._computeProjectedSystemSizePx(
			system,
			context,
			distance
		);
		const distanceTier = this._resolveDistanceTier(levels, distance);
		const screenTier = this._resolveScreenTier(levels, projectedSize);
		const targetTier = Math.max(distanceTier, screenTier);
		const hysteresisFrames = Math.max(0, Math.floor(lod.hysteresisFrames ?? 6));

		const currentTier = clampTier(runtime.lodLevelIndex, levels.length);
		if (targetTier === currentTier) {
			runtime.lodCandidateLevelIndex = -1;
			runtime.lodCandidateFrameCount = 0;
			runtime.lodLevelIndex = currentTier;
			return levels[currentTier];
		}

		if (hysteresisFrames <= 0) {
			runtime.lodLevelIndex = targetTier;
			runtime.lodCandidateLevelIndex = -1;
			runtime.lodCandidateFrameCount = 0;
			return levels[targetTier];
		}

		if (runtime.lodCandidateLevelIndex !== targetTier) {
			runtime.lodCandidateLevelIndex = targetTier;
			runtime.lodCandidateFrameCount = 1;
			return levels[currentTier];
		}

		runtime.lodCandidateFrameCount++;
		if (runtime.lodCandidateFrameCount >= hysteresisFrames) {
			runtime.lodLevelIndex = targetTier;
			runtime.lodCandidateLevelIndex = -1;
			runtime.lodCandidateFrameCount = 0;
			return levels[targetTier];
		}

		return levels[currentTier];
	}

	private _resolveEffectiveMaxParticles(
		system: ParticleSystem,
		level: ParticleLODLevel
	): number {
		const required = Math.max(
			1,
			Math.ceil(system.maxParticles * Math.max(0, level.maxParticlesScale))
		);
		const available = Math.min(
			system.maxParticles,
			this._maxParticlesPerSystem
		);

		if (this._strictFailure && required > available) {
			throw new Error(
				`[ParticleSim:${this._backendTag}] LOD particle budget overflow for "${system.id}" required=${required} available=${available}`
			);
		}

		return Math.min(required, available);
	}

	private _normalizeLODLevels(system: ParticleSystem): ParticleLODLevel[] {
		const levels = system.lod.levels;
		if (!Array.isArray(levels) || levels.length === 0) {
			throw new Error(
				`[ParticleSim:${this._backendTag}] ParticleSystem "${system.id}" has LOD enabled without levels`
			);
		}

		return levels.map((level, index) => {
			const simulationIntervalFrames = Math.max(
				1,
				Math.floor(level.simulationIntervalFrames)
			);
			const spawnScale = sanitizeRange(
				level.spawnScale,
				`LOD level ${index} spawnScale`,
				system.id,
				this._backendTag
			);
			const renderSortRatio = sanitizeRange(
				level.renderSortRatio,
				`LOD level ${index} renderSortRatio`,
				system.id,
				this._backendTag
			);
			const maxParticlesScale = sanitizePositive(
				level.maxParticlesScale,
				`LOD level ${index} maxParticlesScale`,
				system.id,
				this._backendTag
			);

			return {
				distance:
					Number.isFinite(level.distance) ?
						Math.max(0, level.distance)
					:	Number.POSITIVE_INFINITY,
				projectedSize:
					Number.isFinite(level.projectedSize) ?
						Math.max(0, level.projectedSize)
					:	0,
				simulationIntervalFrames,
				spawnScale,
				maxParticlesScale,
				renderSortRatio,
			};
		});
	}

	private _computeDistanceToCamera(
		system: ParticleSystem,
		context: FrameContext
	): number {
		const systemPosition = system.getWorldPosition();
		const cameraPosition = context.camera.getWorldPosition();
		const dx = systemPosition.x - cameraPosition.x;
		const dy = systemPosition.y - cameraPosition.y;
		const dz = systemPosition.z - cameraPosition.z;
		return Math.hypot(dx, dy, dz);
	}

	private _computeProjectedSystemSizePx(
		system: ParticleSystem,
		context: FrameContext,
		distanceToCamera: number
	): number {
		const emit = system.emit ?? {};
		const spawnRadius = Math.max(0, emit.spawnRadius ?? 0);
		let maxStartSize = 0;
		let maxLifetimeScale = 1;
		for (const template of system.templates) {
			const sizeRange = template.sizeRange ?? [0.5, 1];
			maxStartSize = Math.max(maxStartSize, sizeRange[0], sizeRange[1]);
			if ((template.sizeOverLifetime?.length ?? 0) > 0) {
				maxLifetimeScale = Math.max(
					maxLifetimeScale,
					...template.sizeOverLifetime!.map((key) => key.value)
				);
			}
		}
		const estimatedRadius =
			spawnRadius + maxStartSize + Math.max(0, maxLifetimeScale);

		if (estimatedRadius <= 0) return 0;

		if (context.camera.type === CameraType.Orthographic) {
			return estimatedRadius * 2;
		}

		const halfFov = (context.camera.fov * Math.PI) / 360;
		const tanHalfFov = Math.tan(halfFov) || 1e-6;
		const focalLength =
			(Math.max(1, context.attachments.height) * 0.5) / tanHalfFov;
		return (
			(estimatedRadius * focalLength * 2) / Math.max(0.001, distanceToCamera)
		);
	}

	private _resolveDistanceTier(
		levels: ParticleLODLevel[],
		distance: number
	): number {
		for (let i = 0; i < levels.length; i++) {
			if (distance <= levels[i].distance) {
				return i;
			}
		}
		return levels.length - 1;
	}

	private _resolveScreenTier(
		levels: ParticleLODLevel[],
		projectedSize: number
	): number {
		for (let i = 0; i < levels.length; i++) {
			if (projectedSize >= levels[i].projectedSize) {
				return i;
			}
		}
		return levels.length - 1;
	}

	private _getSystemRuntime(system: ParticleSystem): SystemRuntimeState {
		let runtime = this._runtimeBySystemId.get(system.id);
		if (runtime) return runtime;

		runtime = this._core.createRuntime(system);
		this._runtimeBySystemId.set(system.id, runtime);
		return runtime;
	}
}

function clampTier(index: number, length: number): number {
	if (!Number.isFinite(index)) return 0;
	return Math.min(length - 1, Math.max(0, Math.floor(index)));
}

function sanitizeRange(
	value: number,
	name: string,
	systemId: string,
	backendTag: string
): number {
	if (!Number.isFinite(value) || value < 0 || value > 1) {
		throw new Error(
			`[ParticleSim:${backendTag}] ParticleSystem "${systemId}" invalid ${name}=${value}`
		);
	}
	return value;
}

function sanitizePositive(
	value: number,
	name: string,
	systemId: string,
	backendTag: string
): number {
	if (!Number.isFinite(value) || value <= 0) {
		throw new Error(
			`[ParticleSim:${backendTag}] ParticleSystem "${systemId}" invalid ${name}=${value}`
		);
	}
	return value;
}
