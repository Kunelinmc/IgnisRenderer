import { Matrix4 } from "../../maths/Matrix4";
import type { IVector3 } from "../../maths/types";
import {
	ParticleSpaceMode,
	type ParticleCollider,
	type ParticleEmitterParams,
	type ParticleGradientKey,
	type ParticleSubEmitterConfig,
	type ParticleSystem,
} from "../../particles";
import type { RGBA } from "../../utils/Color";
import {
	PARTICLE_TRANSIENT_BATCHES_KEY,
	type FrameContext,
	type ParticleRenderBatch,
	type ParticleRenderItem,
	type ParticleUVRect,
} from "./types";

interface RuntimeParticle {
	position: IVector3;
	velocity: IVector3;
	age: number;
	lifetime: number;
	startSize: number;
	startColor: RGBA;
	rotation: number;
	angularVelocity: number;
}

interface SystemRuntimeState {
	particles: RuntimeParticle[];
	emissionRemainder: number;
	elapsed: number;
	burstCycles: number[];
	randomState: number;
}

interface SpawnOverrides {
	position?: IVector3;
	baseVelocity?: IVector3;
	lifetimeRange?: [number, number];
	speedRange?: [number, number];
	sizeRange?: [number, number];
}

const FULL_UV_RECT: ParticleUVRect = { u0: 0, v0: 0, u1: 1, v1: 1 };
const DEFAULT_RESTITUTION = 0.6;
const DEFAULT_DAMPING = 0.1;
const MS_TO_SECONDS = 1 / 1000;
const MAX_STEP_SECONDS = 0.1;

export class ParticleSimulationStage {
	private _runtimeBySystemId = new Map<string, SystemRuntimeState>();

	public execute(context: FrameContext, deltaTimeMs: number): void {
		const particleSystems = context.scene.particleSystems ?? [];
		if (particleSystems.length === 0) {
			this._runtimeBySystemId.clear();
			context.transient.set(PARTICLE_TRANSIENT_BATCHES_KEY, []);
			return;
		}

		const activeIds = new Set(particleSystems.map((system) => system.id));
		for (const systemId of this._runtimeBySystemId.keys()) {
			if (!activeIds.has(systemId)) {
				this._runtimeBySystemId.delete(systemId);
			}
		}

		const totalDt = Math.max(0, deltaTimeMs * MS_TO_SECONDS);
		const batches: ParticleRenderBatch[] = [];

		for (const system of particleSystems) {
			const runtime = this._getSystemRuntime(system);
			let remaining = totalDt;
			while (remaining > 0) {
				const stepDt = Math.min(MAX_STEP_SECONDS, remaining);
				this._stepSystem(system, runtime, stepDt);
				remaining -= stepDt;
			}

			if (!system.visible) continue;

			const batch = this._buildBatch(system, runtime, context);
			if (batch.particles.length > 0) {
				batches.push(batch);
			}
		}

		context.transient.set(PARTICLE_TRANSIENT_BATCHES_KEY, batches);
	}

	private _stepSystem(
		system: ParticleSystem,
		runtime: SystemRuntimeState,
		dt: number
	): void {
		if (dt <= 0) return;

		runtime.elapsed += dt;
		const spawnCount = this._resolveSpawnCount(system, runtime, dt);
		if (spawnCount > 0) {
			this._spawnParticles(system, runtime, spawnCount);
		}

		const colliders = system.colliders ?? [];
		const gravity = system.gravity;
		const subEmitter = system.subEmitter;

		for (let i = runtime.particles.length - 1; i >= 0; i--) {
			const particle = runtime.particles[i];
			particle.age += dt;
			if (particle.age >= particle.lifetime) {
				if (subEmitter?.enabled !== false && subEmitter) {
					this._spawnSubEmitterParticles(system, runtime, particle, subEmitter);
				}
				runtime.particles.splice(i, 1);
				continue;
			}

			particle.velocity.x += gravity.x * dt;
			particle.velocity.y += gravity.y * dt;
			particle.velocity.z += gravity.z * dt;

			particle.position.x += particle.velocity.x * dt;
			particle.position.y += particle.velocity.y * dt;
			particle.position.z += particle.velocity.z * dt;
			particle.rotation += particle.angularVelocity * dt;

			if (colliders.length === 0) continue;
			for (const collider of colliders) {
				if (collider.enabled === false) continue;
				this._resolveCollider(particle, collider);
			}
		}
	}

	private _resolveSpawnCount(
		system: ParticleSystem,
		runtime: SystemRuntimeState,
		dt: number
	): number {
		const emit = system.emit ?? {};
		const rate = Math.max(0, emit.rate ?? 0);
		runtime.emissionRemainder += rate * dt;

		let spawnCount = Math.floor(runtime.emissionRemainder);
		if (spawnCount > 0) {
			runtime.emissionRemainder -= spawnCount;
		}

		const bursts = emit.bursts ?? [];
		for (let i = 0; i < bursts.length; i++) {
			const burst = bursts[i];
			const cycles = Math.max(1, burst.cycles ?? 1);
			const interval = Math.max(0, burst.interval ?? 0);
			let firedCycles = runtime.burstCycles[i] ?? 0;

			while (firedCycles < cycles) {
				const triggerAt = burst.time + interval * firedCycles;
				if (runtime.elapsed < triggerAt) break;
				spawnCount += Math.max(0, Math.floor(burst.count));
				firedCycles++;
			}

			runtime.burstCycles[i] = firedCycles;
		}

		return spawnCount;
	}

	private _spawnParticles(
		system: ParticleSystem,
		runtime: SystemRuntimeState,
		count: number,
		overrides: SpawnOverrides = {}
	): void {
		const available = Math.max(
			0,
			system.maxParticles - runtime.particles.length
		);
		const spawnCount = Math.min(available, Math.max(0, count));
		if (spawnCount <= 0) return;

		for (let i = 0; i < spawnCount; i++) {
			runtime.particles.push(this._createParticle(system, runtime, overrides));
		}
	}

	private _spawnSubEmitterParticles(
		system: ParticleSystem,
		runtime: SystemRuntimeState,
		parent: RuntimeParticle,
		subEmitter: ParticleSubEmitterConfig
	): void {
		if ((subEmitter.trigger ?? "death") !== "death") return;

		const count = Math.max(0, Math.floor(subEmitter.count ?? 0));
		if (count <= 0) return;

		const inheritScale = subEmitter.inheritVelocityScale ?? 0.5;
		const baseVelocity = {
			x: parent.velocity.x * inheritScale,
			y: parent.velocity.y * inheritScale,
			z: parent.velocity.z * inheritScale,
		};
		this._spawnParticles(system, runtime, count, {
			position: parent.position,
			baseVelocity,
			lifetimeRange: subEmitter.lifetimeRange,
			speedRange: subEmitter.speedRange,
			sizeRange: subEmitter.sizeRange,
		});
	}

	private _createParticle(
		system: ParticleSystem,
		runtime: SystemRuntimeState,
		overrides: SpawnOverrides = {}
	): RuntimeParticle {
		const emit = system.emit;
		const lifetimeRange = overrides.lifetimeRange ??
			emit.lifetimeRange ?? [0.5, 1.5];
		const speedRange = overrides.speedRange ?? emit.speedRange ?? [2, 5];
		const sizeRange = overrides.sizeRange ?? emit.sizeRange ?? [0.5, 1.0];
		const rotationRange = emit.rotationRange ?? [0, 0];
		const angularVelocityRange = emit.angularVelocityRange ?? [0, 0];
		const startColor = emit.startColor ?? { r: 255, g: 255, b: 255, a: 1 };

		const randomDirection = this._randomDirectionInCone(
			runtime,
			emit.direction ?? { x: 0, y: 1, z: 0 },
			Math.max(0, emit.spread ?? 0)
		);
		const speed = this._randomRange(runtime, speedRange[0], speedRange[1]);
		const spawnedVelocity = {
			x: randomDirection.x * speed,
			y: randomDirection.y * speed,
			z: randomDirection.z * speed,
		};
		const velocity = overrides.baseVelocity
			? {
					x: spawnedVelocity.x + overrides.baseVelocity.x,
					y: spawnedVelocity.y + overrides.baseVelocity.y,
					z: spawnedVelocity.z + overrides.baseVelocity.z,
				}
			: spawnedVelocity;

		const spawnOffset = overrides.position
			? { x: 0, y: 0, z: 0 }
			: this._randomSpawnOffset(runtime, Math.max(0, emit.spawnRadius ?? 0));
		const spawnBasePosition = overrides.position
			? overrides.position
			: system.space === ParticleSpaceMode.Local
				? { x: 0, y: 0, z: 0 }
				: system.position;

		return {
			position: {
				x: spawnBasePosition.x + spawnOffset.x,
				y: spawnBasePosition.y + spawnOffset.y,
				z: spawnBasePosition.z + spawnOffset.z,
			},
			velocity,
			age: 0,
			lifetime: Math.max(
				0.01,
				this._randomRange(runtime, lifetimeRange[0], lifetimeRange[1])
			),
			startSize: Math.max(
				0.001,
				this._randomRange(runtime, sizeRange[0], sizeRange[1])
			),
			startColor: {
				r: startColor.r,
				g: startColor.g,
				b: startColor.b,
				a: startColor.a,
			},
			rotation: this._randomRange(runtime, rotationRange[0], rotationRange[1]),
			angularVelocity: this._randomRange(
				runtime,
				angularVelocityRange[0],
				angularVelocityRange[1]
			),
		};
	}

	private _buildBatch(
		system: ParticleSystem,
		runtime: SystemRuntimeState,
		context: FrameContext
	): ParticleRenderBatch {
		const particles: ParticleRenderItem[] = [];
		const cameraView = context.camera.viewMatrix;
		const systemPosition = system.position;

		for (const particle of runtime.particles) {
			const worldPosition =
				system.space === ParticleSpaceMode.Local
					? {
							x: particle.position.x + systemPosition.x,
							y: particle.position.y + systemPosition.y,
							z: particle.position.z + systemPosition.z,
						}
					: {
							x: particle.position.x,
							y: particle.position.y,
							z: particle.position.z,
						};

			const cameraSpace = Matrix4.transformPoint(cameraView, worldPosition);
			const depth = -cameraSpace.z;
			if (depth <= 0) continue;

			const lifeT = Math.max(0, Math.min(1, particle.age / particle.lifetime));
			const size = this._sampleNumberGradient(
				system.sizeOverLifetime,
				lifeT,
				particle.startSize
			);
			const color = this._sampleColorGradient(
				system.colorOverLifetime,
				lifeT,
				particle.startColor
			);

			if (size <= 0 || color.a <= 0) continue;

			particles.push({
				position: worldPosition,
				size,
				color,
				rotation: particle.rotation,
				depth,
				uvRect: this._resolveAtlasUVRect(system, particle),
			});
		}

		particles.sort((left, right) => right.depth - left.depth);

		return {
			systemId: system.id,
			blendMode: system.blendMode,
			texture: system.texture,
			receiveShadows: system.receiveShadows,
			particles,
		};
	}

	private _resolveCollider(
		particle: RuntimeParticle,
		collider: ParticleCollider
	): void {
		switch (collider.type) {
			case "plane":
				this._resolvePlaneCollider(particle, collider);
				break;
			case "sphere":
				this._resolveSphereCollider(particle, collider);
				break;
			case "aabb":
				this._resolveAABBCollider(particle, collider);
				break;
		}
	}

	private _resolvePlaneCollider(
		particle: RuntimeParticle,
		collider: Extract<ParticleCollider, { type: "plane" }>
	): void {
		const normal = normalizeVector(collider.normal);
		const distance =
			particle.position.x * normal.x +
			particle.position.y * normal.y +
			particle.position.z * normal.z +
			collider.constant;
		if (distance >= 0) return;

		particle.position.x -= normal.x * distance;
		particle.position.y -= normal.y * distance;
		particle.position.z -= normal.z * distance;
		this._applyBounce(particle, normal, collider.restitution, collider.damping);
	}

	private _resolveSphereCollider(
		particle: RuntimeParticle,
		collider: Extract<ParticleCollider, { type: "sphere" }>
	): void {
		const dx = particle.position.x - collider.center.x;
		const dy = particle.position.y - collider.center.y;
		const dz = particle.position.z - collider.center.z;
		const radius = Math.max(0.001, collider.radius);
		const distanceSq = dx * dx + dy * dy + dz * dz;
		if (distanceSq >= radius * radius) return;

		const distance = Math.sqrt(distanceSq) || 1;
		const normal = {
			x: dx / distance,
			y: dy / distance,
			z: dz / distance,
		};

		particle.position.x = collider.center.x + normal.x * radius;
		particle.position.y = collider.center.y + normal.y * radius;
		particle.position.z = collider.center.z + normal.z * radius;
		this._applyBounce(particle, normal, collider.restitution, collider.damping);
	}

	private _resolveAABBCollider(
		particle: RuntimeParticle,
		collider: Extract<ParticleCollider, { type: "aabb" }>
	): void {
		const p = particle.position;
		if (
			p.x < collider.min.x ||
			p.x > collider.max.x ||
			p.y < collider.min.y ||
			p.y > collider.max.y ||
			p.z < collider.min.z ||
			p.z > collider.max.z
		) {
			return;
		}

		const distances = [
			{
				axis: "x",
				value: Math.abs(p.x - collider.min.x),
				normal: { x: -1, y: 0, z: 0 },
			},
			{
				axis: "x",
				value: Math.abs(collider.max.x - p.x),
				normal: { x: 1, y: 0, z: 0 },
			},
			{
				axis: "y",
				value: Math.abs(p.y - collider.min.y),
				normal: { x: 0, y: -1, z: 0 },
			},
			{
				axis: "y",
				value: Math.abs(collider.max.y - p.y),
				normal: { x: 0, y: 1, z: 0 },
			},
			{
				axis: "z",
				value: Math.abs(p.z - collider.min.z),
				normal: { x: 0, y: 0, z: -1 },
			},
			{
				axis: "z",
				value: Math.abs(collider.max.z - p.z),
				normal: { x: 0, y: 0, z: 1 },
			},
		];
		distances.sort((left, right) => left.value - right.value);

		const hit = distances[0];
		if (!hit) return;

		if (hit.axis === "x") {
			p.x = hit.normal.x < 0 ? collider.min.x : collider.max.x;
		} else if (hit.axis === "y") {
			p.y = hit.normal.y < 0 ? collider.min.y : collider.max.y;
		} else {
			p.z = hit.normal.z < 0 ? collider.min.z : collider.max.z;
		}

		this._applyBounce(
			particle,
			hit.normal,
			collider.restitution,
			collider.damping
		);
	}

	private _applyBounce(
		particle: RuntimeParticle,
		normal: IVector3,
		restitution: number | undefined,
		damping: number | undefined
	): void {
		const rest = clamp01(restitution ?? DEFAULT_RESTITUTION);
		const damp = clamp01(damping ?? DEFAULT_DAMPING);
		const velocity = particle.velocity;
		const vn =
			velocity.x * normal.x + velocity.y * normal.y + velocity.z * normal.z;
		if (vn < 0) {
			const impulse = -(1 + rest) * vn;
			velocity.x += normal.x * impulse;
			velocity.y += normal.y * impulse;
			velocity.z += normal.z * impulse;
		}
		const dampingScale = 1 - damp;
		velocity.x *= dampingScale;
		velocity.y *= dampingScale;
		velocity.z *= dampingScale;
	}

	private _resolveAtlasUVRect(
		system: ParticleSystem,
		particle: RuntimeParticle
	): ParticleUVRect {
		const atlas = system.atlas;
		if (!atlas) return FULL_UV_RECT;

		const rows = Math.max(1, atlas.rows | 0);
		const columns = Math.max(1, atlas.columns | 0);
		const fps = Math.max(0, atlas.fps);
		const frameCount = rows * columns;
		if (frameCount <= 1 || fps <= 0) return FULL_UV_RECT;

		const rawFrame = Math.floor(particle.age * fps);
		const frame =
			atlas.loop === false
				? Math.min(frameCount - 1, rawFrame)
				: ((rawFrame % frameCount) + frameCount) % frameCount;
		const column = frame % columns;
		const row = Math.floor(frame / columns);
		const u0 = column / columns;
		const v0 = row / rows;

		return {
			u0,
			v0,
			u1: (column + 1) / columns,
			v1: (row + 1) / rows,
		};
	}

	private _sampleNumberGradient(
		keys: ParticleGradientKey<number>[] | undefined,
		t: number,
		fallback: number
	): number {
		if (!keys || keys.length === 0) return fallback;
		const sorted = keys.slice().sort((left, right) => left.t - right.t);
		if (t <= sorted[0].t) return sorted[0].value;
		if (t >= sorted[sorted.length - 1].t) {
			return sorted[sorted.length - 1].value;
		}

		for (let i = 1; i < sorted.length; i++) {
			const left = sorted[i - 1];
			const right = sorted[i];
			if (t > right.t) continue;
			const span = right.t - left.t || 1;
			const localT = (t - left.t) / span;
			return left.value + (right.value - left.value) * localT;
		}

		return fallback;
	}

	private _sampleColorGradient(
		keys: ParticleGradientKey<RGBA>[] | undefined,
		t: number,
		fallback: RGBA
	): RGBA {
		if (!keys || keys.length === 0) {
			return {
				r: fallback.r,
				g: fallback.g,
				b: fallback.b,
				a: fallback.a,
			};
		}
		const sorted = keys.slice().sort((left, right) => left.t - right.t);
		if (t <= sorted[0].t) return cloneColor(sorted[0].value);
		if (t >= sorted[sorted.length - 1].t) {
			return cloneColor(sorted[sorted.length - 1].value);
		}

		for (let i = 1; i < sorted.length; i++) {
			const left = sorted[i - 1];
			const right = sorted[i];
			if (t > right.t) continue;
			const span = right.t - left.t || 1;
			const localT = (t - left.t) / span;
			return {
				r: left.value.r + (right.value.r - left.value.r) * localT,
				g: left.value.g + (right.value.g - left.value.g) * localT,
				b: left.value.b + (right.value.b - left.value.b) * localT,
				a: left.value.a + (right.value.a - left.value.a) * localT,
			};
		}

		return {
			r: fallback.r,
			g: fallback.g,
			b: fallback.b,
			a: fallback.a,
		};
	}

	private _randomDirectionInCone(
		runtime: SystemRuntimeState,
		direction: IVector3,
		spread: number
	): IVector3 {
		const base = normalizeVector(direction);
		if (spread <= 0) return base;

		const random = this._randomUnitVector(runtime);
		const spreadScale = Math.tan(spread);
		return normalizeVector({
			x: base.x + random.x * spreadScale,
			y: base.y + random.y * spreadScale,
			z: base.z + random.z * spreadScale,
		});
	}

	private _randomSpawnOffset(
		runtime: SystemRuntimeState,
		radius: number
	): IVector3 {
		if (radius <= 0) return { x: 0, y: 0, z: 0 };
		const direction = this._randomUnitVector(runtime);
		const distance = Math.cbrt(this._nextRandom(runtime)) * radius;
		return {
			x: direction.x * distance,
			y: direction.y * distance,
			z: direction.z * distance,
		};
	}

	private _randomUnitVector(runtime: SystemRuntimeState): IVector3 {
		const u = this._nextRandom(runtime) * 2 - 1;
		const theta = this._nextRandom(runtime) * Math.PI * 2;
		const r = Math.sqrt(Math.max(0, 1 - u * u));
		return {
			x: r * Math.cos(theta),
			y: u,
			z: r * Math.sin(theta),
		};
	}

	private _randomRange(
		runtime: SystemRuntimeState,
		min: number,
		max: number
	): number {
		const lo = Math.min(min, max);
		const hi = Math.max(min, max);
		return lo + (hi - lo) * this._nextRandom(runtime);
	}

	private _nextRandom(runtime: SystemRuntimeState): number {
		runtime.randomState = (runtime.randomState * 1664525 + 1013904223) >>> 0;
		return runtime.randomState / 0x100000000;
	}

	private _getSystemRuntime(system: ParticleSystem): SystemRuntimeState {
		let runtime = this._runtimeBySystemId.get(system.id);
		if (runtime) return runtime;

		runtime = {
			particles: [],
			emissionRemainder: 0,
			elapsed: 0,
			burstCycles: [],
			randomState: system.seed >>> 0 || 1,
		};
		this._runtimeBySystemId.set(system.id, runtime);
		return runtime;
	}
}

function clamp01(value: number): number {
	return Math.max(0, Math.min(1, value));
}

function normalizeVector(source: IVector3): IVector3 {
	const length = Math.hypot(source.x, source.y, source.z) || 1;
	return {
		x: source.x / length,
		y: source.y / length,
		z: source.z / length,
	};
}

function cloneColor(source: RGBA): RGBA {
	return {
		r: source.r,
		g: source.g,
		b: source.b,
		a: source.a,
	};
}
