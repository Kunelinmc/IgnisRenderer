import { Logger } from "../../foundation/Logger";
import {
	PARTICLE_TRANSIENT_BATCHES_KEY,
	type FrameContext,
	type ParticleRenderBatch,
} from "../../pipeline/types";
import {
	ParticleBlendMode,
	ParticleSpaceMode,
	type ParticleSystem,
} from "../../particles";
import type { IComputeKernel, IComputeRuntime } from "../../backends/IComputeRuntime";
import {
	BufferUsage,
	type BufferDesc,
	type IRenderBuffer,
} from "../../backends/types";
import { ComputeRuntime } from "../../backends/webgpu/ComputeRuntime";
import type { IWebGPUComputeFacade } from "../../backends/webgpu/ComputeFacade";
import {
	WEBGPU_PARTICLE_INSTANCE_FLOATS,
	WEBGPU_PARTICLE_INSTANCE_STRIDE,
} from "../../backends/webgpu/constants";
import {
	WEBGPU_PARTICLE_DRAW_BATCHES_KEY,
	type WebGPUParticleDrawBatch,
} from "../../backends/webgpu/types";
import { ShaderSource } from "../../shaders/ShaderSource";
import { DefaultParticleSimulator } from "./DefaultParticleSimulator";
import type { IParticleSimulator } from "./IParticleSimulator";
import { MAX_STEP_SECONDS } from "./constants";

interface WebGPUParticleSimulatorOptions {
	backend: IWebGPUComputeFacade;
	backendTag?: string;
	computeRuntime?: IComputeRuntime;
	maxParticlesPerSystem?: number;
}

interface ParticleComputeKernels {
	reset: IComputeKernel;
	spawn: IComputeKernel;
	simulate: IComputeKernel;
}

interface SystemGPUState {
	mode: "compute" | "upload";
	stateBuffer: IRenderBuffer;
	instanceBuffer: IRenderBuffer;
	instanceCapacity: number;
	indirectBuffer: IRenderBuffer;
	paramsBuffer: IRenderBuffer;
	spawnCursor: number;
	emissionRemainder: number;
	elapsed: number;
	burstCycles: number[];
	randomState: number;
	frameIndex: number;
	hasSpawned: boolean;
}

interface ParticleSystemPartition {
	compute: ParticleSystem[];
	cpu: ParticleSystem[];
}

const PARTICLE_STATE_STRIDE = 64;
const PARTICLE_SIM_PARAMS_SIZE = 176;
const PARTICLE_SIM_WORKGROUP_SIZE = 64;
const GPU_PARTICLE_SPACE_LOCAL = 0;
const GPU_PARTICLE_SPACE_WORLD = 1;

export class WebGPUParticleSimulator implements IParticleSimulator {
	private _backend: IWebGPUComputeFacade;
	private _cpuSimulator: DefaultParticleSimulator;
	private _computeRuntime: IComputeRuntime | null = null;
	private _ownsComputeRuntime = false;
	private _kernels: ParticleComputeKernels | null = null;
	private _kernelPromise: Promise<ParticleComputeKernels> | null = null;
	private _gpuStateBySystemId = new Map<string, SystemGPUState>();
	private _computeSystemsThisFrame: ParticleSystem[] = [];
	private _cpuSystemsThisFrame: ParticleSystem[] = [];
	private _computeDrawBatchesThisFrame: WebGPUParticleDrawBatch[] = [];
	private _syncCpuContext: FrameContext | null = null;
	private _syncCpuAllFrame = false;
	private _warnedUploadFailure = false;
	private _warnedComputeFailure = false;
	private _computeDisabled = false;
	private _maxParticlesPerSystem: number;

	constructor(options: WebGPUParticleSimulatorOptions) {
		this._backend = options.backend;
		this._cpuSimulator = new DefaultParticleSimulator({
			backendTag: options.backendTag ?? "webgpu-gpu-sim",
			maxParticlesPerSystem: options.maxParticlesPerSystem,
		});
		this._maxParticlesPerSystem = Math.max(
			1,
			Math.floor(options.maxParticlesPerSystem ?? Number.POSITIVE_INFINITY)
		);
		if (options.computeRuntime) {
			this._computeRuntime = options.computeRuntime;
		} else {
			try {
				this._computeRuntime = new ComputeRuntime(options.backend);
				this._ownsComputeRuntime = true;
			} catch {
				this._computeRuntime = null;
			}
		}
	}

	public beginFrame(context: FrameContext): void {
		const systems = context.scene.particleSystems ?? [];
		for (const system of systems) {
			system.updateWorldMatrix(system.parent?.worldMatrix);
		}

		const partition = this._partitionSystems(context, systems);
		this._computeSystemsThisFrame = partition.compute;
		this._cpuSystemsThisFrame = partition.cpu;
		this._computeDrawBatchesThisFrame = [];
		this._syncCpuContext = null;
		this._syncCpuAllFrame = false;

		this._cpuSimulator.beginFrame(
			this._withParticleSystems(context, this._cpuSystemsThisFrame)
		);

		const activeIds = new Set(
			[...this._computeSystemsThisFrame, ...this._cpuSystemsThisFrame].map(
				(system) => system.id
			)
		);
		for (const [systemId, state] of this._gpuStateBySystemId.entries()) {
			if (activeIds.has(systemId)) {
				continue;
			}
			this._destroyState(state);
			this._gpuStateBySystemId.delete(systemId);
		}
	}

	public simulate(context: FrameContext, deltaTimeSeconds: number): void {
		this._syncCpuContext = this._withParticleSystems(
			context,
			context.scene.particleSystems ?? []
		);
		this._cpuSimulator.beginFrame(this._syncCpuContext);
		this._cpuSimulator.simulate(this._syncCpuContext, deltaTimeSeconds);
		this._computeDrawBatchesThisFrame = [];
		this._syncCpuAllFrame = true;
	}

	public async simulateAndEmitRenderBatches(
		context: FrameContext,
		deltaTimeSeconds: number
	): Promise<void> {
		if (this._computeSystemsThisFrame.length > 0) {
			try {
				await this._simulateComputeSystems(context, deltaTimeSeconds);
			} catch (error) {
				this._handleComputeFailure(error);
				this._computeSystemsThisFrame = [];
				this._cpuSystemsThisFrame = context.scene.particleSystems ?? [];
				const cpuContext = this._withParticleSystems(
					context,
					this._cpuSystemsThisFrame
				);
				this._cpuSimulator.beginFrame(cpuContext);
				this._cpuSimulator.simulate(cpuContext, deltaTimeSeconds);
				this._cpuSimulator.emitRenderBatches(cpuContext);
				this._emitDrawBatches(context);
				return;
			}
		}

		const cpuContext = this._withParticleSystems(
			context,
			this._cpuSystemsThisFrame
		);
		this._cpuSimulator.simulate(cpuContext, deltaTimeSeconds);
		this._cpuSimulator.emitRenderBatches(cpuContext);
		this._emitDrawBatches(context);
	}

	public emitRenderBatches(context: FrameContext): void {
		if (this._syncCpuAllFrame) {
			this._cpuSimulator.emitRenderBatches(this._syncCpuContext ?? context);
			this._emitDrawBatches(context);
			this._syncCpuAllFrame = false;
			return;
		}

		const cpuContext = this._withParticleSystems(
			context,
			this._cpuSystemsThisFrame
		);
		this._cpuSimulator.emitRenderBatches(cpuContext);
		this._emitDrawBatches(context);
	}

	public endFrame(): void {
		this._cpuSimulator.endFrame();
		this._computeDrawBatchesThisFrame = [];
		this._syncCpuContext = null;
		this._syncCpuAllFrame = false;
	}

	public destroy(): void {
		for (const state of this._gpuStateBySystemId.values()) {
			this._destroyState(state);
		}
		this._gpuStateBySystemId.clear();
		if (this._kernels) {
			this._kernels.reset.destroy();
			this._kernels.spawn.destroy();
			this._kernels.simulate.destroy();
			this._kernels = null;
		}
		if (this._ownsComputeRuntime) {
			this._computeRuntime?.destroy();
		}
		this._computeRuntime = null;
	}

	private _partitionSystems(
		context: FrameContext,
		systems: readonly ParticleSystem[]
	): ParticleSystemPartition {
		const compute: ParticleSystem[] = [];
		const cpu: ParticleSystem[] = [];
		for (const system of systems) {
			if (this._canUseComputeSimulation(context, system)) {
				compute.push(system);
			} else {
				cpu.push(system);
			}
		}
		return { compute, cpu };
	}

	private _canUseComputeSimulation(
		_context: FrameContext,
		system: ParticleSystem
	): boolean {
		if (this._computeDisabled || !this._computeRuntime) {
			return false;
		}
		if (system.blendMode !== ParticleBlendMode.Additive) {
			return false;
		}
		if (system.templates.length !== 1) {
			return false;
		}
		const template = system.templates[0];
		if (!template || template.shape.kind !== "billboard") {
			return false;
		}
		if (system.lod?.enabled) {
			return false;
		}
		if ((system.colliders?.length ?? 0) > 0) {
			return false;
		}
		if (system.subEmitter && system.subEmitter.enabled !== false) {
			return false;
		}
		if ((template.sizeOverLifetime?.length ?? 0) > 0) {
			return false;
		}
		if ((template.colorOverLifetime?.length ?? 0) > 0) {
			return false;
		}
		return true;
	}

	private async _simulateComputeSystems(
		context: FrameContext,
		deltaTimeSeconds: number
	): Promise<void> {
		const runtime = this._computeRuntime;
		if (!runtime) {
			return;
		}

		const kernels = await this._ensureKernels();
		const totalDt = Math.max(0, deltaTimeSeconds);
		this._computeDrawBatchesThisFrame = [];

		for (const system of this._computeSystemsThisFrame) {
			system.updateWorldMatrix(system.parent?.worldMatrix);
			const maxParticles = this._resolveEffectiveMaxParticles(system);
			const state = this._ensureSystemState(system, maxParticles);
			state.frameIndex++;

			if (totalDt <= 0) {
				if (state.hasSpawned) {
					this._dispatchComputeStep(runtime, kernels, system, state, 0);
				}
			} else {
				let stepBudget = totalDt;
				while (stepBudget > 0) {
					const stepDt = Math.min(MAX_STEP_SECONDS, stepBudget);
					this._dispatchComputeStep(
						runtime,
						kernels,
						system,
						state,
						stepDt
					);
					stepBudget -= stepDt;
				}
			}

			if (!system.visible || !state.hasSpawned) {
				continue;
			}

			const template = system.templates[0];
			const shape = template?.shape;
			this._computeDrawBatchesThisFrame.push({
				systemId: system.id,
				templateIndex: 0,
				templateId: template?.id,
				blendMode:
					shape?.kind === "billboard" ?
						shape.blendMode ?? ParticleBlendMode.Alpha
					:	ParticleBlendMode.Alpha,
				texture: shape?.kind === "billboard" ? shape.texture ?? null : null,
				receiveShadows: template?.receiveShadows ?? true,
				castShadows: false,
				shadowDensity: Math.max(0, template?.shadowDensity ?? 1),
				shadowSoftness: Math.max(0, template?.shadowSoftness ?? 1),
				instanceBuffer: state.instanceBuffer,
				instanceCount: maxParticles,
				indirectBuffer: state.indirectBuffer,
				indirectOffset: 0,
			});
		}
	}

	private _dispatchComputeStep(
		runtime: IComputeRuntime,
		kernels: ParticleComputeKernels,
		system: ParticleSystem,
		state: SystemGPUState,
		deltaTimeSeconds: number
	): void {
		const maxParticles = this._resolveEffectiveMaxParticles(system);
		const spawnCount = Math.min(
			maxParticles,
			this._resolveSpawnCount(system, state, deltaTimeSeconds)
		);
		const spawnStart = state.spawnCursor % maxParticles;
		if (spawnCount > 0) {
			state.spawnCursor = (spawnStart + spawnCount) % maxParticles;
			state.hasSpawned = true;
		}

		state.randomState = nextRandomState(state.randomState);
		const params = this._createParamsBufferData({
			system,
			state,
			deltaTimeSeconds,
			spawnCount,
			spawnStart,
			maxParticles,
		});
		runtime.writeBuffer(state.paramsBuffer, params);

		kernels.reset.dispatch({
			label: `WebGPUParticleReset_${system.id}`,
			resources: {
				indirect: state.indirectBuffer,
			},
			dispatch: { x: 1 },
		});

		if (spawnCount > 0) {
			kernels.spawn.dispatch({
				label: `WebGPUParticleSpawn_${system.id}`,
				resources: {
					particles: state.stateBuffer,
					params: state.paramsBuffer,
				},
				dispatch2D: {
					width: spawnCount,
					height: 1,
				},
			});
		}

		if (state.hasSpawned) {
			kernels.simulate.dispatch({
				label: `WebGPUParticleSimulate_${system.id}`,
				resources: {
					particles: state.stateBuffer,
					instances: state.instanceBuffer,
					indirect: state.indirectBuffer,
					params: state.paramsBuffer,
				},
				dispatch2D: {
					width: maxParticles,
					height: 1,
				},
			});
		}
	}

	private async _ensureKernels(): Promise<ParticleComputeKernels> {
		if (this._kernels) {
			return this._kernels;
		}
		if (this._kernelPromise) {
			return this._kernelPromise;
		}
		const runtime = this._computeRuntime;
		if (!runtime) {
			throw new Error("WebGPU particle compute runtime is unavailable.");
		}

		this._kernelPromise = (async () => {
			const code = await ShaderSource.load("webgpu.particleSimulation.raw");
			let reset: IComputeKernel | null = null;
			let spawn: IComputeKernel | null = null;
			try {
				reset = await runtime.createKernel({
					label: "WebGPUParticleReset",
					code,
					entryPoint: "resetMain",
					language: "wgsl",
					sourceKind: "particle",
					bindings: [
						{ key: "indirect", binding: 2, type: "buffer" },
					],
					workgroupSize: { x: 1 },
				});
				spawn = await runtime.createKernel({
					label: "WebGPUParticleSpawn",
					code,
					entryPoint: "spawnMain",
					language: "wgsl",
					sourceKind: "particle",
					bindings: [
						{ key: "particles", binding: 0, type: "buffer" },
						{ key: "params", binding: 3, type: "buffer" },
					],
					workgroupSize: { x: PARTICLE_SIM_WORKGROUP_SIZE },
				});
				const simulate = await runtime.createKernel({
					label: "WebGPUParticleSimulate",
					code,
					entryPoint: "simulateMain",
					language: "wgsl",
					sourceKind: "particle",
					bindings: [
						{ key: "particles", binding: 0, type: "buffer" },
						{ key: "instances", binding: 1, type: "buffer" },
						{ key: "indirect", binding: 2, type: "buffer" },
						{ key: "params", binding: 3, type: "buffer" },
					],
					workgroupSize: { x: PARTICLE_SIM_WORKGROUP_SIZE },
				});
				this._kernels = { reset, spawn, simulate };
				return this._kernels;
			} catch (error) {
				spawn?.destroy();
				reset?.destroy();
				throw error;
			}
		})();

		try {
			return await this._kernelPromise;
		} finally {
			this._kernelPromise = null;
		}
	}

	private _resolveSpawnCount(
		system: ParticleSystem,
		state: SystemGPUState,
		deltaTimeSeconds: number
	): number {
		if (deltaTimeSeconds <= 0) {
			return 0;
		}

		const emit = system.emit ?? {};
		state.elapsed += deltaTimeSeconds;
		const rate = Math.max(0, emit.rate ?? 0);
		state.emissionRemainder += rate * deltaTimeSeconds;

		let spawnCount = Math.floor(state.emissionRemainder);
		if (spawnCount > 0) {
			state.emissionRemainder -= spawnCount;
		}

		const bursts = emit.bursts ?? [];
		for (let i = 0; i < bursts.length; i++) {
			const burst = bursts[i];
			const cycles = Math.max(1, burst.cycles ?? 1);
			const interval = Math.max(0, burst.interval ?? 0);
			let firedCycles = state.burstCycles[i] ?? 0;

			while (firedCycles < cycles) {
				const triggerAt = burst.time + interval * firedCycles;
				if (state.elapsed < triggerAt) {
					break;
				}
				spawnCount += Math.max(0, Math.floor(burst.count));
				firedCycles++;
			}

			state.burstCycles[i] = firedCycles;
		}

		return Math.max(0, spawnCount);
	}

	private _emitDrawBatches(context: FrameContext): void {
		const cpuBatches =
			context.transient.get(PARTICLE_TRANSIENT_BATCHES_KEY) ?? ([] as const);
		try {
			const uploadedBatches = this._buildUploadedDrawBatches(cpuBatches);
			context.transient.set(WEBGPU_PARTICLE_DRAW_BATCHES_KEY, [
				...this._computeDrawBatchesThisFrame,
				...uploadedBatches,
			]);
		} catch (error) {
			if (!this._warnedUploadFailure) {
				this._warnedUploadFailure = true;
				Logger.warn(
					`[webgpu-particle-gpu-upload-failed] Falling back to CPU particle batches because GPU upload failed: ${String(error)}`,
					{
						scope: "WebGPUParticleSimulator",
						onceKey: "webgpu-particle-gpu-upload-failed",
					}
				);
			}
			context.transient.set(
				WEBGPU_PARTICLE_DRAW_BATCHES_KEY,
				this._computeDrawBatchesThisFrame.slice()
			);
		}
	}

	private _buildUploadedDrawBatches(
		batches: readonly ParticleRenderBatch[]
	): WebGPUParticleDrawBatch[] {
		const drawBatches: WebGPUParticleDrawBatch[] = [];

		for (const batch of batches) {
			const instanceCount = batch.particles.length;
			if (instanceCount <= 0) {
				continue;
			}
			const state = this._ensureUploadState(batch.systemId, instanceCount);
			const instanceData = new Float32Array(
				instanceCount * WEBGPU_PARTICLE_INSTANCE_FLOATS
			);
			let particleOffset = 0;
			for (const particle of batch.particles) {
				const offset = particleOffset * WEBGPU_PARTICLE_INSTANCE_FLOATS;
				instanceData[offset] = particle.position.x;
				instanceData[offset + 1] = particle.position.y;
				instanceData[offset + 2] = particle.position.z;
				instanceData[offset + 3] = Math.max(0.001, particle.size);
				instanceData[offset + 4] = particle.color.r / 255;
				instanceData[offset + 5] = particle.color.g / 255;
				instanceData[offset + 6] = particle.color.b / 255;
				instanceData[offset + 7] = clamp01(particle.color.a);
				instanceData[offset + 8] = particle.uvRect.u0;
				instanceData[offset + 9] = particle.uvRect.v0;
				instanceData[offset + 10] = particle.uvRect.u1;
				instanceData[offset + 11] = particle.uvRect.v1;
				instanceData[offset + 12] = particle.rotation;
				instanceData[offset + 13] = batch.receiveShadows ? 1 : 0;
				instanceData[offset + 14] = 0;
				instanceData[offset + 15] = 0;
				particleOffset++;
			}

			this._backend.writeBuffer(state.instanceBuffer, instanceData);
			const indirectArgs = new Uint32Array([6, instanceCount, 0, 0]);
			this._backend.writeBuffer(state.indirectBuffer, indirectArgs);
			drawBatches.push({
				systemId: batch.systemId,
				templateIndex: batch.templateIndex,
				templateId: batch.templateId,
				blendMode: batch.blendMode,
				texture: batch.texture,
				receiveShadows: batch.receiveShadows,
				castShadows: batch.castShadows,
				shadowDensity: batch.shadowDensity,
				shadowSoftness: batch.shadowSoftness,
				instanceBuffer: state.instanceBuffer,
				instanceCount,
				indirectBuffer: state.indirectBuffer,
				indirectOffset: 0,
			});
		}

		return drawBatches;
	}

	private _ensureSystemState(
		system: ParticleSystem,
		maxParticles: number
	): SystemGPUState {
		const systemId = system.id;
		const existing = this._gpuStateBySystemId.get(systemId);
		const capacity = this._resolveNextCapacity(maxParticles);
		if (
			existing &&
			existing.mode === "compute" &&
			capacity <= existing.instanceCapacity
		) {
			return existing;
		}

		if (existing) {
			this._destroyState(existing);
		}

		const runtime = this._computeRuntime;
		if (!runtime) {
			throw new Error("WebGPU particle compute runtime is unavailable.");
		}

		const state: SystemGPUState = {
			mode: "compute",
			stateBuffer: runtime.createBuffer({
				size: capacity * PARTICLE_STATE_STRIDE,
				usage: BufferUsage.Storage | BufferUsage.CopyDst,
				label: `WebGPUParticleState_${systemId}`,
			}),
			instanceBuffer: runtime.createBuffer({
				size: capacity * WEBGPU_PARTICLE_INSTANCE_STRIDE,
				usage: BufferUsage.Vertex | BufferUsage.Storage | BufferUsage.CopyDst,
				label: `WebGPUParticleInstances_${systemId}`,
			}),
			instanceCapacity: capacity,
			indirectBuffer: runtime.createBuffer({
				size: Uint32Array.BYTES_PER_ELEMENT * 4,
				usage: BufferUsage.Storage | BufferUsage.Indirect | BufferUsage.CopyDst,
				label: `WebGPUParticleIndirect_${systemId}`,
			}),
			paramsBuffer: runtime.createBuffer({
				size: PARTICLE_SIM_PARAMS_SIZE,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: `WebGPUParticleParams_${systemId}`,
			}),
			spawnCursor: 0,
			emissionRemainder: 0,
			elapsed: 0,
			burstCycles: [],
			randomState: system.seed >>> 0 || 1,
			frameIndex: 0,
			hasSpawned: false,
		};
		this._gpuStateBySystemId.set(systemId, state);
		return state;
	}

	private _ensureUploadState(
		systemId: string,
		instanceCount: number
	): SystemGPUState {
		const existing = this._gpuStateBySystemId.get(systemId);
		if (
			existing &&
			existing.mode === "upload" &&
			instanceCount <= existing.instanceCapacity
		) {
			return existing;
		}

		if (existing && existing.mode !== "upload") {
			this._destroyState(existing);
		}

		const capacity = this._resolveNextCapacity(instanceCount);
		const reusable = existing?.mode === "upload" ? existing : null;
		const nextState: SystemGPUState = {
			mode: "upload",
			stateBuffer:
				reusable?.stateBuffer ??
				this._backend.createBuffer({
					size: PARTICLE_STATE_STRIDE,
					usage: BufferUsage.Storage | BufferUsage.CopyDst,
					label: `WebGPUParticleState_${systemId}`,
				}),
			instanceBuffer: this._backend.createBuffer({
				size: capacity * WEBGPU_PARTICLE_INSTANCE_STRIDE,
				usage: BufferUsage.Vertex | BufferUsage.CopyDst,
				label: `WebGPUParticleInstances_${systemId}`,
			}),
			instanceCapacity: capacity,
			indirectBuffer:
				reusable?.indirectBuffer ??
				this._backend.createBuffer({
					size: Uint32Array.BYTES_PER_ELEMENT * 4,
					usage: BufferUsage.CopyDst | BufferUsage.Indirect,
					label: `WebGPUParticleIndirect_${systemId}`,
				}),
			paramsBuffer:
				reusable?.paramsBuffer ??
				this._backend.createBuffer({
					size: PARTICLE_SIM_PARAMS_SIZE,
					usage: BufferUsage.Uniform | BufferUsage.CopyDst,
					label: `WebGPUParticleParams_${systemId}`,
				}),
			spawnCursor: reusable?.spawnCursor ?? 0,
			emissionRemainder: reusable?.emissionRemainder ?? 0,
			elapsed: reusable?.elapsed ?? 0,
			burstCycles: reusable?.burstCycles ?? [],
			randomState: reusable?.randomState ?? 1,
			frameIndex: reusable?.frameIndex ?? 0,
			hasSpawned: reusable?.hasSpawned ?? false,
		};
		if (reusable) {
			reusable.instanceBuffer.destroy();
		}
		this._gpuStateBySystemId.set(systemId, nextState);
		return nextState;
	}

	private _resolveEffectiveMaxParticles(system: ParticleSystem): number {
		return Math.max(
			1,
			Math.min(
				Math.floor(system.maxParticles),
				this._maxParticlesPerSystem
			)
		);
	}

	private _resolveNextCapacity(targetCount: number): number {
		const resolved =
			Number.isFinite(targetCount) ? Math.max(1, Math.floor(targetCount)) : 1;
		const maxCapacity = Math.max(
			256,
			Math.floor(Number.MAX_SAFE_INTEGER / WEBGPU_PARTICLE_INSTANCE_STRIDE)
		);
		if (resolved > maxCapacity) {
			throw new Error(
				`Particle instance request ${resolved} exceeds max capacity ${maxCapacity}.`
			);
		}
		const exponent = Math.ceil(Math.log2(resolved));
		const powerOfTwo = Math.pow(2, exponent);
		return Math.max(
			256,
			Math.min(maxCapacity, Number.isFinite(powerOfTwo) ? powerOfTwo : 256)
		);
	}

	private _createParamsBufferData(input: {
		system: ParticleSystem;
		state: SystemGPUState;
		deltaTimeSeconds: number;
		spawnCount: number;
		spawnStart: number;
		maxParticles: number;
	}): ArrayBuffer {
		const { system, state } = input;
		const emit = system.emit ?? {};
		const worldPosition = system.getWorldPosition();
		const template = system.templates[0];
		const lifetimeRange = normalizeRange(
			template?.lifetimeRange ?? [0.5, 1.5]
		);
		const speedRange = normalizeRange(template?.speedRange ?? [2, 5]);
		const sizeRange = normalizeRange(template?.sizeRange ?? [0.5, 1]);
		const rotationRange = normalizeRange(
			template?.rotationRange ?? [0, 0]
		);
		const angularVelocityRange = normalizeRange(
			template?.angularVelocityRange ?? [0, 0]
		);
		const direction = emit.direction ?? { x: 0, y: 1, z: 0 };
		const shape = template?.shape;
		const atlas = shape?.kind === "billboard" ? shape.atlas : null;
		const buffer = new ArrayBuffer(PARTICLE_SIM_PARAMS_SIZE);
		const view = new DataView(buffer);

		view.setFloat32(0, Math.max(0, input.deltaTimeSeconds), true);
		view.setFloat32(4, state.elapsed, true);
		view.setUint32(8, Math.max(0, input.spawnCount >>> 0), true);
		view.setUint32(12, Math.max(1, input.maxParticles >>> 0), true);
		view.setUint32(16, Math.max(0, input.spawnStart >>> 0), true);
		view.setUint32(20, state.randomState >>> 0, true);
		view.setUint32(
			24,
			system.space === ParticleSpaceMode.World ?
				GPU_PARTICLE_SPACE_WORLD
			:	GPU_PARTICLE_SPACE_LOCAL,
			true
		);
		view.setUint32(
			28,
			(template?.receiveShadows ?? true) ? 1 : 0,
			true
		);

		writeVec4(view, 32, system.gravity.x, system.gravity.y, system.gravity.z, 0);
		writeVec4(view, 48, worldPosition.x, worldPosition.y, worldPosition.z, 0);
		writeVec4(
			view,
			64,
			system.space === ParticleSpaceMode.World ? worldPosition.x : 0,
			system.space === ParticleSpaceMode.World ? worldPosition.y : 0,
			system.space === ParticleSpaceMode.World ? worldPosition.z : 0,
			Math.max(0, emit.spawnRadius ?? 0)
		);
		writeVec4(
			view,
			80,
			finiteOr(direction.x, 0),
			finiteOr(direction.y, 1),
			finiteOr(direction.z, 0),
			Math.max(0, finiteOr(emit.spread, 0))
		);
		writeVec4(
			view,
			96,
			lifetimeRange[0],
			lifetimeRange[1],
			speedRange[0],
			speedRange[1]
		);
		writeVec4(
			view,
			112,
			sizeRange[0],
			sizeRange[1],
			rotationRange[0],
			rotationRange[1]
		);
		writeVec4(
			view,
			128,
			angularVelocityRange[0],
			angularVelocityRange[1],
			Math.max(1, Math.floor(atlas?.rows ?? 1)),
			Math.max(1, Math.floor(atlas?.columns ?? 1))
		);
		writeVec4(
			view,
			144,
			Math.max(0, atlas?.fps ?? 0),
			atlas?.loop === false ? 0 : 1,
			0,
			0
		);
		const startColor =
			template?.startColor ?? { r: 255, g: 255, b: 255, a: 1 };
		writeVec4(
			view,
			160,
			clamp01(startColor.r / 255),
			clamp01(startColor.g / 255),
			clamp01(startColor.b / 255),
			clamp01(startColor.a)
		);

		return buffer;
	}

	private _withParticleSystems(
		context: FrameContext,
		particleSystems: ParticleSystem[]
	): FrameContext {
		return {
			...context,
			scene: {
				...context.scene,
				particleSystems,
			},
		};
	}

	private _handleComputeFailure(error: unknown): void {
		this._computeDisabled = true;
		if (this._warnedComputeFailure) {
			return;
		}
		this._warnedComputeFailure = true;
		Logger.warn(
			`[webgpu-particle-compute-failed] Falling back to CPU particle simulation because GPU compute simulation failed: ${String(error)}`,
			{
				scope: "WebGPUParticleSimulator",
				onceKey: "webgpu-particle-compute-failed",
			}
		);
	}

	private _destroyState(state: SystemGPUState): void {
		state.stateBuffer.destroy();
		state.instanceBuffer.destroy();
		state.indirectBuffer.destroy();
		state.paramsBuffer.destroy();
	}
}

function writeVec4(
	view: DataView,
	byteOffset: number,
	x: number,
	y: number,
	z: number,
	w: number
): void {
	view.setFloat32(byteOffset, finiteOr(x, 0), true);
	view.setFloat32(byteOffset + 4, finiteOr(y, 0), true);
	view.setFloat32(byteOffset + 8, finiteOr(z, 0), true);
	view.setFloat32(byteOffset + 12, finiteOr(w, 0), true);
}

function normalizeRange(range: [number, number]): [number, number] {
	const a = finiteOr(range[0], 0);
	const b = finiteOr(range[1], a);
	return [Math.min(a, b), Math.max(a, b)];
}

function finiteOr(value: number | undefined, fallback: number): number {
	return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function clamp01(value: number): number {
	if (value <= 0) return 0;
	if (value >= 1) return 1;
	return value;
}

function nextRandomState(value: number): number {
	const seed = value >>> 0 || 1;
	return (seed * 1664525 + 1013904223) >>> 0;
}
