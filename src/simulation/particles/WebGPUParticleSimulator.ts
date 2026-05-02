import { Logger } from "../../foundation/Logger";
import {
	PARTICLE_TRANSIENT_BATCHES_KEY,
	type FrameContext,
	type ParticleRenderBatch,
} from "../../pipeline/types";
import { DefaultParticleSimulator } from "./DefaultParticleSimulator";
import type { IParticleSimulator } from "./IParticleSimulator";
import {
	BufferUsage,
	type BufferDesc,
	type IRenderBuffer,
} from "../../renderers/types";
import {
	WEBGPU_PARTICLE_INSTANCE_FLOATS,
	WEBGPU_PARTICLE_INSTANCE_STRIDE,
} from "../../renderers/webgpu/particleLayout";
import {
	WEBGPU_PARTICLE_DRAW_BATCHES_KEY,
	type WebGPUParticleDrawBatch,
} from "../../renderers/webgpu/particleTransient";

interface WebGPUParticleSimulatorBackend {
	createBuffer(desc: BufferDesc): IRenderBuffer;
	writeBuffer(
		buffer: IRenderBuffer,
		data: BufferSource,
		offset?: number
	): void;
}

interface WebGPUParticleSimulatorOptions {
	backend: WebGPUParticleSimulatorBackend;
	backendTag?: string;
	maxParticlesPerSystem?: number;
}

interface SystemGPUState {
	instanceBuffer: IRenderBuffer;
	instanceCapacity: number;
	indirectBuffer: IRenderBuffer;
}

export class WebGPUParticleSimulator implements IParticleSimulator {
	private _backend: WebGPUParticleSimulatorBackend;
	private _cpuSimulator: DefaultParticleSimulator;
	private _gpuStateBySystemId = new Map<string, SystemGPUState>();
	private _warnedUploadFailure = false;

	constructor(options: WebGPUParticleSimulatorOptions) {
		this._backend = options.backend;
		this._cpuSimulator = new DefaultParticleSimulator({
			backendTag: options.backendTag ?? "webgpu-gpu-sim",
			maxParticlesPerSystem: options.maxParticlesPerSystem,
		});
	}

	public beginFrame(context: FrameContext): void {
		this._cpuSimulator.beginFrame(context);
		const activeIds = new Set(
			(context.scene.particleSystems ?? []).map((system) => system.id)
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
		this._cpuSimulator.simulate(context, deltaTimeSeconds);
	}

	public emitRenderBatches(context: FrameContext): void {
		this._cpuSimulator.emitRenderBatches(context);
		const batches =
			context.transient.get(PARTICLE_TRANSIENT_BATCHES_KEY) ?? ([] as const);
		if (batches.length <= 0) {
			context.transient.set(WEBGPU_PARTICLE_DRAW_BATCHES_KEY, []);
			return;
		}

		try {
			const drawBatches = this._buildDrawBatches(batches);
			context.transient.set(WEBGPU_PARTICLE_DRAW_BATCHES_KEY, drawBatches);
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
			context.transient.set(WEBGPU_PARTICLE_DRAW_BATCHES_KEY, []);
		}
	}

	public endFrame(): void {
		this._cpuSimulator.endFrame();
	}

	public destroy(): void {
		for (const state of this._gpuStateBySystemId.values()) {
			this._destroyState(state);
		}
		this._gpuStateBySystemId.clear();
	}

	private _buildDrawBatches(
		batches: readonly ParticleRenderBatch[]
	): WebGPUParticleDrawBatch[] {
		const drawBatches: WebGPUParticleDrawBatch[] = [];

		for (const batch of batches) {
			const instanceCount = batch.particles.length;
			if (instanceCount <= 0) {
				continue;
			}
			const state = this._ensureSystemState(batch.systemId, instanceCount);
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
				blendMode: batch.blendMode,
				texture: batch.texture,
				receiveShadows: batch.receiveShadows,
				instanceBuffer: state.instanceBuffer,
				instanceCount,
				indirectBuffer: state.indirectBuffer,
				indirectOffset: 0,
			});
		}

		return drawBatches;
	}

	private _ensureSystemState(
		systemId: string,
		instanceCount: number
	): SystemGPUState {
		const existing = this._gpuStateBySystemId.get(systemId);
		if (existing && instanceCount <= existing.instanceCapacity) {
			return existing;
		}
		const capacity = this._resolveNextCapacity(instanceCount);
		const nextState: SystemGPUState = {
			instanceBuffer: this._backend.createBuffer({
				size: capacity * WEBGPU_PARTICLE_INSTANCE_STRIDE,
				usage: BufferUsage.Vertex | BufferUsage.CopyDst,
				label: `WebGPUParticleInstances_${systemId}`,
			}),
			instanceCapacity: capacity,
			indirectBuffer:
				existing?.indirectBuffer ??
				this._backend.createBuffer({
					size: Uint32Array.BYTES_PER_ELEMENT * 4,
					usage: BufferUsage.CopyDst | BufferUsage.Indirect,
					label: `WebGPUParticleIndirect_${systemId}`,
				}),
		};
		if (existing) {
			existing.instanceBuffer.destroy();
		}
		this._gpuStateBySystemId.set(systemId, nextState);
		return nextState;
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

	private _destroyState(state: SystemGPUState): void {
		state.instanceBuffer.destroy();
		state.indirectBuffer.destroy();
	}
}

function clamp01(value: number): number {
	if (value <= 0) return 0;
	if (value >= 1) return 1;
	return value;
}
