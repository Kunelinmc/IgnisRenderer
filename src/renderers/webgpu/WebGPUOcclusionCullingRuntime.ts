import type { FrameContext } from "../../pipeline/types";
import type {
	NormalizedOcclusionCullingOptions,
	OcclusionCandidate,
	OcclusionVisibilityProvider,
} from "../../pipeline/OcclusionCulling";
import type { ICommandEncoder } from "../ICommandEncoder";
import {
	BufferUsage,
	type IBindingGroup,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type IShaderModule,
} from "../types";
import type { WebGPUFrameHost } from "./rendergraph/WebGPUFrameHost";
import { tryGetNativeWebGPUCommandEncoder } from "./WebGPUCommandEncoder";
import {
	tryGetWebGPUBuffer,
} from "./WebGPUResourceAccess";
import { ShaderSource } from "../../shaders/ShaderSource";
import { Logger } from "../../foundation/Logger";
import {
	WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WORKGROUP_SIZE,
} from "./constants";
import { ceilDiv } from "../../maths/Misc";

interface VisibilityState {
	signatureA: number;
	signatureB: number;
	lastResultFrame: number;
	occludedStreak: number;
	visible: boolean;
}

interface PendingReadback {
	visibilityGeneration: number;
	frameIndex: number;
	packetIds: string[];
	signaturesA: number[];
	signaturesB: number[];
	options: NormalizedOcclusionCullingOptions;
	buffer: GPUBuffer;
	byteLength: number;
	queued: boolean;
	done: boolean;
}

const WEBGPU_MAP_MODE_READ =
	(globalThis as { GPUMapMode?: { READ?: number } }).GPUMapMode?.READ ?? 0x0001;
const CANDIDATE_FLOATS = 8;
const CANDIDATE_STRIDE_BYTES = CANDIDATE_FLOATS * 4;
const READBACK_ALIGNMENT = 4;

/**
 * Owns WebGPU previous-frame occlusion culling state and snapshots.
 */
export class WebGPUOcclusionCullingRuntime {
	private readonly _backend: WebGPUFrameHost;
	private _frameIndex = 0;
	private _lastCompletedFrameIndex = -1;
	private _visibilityGeneration = 0;
	private _visibilityByPacketId = new Map<string, VisibilityState>();
	private _pendingReadbacks: PendingReadback[] = [];
	private _candidateBuffer: IRenderBuffer | null = null;
	private _resultBuffer: IRenderBuffer | null = null;
	private _paramsBuffer: IRenderBuffer | null = null;
	private _occlusionModule: IShaderModule | null = null;
	private _occlusionPipeline: IComputePipeline | null = null;
	private _warnedKeys = new Set<string>();

	constructor(backend: WebGPUFrameHost) {
		this._backend = backend;
	}

	public beginFrame(context: FrameContext): void {
		this._frameIndex++;
		if (context.incremental?.temporalHistoryReset) {
			this.resetVisibility();
		}
		this._collectCompletedReadbacks();
	}

	public getVisibilityProvider(
		options: NormalizedOcclusionCullingOptions,
	): OcclusionVisibilityProvider {
		this._collectCompletedReadbacks();
		return {
			sourceFrameIndex: this._lastCompletedFrameIndex,
			isPacketVisible: (candidate) => this._isPacketVisible(candidate, options),
		};
	}

	public resetVisibility(): void {
		this._visibilityGeneration++;
		this._visibilityByPacketId.clear();
		for (const pending of this._pendingReadbacks) {
			this._destroyGPUBuffer(pending.buffer);
		}
		this._pendingReadbacks = [];
		this._lastCompletedFrameIndex = -1;
	}

	public invalidateFrameResources(): void {
	}

	public onShaderRuntimeChanged(): void {
		this._destroyPipeline(this._occlusionPipeline);
		this._destroyShaderModule(this._occlusionModule);
		this._occlusionPipeline = null;
		this._occlusionModule = null;
	}

	public destroy(): void {
		this.resetVisibility();
		this.invalidateFrameResources();
		this._destroyBuffer(this._candidateBuffer);
		this._destroyBuffer(this._resultBuffer);
		this._destroyBuffer(this._paramsBuffer);
		this._candidateBuffer = null;
		this._resultBuffer = null;
		this._paramsBuffer = null;
		this.onShaderRuntimeChanged();
	}

	public async recordVisibilityPass(request: {
		context: FrameContext;
		encoder: ICommandEncoder;
		hiZ: IRenderTexture | null | undefined;
		options: NormalizedOcclusionCullingOptions;
	}): Promise<void> {
		const occlusion = request.context.scene.occlusion;
		const candidates = occlusion?.candidates.filter((candidate) => candidate.eligible) ?? [];
		if (!request.hiZ || candidates.length === 0) {
			return;
		}
		try {
			await this._ensureResources(candidates.length);
		} catch (error) {
			this._warnOnce(
				"webgpu-occlusion-encode-failed",
				`WebGPU occlusion culling resources are unavailable; keeping current visibility snapshot. ${String(error)}`,
			);
			return;
		}
		if (
			!this._occlusionPipeline ||
			!this._candidateBuffer ||
			!this._resultBuffer ||
			!this._paramsBuffer
		) {
			this._warnOnce(
				"webgpu-occlusion-encode-failed",
				"WebGPU occlusion culling resources are incomplete; keeping current visibility snapshot.",
			);
			return;
		}
		try {
			this._uploadCandidates(candidates);
			this._backend.writeBuffer(this._paramsBuffer, new Float32Array([0.05, 0, 0, 0]));
			this._recordOcclusionCompute(request.encoder, candidates.length, request.hiZ);
			this._queueReadback(request.encoder, candidates, request.options);
		} catch (error) {
			this._warnOnce(
				"webgpu-occlusion-encode-failed",
				`WebGPU occlusion culling compute pass could not be encoded; keeping current visibility snapshot. ${String(error)}`,
			);
		}
	}

	public scheduleQueuedReadbacks(): void {
		for (const pending of this._pendingReadbacks) {
			if (pending.queued || pending.done) {
				continue;
			}
			pending.queued = true;
			void pending.buffer
				.mapAsync(WEBGPU_MAP_MODE_READ, 0, pending.byteLength)
				.then(() => {
					const mapped = pending.buffer.getMappedRange(0, pending.byteLength);
					this._applyReadback(pending, new Uint32Array(mapped.slice(0)));
				})
				.catch(() => {
					if (pending.visibilityGeneration !== this._visibilityGeneration) {
						pending.done = true;
						return;
					}
					this._warnOnce(
						"webgpu-occlusion-readback-failed",
						"WebGPU occlusion culling readback failed; keeping current visibility snapshot.",
					);
					pending.done = true;
				})
				.finally(() => {
					try {
						pending.buffer.unmap();
					} catch {
						// Ignore unmap races after device loss.
					}
					this._destroyGPUBuffer(pending.buffer);
					pending.done = true;
				});
		}
	}

	private _isPacketVisible(
		candidate: OcclusionCandidate,
		options: NormalizedOcclusionCullingOptions,
	): boolean {
		if (!candidate.eligible) {
			return true;
		}
		const state = this._visibilityByPacketId.get(candidate.packetId);
		if (!state) {
			return true;
		}
		if (
			state.signatureA !== candidate.signatureA ||
			state.signatureB !== candidate.signatureB
		) {
			return true;
		}
		if (this._frameIndex - state.lastResultFrame > options.maxReadbackLatencyFrames) {
			return true;
		}
		return state.visible;
	}

	private async _ensureResources(candidateCount: number): Promise<void> {
		await this._ensureOcclusionResources();
		this._ensureCandidateBuffers(candidateCount);
	}

	private async _ensureOcclusionResources(): Promise<void> {
		if (!this._occlusionModule) {
			const shader = await ShaderSource.load("webgpu.utility.occlusionCulling.composite");
			this._occlusionModule = await this._backend.createShaderModule({
				label: "WebGPUOcclusionCullingShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "unknown",
			});
		}
		if (!this._occlusionPipeline) {
			this._occlusionPipeline = await this._backend.createComputePipeline({
				label: "WebGPUOcclusionCullingPipeline",
				compute: {
					module: this._occlusionModule,
					entryPoint: "csMain",
				},
			});
		}
		if (!this._paramsBuffer) {
			this._paramsBuffer = this._backend.createBuffer({
				label: "WebGPUOcclusionCullingParams",
				size: 16,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
	}

	private _ensureCandidateBuffers(candidateCount: number): void {
		const candidateBytes = candidateCount * CANDIDATE_STRIDE_BYTES;
		const resultBytes = align(candidateCount * 4, READBACK_ALIGNMENT);
		if (!this._candidateBuffer || this._candidateBuffer.size < candidateBytes) {
			this._destroyBuffer(this._candidateBuffer);
			this._candidateBuffer = this._backend.createBuffer({
				label: "WebGPUOcclusionCandidates",
				size: Math.max(CANDIDATE_STRIDE_BYTES, candidateBytes),
				usage: BufferUsage.Storage | BufferUsage.CopyDst,
			});
		}
		if (!this._resultBuffer || this._resultBuffer.size < resultBytes) {
			this._destroyBuffer(this._resultBuffer);
			this._resultBuffer = this._backend.createBuffer({
				label: "WebGPUOcclusionResults",
				size: Math.max(4, resultBytes),
				usage: BufferUsage.Storage | BufferUsage.CopySrc | BufferUsage.CopyDst,
			});
		}
	}

	private _uploadCandidates(candidates: readonly OcclusionCandidate[]): void {
		if (!this._candidateBuffer) {
			return;
		}
		const data = new Float32Array(candidates.length * CANDIDATE_FLOATS);
		for (let index = 0; index < candidates.length; index++) {
			const candidate = candidates[index];
			const offset = index * CANDIDATE_FLOATS;
			data[offset + 0] = candidate.screenRect.x;
			data[offset + 1] = candidate.screenRect.y;
			data[offset + 2] = candidate.screenRect.width;
			data[offset + 3] = candidate.screenRect.height;
			data[offset + 4] = candidate.nearDepth;
			data[offset + 5] = candidate.farDepth;
			data[offset + 6] = candidate.screenAreaPx;
			data[offset + 7] = 0;
		}
		this._backend.writeBuffer(this._candidateBuffer, data);
	}

	private _recordOcclusionCompute(
		encoder: ICommandEncoder,
		candidateCount: number,
		hiZ: IRenderTexture,
	): void {
		if (
			!this._occlusionPipeline ||
			!this._candidateBuffer ||
			!this._paramsBuffer ||
			!this._resultBuffer
		) {
			return;
		}
		const binding = this._createBindingGroup(
			this._occlusionPipeline,
			[
				{ binding: 0, resource: hiZ },
				{ binding: 1, resource: this._candidateBuffer },
				{ binding: 2, resource: this._paramsBuffer },
				{ binding: 3, resource: this._resultBuffer },
			],
			"WebGPUOcclusionCullingBinding",
		);
		encoder.beginComputePass({ label: "WebGPUOcclusionCulling" });
		encoder.setComputePipeline(this._occlusionPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(ceilDiv(candidateCount, 64), 1, 1);
		encoder.endComputePass();
	}

	private _queueReadback(
		encoder: ICommandEncoder,
		candidates: readonly OcclusionCandidate[],
		options: NormalizedOcclusionCullingOptions,
	): void {
		if (!this._resultBuffer || !this._backend.device) {
			this._warnOnce(
				"webgpu-occlusion-readback-failed",
				"WebGPU occlusion culling cannot schedule readback without a result buffer and device.",
			);
			return;
		}
		const nativeEncoder = tryGetNativeWebGPUCommandEncoder(encoder);
		const sourceBuffer = tryGetWebGPUBuffer(this._resultBuffer);
		if (!nativeEncoder || !sourceBuffer) {
			this._warnOnce(
				"webgpu-occlusion-readback-failed",
				"WebGPU occlusion culling cannot access native buffers for readback.",
			);
			return;
		}
		const byteLength = align(candidates.length * 4, READBACK_ALIGNMENT);
		const readback = this._backend.device.createBuffer({
			label: "WebGPUOcclusionReadback",
			size: byteLength,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});
		nativeEncoder.copyBufferToBuffer(sourceBuffer, 0, readback, 0, byteLength);
		this._pendingReadbacks.push({
			visibilityGeneration: this._visibilityGeneration,
			frameIndex: this._frameIndex,
			packetIds: candidates.map((candidate) => candidate.packetId),
			signaturesA: candidates.map((candidate) => candidate.signatureA),
			signaturesB: candidates.map((candidate) => candidate.signatureB),
			options,
			buffer: readback,
			byteLength,
			queued: false,
			done: false,
		});
	}

	private _applyReadback(pending: PendingReadback, results: Uint32Array): void {
		if (pending.visibilityGeneration !== this._visibilityGeneration) {
			pending.done = true;
			return;
		}
		for (let index = 0; index < pending.packetIds.length; index++) {
			const packetId = pending.packetIds[index];
			const previous = this._visibilityByPacketId.get(packetId);
			const gpuVisible = results[index] !== 0;
			const signatureA = pending.signaturesA[index];
			const signatureB = pending.signaturesB[index];
			const signatureChanged =
				!previous ||
				previous.signatureA !== signatureA ||
				previous.signatureB !== signatureB;
			const occludedStreak = gpuVisible
				? 0
				: signatureChanged
					? 1
					: (previous?.occludedStreak ?? 0) + 1;
			this._visibilityByPacketId.set(packetId, {
				signatureA,
				signatureB,
				lastResultFrame: pending.frameIndex,
				occludedStreak,
				visible: gpuVisible || occludedStreak < pending.options.hysteresisFrames,
			});
		}
		this._lastCompletedFrameIndex = Math.max(this._lastCompletedFrameIndex, pending.frameIndex);
		pending.done = true;
	}

	private _collectCompletedReadbacks(): void {
		this._pendingReadbacks = this._pendingReadbacks.filter((pending) => !pending.done);
	}

	private _createBindingGroup(
		pipeline: IComputePipeline,
		entries: Array<{ binding: number; resource: unknown }>,
		label: string,
	): IBindingGroup {
		return this._backend.createBindingGroup({
			pipeline,
			layoutIndex: 0,
			entries: entries as Array<{ binding: number; resource: any }>,
			label,
		});
	}

	private _destroyBuffer(buffer: IRenderBuffer | null): void {
		try {
			buffer?.destroy();
		} catch {
			// Backend reset may already have released the resource.
		}
	}

	private _destroyTexture(texture: IRenderTexture | null): void {
		try {
			texture?.destroy();
		} catch {
			// Backend reset may already have released the resource.
		}
	}

	private _destroyPipeline(pipeline: IComputePipeline | null): void {
		this._destroyBuffer(pipeline as unknown as IRenderBuffer | null);
	}

	private _destroyShaderModule(module: IShaderModule | null): void {
		this._destroyBuffer(module as unknown as IRenderBuffer | null);
	}

	private _destroyGPUBuffer(buffer: GPUBuffer): void {
		try {
			buffer.destroy();
		} catch {
			// Device loss may invalidate explicit destroy.
		}
	}

	private _warnOnce(key: string, message: string): void {
		if (this._warnedKeys.has(key)) {
			return;
		}
		this._warnedKeys.add(key);
		Logger.warn(`[${key}] ${message}`, {
			scope: "WebGPUOcclusionCullingRuntime",
			onceKey: key,
		});
	}
}

function align(value: number, alignment: number): number {
	return Math.ceil(value / alignment) * alignment;
}
