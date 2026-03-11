import {
	BufferUsage,
	type IBindingGroup,
	type IRenderBuffer,
	type IRenderPipeline,
	type IRenderTexture,
	type ISampler,
} from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
import type { DrawPacket } from "../../pipeline/types";
import {
	WEBGPU_MAX_MORPH_TARGETS,
	WEBGPU_MODEL_ANIMATION_UNIFORM_FLOATS,
	WEBGPU_MODEL_BINDING_ANIMATION_PARAMS,
	WEBGPU_MODEL_BINDING_JOINT_MATRICES,
	WEBGPU_MODEL_BINDING_MORPH_NORMAL,
	WEBGPU_MODEL_BINDING_MORPH_POSITION,
	WEBGPU_MODEL_BINDING_MORPH_WEIGHTS,
	WEBGPU_MODEL_UNIFORM_FLOATS,
	packModelUniformData,
	type WebGPUMaterialUniformData,
} from "./";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";

export interface WebGPUModelAnimationBindingState {
	jointMatrices: Float32Array | null;
	morphWeights: Float32Array | null;
	morphTargetCount: number;
	morphPositionBuffer: IRenderBuffer | null;
	morphNormalBuffer: IRenderBuffer | null;
}

interface MaterialBindingEntry {
	uniformBuffer: IRenderBuffer;
	animationParamsBuffer: IRenderBuffer;
	jointMatricesBuffer: IRenderBuffer;
	morphWeightsBuffer: IRenderBuffer;
	bindingGroup: IBindingGroup | null;
	pipeline: IRenderPipeline | null;
	textures: IRenderTexture[];
	samplers: ISampler[];
	morphPositionBuffer: IRenderBuffer;
	morphNormalBuffer: IRenderBuffer;
	prevModelMatrix: DrawPacket["worldMatrix"] | null;
	prevJointMatrices: Float32Array | null;
	prevMorphWeights: Float32Array | null;
	jointMatrixCapacity: number;
	morphWeightCapacity: number;
	lastUsedFrame: number;
}

export class WebGPUMaterialBindingCache {
	private _backend: WebGPUBackend;
	private _layouts: WebGPUPipelineLayouts;
	private _cache = new Map<string, MaterialBindingEntry>();
	private _currentFrame = 0;
	private _fallbackStorageBuffer: IRenderBuffer;

	constructor(backend: WebGPUBackend, layouts: WebGPUPipelineLayouts) {
		this._backend = backend;
		this._layouts = layouts;
		this._fallbackStorageBuffer = this._backend.createBuffer({
			size: 16,
			usage: BufferUsage.Storage | BufferUsage.CopyDst,
			label: "WebGPUFallbackAnimationStorageBuffer",
		});
		this._backend.writeBuffer(this._fallbackStorageBuffer, new Float32Array(4));
	}

	public beginFrame(): void {
		this._currentFrame++;

		for (const [key, entry] of this._cache.entries()) {
			if (entry.lastUsedFrame < this._currentFrame - 5) {
				entry.uniformBuffer.destroy();
				entry.animationParamsBuffer.destroy();
				entry.jointMatricesBuffer.destroy();
				entry.morphWeightsBuffer.destroy();
				this._cache.delete(key);
			}
		}
	}

	public getBinding(
		packet: DrawPacket,
		pipeline: IRenderPipeline,
		materialData: WebGPUMaterialUniformData,
		textures: IRenderTexture[],
		samplers: ISampler[],
		animation: WebGPUModelAnimationBindingState
	): IBindingGroup {
		const cacheKey = `${packet.id}-${materialData.pipelineKey}`;
		let cached = this._cache.get(cacheKey);
		if (!cached) {
			cached = this._createEntry(cacheKey);
			this._cache.set(cacheKey, cached);
		} else {
			cached.lastUsedFrame = this._currentFrame;
		}

		const uniformData = packModelUniformData(
			packet.worldMatrix,
			packet.normalMatrix as any,
			materialData,
			cached.prevModelMatrix ?? packet.worldMatrix
		);
		this._backend.writeBuffer(
			cached.uniformBuffer,
			new Float32Array(uniformData)
		);
		cached.prevModelMatrix = packet.worldMatrix.clone();

		const jointCount = this._resolveJointCount(animation.jointMatrices);
		const morphCount = this._resolveMorphCount(
			animation.morphWeights,
			animation.morphTargetCount
		);
		const jointCapacity = Math.max(
			1,
			jointCount,
			this._resolveJointCount(cached.prevJointMatrices)
		);
		const morphCapacity = Math.max(
			1,
			morphCount,
			cached.prevMorphWeights?.length ?? 0
		);

		let requiresRebind = false;
		requiresRebind =
			this._ensureJointBufferCapacity(cached, cacheKey, jointCapacity) ||
			requiresRebind;
		requiresRebind =
			this._ensureMorphBufferCapacity(cached, cacheKey, morphCapacity) ||
			requiresRebind;

		const jointPayload = this._buildJointPayload(
			animation.jointMatrices,
			cached.prevJointMatrices,
			jointCount,
			jointCapacity
		);
		const morphPayload = this._buildMorphPayload(
			animation.morphWeights,
			cached.prevMorphWeights,
			morphCount,
			morphCapacity
		);
		this._backend.writeBuffer(
			cached.jointMatricesBuffer,
			new Float32Array(jointPayload)
		);
		this._backend.writeBuffer(
			cached.morphWeightsBuffer,
			new Float32Array(morphPayload)
		);
		this._backend.writeBuffer(
			cached.animationParamsBuffer,
			new Float32Array([jointCount, morphCount, jointCapacity, morphCapacity])
		);
		cached.prevJointMatrices =
			jointCount > 0 && animation.jointMatrices ?
				new Float32Array(animation.jointMatrices.subarray(0, jointCount * 16))
			:	null;
		cached.prevMorphWeights =
			morphCount > 0 && animation.morphWeights ?
				new Float32Array(animation.morphWeights.subarray(0, morphCount))
			:	null;

		const morphPositionBuffer =
			animation.morphPositionBuffer ?? this._fallbackStorageBuffer;
		const morphNormalBuffer =
			animation.morphNormalBuffer ?? this._fallbackStorageBuffer;
		if (
			cached.morphPositionBuffer !== morphPositionBuffer ||
			cached.morphNormalBuffer !== morphNormalBuffer
		) {
			requiresRebind = true;
		}

		if (
			!cached.bindingGroup ||
			requiresRebind ||
			cached.pipeline !== pipeline ||
			!areTexturesEqual(cached.textures, textures) ||
			!areSamplersEqual(cached.samplers, samplers) ||
			cached.morphPositionBuffer !== morphPositionBuffer ||
			cached.morphNormalBuffer !== morphNormalBuffer
		) {
			const entries: Array<{ binding: number; resource: any }> = [
				{ binding: 0, resource: cached.uniformBuffer },
			];
			for (let i = 0; i < textures.length; i++) {
				entries.push({ binding: 1 + i * 2, resource: textures[i] });
				entries.push({ binding: 2 + i * 2, resource: samplers[i] });
			}
			entries.push(
				{
					binding: WEBGPU_MODEL_BINDING_ANIMATION_PARAMS,
					resource: cached.animationParamsBuffer,
				},
				{
					binding: WEBGPU_MODEL_BINDING_JOINT_MATRICES,
					resource: cached.jointMatricesBuffer,
				},
				{
					binding: WEBGPU_MODEL_BINDING_MORPH_WEIGHTS,
					resource: cached.morphWeightsBuffer,
				},
				{
					binding: WEBGPU_MODEL_BINDING_MORPH_POSITION,
					resource: morphPositionBuffer,
				},
				{
					binding: WEBGPU_MODEL_BINDING_MORPH_NORMAL,
					resource: morphNormalBuffer,
				}
			);
			cached.bindingGroup = this._backend.createBindingGroup({
				label: `ModelBinding_${cacheKey}`,
				layout: this._layouts.modelBindGroupLayout,
				entries,
			});
			cached.pipeline = pipeline;
			cached.textures = textures.slice();
			cached.samplers = samplers.slice();
			cached.morphPositionBuffer = morphPositionBuffer;
			cached.morphNormalBuffer = morphNormalBuffer;
		}

		return cached.bindingGroup;
	}

	private _createEntry(cacheKey: string): MaterialBindingEntry {
		return {
			uniformBuffer: this._backend.createBuffer({
				size: WEBGPU_MODEL_UNIFORM_FLOATS * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: `ModelUniform_${cacheKey}`,
			}),
			animationParamsBuffer: this._backend.createBuffer({
				size: WEBGPU_MODEL_ANIMATION_UNIFORM_FLOATS * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: `ModelAnimationParams_${cacheKey}`,
			}),
			jointMatricesBuffer: this._backend.createBuffer({
				size: 2 * 16 * 4,
				usage: BufferUsage.Storage | BufferUsage.CopyDst,
				label: `ModelJointMatrices_${cacheKey}`,
			}),
			morphWeightsBuffer: this._backend.createBuffer({
				size: 2 * 4,
				usage: BufferUsage.Storage | BufferUsage.CopyDst,
				label: `ModelMorphWeights_${cacheKey}`,
			}),
			bindingGroup: null,
			pipeline: null,
			textures: [],
			samplers: [],
			morphPositionBuffer: this._fallbackStorageBuffer,
			morphNormalBuffer: this._fallbackStorageBuffer,
			prevModelMatrix: null,
			prevJointMatrices: null,
			prevMorphWeights: null,
			jointMatrixCapacity: 1,
			morphWeightCapacity: 1,
			lastUsedFrame: this._currentFrame,
		};
	}

	private _resolveJointCount(matrices: Float32Array | null): number {
		if (!matrices) return 0;
		return Math.max(0, Math.floor(matrices.length / 16));
	}

	private _resolveMorphCount(
		weights: Float32Array | null,
		targetCount: number
	): number {
		const requested = Math.min(
			WEBGPU_MAX_MORPH_TARGETS,
			Math.max(0, Math.floor(targetCount))
		);
		if (!weights) return requested;
		return Math.min(requested, weights.length);
	}

	private _ensureJointBufferCapacity(
		entry: MaterialBindingEntry,
		cacheKey: string,
		capacity: number
	): boolean {
		if (capacity <= entry.jointMatrixCapacity) {
			return false;
		}
		entry.jointMatricesBuffer.destroy();
		entry.jointMatricesBuffer = this._backend.createBuffer({
			size: capacity * 2 * 16 * 4,
			usage: BufferUsage.Storage | BufferUsage.CopyDst,
			label: `ModelJointMatrices_${cacheKey}`,
		});
		entry.jointMatrixCapacity = capacity;
		return true;
	}

	private _ensureMorphBufferCapacity(
		entry: MaterialBindingEntry,
		cacheKey: string,
		capacity: number
	): boolean {
		if (capacity <= entry.morphWeightCapacity) {
			return false;
		}
		entry.morphWeightsBuffer.destroy();
		entry.morphWeightsBuffer = this._backend.createBuffer({
			size: capacity * 2 * 4,
			usage: BufferUsage.Storage | BufferUsage.CopyDst,
			label: `ModelMorphWeights_${cacheKey}`,
		});
		entry.morphWeightCapacity = capacity;
		return true;
	}

	private _buildJointPayload(
		currentJointMatrices: Float32Array | null,
		prevJointMatrices: Float32Array | null,
		jointCount: number,
		jointCapacity: number
	): Float32Array {
		const result = new Float32Array(jointCapacity * 2 * 16);
		fillIdentityMatrices(result, 0, jointCapacity);
		fillIdentityMatrices(result, jointCapacity * 16, jointCapacity);

		if (jointCount > 0 && currentJointMatrices) {
			result.set(currentJointMatrices.subarray(0, jointCount * 16), 0);
		}

		const prevSource = prevJointMatrices ?? currentJointMatrices;
		const prevCount = Math.min(
			this._resolveJointCount(prevSource),
			jointCapacity
		);
		if (prevSource && prevCount > 0) {
			result.set(prevSource.subarray(0, prevCount * 16), jointCapacity * 16);
		}

		return result;
	}

	private _buildMorphPayload(
		currentMorphWeights: Float32Array | null,
		prevMorphWeights: Float32Array | null,
		morphCount: number,
		morphCapacity: number
	): Float32Array {
		const result = new Float32Array(morphCapacity * 2);
		if (morphCount > 0 && currentMorphWeights) {
			result.set(currentMorphWeights.subarray(0, morphCount), 0);
		}

		const prevSource = prevMorphWeights ?? currentMorphWeights;
		const prevCount = Math.min(prevSource?.length ?? 0, morphCapacity);
		if (prevSource && prevCount > 0) {
			result.set(prevSource.subarray(0, prevCount), morphCapacity);
		}

		return result;
	}
}

function areTexturesEqual(
	left: IRenderTexture[],
	right: IRenderTexture[]
): boolean {
	if (left.length !== right.length) return false;
	for (let i = 0; i < left.length; i++) {
		if (left[i] !== right[i]) return false;
	}
	return true;
}

function areSamplersEqual(left: ISampler[], right: ISampler[]): boolean {
	if (left.length !== right.length) return false;
	for (let i = 0; i < left.length; i++) {
		if (left[i] !== right[i]) return false;
	}
	return true;
}

function fillIdentityMatrices(
	target: Float32Array,
	startFloatOffset: number,
	count: number
): void {
	for (let i = 0; i < count; i++) {
		const base = startFloatOffset + i * 16;
		target[base] = 1;
		target[base + 5] = 1;
		target[base + 10] = 1;
		target[base + 15] = 1;
	}
}
