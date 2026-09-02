import {
	BufferUsage,
	type IBindingGroup,
	type IRenderBuffer,
	type IRenderPipeline,
	type IRenderTexture,
	type ISampler,
} from "../types";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import {
	DRAW_PACKET_FLAG_SHADOW_RECEIVER,
	type DrawPacket,
} from "../../pipeline/types";
import type { Matrix3Arr, Matrix4Arr } from "../../maths/types";
import { Matrix4 } from "../../maths/Matrix4";
import {
	WEBGPU_MODEL_BINDING_ANIMATION_PARAMS,
	WEBGPU_MODEL_BINDING_JOINT_MATRICES,
	WEBGPU_MODEL_BINDING_MORPH_NORMAL,
	WEBGPU_MODEL_BINDING_MORPH_POSITION,
	WEBGPU_MODEL_BINDING_MORPH_WEIGHTS,
	WEBGPU_MODEL_BINDING_MATERIAL_COMMON,
	WEBGPU_MODEL_BINDING_FLAT_MATERIAL,
	WEBGPU_MODEL_BINDING_PBR_MATERIAL,
	WEBGPU_MODEL_BINDING_PHONG_MATERIAL,
	WEBGPU_MODEL_BINDING_SHADER_UNIFORMS,
	WEBGPU_MODEL_BINDING_STATIC_INSTANCES,
	WEBGPU_OBJECT_UNIFORM_BYTE_SIZE,
	WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT,
	createObjectUniformWriter,
	writeObjectUniformData,
	type WebGPUMaterialUniformData,
	type WebGPUObjectUniformWriter,
} from "./";
import type {
	WebGPUAnimationPayloadPool,
	WebGPUSceneAnimationPayload,
} from "./WebGPUAnimationPayloadPool";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";
import type {
	WebGPUMaterialBufferCache,
	WebGPUMaterialBufferLease,
} from "./WebGPUMaterialBufferCache";

type MatrixRows = Matrix4Arr;
type FloatBuffer = Float32Array<ArrayBuffer>;

interface MaterialBindingEntry {
	objectUniformBuffer: IRenderBuffer;
	shaderUniformBuffer: IRenderBuffer | null;
	bindingGroup: IBindingGroup | null;
	materialData: WebGPUMaterialUniformData | null;
	materialLease: WebGPUMaterialBufferLease | null;
	textures: IRenderTexture[];
	samplers: ISampler[];
	morphPositionBuffer: IRenderBuffer;
	morphNormalBuffer: IRenderBuffer;
	objectUniformWriter: WebGPUObjectUniformWriter;
	currentModelMatrix: MatrixRows;
	previousModelMatrix: MatrixRows;
	normalMatrix: MatrixRows;
	shaderUniformBufferSize: number;
	shaderUniformCacheKey: string;
	shaderUniformValueRevision: number;
	animationPayloadGeneration: number;
	hasModelSnapshot: boolean;
	hasPackedUniform: boolean;
	modelFrame: number;
	receiveShadows: boolean;
	lastUsedFrame: number;
}

const MATERIAL_BINDING_CACHE_MAX_ENTRIES = 16_384;
const FALLBACK_UNIFORM_DATA: FloatBuffer = new Float32Array(4);

export class WebGPUMaterialBindingCache {
	private _backend: WebGPUDeviceResourceHost;
	private _layouts: WebGPUPipelineLayouts;
	private _animationPayloads: WebGPUAnimationPayloadPool;
	private _cache = new Map<string, MaterialBindingEntry>();
	private _currentFrame = 0;
	private _fallbackShaderUniformBuffer: IRenderBuffer;
	private _destroyed = false;

	constructor(
		backend: WebGPUDeviceResourceHost,
		layouts: WebGPUPipelineLayouts,
		animationPayloads: WebGPUAnimationPayloadPool,
		private readonly _materialBuffers: WebGPUMaterialBufferCache,
	) {
		this._backend = backend;
		this._layouts = layouts;
		this._animationPayloads = animationPayloads;
		this._fallbackShaderUniformBuffer = this._backend.createBuffer({
			size: 16,
			usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			label: "WebGPUFallbackShaderUniformBuffer",
		});
		this._backend.writeBuffer(
			this._fallbackShaderUniformBuffer,
			FALLBACK_UNIFORM_DATA
		);
	}

	public beginFrame(): void {
		if (this._destroyed) {
			return;
		}
		this._currentFrame++;
	}

	public destroy(): void {
		if (this._destroyed) {
			return;
		}
		this._destroyed = true;
		for (const entry of this._cache.values()) {
			this._destroyBindingGroup(entry.bindingGroup);
			entry.objectUniformBuffer.destroy();
			entry.shaderUniformBuffer?.destroy();
			entry.materialLease?.release();
		}
		this._cache.clear();
		this._fallbackShaderUniformBuffer.destroy();
	}

	public getBinding(
		packet: DrawPacket,
		pipeline: IRenderPipeline,
		materialData: WebGPUMaterialUniformData,
		textures: IRenderTexture[],
		samplers: ISampler[],
		animation: WebGPUSceneAnimationPayload,
		morphPositionBuffer: IRenderBuffer | null,
		morphNormalBuffer: IRenderBuffer | null
	): IBindingGroup {
		const cacheKey = `${packet.submission.id}-${materialData.pipelineKey}`;
		let cached = this._cache.get(cacheKey);
		if (!cached) {
			cached = this._createEntry(cacheKey);
			this._cache.set(cacheKey, cached);
			this._trimCache();
		} else {
			cached.lastUsedFrame = this._currentFrame;
			this._cache.delete(cacheKey);
			this._cache.set(cacheKey, cached);
		}

		this._updateObjectUniform(cached, packet);

		let requiresRebind = this._updateMaterialLease(cached, materialData);
		requiresRebind =
			this._updateShaderUniformBuffer(
				cached,
				cacheKey,
				materialData
			) || requiresRebind;
		if (cached.animationPayloadGeneration !== animation.generation) {
			requiresRebind = true;
		}

		const fallbackStorageBuffer =
			this._animationPayloads.getFallbackStorageBuffer();
		const resolvedMorphPositionBuffer =
			morphPositionBuffer ?? fallbackStorageBuffer;
		const resolvedMorphNormalBuffer = morphNormalBuffer ?? fallbackStorageBuffer;
		if (
			cached.morphPositionBuffer !== resolvedMorphPositionBuffer ||
			cached.morphNormalBuffer !== resolvedMorphNormalBuffer
		) {
			requiresRebind = true;
		}

		if (
			!cached.bindingGroup ||
			requiresRebind ||
			!areTexturesEqual(cached.textures, textures) ||
			!areSamplersEqual(cached.samplers, samplers) ||
			cached.morphPositionBuffer !== resolvedMorphPositionBuffer ||
			cached.morphNormalBuffer !== resolvedMorphNormalBuffer
		) {
			const previousBindingGroup = cached.bindingGroup;
			const materialResources = cached.materialLease?.resources;
			if (!materialResources) {
				throw new Error("WebGPU model binding is missing material buffers.");
			}
			const entries: Array<{ binding: number; resource: any }> = [
				{ binding: 0, resource: cached.objectUniformBuffer },
			];
			for (let i = 0; i < textures.length; i++) {
				entries.push({ binding: 1 + i * 2, resource: textures[i] });
				if (i < WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT) {
					entries.push({ binding: 2 + i * 2, resource: samplers[i] });
				}
			}
			entries.push(
				{
					binding: WEBGPU_MODEL_BINDING_ANIMATION_PARAMS,
					resource: animation.paramsBuffer,
				},
				{
					binding: WEBGPU_MODEL_BINDING_JOINT_MATRICES,
					resource: animation.jointMatricesBuffer,
				},
				{
					binding: WEBGPU_MODEL_BINDING_MORPH_WEIGHTS,
					resource: animation.morphWeightsBuffer,
				},
				{
					binding: WEBGPU_MODEL_BINDING_MORPH_POSITION,
					resource: resolvedMorphPositionBuffer,
				},
				{
					binding: WEBGPU_MODEL_BINDING_MORPH_NORMAL,
					resource: resolvedMorphNormalBuffer,
				},
				{
					binding: WEBGPU_MODEL_BINDING_SHADER_UNIFORMS,
					resource:
						cached.shaderUniformBuffer ??
						this._fallbackShaderUniformBuffer,
				},
				{
					binding: WEBGPU_MODEL_BINDING_STATIC_INSTANCES,
					resource: fallbackStorageBuffer,
				},
				{
					binding: WEBGPU_MODEL_BINDING_MATERIAL_COMMON,
					resource: materialResources.commonBuffer,
				},
			);
			if (materialResources.lightingBuffer) {
				entries.push({
					binding: getLightingMaterialBinding(materialResources.shadingFamily),
					resource: materialResources.lightingBuffer,
				});
			}
			cached.bindingGroup = this._backend.createBindingGroup({
				label: `ModelBinding_${cacheKey}`,
				layout: this._layouts.modelBindGroupLayouts[materialResources.shadingFamily],
				entries,
				cache: false,
			});
			if (previousBindingGroup && previousBindingGroup !== cached.bindingGroup) {
				this._destroyBindingGroup(previousBindingGroup);
			}
			cached.textures = textures.slice();
			cached.samplers = samplers.slice();
			cached.morphPositionBuffer = resolvedMorphPositionBuffer;
			cached.morphNormalBuffer = resolvedMorphNormalBuffer;
			cached.animationPayloadGeneration = animation.generation;
		}

		return cached.bindingGroup;
	}

	private _createEntry(cacheKey: string): MaterialBindingEntry {
		const entry: MaterialBindingEntry = {
			objectUniformBuffer: this._backend.createBuffer({
				size: WEBGPU_OBJECT_UNIFORM_BYTE_SIZE,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: `ObjectUniform_${cacheKey}`,
			}),
			shaderUniformBuffer: null,
			bindingGroup: null,
			materialData: null,
			materialLease: null,
			textures: [],
			samplers: [],
			morphPositionBuffer: this._animationPayloads.getFallbackStorageBuffer(),
			morphNormalBuffer: this._animationPayloads.getFallbackStorageBuffer(),
			objectUniformWriter: createObjectUniformWriter(),
			currentModelMatrix: createMatrixRows(),
			previousModelMatrix: createMatrixRows(),
			normalMatrix: createMatrixRows(),
			shaderUniformBufferSize: 0,
			shaderUniformCacheKey: "none",
			shaderUniformValueRevision: -1,
			animationPayloadGeneration: -1,
			hasModelSnapshot: false,
			hasPackedUniform: false,
			modelFrame: -1,
			receiveShadows: true,
			lastUsedFrame: this._currentFrame,
		};
		return entry;
	}

	private _updateObjectUniform(
		entry: MaterialBindingEntry,
		packet: DrawPacket,
	): void {
		let uniformDirty = !entry.hasPackedUniform;
		const explicitPreviousMatrix = packet.submission.instance.previousWorldMatrix ?? null;

		if (entry.modelFrame !== this._currentFrame) {
			if (explicitPreviousMatrix) {
				uniformDirty =
					copyMatrixToRows(
						explicitPreviousMatrix,
						entry.previousModelMatrix
					) || uniformDirty;
				entry.hasModelSnapshot = true;
			} else if (!entry.hasModelSnapshot) {
				copyMatrixToRows(packet.submission.instance.worldMatrix, entry.currentModelMatrix);
				copyRows(entry.currentModelMatrix, entry.previousModelMatrix);
				entry.hasModelSnapshot = true;
			} else {
				uniformDirty =
					copyRows(entry.currentModelMatrix, entry.previousModelMatrix) ||
					uniformDirty;
			}
			entry.modelFrame = this._currentFrame;
		}

		uniformDirty =
			copyMatrixToRows(packet.submission.instance.worldMatrix, entry.currentModelMatrix) ||
			uniformDirty;
		uniformDirty =
			copyNormalMatrixToRows(packet.submission.instance.normalMatrix, entry.normalMatrix) ||
			uniformDirty;
		const receiveShadows =
			(packet.submission.passFlags & DRAW_PACKET_FLAG_SHADOW_RECEIVER) !== 0;
		if (entry.receiveShadows !== receiveShadows) {
			entry.receiveShadows = receiveShadows;
			uniformDirty = true;
		}

		if (!uniformDirty) {
			return;
		}

		const uniformData = writeObjectUniformData(
			entry.objectUniformWriter,
			entry.currentModelMatrix,
			entry.normalMatrix,
			entry.previousModelMatrix,
			packet.submission.instance.renderLayers,
			receiveShadows
		);
		this._backend.writeBuffer(entry.objectUniformBuffer, uniformData);
		entry.hasPackedUniform = true;
	}

	private _updateMaterialLease(
		entry: MaterialBindingEntry,
		materialData: WebGPUMaterialUniformData,
	): boolean {
		if (entry.materialData === materialData && entry.materialLease) return false;
		const nextLease = this._materialBuffers.acquire(materialData);
		entry.materialLease?.release();
		entry.materialData = materialData;
		entry.materialLease = nextLease;
		return true;
	}

	private _updateShaderUniformBuffer(
		entry: MaterialBindingEntry,
		cacheKey: string,
		materialData: WebGPUMaterialUniformData
	): boolean {
		const shaderUniforms = materialData.shaderUniforms;
		if (!shaderUniforms.data || shaderUniforms.byteLength <= 0) {
			const hadBuffer = entry.shaderUniformBuffer !== null;
			if (entry.shaderUniformBuffer) {
				entry.shaderUniformBuffer.destroy();
				entry.shaderUniformBuffer = null;
			}
			entry.shaderUniformBufferSize = 0;
			entry.shaderUniformCacheKey = "none";
			entry.shaderUniformValueRevision = shaderUniforms.valueRevision;
			return hadBuffer;
		}

		const requiredSize = Math.max(16, shaderUniforms.byteLength);
		let requiresRebind = false;
		if (
			!entry.shaderUniformBuffer ||
			entry.shaderUniformBufferSize !== requiredSize
		) {
			entry.shaderUniformBuffer?.destroy();
			entry.shaderUniformBuffer = this._backend.createBuffer({
				size: requiredSize,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: `ShaderMaterialUniform_${cacheKey}`,
			});
			entry.shaderUniformBufferSize = requiredSize;
			requiresRebind = true;
		}

		const mustWrite =
			entry.shaderUniformCacheKey !== shaderUniforms.cacheKey ||
			entry.shaderUniformValueRevision !== shaderUniforms.valueRevision;
		if (mustWrite) {
			this._backend.writeBuffer(entry.shaderUniformBuffer, shaderUniforms.data);
			entry.shaderUniformCacheKey = shaderUniforms.cacheKey;
			entry.shaderUniformValueRevision = shaderUniforms.valueRevision;
		}
		return requiresRebind;
	}

	private _destroyBindingGroup(group: IBindingGroup | null): void {
		const destroyFn = (group as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(group);
		}
	}

	private _trimCache(): void {
		while (this._cache.size > MATERIAL_BINDING_CACHE_MAX_ENTRIES) {
			const oldest = this._cache.entries().next().value as
				| [string, MaterialBindingEntry]
				| undefined;
			if (!oldest) break;
			const [key, entry] = oldest;
			this._cache.delete(key);
			this._destroyBindingGroup(entry.bindingGroup);
			entry.objectUniformBuffer.destroy();
			entry.shaderUniformBuffer?.destroy();
			entry.materialLease?.release();
		}
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

function createMatrixRows(): MatrixRows {
	return [
		[0, 0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 0, 0],
	];
}

function resolveMatrixRows(matrix: Matrix4 | number[][]): number[][] {
	return matrix instanceof Matrix4 ? matrix.elements : matrix;
}

function copyMatrixToRows(
	source: Matrix4 | number[][],
	target: MatrixRows
): boolean {
	return copyRows(resolveMatrixRows(source), target);
}

function copyNormalMatrixToRows(
	source: Matrix4 | Matrix3Arr,
	target: MatrixRows
): boolean {
	const rows = source instanceof Matrix4 ? source.elements : source;
	let changed = false;
	for (let row = 0; row < 4; row++) {
		for (let column = 0; column < 4; column++) {
			const value =
				row < 3 && column < 3 ? rows[row]?.[column] ?? 0
				: row === 3 && column === 3 ? 1
				: 0;
			if (target[row][column] !== value) {
				target[row][column] = value;
				changed = true;
			}
		}
	}
	return changed;
}

function copyRows(source: number[][], target: MatrixRows): boolean {
	let changed = false;
	for (let row = 0; row < 4; row++) {
		for (let column = 0; column < 4; column++) {
			const value = source[row]?.[column] ?? 0;
			if (target[row][column] !== value) {
				target[row][column] = value;
				changed = true;
			}
		}
	}
	return changed;
}

function getLightingMaterialBinding(
	family: WebGPUMaterialUniformData["shadingFamily"],
): number {
	switch (family) {
		case "pbr":
			return WEBGPU_MODEL_BINDING_PBR_MATERIAL;
		case "phong":
			return WEBGPU_MODEL_BINDING_PHONG_MATERIAL;
		case "flat":
			return WEBGPU_MODEL_BINDING_FLAT_MATERIAL;
		case "unlit":
			throw new Error("Unlit model bindings do not contain a lighting buffer.");
	}
}
