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
import type { Matrix3Arr } from "../../maths/types";
import { Matrix4 } from "../../maths/Matrix4";
import {
	WEBGPU_MODEL_BINDING_ANIMATION_PARAMS,
	WEBGPU_MODEL_BINDING_JOINT_MATRICES,
	WEBGPU_MODEL_BINDING_MORPH_NORMAL,
	WEBGPU_MODEL_BINDING_MORPH_POSITION,
	WEBGPU_MODEL_BINDING_MORPH_WEIGHTS,
	WEBGPU_MODEL_BINDING_SHADER_UNIFORMS,
	WEBGPU_MODEL_BINDING_STATIC_INSTANCES,
	WEBGPU_MODEL_UNIFORM_BYTE_SIZE,
	WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT,
	WEBGPU_TEXTURE_SLOT_COUNT,
	createModelUniformWriter,
	writeModelUniformData,
	type WebGPUMaterialUniformData,
	type WebGPUModelUniformWriter,
} from "./";
import type {
	WebGPUAnimationPayloadPool,
	WebGPUSceneAnimationPayload,
} from "./WebGPUAnimationPayloadPool";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";

type MatrixRows = number[][];
type FloatBuffer = Float32Array<ArrayBuffer>;

interface MaterialBindingEntry {
	uniformBuffer: IRenderBuffer;
	shaderUniformBuffer: IRenderBuffer | null;
	bindingGroup: IBindingGroup | null;
	textures: IRenderTexture[];
	samplers: ISampler[];
	morphPositionBuffer: IRenderBuffer;
	morphNormalBuffer: IRenderBuffer;
	modelUniformWriter: WebGPUModelUniformWriter;
	currentModelMatrix: MatrixRows;
	previousModelMatrix: MatrixRows;
	normalMatrix: MatrixRows;
	materialSnapshot: FloatBuffer;
	shaderUniformBufferSize: number;
	shaderUniformCacheKey: string;
	shaderUniformValueRevision: number;
	animationPayloadGeneration: number;
	hasModelSnapshot: boolean;
	hasMaterialSnapshot: boolean;
	hasPackedUniform: boolean;
	modelFrame: number;
	receiveShadows: boolean;
	lastUsedFrame: number;
}

const MATERIAL_SNAPSHOT_FLOATS = (14 + WEBGPU_TEXTURE_SLOT_COUNT * 2) * 4;
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
		animationPayloads: WebGPUAnimationPayloadPool
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
			entry.uniformBuffer.destroy();
			entry.shaderUniformBuffer?.destroy();
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

		this._updateModelUniform(cached, packet, materialData);

		let requiresRebind = false;
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
			const entries: Array<{ binding: number; resource: any }> = [
				{ binding: 0, resource: cached.uniformBuffer },
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
				}
			);
			cached.bindingGroup = this._backend.createBindingGroup({
				label: `ModelBinding_${cacheKey}`,
				layout: this._layouts.modelBindGroupLayout,
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
			uniformBuffer: this._backend.createBuffer({
				size: WEBGPU_MODEL_UNIFORM_BYTE_SIZE,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: `ModelUniform_${cacheKey}`,
			}),
			shaderUniformBuffer: null,
			bindingGroup: null,
			textures: [],
			samplers: [],
			morphPositionBuffer: this._animationPayloads.getFallbackStorageBuffer(),
			morphNormalBuffer: this._animationPayloads.getFallbackStorageBuffer(),
			modelUniformWriter: createModelUniformWriter(),
			currentModelMatrix: createMatrixRows(),
			previousModelMatrix: createMatrixRows(),
			normalMatrix: createMatrixRows(),
			materialSnapshot: new Float32Array(MATERIAL_SNAPSHOT_FLOATS),
			shaderUniformBufferSize: 0,
			shaderUniformCacheKey: "none",
			shaderUniformValueRevision: -1,
			animationPayloadGeneration: -1,
			hasModelSnapshot: false,
			hasMaterialSnapshot: false,
			hasPackedUniform: false,
			modelFrame: -1,
			receiveShadows: true,
			lastUsedFrame: this._currentFrame,
		};
		return entry;
	}

	private _updateModelUniform(
		entry: MaterialBindingEntry,
		packet: DrawPacket,
		materialData: WebGPUMaterialUniformData
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
		uniformDirty =
			updateMaterialSnapshot(
				entry.materialSnapshot,
				materialData,
				entry.hasMaterialSnapshot
			) || uniformDirty;
		entry.hasMaterialSnapshot = true;
		const receiveShadows =
			(packet.submission.passFlags & DRAW_PACKET_FLAG_SHADOW_RECEIVER) !== 0;
		if (entry.receiveShadows !== receiveShadows) {
			entry.receiveShadows = receiveShadows;
			uniformDirty = true;
		}

		if (!uniformDirty) {
			return;
		}

		const uniformData = writeModelUniformData(
			entry.modelUniformWriter,
			entry.currentModelMatrix,
			entry.normalMatrix,
			materialData,
			entry.previousModelMatrix,
			packet.submission.instance.renderLayers,
			receiveShadows
		);
		this._backend.writeBuffer(entry.uniformBuffer, uniformData);
		entry.hasPackedUniform = true;
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
			entry.uniformBuffer.destroy();
			entry.shaderUniformBuffer?.destroy();
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

function copyRows(source: MatrixRows, target: MatrixRows): boolean {
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

function updateMaterialSnapshot(
	target: FloatBuffer,
	materialData: WebGPUMaterialUniformData,
	hasSnapshot: boolean
): boolean {
	let offset = 0;
	let changed = !hasSnapshot;
	for (const values of [
		materialData.baseColorFactor,
		materialData.emissiveFactor,
		materialData.surfaceParams0,
		materialData.surfaceParams1,
		materialData.surfaceParams2,
		materialData.surfaceParams3,
		materialData.specularColorFactor,
		materialData.phongAmbientShininess,
		materialData.phongSpecularShading,
		materialData.sheenColorClearcoatNormalScale,
		materialData.attenuationColor,
		materialData.anisotropyParams,
		materialData.materialFlags,
		materialData.pbrMasks,
	]) {
		changed = writeSnapshotVec4(target, offset, values) || changed;
		offset += 4;
	}

	for (let i = 0; i < WEBGPU_TEXTURE_SLOT_COUNT; i++) {
		const slot = materialData.textureSlots[i];
		changed =
			writeSnapshotVec4(
				target,
				offset,
				slot?.transformA ?? ZERO_VEC4
			) || changed;
		offset += 4;
		changed =
			writeSnapshotVec4(
				target,
				offset,
				slot?.transformB ?? ZERO_VEC4
			) || changed;
		offset += 4;
	}

	return changed;
}

const ZERO_VEC4: [number, number, number, number] = [0, 0, 0, 0];

function writeSnapshotVec4(
	target: FloatBuffer,
	offset: number,
	values: readonly number[]
): boolean {
	let changed = false;
	for (let i = 0; i < 4; i++) {
		const value = values[i] ?? 0;
		if (target[offset + i] !== value) {
			target[offset + i] = value;
			changed = true;
		}
	}
	return changed;
}
