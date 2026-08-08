import {
	BufferUsage,
	type IBindingGroup,
	type IRenderBuffer,
	type IRenderPipeline,
	type IRenderTexture,
	type ISampler,
} from "../types";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type { DrawPacket } from "../../pipeline/types";
import type { Matrix3Arr } from "../../maths/types";
import { Matrix4 } from "../../maths/Matrix4";
import {
	WEBGPU_MAX_MORPH_TARGETS,
	WEBGPU_MODEL_ANIMATION_UNIFORM_FLOATS,
	WEBGPU_MODEL_BINDING_ANIMATION_PARAMS,
	WEBGPU_MODEL_BINDING_JOINT_MATRICES,
	WEBGPU_MODEL_BINDING_MORPH_NORMAL,
	WEBGPU_MODEL_BINDING_MORPH_POSITION,
	WEBGPU_MODEL_BINDING_MORPH_WEIGHTS,
	WEBGPU_MODEL_BINDING_ANISOTROPY_TEXTURE,
	WEBGPU_MODEL_BINDING_SHADER_UNIFORMS,
	WEBGPU_MODEL_UNIFORM_BYTE_SIZE,
	WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT,
	WEBGPU_TEXTURE_SLOT_COUNT,
	createModelUniformWriter,
	writeModelUniformData,
	type WebGPUMaterialUniformData,
	type WebGPUModelUniformWriter,
} from "./";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";

export interface WebGPUModelAnimationBindingState {
	jointMatrices: Float32Array | null;
	morphWeights: Float32Array | null;
	morphTargetCount: number;
	morphPositionBuffer: IRenderBuffer | null;
	morphNormalBuffer: IRenderBuffer | null;
}

type MatrixRows = number[][];
type FloatBuffer = Float32Array<ArrayBuffer>;

interface MaterialBindingEntry {
	uniformBuffer: IRenderBuffer;
	shaderUniformBuffer: IRenderBuffer | null;
	animationParamsBuffer: IRenderBuffer;
	jointMatricesBuffer: IRenderBuffer;
	morphWeightsBuffer: IRenderBuffer;
	bindingGroup: IBindingGroup | null;
	textures: IRenderTexture[];
	samplers: ISampler[];
	anisotropyTexture: IRenderTexture | null;
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
	hasModelSnapshot: boolean;
	hasMaterialSnapshot: boolean;
	hasPackedUniform: boolean;
	modelFrame: number;
	receiveShadows: boolean;
	animationParamsData: FloatBuffer;
	jointPayload: FloatBuffer;
	morphPayload: FloatBuffer;
	currentJointMatrices: FloatBuffer | null;
	previousJointMatrices: FloatBuffer | null;
	currentMorphWeights: FloatBuffer | null;
	previousMorphWeights: FloatBuffer | null;
	currentJointCount: number;
	previousJointCount: number;
	currentMorphCount: number;
	previousMorphCount: number;
	animationFrame: number;
	animationParamsWritten: boolean;
	jointPayloadWritten: boolean;
	morphPayloadWritten: boolean;
	jointMatrixCapacity: number;
	morphWeightCapacity: number;
	lastUsedFrame: number;
}

const MATERIAL_SNAPSHOT_FLOATS = (15 + WEBGPU_TEXTURE_SLOT_COUNT * 2) * 4;
const FALLBACK_STORAGE_DATA: FloatBuffer = new Float32Array(4);
const FALLBACK_UNIFORM_DATA: FloatBuffer = new Float32Array(4);

export class WebGPUMaterialBindingCache {
	private _backend: WebGPUDeviceResourceHost;
	private _layouts: WebGPUPipelineLayouts;
	private _cache = new Map<string, MaterialBindingEntry>();
	private _currentFrame = 0;
	private _fallbackStorageBuffer: IRenderBuffer;
	private _fallbackShaderUniformBuffer: IRenderBuffer;
	private _destroyed = false;

	constructor(backend: WebGPUDeviceResourceHost, layouts: WebGPUPipelineLayouts) {
		this._backend = backend;
		this._layouts = layouts;
		this._fallbackStorageBuffer = this._backend.createBuffer({
			size: 16,
			usage: BufferUsage.Storage | BufferUsage.CopyDst,
			label: "WebGPUFallbackAnimationStorageBuffer",
		});
		this._fallbackShaderUniformBuffer = this._backend.createBuffer({
			size: 16,
			usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			label: "WebGPUFallbackShaderUniformBuffer",
		});
		this._backend.writeBuffer(this._fallbackStorageBuffer, FALLBACK_STORAGE_DATA);
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

		for (const [key, entry] of this._cache.entries()) {
			if (entry.lastUsedFrame < this._currentFrame - 5) {
				this._destroyBindingGroup(entry.bindingGroup);
				entry.bindingGroup = null;
				entry.uniformBuffer.destroy();
				entry.shaderUniformBuffer?.destroy();
				entry.animationParamsBuffer.destroy();
				entry.jointMatricesBuffer.destroy();
				entry.morphWeightsBuffer.destroy();
				this._cache.delete(key);
			}
		}
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
			entry.animationParamsBuffer.destroy();
			entry.jointMatricesBuffer.destroy();
			entry.morphWeightsBuffer.destroy();
		}
		this._cache.clear();
		this._fallbackStorageBuffer.destroy();
		this._fallbackShaderUniformBuffer.destroy();
	}

	public getBinding(
		packet: DrawPacket,
		pipeline: IRenderPipeline,
		materialData: WebGPUMaterialUniformData,
		textures: IRenderTexture[],
		samplers: ISampler[],
		anisotropyTexture: IRenderTexture,
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

		this._updateModelUniform(cached, packet, materialData);

		const jointCount = this._resolveJointCount(animation.jointMatrices);
		const morphCount = this._resolveMorphCount(
			animation.morphWeights,
			animation.morphTargetCount
		);
		const jointCapacity = Math.max(1, jointCount);
		const morphCapacity = Math.max(1, morphCount);

		let requiresRebind = false;
		requiresRebind =
			this._updateShaderUniformBuffer(
				cached,
				cacheKey,
				materialData
			) || requiresRebind;
		requiresRebind =
			this._ensureJointBufferCapacity(cached, cacheKey, jointCapacity) ||
			requiresRebind;
		requiresRebind =
			this._ensureMorphBufferCapacity(cached, cacheKey, morphCapacity) ||
			requiresRebind;

		this._updateAnimationBuffers(cached, animation, jointCount, morphCount);

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
			!areTexturesEqual(cached.textures, textures) ||
			!areSamplersEqual(cached.samplers, samplers) ||
			cached.anisotropyTexture !== anisotropyTexture ||
			cached.morphPositionBuffer !== morphPositionBuffer ||
			cached.morphNormalBuffer !== morphNormalBuffer
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
				},
				{
					binding: WEBGPU_MODEL_BINDING_SHADER_UNIFORMS,
					resource:
						cached.shaderUniformBuffer ??
						this._fallbackShaderUniformBuffer,
				},
				{
					binding: WEBGPU_MODEL_BINDING_ANISOTROPY_TEXTURE,
					resource: anisotropyTexture,
				}
			);
			cached.bindingGroup = this._backend.createBindingGroup({
				label: `ModelBinding_${cacheKey}`,
				layout: this._layouts.modelBindGroupLayout,
				entries,
			});
			if (previousBindingGroup && previousBindingGroup !== cached.bindingGroup) {
				this._destroyBindingGroup(previousBindingGroup);
			}
			cached.textures = textures.slice();
			cached.samplers = samplers.slice();
			cached.anisotropyTexture = anisotropyTexture;
			cached.morphPositionBuffer = morphPositionBuffer;
			cached.morphNormalBuffer = morphNormalBuffer;
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
			textures: [],
			samplers: [],
			anisotropyTexture: null,
			morphPositionBuffer: this._fallbackStorageBuffer,
			morphNormalBuffer: this._fallbackStorageBuffer,
			modelUniformWriter: createModelUniformWriter(),
			currentModelMatrix: createMatrixRows(),
			previousModelMatrix: createMatrixRows(),
			normalMatrix: createMatrixRows(),
			materialSnapshot: new Float32Array(MATERIAL_SNAPSHOT_FLOATS),
			shaderUniformBufferSize: 0,
			shaderUniformCacheKey: "none",
			shaderUniformValueRevision: -1,
			hasModelSnapshot: false,
			hasMaterialSnapshot: false,
			hasPackedUniform: false,
			modelFrame: -1,
			receiveShadows: true,
			animationParamsData: new Float32Array([0, 0, 1, 1]),
			jointPayload: createJointPayload(1),
			morphPayload: new Float32Array(2),
			currentJointMatrices: null,
			previousJointMatrices: null,
			currentMorphWeights: null,
			previousMorphWeights: null,
			currentJointCount: 0,
			previousJointCount: 0,
			currentMorphCount: 0,
			previousMorphCount: 0,
			animationFrame: -1,
			animationParamsWritten: true,
			jointPayloadWritten: false,
			morphPayloadWritten: false,
			jointMatrixCapacity: 1,
			morphWeightCapacity: 1,
			lastUsedFrame: this._currentFrame,
		};
		this._backend.writeBuffer(entry.animationParamsBuffer, entry.animationParamsData);
		return entry;
	}

	private _updateModelUniform(
		entry: MaterialBindingEntry,
		packet: DrawPacket,
		materialData: WebGPUMaterialUniformData
	): void {
		let uniformDirty = !entry.hasPackedUniform;
		const explicitPreviousMatrix = packet.previousWorldMatrix ?? null;

		if (entry.modelFrame !== this._currentFrame) {
			if (explicitPreviousMatrix) {
				uniformDirty =
					copyMatrixToRows(
						explicitPreviousMatrix,
						entry.previousModelMatrix
					) || uniformDirty;
				entry.hasModelSnapshot = true;
			} else if (!entry.hasModelSnapshot) {
				copyMatrixToRows(packet.worldMatrix, entry.currentModelMatrix);
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
			copyMatrixToRows(packet.worldMatrix, entry.currentModelMatrix) ||
			uniformDirty;
		uniformDirty =
			copyNormalMatrixToRows(packet.normalMatrix, entry.normalMatrix) ||
			uniformDirty;
		uniformDirty =
			updateMaterialSnapshot(
				entry.materialSnapshot,
				materialData,
				entry.hasMaterialSnapshot
			) || uniformDirty;
		entry.hasMaterialSnapshot = true;
		const receiveShadows = packet.primitive?.receiveShadows !== false;
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
			packet.meshInstance?.renderLayers ?? 1,
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

	private _updateAnimationBuffers(
		entry: MaterialBindingEntry,
		animation: WebGPUModelAnimationBindingState,
		jointCount: number,
		morphCount: number
	): void {
		let jointPayloadDirty = false;
		let morphPayloadDirty = false;

		if (entry.animationFrame !== this._currentFrame) {
			if (entry.currentJointMatrices && entry.currentJointCount > 0) {
				this._ensureJointSnapshots(entry);
				jointPayloadDirty =
					copyFloatPrefix(
						entry.currentJointMatrices,
						entry.previousJointMatrices!,
						entry.currentJointCount * 16
					) || jointPayloadDirty;
				entry.previousJointCount = entry.currentJointCount;
			} else if (entry.previousJointCount !== 0) {
				entry.previousJointCount = 0;
				jointPayloadDirty = true;
			}

			if (entry.currentMorphWeights && entry.currentMorphCount > 0) {
				this._ensureMorphSnapshots(entry);
				morphPayloadDirty =
					copyFloatPrefix(
						entry.currentMorphWeights,
						entry.previousMorphWeights!,
						entry.currentMorphCount
					) || morphPayloadDirty;
				entry.previousMorphCount = entry.currentMorphCount;
			} else if (entry.previousMorphCount !== 0) {
				entry.previousMorphCount = 0;
				morphPayloadDirty = true;
			}

			entry.animationFrame = this._currentFrame;
		}

		if (jointCount > 0 && animation.jointMatrices) {
			const hadPreviousJointState = entry.currentJointCount > 0;
			this._ensureJointSnapshots(entry);
			jointPayloadDirty =
				copyFloatPrefix(
					animation.jointMatrices,
					entry.currentJointMatrices!,
					jointCount * 16
				) || jointPayloadDirty;
			if (entry.currentJointCount !== jointCount) {
				entry.currentJointCount = jointCount;
				jointPayloadDirty = true;
			}
			if (!hadPreviousJointState || entry.previousJointCount <= 0) {
				copyFloatPrefix(
					entry.currentJointMatrices!,
					entry.previousJointMatrices!,
					jointCount * 16
				);
				entry.previousJointCount = jointCount;
				jointPayloadDirty = true;
			}
		} else if (entry.currentJointCount !== 0) {
			entry.currentJointCount = 0;
			jointPayloadDirty = true;
		}

		if (morphCount > 0) {
			const hadPreviousMorphState = entry.currentMorphCount > 0;
			this._ensureMorphSnapshots(entry);
			if (animation.morphWeights) {
				morphPayloadDirty =
					copyFloatPrefix(
						animation.morphWeights,
						entry.currentMorphWeights!,
						morphCount
					) || morphPayloadDirty;
			} else {
				morphPayloadDirty =
					zeroFloatPrefix(entry.currentMorphWeights!, morphCount) ||
					morphPayloadDirty;
			}
			if (entry.currentMorphCount !== morphCount) {
				entry.currentMorphCount = morphCount;
				morphPayloadDirty = true;
			}
			if (!hadPreviousMorphState || entry.previousMorphCount <= 0) {
				copyFloatPrefix(
					entry.currentMorphWeights!,
					entry.previousMorphWeights!,
					morphCount
				);
				entry.previousMorphCount = morphCount;
				morphPayloadDirty = true;
			}
		} else if (entry.currentMorphCount !== 0) {
			entry.currentMorphCount = 0;
			morphPayloadDirty = true;
		}

		this._writeAnimationParamsIfNeeded(entry, jointCount, morphCount);

		if (jointCount > 0 && (jointPayloadDirty || !entry.jointPayloadWritten)) {
			this._writeJointPayload(entry, jointCount);
		}
		if (morphCount > 0 && (morphPayloadDirty || !entry.morphPayloadWritten)) {
			this._writeMorphPayload(entry, morphCount);
		}
	}

	private _writeAnimationParamsIfNeeded(
		entry: MaterialBindingEntry,
		jointCount: number,
		morphCount: number
	): void {
		const params = entry.animationParamsData;
		let dirty = !entry.animationParamsWritten;
		dirty = setArrayValue(params, 0, jointCount) || dirty;
		dirty = setArrayValue(params, 1, morphCount) || dirty;
		dirty = setArrayValue(params, 2, entry.jointMatrixCapacity) || dirty;
		dirty = setArrayValue(params, 3, entry.morphWeightCapacity) || dirty;
		if (!dirty) {
			return;
		}
		this._backend.writeBuffer(entry.animationParamsBuffer, params);
		entry.animationParamsWritten = true;
	}

	private _writeJointPayload(
		entry: MaterialBindingEntry,
		jointCount: number
	): void {
		const capacity = entry.jointMatrixCapacity;
		const payload = entry.jointPayload;
		payload.fill(0);
		fillIdentityMatrices(payload, 0, capacity);
		fillIdentityMatrices(payload, capacity * 16, capacity);
		if (entry.currentJointMatrices && jointCount > 0) {
			payload.set(entry.currentJointMatrices.subarray(0, jointCount * 16), 0);
		}
		const prevCount = Math.min(entry.previousJointCount, jointCount);
		if (entry.previousJointMatrices && prevCount > 0) {
			payload.set(
				entry.previousJointMatrices.subarray(0, prevCount * 16),
				capacity * 16
			);
		}
		this._backend.writeBuffer(entry.jointMatricesBuffer, payload);
		entry.jointPayloadWritten = true;
	}

	private _writeMorphPayload(
		entry: MaterialBindingEntry,
		morphCount: number
	): void {
		const capacity = entry.morphWeightCapacity;
		const payload = entry.morphPayload;
		payload.fill(0);
		if (entry.currentMorphWeights && morphCount > 0) {
			payload.set(entry.currentMorphWeights.subarray(0, morphCount), 0);
		}
		const prevCount = Math.min(entry.previousMorphCount, morphCount);
		if (entry.previousMorphWeights && prevCount > 0) {
			payload.set(
				entry.previousMorphWeights.subarray(0, prevCount),
				capacity
			);
		}
		this._backend.writeBuffer(entry.morphWeightsBuffer, payload);
		entry.morphPayloadWritten = true;
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
		entry.jointPayload = createJointPayload(capacity);
		entry.currentJointMatrices = resizeFloatArray(
			entry.currentJointMatrices,
			capacity * 16
		);
		entry.previousJointMatrices = resizeFloatArray(
			entry.previousJointMatrices,
			capacity * 16
		);
		entry.jointMatrixCapacity = capacity;
		entry.jointPayloadWritten = false;
		entry.animationParamsWritten = false;
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
		entry.morphPayload = new Float32Array(capacity * 2);
		entry.currentMorphWeights = resizeFloatArray(
			entry.currentMorphWeights,
			capacity
		);
		entry.previousMorphWeights = resizeFloatArray(
			entry.previousMorphWeights,
			capacity
		);
		entry.morphWeightCapacity = capacity;
		entry.morphPayloadWritten = false;
		entry.animationParamsWritten = false;
		return true;
	}

	private _ensureJointSnapshots(entry: MaterialBindingEntry): void {
		const length = entry.jointMatrixCapacity * 16;
		if (!entry.currentJointMatrices) {
			entry.currentJointMatrices = new Float32Array(length);
		}
		if (!entry.previousJointMatrices) {
			entry.previousJointMatrices = new Float32Array(length);
		}
	}

	private _ensureMorphSnapshots(entry: MaterialBindingEntry): void {
		const length = entry.morphWeightCapacity;
		if (!entry.currentMorphWeights) {
			entry.currentMorphWeights = new Float32Array(length);
		}
		if (!entry.previousMorphWeights) {
			entry.previousMorphWeights = new Float32Array(length);
		}
	}

	private _destroyBindingGroup(group: IBindingGroup | null): void {
		const destroyFn = (group as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(group);
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
		materialData.anisotropyTexture.transformA,
		materialData.anisotropyTexture.transformB,
		materialData.materialFlags,
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

function setArrayValue(
	target: FloatBuffer,
	index: number,
	value: number
): boolean {
	if (target[index] === value) {
		return false;
	}
	target[index] = value;
	return true;
}

function copyFloatPrefix(
	source: Float32Array,
	target: FloatBuffer,
	count: number
): boolean {
	let changed = false;
	for (let i = 0; i < count; i++) {
		const value = source[i] ?? 0;
		if (target[i] !== value) {
			target[i] = value;
			changed = true;
		}
	}
	return changed;
}

function zeroFloatPrefix(target: FloatBuffer, count: number): boolean {
	let changed = false;
	for (let i = 0; i < count; i++) {
		if (target[i] !== 0) {
			target[i] = 0;
			changed = true;
		}
	}
	return changed;
}

function resizeFloatArray(
	source: Float32Array | null,
	length: number
): FloatBuffer {
	const target = new Float32Array(length);
	if (source) {
		target.set(source.subarray(0, Math.min(source.length, length)));
	}
	return target;
}

function createJointPayload(jointCapacity: number): FloatBuffer {
	const payload = new Float32Array(jointCapacity * 2 * 16);
	fillIdentityMatrices(payload, 0, jointCapacity);
	fillIdentityMatrices(payload, jointCapacity * 16, jointCapacity);
	return payload;
}

function fillIdentityMatrices(
	target: FloatBuffer,
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
