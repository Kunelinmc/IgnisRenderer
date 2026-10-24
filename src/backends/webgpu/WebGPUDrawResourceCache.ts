import { Matrix4 } from "../../maths/Matrix4";
import type { Matrix3Arr, Matrix4Arr } from "../../maths/types";
import type { Material } from "../../materials/Material";
import { ShaderMaterial } from "../../materials/ShaderMaterial";
import { DRAW_PACKET_FLAG_SHADOW_RECEIVER, type DrawPacket } from "../../pipeline/types";
import {
	BufferUsage,
	type IBindingGroup,
	type IRenderBuffer,
} from "../types";
import {
	WEBGPU_FLAT_MATERIAL_UNIFORM_BYTE_SIZE,
	WEBGPU_MATERIAL_COMMON_UNIFORM_BYTE_SIZE,
	WEBGPU_PBR_MATERIAL_UNIFORM_BYTE_SIZE,
	WEBGPU_PHONG_MATERIAL_UNIFORM_BYTE_SIZE,
	WEBGPU_OBJECT_UNIFORM_BYTE_SIZE,
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
	WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT,
} from "./constants";
import { WEBGPU_OBJECT_UNIFORM_LAYOUT } from "./bufferLayouts";
import { createWebGPUMaterialUniformData } from "./material";
import {
	createObjectUniformWriter,
	writeObjectUniformData,
	packFlatMaterialUniformData,
	packMaterialCommonUniformData,
	packPBRMaterialUniformData,
	packPhongMaterialUniformData,
	type WebGPUObjectUniformWriter,
} from "./packing";
import type { WebGPUMaterialUniformData } from "./types";
import type {
	WebGPUAnimationPayloadPool,
	WebGPUSceneAnimationPayload,
} from "./WebGPUAnimationPayloadPool";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";
import type { WebGPUTextureRegistry } from "./WebGPUTextureRegistry";
import type {
	WebGPUDepthStaticBindingState,
	WebGPUDrawModelBinding,
	WebGPUMaterialBufferResources,
	WebGPUMaterialResource,
} from "./WebGPUResourceContracts";

type FloatBuffer = Float32Array<ArrayBuffer>;
type MatrixRows = Matrix4Arr;

interface ObjectState {
	readonly slot: number;
	readonly arena: IRenderBuffer;
	readonly offset: number;
	readonly dynamicOffsets: readonly [number];
	objectUniformWriter: WebGPUObjectUniformWriter;
	previousModelMatrix: MatrixRows;
	uploadPending: boolean;
}

interface SceneBindingEntry {
	readonly group: IBindingGroup;
	readonly material: WebGPUMaterialResource;
	readonly arena: IRenderBuffer;
	readonly animationGeneration: number;
	readonly animationParams: IRenderBuffer;
	readonly jointMatrices: IRenderBuffer;
	readonly morphWeights: IRenderBuffer;
	readonly morphPosition: IRenderBuffer;
	readonly morphNormal: IRenderBuffer;
	refs: number;
}

interface DepthBindingEntry {
	readonly group: IBindingGroup;
	readonly handles: unknown[];
	readonly sizes: number[];
}

type BindingEntry = SceneBindingEntry | DepthBindingEntry;

interface DrawEntry {
	readonly object: ObjectState | null;
	readonly scene: [SceneBindingEntry | null, SceneBindingEntry | null];
	depth: DepthBindingEntry | null;
	lastUsedFrame: number;
}

interface MaterialEntry {
	readonly revision: number;
	readonly promise: Promise<WebGPUMaterialResource>;
}

interface MaterialState {
	refreshedFrame: number;
	readonly variants: [MaterialEntry | null, MaterialEntry | null];
}

const OBJECT_MODEL_FLOAT_OFFSET = WEBGPU_OBJECT_UNIFORM_LAYOUT.byteOffsetOf("modelMatrix") / 4;
const OBJECT_NORMAL_FLOAT_OFFSET = WEBGPU_OBJECT_UNIFORM_LAYOUT.byteOffsetOf("normalMatrix") / 4;
const OBJECT_INSTANCE_FLOAT_OFFSET = WEBGPU_OBJECT_UNIFORM_LAYOUT.byteOffsetOf("instanceData") / 4;
const MAX_DRAW_ENTRIES = 16_384;
const OBJECT_ARENA_CHUNK_SLOTS = 1024;
const ZERO_DYNAMIC_OFFSETS = [0] as const;
const DEPTH_BINDINGS = [0, 34, 35, 36, 37, 40];
const FALLBACK_UNIFORM_DATA: FloatBuffer = new Float32Array(4);

/** @internal WebGPU frame services cache draw and material records; use Renderer for drawing. */
export class WebGPUDrawResourceCache {
	private readonly _draws = new Map<string | IRenderBuffer, DrawEntry>();
	private _materials = new WeakMap<Material, MaterialState>();
	// Track allocated records for teardown without a second material lookup cache.
	private readonly _liveMaterials = new Set<WebGPUMaterialResource>();
	// Draws sharing a snapshot, animation payload, morph set, and arena reuse one group.
	private readonly _sceneBindings =
		new WeakMap<WebGPUMaterialResource, SceneBindingEntry[]>();
	private readonly _retiredBindings: BindingEntry[] = [];
	private readonly _objectArenas: IRenderBuffer[] = [];
	private readonly _freeObjectSlots: number[] = [];
	private readonly _objectSlotStride: number;
	private _nextObjectSlot = 0;
	private _fallbackShaderUniformBuffer: IRenderBuffer | null = null;
	private _currentFrame = 0;
	private _frameHits = 0;
	private _frameResolves = 0;
	private _destroyed = false;

	constructor(
		private readonly _backend: WebGPUDeviceResourceHost,
		private readonly _layouts: WebGPUPipelineLayouts,
		private readonly _animationPayloads: WebGPUAnimationPayloadPool,
		private readonly _textures: WebGPUTextureRegistry,
	) {
		const alignment =
			_backend.device?.limits?.minUniformBufferOffsetAlignment ?? 256;
		this._objectSlotStride =
			Math.ceil(WEBGPU_OBJECT_UNIFORM_BYTE_SIZE / alignment) * alignment;
	}

	/** @internal WebGPU orchestration advances draw history; use Renderer.renderFrame(). */
	public beginFrame(): void {
		if (this._destroyed) return;
		this._currentFrame++;
		this._frameHits = 0;
		this._frameResolves = 0;
		for (const binding of this._retiredBindings) this._destroyBinding(binding);
		this._retiredBindings.length = 0;
		this._trimDraws();
	}

	/** @internal WebGPU teardown releases owned records once; use Renderer.destroy(). */
	public destroy(): void {
		if (this._destroyed) return;
		this._destroyed = true;
		// Release draw references first so shared scene entries retire exactly once.
		for (const entry of this._draws.values()) this._destroyDraw(entry);
		this._draws.clear();
		for (const binding of this._retiredBindings) this._destroyBinding(binding);
		this._retiredBindings.length = 0;
		for (const entry of this._liveMaterials) this._destroyMaterialBuffers(entry);
		for (const arena of this._objectArenas) arena.destroy();
		this._objectArenas.length = 0;
		this._freeObjectSlots.length = 0;
		this._materials = new WeakMap();
		this._fallbackShaderUniformBuffer?.destroy();
		this._fallbackShaderUniformBuffer = null;
	}

	/** @internal WebGPU frame preparation diagnostics; public rendering uses Renderer. */
	public getDebugStats() {
		let references = 0;
		for (const entry of this._liveMaterials) references += entry.references;
		return {
			materialSnapshots: { frameHits: this._frameHits, frameResolves: this._frameResolves },
			materialBuffers: { entries: this._liveMaterials.size, references },
			objectArena: {
				chunks: this._objectArenas.length,
				slots: this._nextObjectSlot,
				freeSlots: this._freeObjectSlots.length,
			},
		};
	}

	/**
	 * @internal WebGPU preparation resolves one material record per revision and wireframe
	 * variant. Its buffers are allocated only when retained. Use Material for authoring.
	 */
	public resolveMaterial(
		material: Material,
		wireframe: boolean,
	): Promise<WebGPUMaterialResource> {
		const { variants } = this._getMaterialState(material);
		const revision = material._getRevisionInternal();
		const index = wireframe ? 1 : 0;
		const cached = variants[index];
		if (cached?.revision === revision) {
			this._frameHits++;
			return cached.promise;
		}
		this._frameResolves++;
		const data = createWebGPUMaterialUniformData(material, wireframe);
		const promise = this._resolveResources(revision, wireframe, data);
		const entry: MaterialEntry = { revision, promise };
		variants[index] = entry;
		void promise.catch(() => {
			if (this._materials.get(material)?.variants[index] === entry) variants[index] = null;
		});
		return promise;
	}

	/** @internal WebGPU preparation refreshes inputs once per frame; observe Material.revision. */
	public refreshMaterialRevision(material: Material): number {
		this._getMaterialState(material);
		return material._getRevisionInternal();
	}

	private _getMaterialState(material: Material): MaterialState {
		this._assertActive();
		let state = this._materials.get(material);
		if (!state) {
			state = { refreshedFrame: -1, variants: [null, null] };
			this._materials.set(material, state);
		}
		if (material instanceof ShaderMaterial || state.refreshedFrame !== this._currentFrame) {
			material.refreshRevision();
			state.refreshedFrame = this._currentFrame;
		}
		return state;
	}

	/** @internal WebGPU scene preparation resolves a draw's model binding; use Renderer. */
	public getMaterialBinding(
		packet: DrawPacket,
		snapshot: WebGPUMaterialResource,
		animation: WebGPUSceneAnimationPayload,
		morphPositionBuffer: IRenderBuffer | null = null,
		morphNormalBuffer: IRenderBuffer | null = null,
	): WebGPUDrawModelBinding {
		const draw = this._getDraw(packet);
		const object = draw.object!;
		const index = snapshot.wireframe ? 1 : 0;
		const cached = draw.scene[index];
		const fallbackStorageBuffer = this._animationPayloads.getFallbackStorageBuffer();
		const morphPosition = morphPositionBuffer ?? fallbackStorageBuffer;
		const morphNormal = morphNormalBuffer ?? fallbackStorageBuffer;
		if (cached?.material === snapshot && cached.arena === object.arena &&
			cached.animationGeneration === animation.generation &&
			cached.animationParams === animation.paramsBuffer &&
			cached.jointMatrices === animation.jointMatricesBuffer &&
			cached.morphWeights === animation.morphWeightsBuffer &&
			cached.morphPosition === morphPosition && cached.morphNormal === morphNormal) {
			return { group: cached.group, dynamicOffsets: object.dynamicOffsets };
		}
		const entry = this._acquireSceneBinding(
			snapshot, animation, morphPosition, morphNormal, object.arena,
		);
		if (cached) this._releaseSceneBinding(cached);
		draw.scene[index] = entry;
		return { group: entry.group, dynamicOffsets: object.dynamicOffsets };
	}

	private _acquireSceneBinding(
		snapshot: WebGPUMaterialResource,
		animation: WebGPUSceneAnimationPayload,
		morphPosition: IRenderBuffer,
		morphNormal: IRenderBuffer,
		arena: IRenderBuffer,
	): SceneBindingEntry {
		let entries = this._sceneBindings.get(snapshot);
		if (!entries) {
			entries = [];
			this._sceneBindings.set(snapshot, entries);
		}
		for (const entry of entries) {
			if (entry.arena !== arena ||
				entry.animationGeneration !== animation.generation ||
				entry.animationParams !== animation.paramsBuffer ||
				entry.jointMatrices !== animation.jointMatricesBuffer ||
				entry.morphWeights !== animation.morphWeightsBuffer ||
				entry.morphPosition !== morphPosition ||
				entry.morphNormal !== morphNormal) continue;
			entry.refs++;
			return entry;
		}
		const entry = this._createSceneBinding(
			snapshot, animation, morphPosition, morphNormal, arena,
		);
		entry.refs = 1;
		entries.push(entry);
		return entry;
	}

	private _createSceneBinding(
		snapshot: WebGPUMaterialResource,
		animation: WebGPUSceneAnimationPayload,
		morphPosition: IRenderBuffer,
		morphNormal: IRenderBuffer,
		arena: IRenderBuffer,
	): SceneBindingEntry {
		const materialResources = this.retainMaterial(snapshot);
		let group: IBindingGroup;
		try {
			const entries: Array<{ binding: number; resource: any }> = [
				{
					binding: 0,
					resource: {
						buffer: arena,
						offset: 0,
						size: WEBGPU_OBJECT_UNIFORM_BYTE_SIZE,
					},
				},
			];
			for (let i = 0; i < snapshot.textures.length; i++) {
				entries.push({ binding: 1 + i * 2, resource: snapshot.textures[i] });
				if (i < WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT) {
					entries.push({ binding: 2 + i * 2, resource: snapshot.samplers[i] });
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
					resource: morphPosition,
				},
				{
					binding: WEBGPU_MODEL_BINDING_MORPH_NORMAL,
					resource: morphNormal,
				},
				{
					binding: WEBGPU_MODEL_BINDING_SHADER_UNIFORMS,
					resource:
						materialResources.shaderUniformBuffer ??
						this._getFallbackShaderUniformBuffer(),
				},
				{
					binding: WEBGPU_MODEL_BINDING_STATIC_INSTANCES,
					resource: this._animationPayloads.getFallbackStorageBuffer(),
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
			group = this._backend.createBindingGroup({
				label: "ModelBinding_" + snapshot.data.pipelineKey,
				layout: this._layouts.modelBindGroupLayouts[materialResources.shadingFamily],
				entries, cache: false,
			});
		} catch (error) {
			this.releaseMaterial(snapshot);
			throw error;
		}
		return {
			group, material: snapshot, arena,
			animationGeneration: animation.generation,
			animationParams: animation.paramsBuffer,
			jointMatrices: animation.jointMatricesBuffer,
			morphWeights: animation.morphWeightsBuffer,
			morphPosition, morphNormal, refs: 0,
		};
	}

	private _releaseSceneBinding(entry: SceneBindingEntry, retire = true): void {
		if (--entry.refs > 0) return;
		const entries = this._sceneBindings.get(entry.material);
		const index = entries?.indexOf(entry) ?? -1;
		if (index >= 0) entries!.splice(index, 1);
		if (retire) this._retiredBindings.push(entry);
		else this._destroyBinding(entry);
	}

	/** @internal WebGPU depth preparation borrows deformation and static buffers; use Renderer. */
	public getDepthBinding(
		packet: DrawPacket,
		animation: WebGPUSceneAnimationPayload,
		morphPositionBuffer: IRenderBuffer | null,
		staticState?: WebGPUDepthStaticBindingState,
	): WebGPUDrawModelBinding {
		const draw = this._getDraw(packet, staticState);
		const fallback = this._animationPayloads.getFallbackStorageBuffer();
		const objectBuffer = staticState?.objectBuffer ?? draw.object!.arena;
		const objectOffsets = staticState ? ZERO_DYNAMIC_OFFSETS : draw.object!.dynamicOffsets;
		const resources = [
			objectBuffer,
			animation.paramsBuffer, animation.jointMatricesBuffer, animation.morphWeightsBuffer,
			morphPositionBuffer ?? fallback, staticState?.instanceBuffer ?? fallback,
		];
		const cached = draw.depth;
		if (cached && resources.every(
			(resource, index) => (resource._gpuResource ?? resource) === cached.handles[index] &&
				resource.size === cached.sizes[index],
		)) return { group: cached.group, dynamicOffsets: objectOffsets };
		const group = this._backend.createBindingGroup({
			label: "WebGPUDepthModel_" + packet.submission.id,
			layout: this._layouts.depthModelBindGroupLayout,
			entries: resources.map((resource, index) => ({
				binding: DEPTH_BINDINGS[index],
				resource: index === 0 ? {
					buffer: resource,
					offset: 0,
					size: WEBGPU_OBJECT_UNIFORM_BYTE_SIZE,
				} : resource,
			})),
			cache: false,
		});
		if (cached) this._retiredBindings.push(cached);
		draw.depth = {
			group,
			handles: resources.map((resource) => resource._gpuResource ?? resource),
			sizes: resources.map((resource) => resource.size),
		};
		return { group, dynamicOffsets: objectOffsets };
	}

	private _getDraw(packet: DrawPacket, staticState?: WebGPUDepthStaticBindingState): DrawEntry {
		this._assertActive();
		const key = staticState?.instanceBuffer ?? packet.submission.id;
		let entry = this._draws.get(key);
		const previousFrame = entry?.lastUsedFrame ?? -1;
		if (!entry) {
			entry = {
				object: staticState ? null : this._createObjectState(),
				scene: [null, null], depth: null, lastUsedFrame: this._currentFrame,
			};
			this._draws.set(key, entry);
			this._trimDraws();
		} else if (previousFrame !== this._currentFrame) {
			entry.lastUsedFrame = this._currentFrame;
			this._draws.delete(key);
			this._draws.set(key, entry);
		}
		if (entry.object) this._updateObjectBuffer(packet, entry.object, previousFrame);
		return entry;
	}

	private _trimDraws(): void {
		while (this._draws.size > MAX_DRAW_ENTRIES) {
			const oldest = this._draws.entries().next().value;
			if (!oldest || oldest[1].lastUsedFrame === this._currentFrame) break;
			this._draws.delete(oldest[0]);
			this._destroyDraw(oldest[1]);
		}
	}

	private _destroyDraw(entry: DrawEntry): void {
		for (const binding of entry.scene) {
			if (binding) this._releaseSceneBinding(binding, false);
		}
		if (entry.depth) this._destroyBinding(entry.depth);
		if (entry.object) this._freeObjectSlots.push(entry.object.slot);
	}

	private _destroyBinding(entry: BindingEntry): void {
		(entry.group as { destroy?: () => void }).destroy?.();
		if ("material" in entry) this.releaseMaterial(entry.material);
	}

	/**
	 * @internal WebGPU dynamic and static bindings retain a cache-created material record.
	 * Returns its shared buffers without creating a reference handle. Each successful retain
	 * must be balanced by one releaseMaterial() call. Use Material for public authoring.
	 */
	public retainMaterial(material: WebGPUMaterialResource): WebGPUMaterialBufferResources {
		this._assertActive();
		if (!material.buffers) {
			material.buffers = this._createMaterialBuffers(material.data);
			this._liveMaterials.add(material);
		}
		material.references++;
		return material.buffers;
	}

	/**
	 * @internal WebGPU binding disposal releases one previously retained material reference.
	 * The last release destroys its buffers. Releases after cache teardown are harmless;
	 * duplicate releases on a live cache are invalid. Use Renderer for public lifetime control.
	 */
	public releaseMaterial(material: WebGPUMaterialResource): void {
		if (this._destroyed) return;
		if (material.references <= 0) throw new Error("WebGPU material has no retained reference.");
		if (--material.references === 0) this._destroyMaterialBuffers(material);
	}

	private _destroyMaterialBuffers(entry: WebGPUMaterialResource): void {
		entry.buffers?.commonBuffer.destroy();
		entry.buffers?.lightingBuffer?.destroy();
		entry.buffers?.shaderUniformBuffer?.destroy();
		entry.buffers = null;
		entry.references = 0;
		this._liveMaterials.delete(entry);
	}

	private async _resolveResources(
		revision: number,
		wireframe: boolean,
		data: WebGPUMaterialUniformData,
	): Promise<WebGPUMaterialResource> {
		const textures = await Promise.all(data.common.textureSlots.map((slot, index) =>
			this._textures.getTextureForSlotAsync(slot.map, index, slot.color, slot.forceSRGB),
		));
		this._assertActive();
		const samplers = data.common.textureSlots.map((slot) =>
			this._textures.getSamplerForTexture(slot.map),
		);
		return { revision, wireframe, data, textures, samplers, buffers: null, references: 0 };
	}

	private _createMaterialBuffers(data: WebGPUMaterialUniformData): WebGPUMaterialBufferResources {
		const commonBuffer = this._createBuffer(
			WEBGPU_MATERIAL_COMMON_UNIFORM_BYTE_SIZE,
			`WebGPUMaterialCommon_${data.shadingFamily}`,
			packMaterialCommonUniformData(data.common),
		);
		let lightingBuffer: IRenderBuffer | null = null;
		let shaderUniformBuffer: IRenderBuffer | null = null;
		try {
			switch (data.shadingFamily) {
				case "pbr":
					lightingBuffer = this._createBuffer(
						WEBGPU_PBR_MATERIAL_UNIFORM_BYTE_SIZE,
						"WebGPUPBRMaterial",
						packPBRMaterialUniformData(data.lighting),
					);
					break;
				case "phong":
					lightingBuffer = this._createBuffer(
						WEBGPU_PHONG_MATERIAL_UNIFORM_BYTE_SIZE,
						"WebGPUPhongMaterial",
						packPhongMaterialUniformData(data.lighting),
					);
					break;
				case "flat":
					lightingBuffer = this._createBuffer(
						WEBGPU_FLAT_MATERIAL_UNIFORM_BYTE_SIZE,
						"WebGPUFlatMaterial",
						packFlatMaterialUniformData(data.lighting),
					);
					break;
				case "unlit":
					break;
			}
			const uniforms = data.shaderUniforms;
			if (uniforms.data && uniforms.byteLength > 0) {
				shaderUniformBuffer = this._createBuffer(
					Math.max(16, uniforms.byteLength), "ShaderMaterialUniform_" + data.pipelineKey,
					uniforms.data,
				);
			}
		} catch (error) {
			commonBuffer.destroy();
			lightingBuffer?.destroy();
			shaderUniformBuffer?.destroy();
			throw error;
		}
		return {
			shadingFamily: data.shadingFamily,
			commonBuffer,
			lightingBuffer,
			shaderUniformBuffer,
		};
	}

	private _createBuffer(
		size: number,
		label: string,
		data: Float32Array<ArrayBuffer> | Uint8Array<ArrayBuffer>,
	): IRenderBuffer {
		const buffer = this._backend.createBuffer({
			size,
			usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			label,
		});
		try {
			this._backend.writeBuffer(buffer, data);
		} catch (error) {
			buffer.destroy();
			throw error;
		}
		return buffer;
	}

	private _createObjectState(): ObjectState {
		const slot = this._freeObjectSlots.pop() ?? this._nextObjectSlot++;
		const chunkIndex = Math.floor(slot / OBJECT_ARENA_CHUNK_SLOTS);
		const chunkSlot = slot - chunkIndex * OBJECT_ARENA_CHUNK_SLOTS;
		while (this._objectArenas.length <= chunkIndex) {
			this._objectArenas.push(this._backend.createBuffer({
				size: OBJECT_ARENA_CHUNK_SLOTS * this._objectSlotStride,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: `WebGPUObjectUniformArena_${this._objectArenas.length}`,
			}));
		}
		const offset = chunkSlot * this._objectSlotStride;
		return {
			slot,
			arena: this._objectArenas[chunkIndex],
			offset,
			dynamicOffsets: [offset],
			objectUniformWriter: createObjectUniformWriter(),
			previousModelMatrix: createMatrixRows(),
			uploadPending: true,
		};
	}

	private _updateObjectBuffer(
		packet: DrawPacket, entry: ObjectState, previousFrame: number,
	): void {
		const instance = packet.submission.instance;
		const packed = entry.objectUniformWriter.toFloat32Array();
		const receiveShadows = (packet.submission.passFlags & DRAW_PACKET_FLAG_SHADOW_RECEIVER) !== 0;
		let uniformDirty = entry.uploadPending ||
			packed[OBJECT_INSTANCE_FLOAT_OFFSET] !==
				Math.fround(Math.max(0, Math.floor(instance.renderLayers)) >>> 0) ||
			packed[OBJECT_INSTANCE_FLOAT_OFFSET + 1] !== (receiveShadows ? 1 : 0) ||
			!matrixMatchesPacked(instance.worldMatrix, packed, OBJECT_MODEL_FLOAT_OFFSET, 4) ||
			!matrixMatchesPacked(instance.normalMatrix, packed, OBJECT_NORMAL_FLOAT_OFFSET, 3);

		// The owning draw advances history before any pass attempts an upload.
		if (previousFrame !== this._currentFrame) {
			if (instance.previousWorldMatrix) {
				uniformDirty = copyMatrixToRows(instance.previousWorldMatrix, entry.previousModelMatrix) ||
					uniformDirty;
			} else if (previousFrame < 0) {
				copyMatrixToRows(instance.worldMatrix, entry.previousModelMatrix);
			} else {
				uniformDirty = copyPackedModelToRows(packed, entry.previousModelMatrix) || uniformDirty;
			}
		}
		if (!uniformDirty) return;
		// Keep failed packing/uploads pending without advancing history again on retry.
		entry.uploadPending = true;
		const uniformData = writeObjectUniformData(
			entry.objectUniformWriter, instance.worldMatrix, instance.normalMatrix,
			entry.previousModelMatrix, instance.renderLayers, receiveShadows,
		);
		this._backend.writeBuffer(entry.arena, uniformData, entry.offset);
		entry.uploadPending = false;
	}

	private _getFallbackShaderUniformBuffer(): IRenderBuffer {
		return this._fallbackShaderUniformBuffer ??= this._createBuffer(
			16, "WebGPUFallbackShaderUniformBuffer", FALLBACK_UNIFORM_DATA,
		);
	}

	private _assertActive(): void {
		if (this._destroyed) throw new Error("WebGPU draw resource cache is destroyed.");
	}
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

function matrixMatchesPacked(
	matrix: Matrix4 | Matrix3Arr,
	packed: FloatBuffer,
	offset: number,
	dimension: 3 | 4,
): boolean {
	const rows = matrix instanceof Matrix4 ? matrix.elements : matrix;
	for (let column = 0; column < dimension; column++) {
		for (let row = 0; row < dimension; row++) {
			// Compare against the canonical column-major, float32 payload.
			if (packed[offset + column * 4 + row] !==
				Math.fround(rows[row][column])) return false;
		}
	}
	return true;
}

function copyPackedModelToRows(packed: FloatBuffer, target: MatrixRows): boolean {
	let changed = false;
	for (let row = 0; row < 4; row++) {
		for (let column = 0; column < 4; column++) {
			const value = packed[OBJECT_MODEL_FLOAT_OFFSET + column * 4 + row];
			if (target[row][column] === value) continue;
			target[row][column] = value;
			changed = true;
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
