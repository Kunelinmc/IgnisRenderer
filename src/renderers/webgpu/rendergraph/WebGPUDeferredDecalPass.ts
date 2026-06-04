import {
	DECAL_CHANNELS,
	resolveDecalChannelBlendMode,
	type DecalBlendMode,
} from "../../../decals";
import { ShadingModel, type Material } from "../../../materials/Material";
import type {
	DecalPacket,
	FrameContext,
} from "../../../pipeline/types";
import type { ICommandEncoder } from "../../ICommandEncoder";
import {
	BufferUsage,
	TextureFormat,
	TextureUsage,
	type IBindingGroup,
	type IRenderBuffer,
	type IRenderTexture,
	type ISampler,
} from "../../types";
import type { WebGPUBackend } from "../../WebGPUBackend";
import {
	WEBGPU_MODEL_BINDING_ANISOTROPY_TEXTURE,
	WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT,
	WEBGPU_TEXTURE_SLOT_COUNT,
} from "../constants";
import {
	createWebGPUMaterialUniformData,
} from "../material";
import { packMatrix4ForWGSL } from "../packing";
import type {
	WebGPUMaterialUniformData,
	WebGPUTextureSlotData,
} from "../types";
import type {
	WebGPUPreparedFrameResources,
	WebGPURenderResources,
} from "../WebGPURenderResources";
import type { WebGPUFrameTargets } from "../WebGPUPostProcessContracts";

export interface WebGPUDeferredDecalPassCallbacks {
	getEncoder(): ICommandEncoder | null;
	getFrameTargets(): WebGPUFrameTargets | null;
	requireFrameResources(): WebGPUPreparedFrameResources;
	resolveDirtyRects(
		context: FrameContext,
		width: number,
		height: number
	): Array<{ x: number; y: number; width: number; height: number }>;
}

interface DecalTargetRef {
	texture: IRenderTexture;
	format: TextureFormat;
	label: string;
}

interface DecalMaterialBindingCacheEntry {
	group: IBindingGroup;
	textures: IRenderTexture[];
	samplers: ISampler[];
	anisotropyTexture: IRenderTexture;
}

const DECAL_MODE_VALUE: Record<DecalBlendMode, number> = {
	disabled: 0,
	lerp: 1,
	replace: 2,
	multiply: 3,
	add: 4,
	normal: 5,
};

const DECAL_UNIFORM_FLOATS =
	16 +
	16 +
	4 +
	15 * 4 +
	WEBGPU_TEXTURE_SLOT_COUNT * 4 +
	WEBGPU_TEXTURE_SLOT_COUNT * 4 +
	5 * 4;
const DECAL_UNIFORM_BYTES = DECAL_UNIFORM_FLOATS * 4;
const DECAL_LAYER_MASK_SUPPORTED_BITS = 0x7ff;
const DECAL_REQUIRED_FRAGMENT_SAMPLED_TEXTURES =
	WEBGPU_TEXTURE_SLOT_COUNT + 1 + 11;
const DECAL_REQUIRED_FRAGMENT_SAMPLERS =
	WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT;

/**
 * Applies scene-graph decals by modifying deferred G-buffer channels.
 */
export class WebGPUDeferredDecalPass {
	private readonly _backend: WebGPUBackend;
	private readonly _resources: WebGPURenderResources;
	private readonly _callbacks: WebGPUDeferredDecalPassCallbacks;
	private _uniformBuffer: IRenderBuffer | null = null;
	private _snapshotTextures: IRenderTexture[] = [];
	private _snapshotReadBinding: IBindingGroup | null = null;
	private _snapshotKey = "";
	private _gbufferWriteBinding: IBindingGroup | null = null;
	private _gbufferWriteSources: IRenderTexture[] = [];
	private _materialBindings = new WeakMap<
		Material,
		DecalMaterialBindingCacheEntry
	>();
	private _materialBindingEntries = new Set<DecalMaterialBindingCacheEntry>();

	public constructor(
		backend: WebGPUBackend,
		resources: WebGPURenderResources,
		callbacks: WebGPUDeferredDecalPassCallbacks
	) {
		this._backend = backend;
		this._resources = resources;
		this._callbacks = callbacks;
	}

	public destroyBindings(): void {
		this._destroyBindingGroup(this._snapshotReadBinding);
		this._snapshotReadBinding = null;
		this._destroyBindingGroup(this._gbufferWriteBinding);
		this._gbufferWriteBinding = null;
		this._gbufferWriteSources = [];
		for (const entry of this._materialBindingEntries) {
			this._destroyBindingGroup(entry.group);
		}
		this._materialBindings = new WeakMap();
		this._materialBindingEntries.clear();
		this._uniformBuffer?.destroy();
		this._uniformBuffer = null;
		for (const texture of this._snapshotTextures) {
			texture.destroy();
		}
		this._snapshotTextures = [];
		this._snapshotKey = "";
	}

	public async recordDecalPass(context: FrameContext): Promise<number> {
		const encoder = this._callbacks.getEncoder();
		const targets = this._callbacks.getFrameTargets();
		const decalPackets = context.scene.decalPackets;
		if (
			!encoder ||
			!targets ||
			decalPackets.length <= 0 ||
			typeof encoder.copyTextureToTexture !== "function" ||
			!this._deviceSupportsDecalPipeline()
		) {
			return 0;
		}

		const targetRefs = resolveDecalTargets(targets);
		if (!targetRefs) {
			return 0;
		}
		const activeDecals = decalPackets.filter((packet) =>
			hasSupportedLayerMask(packet.receiverLayerMask)
		);
		if (activeDecals.length <= 0) {
			return 0;
		}

		this._ensureSnapshotTextures(targetRefs);
		const snapshotReadBinding = this._getSnapshotReadBinding();
		const gbufferWriteBinding = this._getGBufferWriteBinding(targetRefs);
		const uniformBuffer = this._getUniformBuffer();
		const pipeline = await this._resources.getDecalPipeline();
		const frameResources = this._callbacks.requireFrameResources();
		const dirtyRects = this._callbacks.resolveDirtyRects(
			context,
			targetRefs[0].texture.width,
			targetRefs[0].texture.height
		);

		let drawCount = 0;
		for (const packet of activeDecals) {
			const materialData = createDecalMaterialUniformData(packet.material);
			const decalBinding = this._getMaterialBinding(
				packet.material,
				materialData,
				uniformBuffer
			);
			this._backend.writeBuffer(
				uniformBuffer,
				createDecalUniformData(packet, materialData)
			);
			this._copyTargetsToSnapshots(encoder, targetRefs);
			encoder.beginRenderPass({
				label: `WebGPUDeferredDecal_${packet.id}`,
				colorAttachments: targetRefs.slice(0, 7).map((target) => ({
					view: target.texture,
					loadOp: "load" as const,
					storeOp: "store" as const,
				})),
			});
			encoder.setPipeline(pipeline);
			encoder.setBindingGroup(0, frameResources.decalFrameBinding);
			encoder.setBindingGroup(1, snapshotReadBinding);
			encoder.setBindingGroup(2, decalBinding);
			encoder.setBindingGroup(3, gbufferWriteBinding);
			for (const rect of dirtyRects) {
				encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
				encoder.draw(3);
				drawCount++;
			}
			encoder.endRenderPass();
		}
		return drawCount;
	}

	private _getUniformBuffer(): IRenderBuffer {
		if (!this._uniformBuffer) {
			this._uniformBuffer = this._backend.createBuffer({
				size: DECAL_UNIFORM_BYTES,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: "WebGPUDecalUniform",
			});
		}
		return this._uniformBuffer;
	}

	private _deviceSupportsDecalPipeline(): boolean {
		const limits = this._backend.device?.limits;
		const maxSampledTextures =
			limits?.maxSampledTexturesPerShaderStage ?? Number.POSITIVE_INFINITY;
		const maxSamplers =
			limits?.maxSamplersPerShaderStage ?? Number.POSITIVE_INFINITY;
		return (
			maxSampledTextures >= DECAL_REQUIRED_FRAGMENT_SAMPLED_TEXTURES &&
			maxSamplers >= DECAL_REQUIRED_FRAGMENT_SAMPLERS
		);
	}

	private _ensureSnapshotTextures(targets: readonly DecalTargetRef[]): void {
		const snapshotKey = targets
			.map((target) =>
				[
					target.texture.width,
					target.texture.height,
					target.format,
					target.label,
				].join(":")
			)
			.join("|");
		if (
			this._snapshotKey === snapshotKey &&
			this._snapshotTextures.length === targets.length
		) {
			return;
		}
		this._destroyBindingGroup(this._snapshotReadBinding);
		this._snapshotReadBinding = null;
		for (const texture of this._snapshotTextures) {
			texture.destroy();
		}
		this._snapshotTextures = targets.map((target) =>
			this._backend.createTexture({
				width: target.texture.width,
				height: target.texture.height,
				format: target.format,
				usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
				label: `WebGPUDecalSnapshot_${target.label}`,
			})
		);
		this._snapshotKey = snapshotKey;
	}

	private _getSnapshotReadBinding(): IBindingGroup {
		if (!this._snapshotReadBinding) {
			this._snapshotReadBinding = this._backend.createBindingGroup({
				layout: this._resources.getGBufferReadLayout(),
				entries: this._snapshotTextures.map((resource, binding) => ({
					binding,
					resource,
				})),
				label: "WebGPUDecalSnapshotReadBinding",
			});
		}
		return this._snapshotReadBinding;
	}

	private _getGBufferWriteBinding(
		targets: readonly DecalTargetRef[]
	): IBindingGroup {
		const sources = targets.slice(7, 11).map((target) => target.texture);
		if (
			this._gbufferWriteBinding &&
			this._gbufferWriteSources.length === sources.length &&
			this._gbufferWriteSources.every(
				(source, index) => source === sources[index]
			)
		) {
			return this._gbufferWriteBinding;
		}
		this._destroyBindingGroup(this._gbufferWriteBinding);
		this._gbufferWriteBinding = this._backend.createBindingGroup({
			layout: this._resources.getGBufferWriteLayout(),
			entries: sources.map((resource, binding) => ({
				binding,
				resource,
			})),
			label: "WebGPUDecalGBufferWriteBinding",
		});
		this._gbufferWriteSources = sources;
		return this._gbufferWriteBinding;
	}

	private _getMaterialBinding(
		material: Material,
		materialData: WebGPUMaterialUniformData,
		uniformBuffer: IRenderBuffer
	): IBindingGroup {
		const textures = materialData.textureSlots.map((slot, index) =>
			this._resources.getTextureForSlot(slot.map, index)
		);
		const samplers = materialData.textureSlots.map((slot) =>
			this._resources.getSamplerForTexture(slot.map)
		);
		const anisotropyTexture = this._resources.getTextureForSlot(
			materialData.anisotropyTexture.map,
			-1
		);
		const cached = this._materialBindings.get(material);
		if (
			cached &&
			areTexturesEqual(cached.textures, textures) &&
			areSamplersEqual(cached.samplers, samplers) &&
			cached.anisotropyTexture === anisotropyTexture
		) {
			return cached.group;
		}

		if (cached) {
			this._destroyBindingGroup(cached.group);
			this._materialBindingEntries.delete(cached);
		}
		const entries: Array<{ binding: number; resource: any }> = [
			{ binding: 0, resource: uniformBuffer },
		];
		for (let i = 0; i < WEBGPU_TEXTURE_SLOT_COUNT; i++) {
			entries.push({ binding: 1 + i * 2, resource: textures[i] });
			if (i < WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT) {
				entries.push({ binding: 2 + i * 2, resource: samplers[i] });
			}
		}
		entries.push({
			binding: WEBGPU_MODEL_BINDING_ANISOTROPY_TEXTURE,
			resource: anisotropyTexture,
		});
		const group = this._backend.createBindingGroup({
			layout: this._resources.getDecalBindGroupLayout(),
			entries,
			label: `WebGPUDecalMaterialBinding_${material.name}`,
		});
		const entry = {
			group,
			textures,
			samplers,
			anisotropyTexture,
		};
		this._materialBindings.set(material, entry);
		this._materialBindingEntries.add(entry);
		return group;
	}

	private _copyTargetsToSnapshots(
		encoder: ICommandEncoder,
		targets: readonly DecalTargetRef[]
	): void {
		for (let i = 0; i < targets.length; i++) {
			encoder.copyTextureToTexture!(
				{ texture: targets[i].texture },
				{ texture: this._snapshotTextures[i] },
				{
					width: targets[i].texture.width,
					height: targets[i].texture.height,
					depthOrArrayLayers: 1,
				}
			);
		}
	}

	private _destroyBindingGroup(group: IBindingGroup | null): void {
		const destroyFn = (group as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(group);
		}
	}
}

function createDecalMaterialUniformData(
	material: Material
): WebGPUMaterialUniformData {
	const data = createWebGPUMaterialUniformData(material, false);
	if (
		material.shading === ShadingModel.Phong ||
		material.shading === ShadingModel.Gouraud ||
		material.shading === ShadingModel.Flat
	) {
		const shininess = Math.max(0, data.phongAmbientShininess[3]);
		const roughness = Math.max(
			0.04,
			Math.min(1, Math.sqrt(2 / Math.max(2, shininess + 2)))
		);
		data.surfaceParams0 = [roughness, 0, 0.5, data.surfaceParams0[3]];
		data.specularColorFactor = [
			data.phongSpecularShading[0],
			data.phongSpecularShading[1],
			data.phongSpecularShading[2],
			1,
		];
	}
	return data;
}

function createDecalUniformData(
	packet: DecalPacket,
	materialData: WebGPUMaterialUniformData
): Float32Array<ArrayBuffer> {
	const data = new Float32Array(DECAL_UNIFORM_FLOATS);
	let cursor = 0;
	cursor = writePackedMatrix(data, cursor, packet.inverseWorldMatrix);
	cursor = writePackedMatrix(data, cursor, packet.worldMatrix);
	cursor = writeVec4(data, cursor, [
		packet.opacity,
		packet.edgeFade,
		packet.receiverLayerMask & DECAL_LAYER_MASK_SUPPORTED_BITS,
		0,
	]);
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
		cursor = writeVec4(data, cursor, values);
	}
	for (let i = 0; i < WEBGPU_TEXTURE_SLOT_COUNT; i++) {
		cursor = writeVec4(
			data,
			cursor,
			resolveTextureSlot(materialData.textureSlots, i).transformA
		);
	}
	for (let i = 0; i < WEBGPU_TEXTURE_SLOT_COUNT; i++) {
		cursor = writeVec4(
			data,
			cursor,
			resolveTextureSlot(materialData.textureSlots, i).transformB
		);
	}
	for (let modeOffset = 0; modeOffset < 20; modeOffset += 4) {
		cursor = writeVec4(data, cursor, [
			encodeBlendMode(packet, modeOffset),
			encodeBlendMode(packet, modeOffset + 1),
			encodeBlendMode(packet, modeOffset + 2),
			encodeBlendMode(packet, modeOffset + 3),
		]);
	}
	return data;
}

function encodeBlendMode(packet: DecalPacket, channelIndex: number): number {
	const channel = DECAL_CHANNELS[channelIndex];
	if (!channel) {
		return 0;
	}
	return DECAL_MODE_VALUE[
		resolveDecalChannelBlendMode(packet.channelBlendModes, channel)
	];
}

function writePackedMatrix(
	target: Float32Array,
	offset: number,
	matrix: Parameters<typeof packMatrix4ForWGSL>[0]
): number {
	target.set(packMatrix4ForWGSL(matrix), offset);
	return offset + 16;
}

function writeVec4(
	target: Float32Array,
	offset: number,
	values: readonly number[]
): number {
	target[offset] = values[0] ?? 0;
	target[offset + 1] = values[1] ?? 0;
	target[offset + 2] = values[2] ?? 0;
	target[offset + 3] = values[3] ?? 0;
	return offset + 4;
}

function resolveTextureSlot(
	slots: readonly WebGPUTextureSlotData[],
	index: number
): WebGPUTextureSlotData {
	return slots[index] ?? {
		map: null,
		transformA: [0, 0, 1, 1],
		transformB: [0, 0, 1, 0],
	};
}

function resolveDecalTargets(
	targets: WebGPUFrameTargets
): DecalTargetRef[] | null {
	if (
		!targets.gAlbedoAlpha ||
		!targets.gNormalRoughMetal ||
		!targets.gEmissiveOcclusion ||
		!targets.gMotionDepth ||
		!targets.gSpecular ||
		!targets.gCoatSheen ||
		!targets.gSheenReflectance ||
		!targets.gMaterialExt0 ||
		!targets.gMaterialExt1 ||
		!targets.gMaterialExt2 ||
		!targets.gMaterialExt3
	) {
		return null;
	}
	return [
		{
			texture: targets.gAlbedoAlpha,
			format: TextureFormat.RGBA8Unorm,
			label: "AlbedoAlpha",
		},
		{
			texture: targets.gNormalRoughMetal,
			format: TextureFormat.RGBA16Float,
			label: "NormalRoughMetal",
		},
		{
			texture: targets.gEmissiveOcclusion,
			format: TextureFormat.RGBA16Float,
			label: "EmissiveOcclusion",
		},
		{
			texture: targets.gMotionDepth,
			format: TextureFormat.RGBA16Float,
			label: "MotionDepth",
		},
		{
			texture: targets.gSpecular,
			format: TextureFormat.RGBA16Float,
			label: "Specular",
		},
		{
			texture: targets.gCoatSheen,
			format: TextureFormat.RGBA16Float,
			label: "CoatSheen",
		},
		{
			texture: targets.gSheenReflectance,
			format: TextureFormat.RGBA16Float,
			label: "SheenReflectance",
		},
		{
			texture: targets.gMaterialExt0,
			format: TextureFormat.RGBA16Float,
			label: "MaterialExt0",
		},
		{
			texture: targets.gMaterialExt1,
			format: TextureFormat.RGBA16Float,
			label: "MaterialExt1",
		},
		{
			texture: targets.gMaterialExt2,
			format: TextureFormat.RGBA16Float,
			label: "MaterialExt2",
		},
		{
			texture: targets.gMaterialExt3,
			format: TextureFormat.RGBA16Float,
			label: "MaterialExt3",
		},
	];
}

function hasSupportedLayerMask(mask: number): boolean {
	return (mask & DECAL_LAYER_MASK_SUPPORTED_BITS) !== 0;
}

function areTexturesEqual(
	left: readonly IRenderTexture[],
	right: readonly IRenderTexture[]
): boolean {
	if (left.length !== right.length) return false;
	for (let i = 0; i < left.length; i++) {
		if (left[i] !== right[i]) return false;
	}
	return true;
}

function areSamplersEqual(
	left: readonly ISampler[],
	right: readonly ISampler[]
): boolean {
	if (left.length !== right.length) return false;
	for (let i = 0; i < left.length; i++) {
		if (left[i] !== right[i]) return false;
	}
	return true;
}
