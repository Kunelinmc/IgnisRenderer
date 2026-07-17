import {
	DECAL_CHANNELS,
	resolveDecalChannelBlendMode,
	type DecalBlendMode,
} from "../../../decals";
import { ShadingModel, type Material } from "../../../materials/Material";
import type {
	DecalPacket,
	DrawPacket,
	FrameContext,
} from "../../../pipeline/types";
import type { DirtyRect } from "../../../pipeline/incremental";
import { computePacketScreenRect } from "../../../pipeline/screenBounds";
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
import type { WebGPUFrameHost } from "./WebGPUFrameHost";
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
} from "../WebGPUResourceContracts";
import type {
	WebGPUDeferredResourceProvider,
	WebGPUTextureResourceProvider,
} from "../WebGPUResourceContracts";
import type { WebGPUFrameTargets } from "../WebGPUPostProcessContracts";
import type { WebGPUFrameGraphRecordingContext } from "./WebGPUFrameGraphRecordingContext";

export interface WebGPUDeferredDecalPassCallbacks {
	readonly recordingContext: WebGPUFrameGraphRecordingContext;
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

interface DecalWorkItem {
	packet: DecalPacket;
	materialData: WebGPUMaterialUniformData;
	binding: IBindingGroup;
	rects: DirtyRect[];
	unionRect: DirtyRect;
}

type DecalExecutionSegment =
	| {
			kind: "batch";
			items: DecalWorkItem[];
		}
	| {
			kind: "fallback";
			item: DecalWorkItem;
		};

interface DecalBatchData {
	rect: DirtyRect;
	decalUniforms: Float32Array;
	tileHeaders: Uint32Array;
	tileDecalIndices: Uint32Array;
	tileColumns: number;
	tileRows: number;
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
const DECAL_BATCH_REQUIRED_STORAGE_TEXTURES = 11;
const DECAL_BATCH_REQUIRED_STORAGE_BUFFERS = 3;
const DECAL_BATCH_TILE_SIZE = 16;
const DECAL_BATCH_WORKGROUP_SIZE = 8;

/**
 * Applies scene-graph decals by modifying deferred G-buffer channels.
 */
export class WebGPUDeferredDecalPass {
	private readonly _host: WebGPUFrameHost;
	private readonly _resources: WebGPUDeferredResourceProvider &
		WebGPUTextureResourceProvider;
	private readonly _recordingContext: WebGPUFrameGraphRecordingContext;
	private _uniformBuffer: IRenderBuffer | null = null;
	private _snapshotTextures: IRenderTexture[] = [];
	private _snapshotReadBinding: IBindingGroup | null = null;
	private _snapshotKey = "";
	private _batchParamsBuffer: IRenderBuffer | null = null;
	private _batchDecalsBuffer: IRenderBuffer | null = null;
	private _batchTileHeadersBuffer: IRenderBuffer | null = null;
	private _batchTileIndicesBuffer: IRenderBuffer | null = null;
	private _batchBinding: IBindingGroup | null = null;
	private _batchBindingSources: unknown[] = [];
	private _outputBinding: IBindingGroup | null = null;
	private _outputBindingSources: unknown[] = [];
	private _materialBindings = new WeakMap<
		Material,
		DecalMaterialBindingCacheEntry
	>();
	private _materialBindingEntries = new Set<DecalMaterialBindingCacheEntry>();

	public constructor(
		host: WebGPUFrameHost,
		resources: WebGPUDeferredResourceProvider & WebGPUTextureResourceProvider,
		callbacks: WebGPUDeferredDecalPassCallbacks
	) {
		this._host = host;
		this._resources = resources;
		this._recordingContext = callbacks.recordingContext;
	}

	public destroyBindings(): void {
		this._destroyBindingGroup(this._snapshotReadBinding);
		this._snapshotReadBinding = null;
		this._destroyBindingGroup(this._batchBinding);
		this._batchBinding = null;
		this._batchBindingSources = [];
		this._destroyBindingGroup(this._outputBinding);
		this._outputBinding = null;
		this._outputBindingSources = [];
		for (const entry of this._materialBindingEntries) {
			this._destroyBindingGroup(entry.group);
		}
		this._materialBindings = new WeakMap();
		this._materialBindingEntries.clear();
		this._uniformBuffer?.destroy();
		this._uniformBuffer = null;
		this._batchParamsBuffer?.destroy();
		this._batchParamsBuffer = null;
		this._batchDecalsBuffer?.destroy();
		this._batchDecalsBuffer = null;
		this._batchTileHeadersBuffer?.destroy();
		this._batchTileHeadersBuffer = null;
		this._batchTileIndicesBuffer?.destroy();
		this._batchTileIndicesBuffer = null;
		for (const texture of this._snapshotTextures) {
			texture.destroy();
		}
		this._snapshotTextures = [];
		this._snapshotKey = "";
	}

	public async recordDecalPass(context: FrameContext): Promise<number> {
		const encoder = this._recordingContext.getEncoder();
		const targets = this._recordingContext.getFrameTargets();
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
		const uniformBuffer = this._getUniformBuffer();
		const frameResources = this._recordingContext.requireFrameResources();
		const dirtyRects = this._recordingContext.resolveDirtyRects(
			context,
			targetRefs[0].texture.width,
			targetRefs[0].texture.height
		);
		const workItems = await this._createWorkItems(
			context,
			activeDecals,
			dirtyRects,
			targetRefs[0].texture.width,
			targetRefs[0].texture.height,
			uniformBuffer
		);
		if (workItems.length <= 0) {
			return 0;
		}

		const segments = this._buildExecutionSegments(workItems);
		const batchSupported = this._deviceSupportsDecalBatchPipeline();

		let drawCount = 0;
		for (const segment of segments) {
			if (segment.kind === "batch" && batchSupported) {
				drawCount += await this._recordBatchSegment(
					encoder,
					targetRefs,
					snapshotReadBinding,
					frameResources.decalFrameBinding,
					segment.items
				);
				continue;
			}

			if (segment.kind === "batch") {
				for (const item of segment.items) {
					drawCount += await this._recordFallbackDecal(
						encoder,
						targetRefs,
						snapshotReadBinding,
						this._getGBufferWriteBinding(targetRefs),
						frameResources.decalFrameBinding,
						item,
						uniformBuffer
					);
				}
				continue;
			}

			drawCount += await this._recordFallbackDecal(
				encoder,
				targetRefs,
				snapshotReadBinding,
				this._getGBufferWriteBinding(targetRefs),
				frameResources.decalFrameBinding,
				segment.item,
				uniformBuffer
			);
		}
		return drawCount;
	}

	private async _createWorkItems(
		context: FrameContext,
		packets: readonly DecalPacket[],
		dirtyRects: readonly DirtyRect[],
		width: number,
		height: number,
		uniformBuffer: IRenderBuffer
	): Promise<DecalWorkItem[]> {
		if (dirtyRects.length <= 0) {
			return [];
		}
		const fullScreenRect = { x: 0, y: 0, width, height };
		const items: DecalWorkItem[] = [];
		for (const packet of packets) {
			const screenRect = computePacketScreenRect(
				packet,
				context.viewCamera,
				width,
				height
			);
			if (
				screenRect &&
				!this._hasReceiverCandidate(context, packet, screenRect)
			) {
				continue;
			}
			const decalRect = screenRect ?? fullScreenRect;
			const rects = intersectDirtyRects(dirtyRects, decalRect, width, height);
			if (rects.length <= 0) {
				continue;
			}
			const unionRect = unionDirtyRects(rects);
			const materialData = createDecalMaterialUniformData(packet.material);
			items.push({
				packet,
				materialData,
				binding: await this._getMaterialBinding(
					packet.material,
					materialData,
					uniformBuffer
				),
				rects,
				unionRect,
			});
		}
		return items;
	}

	private _hasReceiverCandidate(
		context: FrameContext,
		packet: DecalPacket,
		rect: DirtyRect
	): boolean {
		const spatialIndex = context.scene.spatialIndex;
		if (!spatialIndex) {
			return true;
		}
		const candidates = spatialIndex.queryOpaquePackets(rect);
		for (const candidate of candidates) {
			const renderLayers = candidate.meshInstance.renderLayers ?? 1;
			if ((renderLayers & packet.receiverLayerMask) === 0) {
				continue;
			}
			if (boundingSpheresIntersect(candidate, packet)) {
				return true;
			}
		}
		return false;
	}

	private _buildExecutionSegments(
		items: readonly DecalWorkItem[]
	): DecalExecutionSegment[] {
		const segments: DecalExecutionSegment[] = [];
		let index = 0;
		while (index < items.length) {
			const item = items[index];
			const batchItems = [item];
			let nextIndex = index + 1;
			while (
				nextIndex < items.length &&
				this._canBatchTogether(batchItems[0], items[nextIndex])
			) {
				batchItems.push(items[nextIndex]);
				nextIndex++;
			}
			if (batchItems.length > 1) {
				segments.push({ kind: "batch", items: batchItems });
				index = nextIndex;
				continue;
			}
			segments.push({ kind: "fallback", item });
			index++;
		}
		return segments;
	}

	private _canBatchTogether(
		left: DecalWorkItem,
		right: DecalWorkItem
	): boolean {
		return left.packet.material === right.packet.material;
	}

	private async _recordBatchSegment(
		encoder: ICommandEncoder,
		targets: readonly DecalTargetRef[],
		snapshotReadBinding: IBindingGroup,
		frameBinding: IBindingGroup,
		items: readonly DecalWorkItem[]
	): Promise<number> {
		let dispatchCount = 0;
		const dispatchRects = mergeSegmentRects(items);
		const pipeline = await this._resources.getDecalBatchPipeline();
		for (const rect of dispatchRects) {
			const batchData = this._createBatchData(items, rect);
			if (!batchData) {
				continue;
			}
			this._copyTargetsToSnapshots(encoder, targets, rect);
			this._writeBatchBuffers(batchData);
			const binding = this._getBatchBinding(targets);
			encoder.beginComputePass({
				label: `WebGPUDeferredDecalBatch_${items[0].packet.id}`,
			});
			encoder.setComputePipeline(pipeline);
			encoder.setBindingGroup(0, frameBinding);
			encoder.setBindingGroup(1, snapshotReadBinding);
			encoder.setBindingGroup(2, items[0].binding);
			encoder.setBindingGroup(3, binding);
			encoder.dispatchWorkgroups(
				Math.ceil(rect.width / DECAL_BATCH_WORKGROUP_SIZE),
				Math.ceil(rect.height / DECAL_BATCH_WORKGROUP_SIZE),
				1
			);
			encoder.endComputePass();
			dispatchCount++;
		}
		return dispatchCount;
	}

	private async _recordFallbackDecal(
		encoder: ICommandEncoder,
		targets: readonly DecalTargetRef[],
		snapshotReadBinding: IBindingGroup,
		gbufferWriteBinding: IBindingGroup,
		frameBinding: IBindingGroup,
		item: DecalWorkItem,
		uniformBuffer: IRenderBuffer
	): Promise<number> {
		const pipeline = await this._resources.getDecalPipeline();
		this._host.writeBuffer(
			uniformBuffer,
			createDecalUniformData(item.packet, item.materialData)
		);
		const copyRect =
			typeof encoder.setScissorRect === "function" ?
				item.unionRect
			:	{
					x: 0,
					y: 0,
					width: targets[0].texture.width,
					height: targets[0].texture.height,
				};
		this._copyTargetsToSnapshots(encoder, targets, copyRect);
		encoder.beginRenderPass({
			label: `WebGPUDeferredDecal_${item.packet.id}`,
			colorAttachments: targets.slice(0, 7).map((target) => ({
				view: target.texture,
				loadOp: "load" as const,
				storeOp: "store" as const,
			})),
		});
		encoder.setPipeline(pipeline);
		encoder.setBindingGroup(0, frameBinding);
		encoder.setBindingGroup(1, snapshotReadBinding);
		encoder.setBindingGroup(2, item.binding);
		encoder.setBindingGroup(3, gbufferWriteBinding);
		for (const rect of item.rects) {
			encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
			encoder.draw(3);
		}
		encoder.endRenderPass();
		return item.rects.length;
	}

	private _createBatchData(
		items: readonly DecalWorkItem[],
		rect: DirtyRect
	): DecalBatchData | null {
		const tileColumns = Math.max(1, Math.ceil(rect.width / DECAL_BATCH_TILE_SIZE));
		const tileRows = Math.max(1, Math.ceil(rect.height / DECAL_BATCH_TILE_SIZE));
		const tileCount = tileColumns * tileRows;
		const tileLists: number[][] = Array.from({ length: tileCount }, () => []);
		const included = new Set<number>();
		for (let decalIndex = 0; decalIndex < items.length; decalIndex++) {
			for (const itemRect of items[decalIndex].rects) {
				const clipped = intersectDirtyRect(itemRect, rect);
				if (!clipped) {
					continue;
				}
				included.add(decalIndex);
				const minTileX = Math.max(
					0,
					Math.floor((clipped.x - rect.x) / DECAL_BATCH_TILE_SIZE)
				);
				const minTileY = Math.max(
					0,
					Math.floor((clipped.y - rect.y) / DECAL_BATCH_TILE_SIZE)
				);
				const maxTileX = Math.min(
					tileColumns - 1,
					Math.floor(
						(clipped.x + clipped.width - 1 - rect.x) / DECAL_BATCH_TILE_SIZE
					)
				);
				const maxTileY = Math.min(
					tileRows - 1,
					Math.floor(
						(clipped.y + clipped.height - 1 - rect.y) / DECAL_BATCH_TILE_SIZE
					)
				);
				for (let tileY = minTileY; tileY <= maxTileY; tileY++) {
					for (let tileX = minTileX; tileX <= maxTileX; tileX++) {
						const list = tileLists[tileY * tileColumns + tileX];
						if (list[list.length - 1] !== decalIndex) {
							list.push(decalIndex);
						}
					}
				}
			}
		}
		if (included.size <= 0) {
			return null;
		}

		const decalUniforms = new Float32Array(items.length * DECAL_UNIFORM_FLOATS);
		for (let index = 0; index < items.length; index++) {
			decalUniforms.set(
				createDecalUniformData(items[index].packet, items[index].materialData),
				index * DECAL_UNIFORM_FLOATS
			);
		}

		let indexCount = 0;
		for (const list of tileLists) {
			indexCount += list.length;
		}
		const tileHeaders = new Uint32Array(tileCount * 4);
		const tileDecalIndices = new Uint32Array(Math.max(1, indexCount));
		let cursor = 0;
		for (let tileIndex = 0; tileIndex < tileLists.length; tileIndex++) {
			const list = tileLists[tileIndex];
			tileHeaders[tileIndex * 4] = cursor;
			tileHeaders[tileIndex * 4 + 1] = list.length;
			for (const decalIndex of list) {
				tileDecalIndices[cursor] = decalIndex;
				cursor++;
			}
		}

		return {
			rect,
			decalUniforms,
			tileHeaders,
			tileDecalIndices,
			tileColumns,
			tileRows,
		};
	}

	private _writeBatchBuffers(data: DecalBatchData): void {
		const params = new Uint32Array(8);
		params[0] = data.rect.x;
		params[1] = data.rect.y;
		params[2] = data.rect.width;
		params[3] = data.rect.height;
		params[4] = DECAL_BATCH_TILE_SIZE;
		params[5] = data.tileColumns;
		params[6] = data.tileRows;
		this._batchParamsBuffer = this._ensureBuffer(
			this._batchParamsBuffer,
			params.byteLength,
			BufferUsage.Uniform | BufferUsage.CopyDst,
			"WebGPUDecalBatchParams"
		);
		this._batchDecalsBuffer = this._ensureBuffer(
			this._batchDecalsBuffer,
			data.decalUniforms.byteLength,
			BufferUsage.Storage | BufferUsage.CopyDst,
			"WebGPUDecalBatchUniforms"
		);
		this._batchTileHeadersBuffer = this._ensureBuffer(
			this._batchTileHeadersBuffer,
			data.tileHeaders.byteLength,
			BufferUsage.Storage | BufferUsage.CopyDst,
			"WebGPUDecalBatchTileHeaders"
		);
		this._batchTileIndicesBuffer = this._ensureBuffer(
			this._batchTileIndicesBuffer,
			data.tileDecalIndices.byteLength,
			BufferUsage.Storage | BufferUsage.CopyDst,
			"WebGPUDecalBatchTileIndices"
		);
		this._host.writeBuffer(this._batchParamsBuffer, toBufferSource(params));
		this._host.writeBuffer(
			this._batchDecalsBuffer,
			toBufferSource(data.decalUniforms)
		);
		this._host.writeBuffer(
			this._batchTileHeadersBuffer,
			toBufferSource(data.tileHeaders)
		);
		this._host.writeBuffer(
			this._batchTileIndicesBuffer,
			toBufferSource(data.tileDecalIndices)
		);
	}

	private _ensureBuffer(
		buffer: IRenderBuffer | null,
		size: number,
		usage: BufferUsage,
		label: string
	): IRenderBuffer {
		const resolvedSize = Math.max(16, alignTo(size, 16));
		if (buffer && buffer.size >= resolvedSize) {
			return buffer;
		}
		buffer?.destroy();
		this._destroyBindingGroup(this._batchBinding);
		this._batchBinding = null;
		this._batchBindingSources = [];
		return this._host.createBuffer({
			size: resolvedSize,
			usage,
			label,
		});
	}

	private _getBatchBinding(targets: readonly DecalTargetRef[]): IBindingGroup {
		const buffers = this._getBatchBuffers();
		const sources = [
			buffers.params,
			buffers.decals,
			buffers.tileHeaders,
			buffers.tileIndices,
			...targets.map((target) => target.texture),
		];
		if (
			this._batchBinding &&
			this._batchBindingSources.length === sources.length &&
			this._batchBindingSources.every(
				(source, index) => source === sources[index]
			)
		) {
			return this._batchBinding;
		}
		this._destroyBindingGroup(this._batchBinding);
		this._batchBinding = this._host.createBindingGroup({
			layout: this._resources.getDecalBatchBindGroupLayout(),
			entries: sources.map((resource, binding) => ({
				binding,
				resource,
			})),
			label: "WebGPUDecalBatchBinding",
		});
		this._batchBindingSources = sources;
		return this._batchBinding;
	}

	private _getBatchBuffers(): {
		params: IRenderBuffer;
		decals: IRenderBuffer;
		tileHeaders: IRenderBuffer;
		tileIndices: IRenderBuffer;
	} {
		const params = this._ensureBuffer(
			this._batchParamsBuffer,
			16,
			BufferUsage.Uniform | BufferUsage.CopyDst,
			"WebGPUDecalBatchParams"
		);
		this._batchParamsBuffer = params;
		const decals = this._ensureBuffer(
			this._batchDecalsBuffer,
			16,
			BufferUsage.Storage | BufferUsage.CopyDst,
			"WebGPUDecalBatchUniforms"
		);
		this._batchDecalsBuffer = decals;
		const tileHeaders = this._ensureBuffer(
			this._batchTileHeadersBuffer,
			16,
			BufferUsage.Storage | BufferUsage.CopyDst,
			"WebGPUDecalBatchTileHeaders"
		);
		this._batchTileHeadersBuffer = tileHeaders;
		const tileIndices = this._ensureBuffer(
			this._batchTileIndicesBuffer,
			16,
			BufferUsage.Storage | BufferUsage.CopyDst,
			"WebGPUDecalBatchTileIndices"
		);
		this._batchTileIndicesBuffer = tileIndices;
		return {
			params,
			decals,
			tileHeaders,
			tileIndices,
		};
	}

	private _getUniformBuffer(): IRenderBuffer {
		if (!this._uniformBuffer) {
			this._uniformBuffer = this._host.createBuffer({
				size: DECAL_UNIFORM_BYTES,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: "WebGPUDecalUniform",
			});
		}
		return this._uniformBuffer;
	}

	private _deviceSupportsDecalPipeline(): boolean {
		const limits = this._host.device?.limits;
		const maxSampledTextures =
			limits?.maxSampledTexturesPerShaderStage ?? Number.POSITIVE_INFINITY;
		const maxSamplers =
			limits?.maxSamplersPerShaderStage ?? Number.POSITIVE_INFINITY;
		return (
			maxSampledTextures >= DECAL_REQUIRED_FRAGMENT_SAMPLED_TEXTURES &&
			maxSamplers >= DECAL_REQUIRED_FRAGMENT_SAMPLERS
		);
	}

	private _deviceSupportsDecalBatchPipeline(): boolean {
		const limits = this._host.device?.limits;
		const maxStorageTextures =
			limits?.maxStorageTexturesPerShaderStage ?? Number.POSITIVE_INFINITY;
		const maxStorageBuffers =
			limits?.maxStorageBuffersPerShaderStage ?? Number.POSITIVE_INFINITY;
		return (
			this._deviceSupportsDecalPipeline() &&
			maxStorageTextures >= DECAL_BATCH_REQUIRED_STORAGE_TEXTURES &&
			maxStorageBuffers >= DECAL_BATCH_REQUIRED_STORAGE_BUFFERS
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
			this._host.createTexture({
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
			this._snapshotReadBinding = this._host.createBindingGroup({
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
		const outputTargets = targets.slice(7, 11);
		const sources = outputTargets.map((target) => target.texture);
		if (
			this._outputBinding &&
			this._outputBindingSources.length === sources.length &&
			this._outputBindingSources.every(
				(source, index) => source === sources[index]
			)
		) {
			return this._outputBinding;
		}
		this._destroyBindingGroup(this._outputBinding);
		this._outputBinding = this._host.createBindingGroup({
			layout: this._resources.getDecalOutputBindGroupLayout(),
			entries: sources.map((resource, index) => ({
				binding: 11 + index,
				resource,
			})),
			label: "WebGPUDecalOutputBinding",
		});
		this._outputBindingSources = sources;
		return this._outputBinding;
	}

	private async _getMaterialBinding(
		material: Material,
		materialData: WebGPUMaterialUniformData,
		uniformBuffer: IRenderBuffer
	): Promise<IBindingGroup> {
		const textures = await Promise.all(
			materialData.textureSlots.map((slot, index) =>
				this._resources.getTextureForSlotAsync(slot.map, index)
			)
		);
		const samplers = materialData.textureSlots.map((slot) =>
			this._resources.getSamplerForTexture(slot.map)
		);
		const anisotropyTexture = await this._resources.getTextureForSlotAsync(
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
		const group = this._host.createBindingGroup({
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
		targets: readonly DecalTargetRef[],
		rect: DirtyRect
	): void {
		for (let i = 0; i < targets.length; i++) {
			encoder.copyTextureToTexture!(
				{
					texture: targets[i].texture,
					origin: { x: rect.x, y: rect.y },
				},
				{
					texture: this._snapshotTextures[i],
					origin: { x: rect.x, y: rect.y },
				},
				{
					width: rect.width,
					height: rect.height,
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

function intersectDirtyRects(
	rects: readonly DirtyRect[],
	bounds: DirtyRect,
	width: number,
	height: number
): DirtyRect[] {
	const viewport = { x: 0, y: 0, width, height };
	const clampedBounds = intersectDirtyRect(bounds, viewport);
	if (!clampedBounds) {
		return [];
	}
	const result: DirtyRect[] = [];
	for (const rect of rects) {
		const clipped = intersectDirtyRect(rect, clampedBounds);
		if (clipped) {
			result.push(clipped);
		}
	}
	return result;
}

function intersectDirtyRect(
	left: DirtyRect,
	right: DirtyRect
): DirtyRect | null {
	const x = Math.max(left.x, right.x);
	const y = Math.max(left.y, right.y);
	const maxX = Math.min(left.x + left.width, right.x + right.width);
	const maxY = Math.min(left.y + left.height, right.y + right.height);
	const width = maxX - x;
	const height = maxY - y;
	if (width <= 0 || height <= 0) {
		return null;
	}
	return {
		x,
		y,
		width,
		height,
	};
}

function unionDirtyRects(rects: readonly DirtyRect[]): DirtyRect {
	let minX = Number.POSITIVE_INFINITY;
	let minY = Number.POSITIVE_INFINITY;
	let maxX = Number.NEGATIVE_INFINITY;
	let maxY = Number.NEGATIVE_INFINITY;
	for (const rect of rects) {
		minX = Math.min(minX, rect.x);
		minY = Math.min(minY, rect.y);
		maxX = Math.max(maxX, rect.x + rect.width);
		maxY = Math.max(maxY, rect.y + rect.height);
	}
	return {
		x: Math.max(0, Math.floor(minX)),
		y: Math.max(0, Math.floor(minY)),
		width: Math.max(0, Math.ceil(maxX) - Math.max(0, Math.floor(minX))),
		height: Math.max(0, Math.ceil(maxY) - Math.max(0, Math.floor(minY))),
	};
}

function mergeSegmentRects(items: readonly DecalWorkItem[]): DirtyRect[] {
	const rects: DirtyRect[] = [];
	for (const item of items) {
		for (const rect of item.rects) {
			rects.push(rect);
		}
	}
	rects.sort((left, right) => left.y - right.y || left.x - right.x);
	const merged: DirtyRect[] = [];
	for (const rect of rects) {
		const previous = merged[merged.length - 1];
		if (previous && dirtyRectsIntersectOrTouch(previous, rect)) {
			merged[merged.length - 1] = unionDirtyRects([previous, rect]);
			continue;
		}
		merged.push(rect);
	}
	return merged;
}

function dirtyRectsIntersectOrTouch(left: DirtyRect, right: DirtyRect): boolean {
	return (
		left.x <= right.x + right.width &&
		left.x + left.width >= right.x &&
		left.y <= right.y + right.height &&
		left.y + left.height >= right.y
	);
}

function boundingSpheresIntersect(
	left: DrawPacket,
	right: DecalPacket
): boolean {
	const dx = left.worldBounds.center.x - right.worldBounds.center.x;
	const dy = left.worldBounds.center.y - right.worldBounds.center.y;
	const dz = left.worldBounds.center.z - right.worldBounds.center.z;
	const radius = left.worldBounds.radius + right.worldBounds.radius;
	return dx * dx + dy * dy + dz * dz <= radius * radius;
}

function alignTo(value: number, alignment: number): number {
	return Math.ceil(Math.max(0, value) / alignment) * alignment;
}

function toBufferSource(value: ArrayBufferView): BufferSource {
	return value as unknown as BufferSource;
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
