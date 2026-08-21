import {
	DECAL_CHANNELS,
	resolveDecalChannelBlendMode,
	type DecalBlendMode,
} from "../../../decals";
import { Logger } from "../../../foundation/Logger";
import { ShadingModel, type Material } from "../../../materials/Material";
import { ShaderMaterial } from "../../../materials/ShaderMaterial";
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
	WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT,
	WEBGPU_TEXTURE_SLOT_COUNT,
} from "../constants";
import {
	createWebGPUMaterialUniformData,
} from "../material";
import {
	packMatrix4ForWGSL,
	packNormalMatrix4ForWGSL,
} from "../packing";
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
import type { WebGPUFrameTargets } from "../WebGPUFrameTargetContracts";
import type { WebGPUFrameExecutionContext } from "./WebGPUFrameExecutionContext";

interface DecalTargetRef {
	texture: IRenderTexture;
	format: TextureFormat;
	label: string;
}

interface DecalMaterialBindingCacheEntry {
	material: Material;
	uniformBuffer: IRenderBuffer;
	group: IBindingGroup;
	textures: IRenderTexture[];
	samplers: ISampler[];
}

interface DecalBatchBufferSet {
	params: IRenderBuffer;
	decals: IRenderBuffer;
	tileHeaders: IRenderBuffer;
	tileIndices: IRenderBuffer;
	binding: IBindingGroup | null;
	bindingSources: unknown[];
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
	16 +
	4 +
	14 * 4 +
	WEBGPU_TEXTURE_SLOT_COUNT * 4 +
	WEBGPU_TEXTURE_SLOT_COUNT * 4 +
	5 * 4;
const DECAL_UNIFORM_BYTES = DECAL_UNIFORM_FLOATS * 4;
const DECAL_LAYER_MASK_SUPPORTED_BITS = 0x7ff;
const DECAL_REQUIRED_FRAGMENT_SAMPLED_TEXTURES =
	WEBGPU_TEXTURE_SLOT_COUNT + 11;
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
	private readonly _resources: WebGPUDeferredResourceProvider & WebGPUTextureResourceProvider;
	private _frame: Pick<
		WebGPUFrameExecutionContext,
		"commands" | "targets" | "resources" | "dirtyRects"
	> | null = null;
	private _uniformBuffers: IRenderBuffer[] = [];
	private _uniformBufferCursor = 0;
	private _snapshotTextures: IRenderTexture[] = [];
	private _snapshotReadBinding: IBindingGroup | null = null;
	private _snapshotKey = "";
	private _batchBufferSets: DecalBatchBufferSet[] = [];
	private _batchBufferCursor = 0;
	private _outputBinding: IBindingGroup | null = null;
	private _outputBindingSources: unknown[] = [];
	private _materialBindingSlots: DecalMaterialBindingCacheEntry[] = [];

	constructor(
		host: WebGPUFrameHost,
		resources: WebGPUDeferredResourceProvider & WebGPUTextureResourceProvider,
	) {
		this._host = host;
		this._resources = resources;
	}

	public bindFrame(frame: WebGPUFrameExecutionContext): void {
		this._frame = frame;
	}

	public closeFrame(): void {
		this._frame = null;
	}

	public destroyBindings(): void {
		this._destroyBindingGroup(this._snapshotReadBinding);
		this._snapshotReadBinding = null;
		this._destroyBindingGroup(this._outputBinding);
		this._outputBinding = null;
		this._outputBindingSources = [];
		for (const entry of this._materialBindingSlots) {
			this._destroyBindingGroup(entry.group);
		}
		this._materialBindingSlots = [];
		for (const buffer of this._uniformBuffers) {
			buffer.destroy();
		}
		this._uniformBuffers = [];
		this._uniformBufferCursor = 0;
		for (const buffers of this._batchBufferSets) {
			this._destroyBindingGroup(buffers.binding);
			buffers.params.destroy();
			buffers.decals.destroy();
			buffers.tileHeaders.destroy();
			buffers.tileIndices.destroy();
		}
		this._batchBufferSets = [];
		this._batchBufferCursor = 0;
		for (const texture of this._snapshotTextures) {
			texture.destroy();
		}
		this._snapshotTextures = [];
		this._snapshotKey = "";
	}

	public async recordDecalPass(context: FrameContext): Promise<number> {
		const frame = this._requireFrame();
		const encoder = frame.commands.encoder;
		const targets = frame.targets.frameTargets;
		const supportsDecals = this._deviceSupportsDecalPipeline();
		const decalPackets = context.scene.decalPackets;
		this._warnUnsupportedDeferredChannels(decalPackets);
		if (
			!encoder ||
			!targets ||
			decalPackets.length <= 0 ||
			typeof encoder.copyTextureToTexture !== "function" ||
			!supportsDecals
		) {
			if (decalPackets.length > 0 && !supportsDecals) {
				const key = "webgpu-decal-sampled-texture-limit";
				Logger.warn(
					`[${key}] WebGPU device limits cannot bind the full decal ` +
						"material surface; decals are skipped for this frame.",
					{ scope: "WebGPUDeferredDecalPass", onceKey: key },
				);
			}
			return 0;
		}

		const targetRefs = resolveDecalTargets(targets);
		if (!targetRefs) {
			return 0;
		}
		const activeDecals = decalPackets.filter((packet) =>
			hasSupportedLayerMask(packet.receiverLayerMask),
		);
		if (activeDecals.length <= 0) {
			return 0;
		}

		this._uniformBufferCursor = 0;
		this._batchBufferCursor = 0;
		this._ensureSnapshotTextures(targetRefs);
		const snapshotReadBinding = this._getSnapshotReadBinding();
		const frameResources = frame.resources;
		const dirtyRects = frame.dirtyRects.resolveDirtyRects(
			context,
			targetRefs[0].texture.width,
			targetRefs[0].texture.height,
		);
		const workItems = await this._createWorkItems(
			context,
			activeDecals,
			dirtyRects,
			targetRefs[0].texture.width,
			targetRefs[0].texture.height,
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
					segment.items,
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
			);
		}
		return drawCount;
	}

	/** Prepares deferred decal targets, bindings, and pipelines before recording. */
	public async preflight(context: FrameContext): Promise<void> {
		const decalPackets = context.scene.decalPackets;
		if (decalPackets.length <= 0) {
			return;
		}
		this._warnUnsupportedDeferredChannels(decalPackets);
		const frame = this._requireFrame();
		const targets = frame.targets.frameTargets;
		const targetRefs = resolveDecalTargets(targets);
		if (!targetRefs || !this._deviceSupportsDecalPipeline()) {
			return;
		}
		this._ensureSnapshotTextures(targetRefs);
		this._getSnapshotReadBinding();
		this._getGBufferWriteBinding(targetRefs);
		this._uniformBufferCursor = 0;
		const dirtyRects = frame.dirtyRects.resolveDirtyRects(
			context,
			targetRefs[0].texture.width,
			targetRefs[0].texture.height,
		);
		await this._createWorkItems(
			context,
			decalPackets.filter((packet) => hasSupportedLayerMask(packet.receiverLayerMask)),
			dirtyRects,
			targetRefs[0].texture.width,
			targetRefs[0].texture.height,
		);
		this._uniformBufferCursor = 0;
		await this._resources.getDecalPipeline();
		if (this._deviceSupportsDecalBatchPipeline()) {
			await this._resources.getDecalBatchPipeline();
		}
	}

	private _warnUnsupportedDeferredChannels(packets: readonly DecalPacket[]): void {
		for (const packet of packets) {
			const material = packet.material as any;
			if ((material.transmissionFactor ?? 0) > 0 || material.transmissionMap) {
				Logger.warn(
					"[webgpu-deferred-decal-transmission-unsupported] " +
						"Deferred decals ignore transmission channels.",
					{
						scope: "WebGPUDeferredDecalPass",
						onceKey: "webgpu-deferred-decal-transmission-unsupported",
					},
				);
			}
			if (
				(material.thicknessFactor ?? 0) > 0 ||
				material.thicknessMap ||
				Number.isFinite(material.attenuationDistance)
			) {
				Logger.warn(
					"[webgpu-deferred-decal-volume-unsupported] " +
						"Deferred decals ignore thickness and attenuation channels.",
					{
						scope: "WebGPUDeferredDecalPass",
						onceKey: "webgpu-deferred-decal-volume-unsupported",
					},
				);
			}
		}
	}

	private async _createWorkItems(
		context: FrameContext,
		packets: readonly DecalPacket[],
		dirtyRects: readonly DirtyRect[],
		width: number,
		height: number,
	): Promise<DecalWorkItem[]> {
		if (dirtyRects.length <= 0) {
			return [];
		}
		const fullScreenRect = { x: 0, y: 0, width, height };
		const items: DecalWorkItem[] = [];
		for (const packet of packets) {
			if (packet.material instanceof ShaderMaterial) {
				continue;
			}
			const screenRect = computePacketScreenRect(packet, context.viewCamera, width, height);
			if (screenRect && !this._hasReceiverCandidate(context, packet, screenRect)) {
				continue;
			}
			const decalRect = screenRect ?? fullScreenRect;
			const rects = intersectDirtyRects(dirtyRects, decalRect, width, height);
			if (rects.length <= 0) {
				continue;
			}
			const unionRect = unionDirtyRects(rects);
			const materialData = createDecalMaterialUniformData(packet.material);
			const uniformSlot = this._uniformBufferCursor++;
			const uniformBuffer = this._getUniformBuffer(uniformSlot);
			this._host.writeBuffer(uniformBuffer, createDecalUniformData(packet, materialData));
			items.push({
				packet,
				materialData,
				binding: await this._getMaterialBinding(
					packet.material,
					materialData,
					uniformBuffer,
					uniformSlot,
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
		rect: DirtyRect,
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

	private _buildExecutionSegments(items: readonly DecalWorkItem[]): DecalExecutionSegment[] {
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

	private _canBatchTogether(left: DecalWorkItem, right: DecalWorkItem): boolean {
		return left.packet.material === right.packet.material;
	}

	private async _recordBatchSegment(
		encoder: ICommandEncoder,
		targets: readonly DecalTargetRef[],
		snapshotReadBinding: IBindingGroup,
		frameBinding: IBindingGroup,
		items: readonly DecalWorkItem[],
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
			const buffers = this._writeBatchBuffers(batchData);
			const binding = this._getBatchBinding(buffers, targets);
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
				1,
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
	): Promise<number> {
		const pipeline = await this._resources.getDecalPipeline();
		const copyRect =
			typeof encoder.setScissorRect === "function"
				? item.unionRect
				: {
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
		rect: DirtyRect,
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
					Math.floor((clipped.x - rect.x) / DECAL_BATCH_TILE_SIZE),
				);
				const minTileY = Math.max(
					0,
					Math.floor((clipped.y - rect.y) / DECAL_BATCH_TILE_SIZE),
				);
				const maxTileX = Math.min(
					tileColumns - 1,
					Math.floor((clipped.x + clipped.width - 1 - rect.x) / DECAL_BATCH_TILE_SIZE),
				);
				const maxTileY = Math.min(
					tileRows - 1,
					Math.floor((clipped.y + clipped.height - 1 - rect.y) / DECAL_BATCH_TILE_SIZE),
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
				index * DECAL_UNIFORM_FLOATS,
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

	private _writeBatchBuffers(data: DecalBatchData): DecalBatchBufferSet {
		const params = new Uint32Array(8);
		params[0] = data.rect.x;
		params[1] = data.rect.y;
		params[2] = data.rect.width;
		params[3] = data.rect.height;
		params[4] = DECAL_BATCH_TILE_SIZE;
		params[5] = data.tileColumns;
		params[6] = data.tileRows;
		const slot = this._batchBufferCursor++;
		let buffers = this._batchBufferSets[slot];
		if (!buffers) {
			buffers = {
				params: this._createBuffer(
					params.byteLength,
					BufferUsage.Uniform | BufferUsage.CopyDst,
					`WebGPUDecalBatchParams_${slot}`,
				),
				decals: this._createBuffer(
					data.decalUniforms.byteLength,
					BufferUsage.Storage | BufferUsage.CopyDst,
					`WebGPUDecalBatchUniforms_${slot}`,
				),
				tileHeaders: this._createBuffer(
					data.tileHeaders.byteLength,
					BufferUsage.Storage | BufferUsage.CopyDst,
					`WebGPUDecalBatchTileHeaders_${slot}`,
				),
				tileIndices: this._createBuffer(
					data.tileDecalIndices.byteLength,
					BufferUsage.Storage | BufferUsage.CopyDst,
					`WebGPUDecalBatchTileIndices_${slot}`,
				),
				binding: null,
				bindingSources: [],
			};
			this._batchBufferSets[slot] = buffers;
		}
		buffers.params = this._resizeBatchBuffer(
			buffers,
			"params",
			params.byteLength,
			BufferUsage.Uniform | BufferUsage.CopyDst,
			`WebGPUDecalBatchParams_${slot}`,
		);
		buffers.decals = this._resizeBatchBuffer(
			buffers,
			"decals",
			data.decalUniforms.byteLength,
			BufferUsage.Storage | BufferUsage.CopyDst,
			`WebGPUDecalBatchUniforms_${slot}`,
		);
		buffers.tileHeaders = this._resizeBatchBuffer(
			buffers,
			"tileHeaders",
			data.tileHeaders.byteLength,
			BufferUsage.Storage | BufferUsage.CopyDst,
			`WebGPUDecalBatchTileHeaders_${slot}`,
		);
		buffers.tileIndices = this._resizeBatchBuffer(
			buffers,
			"tileIndices",
			data.tileDecalIndices.byteLength,
			BufferUsage.Storage | BufferUsage.CopyDst,
			`WebGPUDecalBatchTileIndices_${slot}`,
		);
		this._host.writeBuffer(buffers.params, toBufferSource(params));
		this._host.writeBuffer(buffers.decals, toBufferSource(data.decalUniforms));
		this._host.writeBuffer(buffers.tileHeaders, toBufferSource(data.tileHeaders));
		this._host.writeBuffer(buffers.tileIndices, toBufferSource(data.tileDecalIndices));
		return buffers;
	}

	private _createBuffer(size: number, usage: BufferUsage, label: string): IRenderBuffer {
		const resolvedSize = Math.max(16, alignTo(size, 16));
		return this._host.createBuffer({
			size: resolvedSize,
			usage,
			label,
		});
	}

	private _resizeBatchBuffer(
		buffers: DecalBatchBufferSet,
		key: "params" | "decals" | "tileHeaders" | "tileIndices",
		size: number,
		usage: BufferUsage,
		label: string,
	): IRenderBuffer {
		const current = buffers[key];
		const resolvedSize = Math.max(16, alignTo(size, 16));
		if (current.size >= resolvedSize) {
			return current;
		}
		current.destroy();
		this._destroyBindingGroup(buffers.binding);
		buffers.binding = null;
		buffers.bindingSources = [];
		return this._createBuffer(size, usage, label);
	}

	private _getBatchBinding(
		buffers: DecalBatchBufferSet,
		targets: readonly DecalTargetRef[],
	): IBindingGroup {
		const sources = [
			buffers.params,
			buffers.decals,
			buffers.tileHeaders,
			buffers.tileIndices,
			...targets.map((target) => target.texture),
		];
		if (
			buffers.binding &&
			buffers.bindingSources.length === sources.length &&
			buffers.bindingSources.every((source, index) => source === sources[index])
		) {
			return buffers.binding;
		}
		this._destroyBindingGroup(buffers.binding);
		buffers.binding = this._host.createBindingGroup({
			layout: this._resources.getDecalBatchBindGroupLayout(),
			entries: sources.map((resource, binding) => ({
				binding,
				resource,
			})),
			label: "WebGPUDecalBatchBinding",
		});
		buffers.bindingSources = sources;
		return buffers.binding;
	}

	private _getUniformBuffer(slot: number): IRenderBuffer {
		if (!this._uniformBuffers[slot]) {
			this._uniformBuffers[slot] = this._host.createBuffer({
				size: DECAL_UNIFORM_BYTES,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: `WebGPUDecalUniform_${slot}`,
			});
		}
		return this._uniformBuffers[slot];
	}

	private _deviceSupportsDecalPipeline(): boolean {
		const limits = this._host.device?.limits;
		const maxSampledTextures =
			limits?.maxSampledTexturesPerShaderStage ?? Number.POSITIVE_INFINITY;
		const maxSamplers = limits?.maxSamplersPerShaderStage ?? Number.POSITIVE_INFINITY;
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
				[target.texture.width, target.texture.height, target.format, target.label].join(
					":",
				),
			)
			.join("|");
		if (this._snapshotKey === snapshotKey && this._snapshotTextures.length === targets.length) {
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
			}),
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

	private _getGBufferWriteBinding(targets: readonly DecalTargetRef[]): IBindingGroup {
		const outputTargets = targets.slice(7, 9);
		const sources = outputTargets.map((target) => target.texture);
		if (
			this._outputBinding &&
			this._outputBindingSources.length === sources.length &&
			this._outputBindingSources.every((source, index) => source === sources[index])
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
		uniformBuffer: IRenderBuffer,
		slot: number,
	): Promise<IBindingGroup> {
		const textures = await Promise.all(
			materialData.textureSlots.map((slot, index) =>
				this._resources.getTextureForSlotAsync(slot.map, index),
			),
		);
		const samplers = materialData.textureSlots.map((slot) =>
			this._resources.getSamplerForTexture(slot.map),
		);
		const cached = this._materialBindingSlots[slot];
		if (
			cached &&
			cached.material === material &&
			cached.uniformBuffer === uniformBuffer &&
			areTexturesEqual(cached.textures, textures) &&
			areSamplersEqual(cached.samplers, samplers)
		) {
			return cached.group;
		}

		if (cached) {
			this._destroyBindingGroup(cached.group);
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
		const group = this._host.createBindingGroup({
			layout: this._resources.getDecalBindGroupLayout(),
			entries,
			label: `WebGPUDecalMaterialBinding_${material.name}`,
		});
		const entry = {
			material,
			uniformBuffer,
			group,
			textures,
			samplers,
		};
		this._materialBindingSlots[slot] = entry;
		return group;
	}

	private _copyTargetsToSnapshots(
		encoder: ICommandEncoder,
		targets: readonly DecalTargetRef[],
		rect: DirtyRect,
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
				},
			);
		}
	}

	private _destroyBindingGroup(group: IBindingGroup | null): void {
		const destroyFn = (group as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(group);
		}
	}

	private _requireFrame() {
		if (!this._frame) {
			throw new Error("WebGPU deferred decal frame is not active.");
		}
		return this._frame;
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
	cursor = writePackedNormalMatrix(data, cursor, packet.normalMatrix);
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
		materialData.materialFlags,
	]) {
		cursor = writeVec4(data, cursor, values);
	}
	cursor = writeU32Vec4(data, cursor, materialData.pbrMasks);
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

function writeU32Vec4(
	target: Float32Array,
	cursor: number,
	values: readonly number[]
): number {
	const words = new Uint32Array(target.buffer, target.byteOffset, target.length);
	for (let index = 0; index < 4; index++) {
		words[cursor + index] = (values[index] ?? 0) >>> 0;
	}
	return cursor + 4;
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

function writePackedNormalMatrix(
	target: Float32Array,
	offset: number,
	matrix: DecalPacket["normalMatrix"]
): number {
	target.set(packNormalMatrix4ForWGSL(matrix), offset);
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
			format: TextureFormat.RGBA8Unorm,
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
			format: TextureFormat.RGBA8Unorm,
			label: "SheenReflectance",
		},
		{
			texture: targets.gMaterialExt0,
			format: TextureFormat.RGBA16Float,
			label: "MaterialExt0",
		},
		{
			texture: targets.gMaterialExt3,
			format: TextureFormat.RGBA16Uint,
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
