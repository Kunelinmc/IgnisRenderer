import { LightType, type ShadowCastingLight } from "../../lights";
import type {
	PagedShadowLayoutMetadata,
	ShadowRenderSet,
} from "../../lights/shadows/ShadowMapping";
import { Matrix4 } from "../../maths/Matrix4";
import type { IVector3 } from "../../maths/types";
import type {
	DrawPacket,
	FrameContext,
} from "../../pipeline/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import { ShaderSource } from "../../shaders/ShaderSource";
import {
	BufferUsage,
	TextureFormat,
	TextureUsage,
	type IBindingGroup,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type IShaderModule,
} from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
import type { WebGPUShadowPass } from "./WebGPUShadowPass";

export const WEBGPU_PAGED_SHADOW_NON_RESIDENT = 0xffffffff;
const PAGE_METADATA_UINTS = 8;
const PAGE_REQUEST_RECORD_UINTS = 8;
const PAGE_RESIDENCY_STATE_UINTS = 8;
const DIRTY_PHYSICAL_PAGE_RECORD_UINTS = 8;
const PAGE_LAYOUT_UINTS = 8;
const PAGE_REQUEST_PARAMS_UINTS = 4;
const PAGE_ALLOC_PARAMS_UINTS = 8;
const PAGE_DIRTY_PARAMS_UINTS = 4;
const PAGE_FEEDBACK_PARAMS_UINTS = 4;
const DEFAULT_FALLBACK_PAGE_SIZE = 1;

export interface WebGPUPagedShadowFrameRequest {
	context: FrameContext;
	encoder: ICommandEncoder | null;
	renderSets: ReadonlyMap<ShadowCastingLight, ShadowRenderSet>;
	shadowCasterPackets: readonly DrawPacket[];
	shadowTransmitterPackets: readonly DrawPacket[];
}

export interface WebGPUPagedShadowResources {
	pageTable: IRenderBuffer;
	physicalDepthAtlas: IRenderTexture;
	physicalTransmittanceAtlas: IRenderTexture | null;
	pageMetadataBuffer: IRenderBuffer;
	pageRequestFlags: IRenderBuffer;
	compactedRequests: IRenderBuffer;
	residencyState: IRenderBuffer;
	freeList: IRenderBuffer;
	counters: IRenderBuffer;
	dirtyPhysicalPages: IRenderBuffer;
	feedbackFlags: IRenderBuffer;
	nextFeedbackFlags: IRenderBuffer;
	pageSize: number;
	physicalGridSize: number;
	physicalAtlasSize: number;
}

export interface WebGPUPagedShadowPageRequest {
	key: string;
	lightId: string;
	shadowMapId: string;
	cascadeIndex: number;
	viewProjection: Matrix4;
	pageX: number;
	pageY: number;
	pageGridSize: number;
	pageTableIndex: number;
	priority: number;
}

export interface WebGPUPagedShadowResidentPage {
	key: string;
	request: WebGPUPagedShadowPageRequest;
	physicalPageIndex: number;
	dirty: boolean;
	lastUsedFrame: number;
	viewProjection: Matrix4;
	viewportX: number;
	viewportY: number;
	viewportSize: number;
}

export interface WebGPUPagedShadowDebugState {
	frameId: number;
	requestCount: number;
	residentCount: number;
	dirtyCount: number;
	pageTableLength: number;
	physicalPageCount: number;
	pageSize: number;
	physicalGridSize: number;
	gpuRequestBufferSize: number;
	gpuDirtyBufferSize: number;
	feedbackFlagCount: number;
}

export interface WebGPUPagedShadowRenderSetLayout {
	renderSet: ShadowRenderSet;
	metadata: PagedShadowLayoutMetadata;
	pageTableBase: number;
	pageTableCascadeStride: number;
	cascadeCount: number;
}

/**
 * WebGPU paged shadow runtime.
 *
 * V1 uses CPU conservative page requests and CPU page-table allocation. The
 * frame graph pass names are retained so GPU feedback/compute allocation can
 * replace these hooks later without changing public `PagedShadowMap` APIs.
 */
export class WebGPUPagedShadowRuntime {
	private _backend: WebGPUBackend;
	private _shadowPass: WebGPUShadowPass;
	private _frameId = 0;
	private _lastRequest: WebGPUPagedShadowFrameRequest | null = null;
	private _pageTableBuffer: IRenderBuffer | null = null;
	private _pageMetadataBuffer: IRenderBuffer | null = null;
	private _physicalDepthAtlas: IRenderTexture | null = null;
	private _fallbackPageTableBuffer: IRenderBuffer | null = null;
	private _fallbackMetadataBuffer: IRenderBuffer | null = null;
	private _fallbackDepthAtlas: IRenderTexture | null = null;
	private _fallbackPageRequestFlags: IRenderBuffer | null = null;
	private _fallbackCompactedRequests: IRenderBuffer | null = null;
	private _fallbackResidencyState: IRenderBuffer | null = null;
	private _fallbackFreeList: IRenderBuffer | null = null;
	private _fallbackCounters: IRenderBuffer | null = null;
	private _fallbackDirtyPhysicalPages: IRenderBuffer | null = null;
	private _fallbackFeedbackFlags: IRenderBuffer | null = null;
	private _fallbackNextFeedbackFlags: IRenderBuffer | null = null;
	private _pageRequestFlagsBuffer: IRenderBuffer | null = null;
	private _compactedRequestsBuffer: IRenderBuffer | null = null;
	private _residencyStateBuffer: IRenderBuffer | null = null;
	private _freeListBuffer: IRenderBuffer | null = null;
	private _countersBuffer: IRenderBuffer | null = null;
	private _dirtyPhysicalPagesBuffer: IRenderBuffer | null = null;
	private _feedbackFlagsBuffer: IRenderBuffer | null = null;
	private _nextFeedbackFlagsBuffer: IRenderBuffer | null = null;
	private _layoutBuffer: IRenderBuffer | null = null;
	private _casterBoundsBuffer: IRenderBuffer | null = null;
	private _cascadeViewProjectionBuffer: IRenderBuffer | null = null;
	private _requestParamsBuffer: IRenderBuffer | null = null;
	private _compactParamsBuffer: IRenderBuffer | null = null;
	private _allocationParamsBuffer: IRenderBuffer | null = null;
	private _dirtyParamsBuffer: IRenderBuffer | null = null;
	private _feedbackParamsBuffer: IRenderBuffer | null = null;
	private _pageTable = new Uint32Array([WEBGPU_PAGED_SHADOW_NON_RESIDENT]);
	private _pageMetadata = new Uint32Array(PAGE_METADATA_UINTS);
	private _pageRequestFlags = new Uint32Array(1);
	private _compactedRequests = new Uint32Array(PAGE_REQUEST_RECORD_UINTS);
	private _residencyState = createNonResidentUint32Array(PAGE_RESIDENCY_STATE_UINTS);
	private _freeList = new Uint32Array([0]);
	private _counters = new Uint32Array(4);
	private _dirtyPhysicalPages = new Uint32Array(DIRTY_PHYSICAL_PAGE_RECORD_UINTS);
	private _feedbackFlags = new Uint32Array(1);
	private _nextFeedbackFlags = new Uint32Array(1);
	private _layoutData = new Uint32Array(PAGE_LAYOUT_UINTS);
	private _casterBoundsData = new Float32Array(4);
	private _cascadeViewProjectionData = new Float32Array(16);
	private _requests: WebGPUPagedShadowPageRequest[] = [];
	private _residentPages = new Map<string, WebGPUPagedShadowResidentPage>();
	private _physicalToKey: Array<string | null> = [];
	private _layouts: WebGPUPagedShadowRenderSetLayout[] = [];
	private _requestBufferCapacity = 1;
	private _layoutCapacity = 1;
	private _casterCapacity = 1;
	private _cascadeCapacity = 1;
	private _computeShaderModules = new Map<string, IShaderModule>();
	private _computePipelines = new Map<string, IComputePipeline>();
	private _computeBindGroups = new Map<string, IBindingGroup>();
	private _physicalPageCount = 1;
	private _pageSize = DEFAULT_FALLBACK_PAGE_SIZE;
	private _physicalGridSize = 1;
	private _physicalAtlasSize = DEFAULT_FALLBACK_PAGE_SIZE;
	private _resourcesDirty = true;
	private _tableDirty = true;
	private _gpuResourcesDirty = true;
	private _gpuAllocationAuthoritative = false;

	public constructor(backend: WebGPUBackend, shadowPass: WebGPUShadowPass) {
		this._backend = backend;
		this._shadowPass = shadowPass;
	}

	/**
	 * @internal WebGPU frame graph preparation hook.
	 */
	public prepareFrame(request: WebGPUPagedShadowFrameRequest): void {
		this._lastRequest = request;
		this._frameId++;
		const layouts = this._resolvePagedRenderSetLayouts(request);
		this._layouts = layouts;
		this._prepareResourceShape(layouts);
		for (const layout of layouts) {
			layout.metadata.physicalAtlasSize = this._physicalAtlasSize;
			layout.metadata.physicalGridSize = this._physicalGridSize;
			layout.metadata.physicalPageSize = this._pageSize;
		}
		this._requests = collectWebGPUPagedShadowPageRequests(
			request,
			layouts
		);
		this._updateGpuFrameInputs(request, layouts);
		this._tableDirty = true;
	}

	/**
	 * @internal WebGPU page request compute pass hook.
	 */
	public async recordPageMarkPass(
		request: WebGPUPagedShadowFrameRequest
	): Promise<void> {
		this._lastRequest = request;
		this._writePageBuffersIfNeeded();
		this._resetGpuRequestBuffers();
		await this._recordComputePass(
			request.encoder,
			"pagedShadowRequestMark",
			"WebGPUPagedShadowRequestMark",
			[
				this._requestParamsBuffer,
				this._pageRequestFlagsBuffer,
				this._feedbackFlagsBuffer,
				this._layoutBuffer,
				this._casterBoundsBuffer,
				this._cascadeViewProjectionBuffer,
			],
			Math.max(1, Math.ceil(Math.max(1, request.shadowCasterPackets.length) / 64)),
			1,
			1
		);
		await this._recordComputePass(
			request.encoder,
			"pagedShadowRequestCompact",
			"WebGPUPagedShadowRequestCompact",
			[
				this._compactParamsBuffer,
				this._pageRequestFlagsBuffer,
				this._countersBuffer,
				this._compactedRequestsBuffer,
			],
			Math.max(1, Math.ceil(this._pageTable.length / 64)),
			1,
			1
		);
	}

	/**
	 * @internal WebGPU page allocation pass hook.
	 */
	public async recordPageAllocationPass(
		request: WebGPUPagedShadowFrameRequest
	): Promise<void> {
		this._lastRequest = request;
		const useGpuAllocation = !!request.encoder;
		await this._recordComputePass(
			request.encoder,
			"pagedShadowResidencyAllocate",
			"WebGPUPagedShadowResidencyAllocate",
			[
				this._allocationParamsBuffer,
				this._pageTableBuffer,
				this._residencyStateBuffer,
				this._compactedRequestsBuffer,
				this._countersBuffer,
			],
			1,
			1,
			1
		);
		await this._recordComputePass(
			request.encoder,
			"pagedShadowDirtyCompact",
			"WebGPUPagedShadowDirtyCompact",
			[
				this._dirtyParamsBuffer,
				this._residencyStateBuffer,
				this._countersBuffer,
				this._dirtyPhysicalPagesBuffer,
			],
			Math.max(1, Math.ceil(this._physicalPageCount / 64)),
			1,
			1
		);
		if (useGpuAllocation) {
			this._gpuAllocationAuthoritative = true;
		}
		if (this._requests.length <= 0) {
			this._writePageBuffersIfNeeded();
			return;
		}

		const requestedKeys = new Set<string>();
		const maxPagesPerFrame = resolveMaxPagesPerFrame(request.renderSets);
		let allocatedThisFrame = 0;
		for (const pageRequest of this._requests) {
			requestedKeys.add(pageRequest.key);
			let page = this._residentPages.get(pageRequest.key);
			if (page) {
				page.request = pageRequest;
				page.lastUsedFrame = this._frameId;
				if (!page.viewProjection) {
					page.dirty = true;
				}
				continue;
			}
			if (allocatedThisFrame >= maxPagesPerFrame) {
				continue;
			}
			const physicalPageIndex = this._allocatePhysicalPage(requestedKeys);
			if (physicalPageIndex < 0) {
				continue;
			}
			page = this._createResidentPage(pageRequest, physicalPageIndex);
			this._residentPages.set(pageRequest.key, page);
			this._physicalToKey[physicalPageIndex] = pageRequest.key;
			allocatedThisFrame++;
		}

		this._evictExpiredPages(requestedKeys, resolveMaxCacheFrames(request.renderSets));
		this._rebuildPageTable();
		this._writePageBuffersIfNeeded();
	}

	/**
	 * @internal WebGPU delayed screen-feedback compute hook.
	 */
	public async recordFeedbackPass(
		request: WebGPUPagedShadowFrameRequest
	): Promise<void> {
		this._lastRequest = request;
		await this._recordComputePass(
			request.encoder,
			"pagedShadowFeedback",
			"WebGPUPagedShadowFeedback",
			[
				this._feedbackParamsBuffer,
				this._nextFeedbackFlagsBuffer,
			],
			Math.max(1, Math.ceil(Math.max(1, request.context.attachments.width) / 8)),
			Math.max(1, Math.ceil(Math.max(1, request.context.attachments.height) / 8)),
			1
		);
		this._swapFeedbackBuffers();
	}

	/**
	 * @internal WebGPU frame graph depth pass hook.
	 */
	public async recordDepthPass(
		request: WebGPUPagedShadowFrameRequest
	): Promise<void> {
		this._lastRequest = request;
		const residentPages = Array.from(this._residentPages.values());
		const dirtyPages = residentPages.filter(
			(page) => page.dirty
		);
		if (dirtyPages.length <= 0) {
			return;
		}
		const resources = this.getResources();
		await this._shadowPass.renderPagedDepthPages(
			request.context,
			dirtyPages,
			resources.physicalDepthAtlas,
			request.encoder,
			request.shadowCasterPackets
		);
		for (const page of dirtyPages) {
			page.dirty = false;
		}
		this._rebuildPageTable();
		this._writePageBuffersIfNeeded();
	}

	/**
	 * @internal WebGPU resource query hook for shadow samplers and debug tools.
	 */
	public getResources(): WebGPUPagedShadowResources {
		if (
			!this._pageTableBuffer ||
			!this._pageMetadataBuffer ||
			!this._physicalDepthAtlas ||
			!this._pageRequestFlagsBuffer ||
			!this._compactedRequestsBuffer ||
			!this._residencyStateBuffer ||
			!this._freeListBuffer ||
			!this._countersBuffer ||
			!this._dirtyPhysicalPagesBuffer ||
			!this._feedbackFlagsBuffer ||
			!this._nextFeedbackFlagsBuffer
		) {
			return this._getFallbackResources();
		}
		return {
			pageTable: this._pageTableBuffer,
			physicalDepthAtlas: this._physicalDepthAtlas,
			physicalTransmittanceAtlas: null,
			pageMetadataBuffer: this._pageMetadataBuffer,
			pageRequestFlags: this._pageRequestFlagsBuffer,
			compactedRequests: this._compactedRequestsBuffer,
			residencyState: this._residencyStateBuffer,
			freeList: this._freeListBuffer,
			counters: this._countersBuffer,
			dirtyPhysicalPages: this._dirtyPhysicalPagesBuffer,
			feedbackFlags: this._feedbackFlagsBuffer,
			nextFeedbackFlags: this._nextFeedbackFlagsBuffer,
			pageSize: this._pageSize,
			physicalGridSize: this._physicalGridSize,
			physicalAtlasSize: this._physicalAtlasSize,
		};
	}

	public getDebugState(): WebGPUPagedShadowDebugState {
		return {
			frameId: this._frameId,
			requestCount: this._requests.length,
			residentCount: this._residentPages.size,
			dirtyCount: Array.from(this._residentPages.values()).filter((page) => page.dirty).length,
			pageTableLength: this._pageTable.length,
			physicalPageCount: this._physicalPageCount,
			pageSize: this._pageSize,
			physicalGridSize: this._physicalGridSize,
			gpuRequestBufferSize: this._compactedRequests.byteLength,
			gpuDirtyBufferSize: this._dirtyPhysicalPages.byteLength,
			feedbackFlagCount: this._feedbackFlags.length,
		};
	}

	/**
	 * @internal WebGPU runtime lifecycle hook.
	 */
	public destroy(): void {
		this._lastRequest = null;
		this._pageTableBuffer?.destroy();
		this._pageMetadataBuffer?.destroy();
		this._physicalDepthAtlas?.destroy();
		this._pageRequestFlagsBuffer?.destroy();
		this._compactedRequestsBuffer?.destroy();
		this._residencyStateBuffer?.destroy();
		this._freeListBuffer?.destroy();
		this._countersBuffer?.destroy();
		this._dirtyPhysicalPagesBuffer?.destroy();
		this._feedbackFlagsBuffer?.destroy();
		this._nextFeedbackFlagsBuffer?.destroy();
		this._layoutBuffer?.destroy();
		this._casterBoundsBuffer?.destroy();
		this._cascadeViewProjectionBuffer?.destroy();
		this._requestParamsBuffer?.destroy();
		this._compactParamsBuffer?.destroy();
		this._allocationParamsBuffer?.destroy();
		this._dirtyParamsBuffer?.destroy();
		this._feedbackParamsBuffer?.destroy();
		this._fallbackPageTableBuffer?.destroy();
		this._fallbackMetadataBuffer?.destroy();
		this._fallbackDepthAtlas?.destroy();
		this._fallbackPageRequestFlags?.destroy();
		this._fallbackCompactedRequests?.destroy();
		this._fallbackResidencyState?.destroy();
		this._fallbackFreeList?.destroy();
		this._fallbackCounters?.destroy();
		this._fallbackDirtyPhysicalPages?.destroy();
		this._fallbackFeedbackFlags?.destroy();
		this._fallbackNextFeedbackFlags?.destroy();
		this._pageTableBuffer = null;
		this._pageMetadataBuffer = null;
		this._physicalDepthAtlas = null;
		this._pageRequestFlagsBuffer = null;
		this._compactedRequestsBuffer = null;
		this._residencyStateBuffer = null;
		this._freeListBuffer = null;
		this._countersBuffer = null;
		this._dirtyPhysicalPagesBuffer = null;
		this._feedbackFlagsBuffer = null;
		this._nextFeedbackFlagsBuffer = null;
		this._layoutBuffer = null;
		this._casterBoundsBuffer = null;
		this._cascadeViewProjectionBuffer = null;
		this._requestParamsBuffer = null;
		this._compactParamsBuffer = null;
		this._allocationParamsBuffer = null;
		this._dirtyParamsBuffer = null;
		this._feedbackParamsBuffer = null;
		this._fallbackPageTableBuffer = null;
		this._fallbackMetadataBuffer = null;
		this._fallbackDepthAtlas = null;
		this._fallbackPageRequestFlags = null;
		this._fallbackCompactedRequests = null;
		this._fallbackResidencyState = null;
		this._fallbackFreeList = null;
		this._fallbackCounters = null;
		this._fallbackDirtyPhysicalPages = null;
		this._fallbackFeedbackFlags = null;
		this._fallbackNextFeedbackFlags = null;
		this._requests = [];
		this._residentPages.clear();
		this._layouts = [];
		this._physicalToKey = [];
		this._pageTable = new Uint32Array([WEBGPU_PAGED_SHADOW_NON_RESIDENT]);
		this._pageMetadata = new Uint32Array(PAGE_METADATA_UINTS);
		this._pageRequestFlags = new Uint32Array(1);
		this._compactedRequests = new Uint32Array(PAGE_REQUEST_RECORD_UINTS);
		this._residencyState = createNonResidentUint32Array(PAGE_RESIDENCY_STATE_UINTS);
		this._freeList = new Uint32Array([0]);
		this._counters = new Uint32Array(4);
		this._dirtyPhysicalPages = new Uint32Array(DIRTY_PHYSICAL_PAGE_RECORD_UINTS);
		this._feedbackFlags = new Uint32Array(1);
		this._nextFeedbackFlags = new Uint32Array(1);
		this._computeShaderModules.clear();
		this._computePipelines.clear();
		this._computeBindGroups.clear();
		this._resourcesDirty = true;
		this._tableDirty = true;
		this._gpuResourcesDirty = true;
		this._gpuAllocationAuthoritative = false;
	}

	private _resolvePagedRenderSetLayouts(
		request: WebGPUPagedShadowFrameRequest
	): WebGPUPagedShadowRenderSetLayout[] {
		const layouts: WebGPUPagedShadowRenderSetLayout[] = [];
		let pageTableCursor = 0;
		for (const [light, renderSet] of request.renderSets) {
			if (
				light.type !== LightType.Directional ||
				renderSet.storageMode !== "paged" ||
				!renderSet.layout.paged
			) {
				continue;
			}
			const metadata = renderSet.layout.paged;
			const pageGridSize = Math.max(1, metadata.pageGridSize | 0);
			const cascadeCount = Math.max(1, Math.min(renderSet.slices.length, 4));
			const pageTableCascadeStride = pageGridSize * pageGridSize;
			metadata.pageTableBase = pageTableCursor;
			metadata.pageTableCascadeStride = pageTableCascadeStride;
			metadata.physicalAtlasSize = this._physicalAtlasSize;
			metadata.physicalGridSize = this._physicalGridSize;
			metadata.physicalPageSize = metadata.pageSize;
			layouts.push({
				renderSet,
				metadata,
				pageTableBase: pageTableCursor,
				pageTableCascadeStride,
				cascadeCount,
			});
			pageTableCursor += pageTableCascadeStride * cascadeCount;
		}
		if (pageTableCursor <= 0) {
			this._pageTable = new Uint32Array([WEBGPU_PAGED_SHADOW_NON_RESIDENT]);
			return [];
		}
		if (this._pageTable.length !== pageTableCursor) {
			this._pageTable = new Uint32Array(pageTableCursor);
			this._pageTable.fill(WEBGPU_PAGED_SHADOW_NON_RESIDENT);
			this._tableDirty = true;
		}
		return layouts;
	}

	private _prepareResourceShape(
		layouts: readonly WebGPUPagedShadowRenderSetLayout[]
	): void {
		let pageSize = DEFAULT_FALLBACK_PAGE_SIZE;
		let physicalPageCount = 1;
		for (const layout of layouts) {
			pageSize = Math.max(pageSize, layout.metadata.pageSize | 0);
			physicalPageCount = Math.max(
				physicalPageCount,
				layout.metadata.physicalPageCount | 0
			);
		}
		pageSize = Math.max(1, pageSize);
		physicalPageCount = Math.max(1, physicalPageCount);
		const physicalGridSize = Math.max(1, Math.ceil(Math.sqrt(physicalPageCount)));
		const physicalAtlasSize = physicalGridSize * pageSize;
		const requestBufferCapacity = Math.max(
			1,
			...layouts.map((layout) => layout.metadata.maxPagesPerFrame | 0),
			Math.min(Math.max(1, this._pageTable.length), physicalPageCount)
		);
		const layoutCapacity = Math.max(1, layouts.length);
		const casterCapacity = Math.max(
			1,
			this._lastRequest?.shadowCasterPackets.length ?? 1
		);
		const cascadeCapacity = Math.max(1, layoutCapacity * 4);
		if (
			this._pageSize === pageSize &&
			this._physicalPageCount === physicalPageCount &&
			this._physicalGridSize === physicalGridSize &&
			this._physicalAtlasSize === physicalAtlasSize &&
			this._requestBufferCapacity === requestBufferCapacity &&
			this._layoutCapacity === layoutCapacity &&
			this._casterCapacity >= casterCapacity &&
			this._cascadeCapacity >= cascadeCapacity &&
			this._pageTableBuffer &&
			this._pageTableBuffer.size >= Math.max(4, this._pageTable.byteLength) &&
			this._pageMetadataBuffer &&
			this._physicalDepthAtlas &&
			this._pageRequestFlagsBuffer &&
			this._compactedRequestsBuffer &&
			this._residencyStateBuffer &&
			this._freeListBuffer &&
			this._countersBuffer &&
			this._dirtyPhysicalPagesBuffer &&
			this._feedbackFlagsBuffer &&
			this._nextFeedbackFlagsBuffer
		) {
			return;
		}

		this._pageSize = pageSize;
		this._physicalPageCount = physicalPageCount;
		this._physicalGridSize = physicalGridSize;
		this._physicalAtlasSize = physicalAtlasSize;
		this._requestBufferCapacity = requestBufferCapacity;
		this._layoutCapacity = layoutCapacity;
		this._casterCapacity = Math.max(this._casterCapacity, casterCapacity);
		this._cascadeCapacity = cascadeCapacity;
		this._residentPages.clear();
		this._physicalToKey = new Array(physicalPageCount).fill(null);
		this._pageMetadata = new Uint32Array(
			Math.max(1, physicalPageCount) * PAGE_METADATA_UINTS
		);
		this._pageRequestFlags = new Uint32Array(Math.max(1, this._pageTable.length));
		this._compactedRequests = new Uint32Array(
			requestBufferCapacity * PAGE_REQUEST_RECORD_UINTS
		);
		this._residencyState = createNonResidentUint32Array(
			physicalPageCount * PAGE_RESIDENCY_STATE_UINTS
		);
		this._freeList = new Uint32Array(physicalPageCount);
		for (let index = 0; index < physicalPageCount; index++) {
			this._freeList[index] = index;
		}
		this._counters = new Uint32Array(4);
		this._dirtyPhysicalPages = new Uint32Array(
			physicalPageCount * DIRTY_PHYSICAL_PAGE_RECORD_UINTS
		);
		this._feedbackFlags = new Uint32Array(Math.max(1, this._pageTable.length));
		this._nextFeedbackFlags = new Uint32Array(Math.max(1, this._pageTable.length));
		this._layoutData = new Uint32Array(layoutCapacity * PAGE_LAYOUT_UINTS);
		this._casterBoundsData = new Float32Array(this._casterCapacity * 4);
		this._cascadeViewProjectionData = new Float32Array(this._cascadeCapacity * 16);
		this._pageTableBuffer?.destroy();
		this._pageMetadataBuffer?.destroy();
		this._physicalDepthAtlas?.destroy();
		this._pageRequestFlagsBuffer?.destroy();
		this._compactedRequestsBuffer?.destroy();
		this._residencyStateBuffer?.destroy();
		this._freeListBuffer?.destroy();
		this._countersBuffer?.destroy();
		this._dirtyPhysicalPagesBuffer?.destroy();
		this._feedbackFlagsBuffer?.destroy();
		this._nextFeedbackFlagsBuffer?.destroy();
		this._layoutBuffer?.destroy();
		this._casterBoundsBuffer?.destroy();
		this._cascadeViewProjectionBuffer?.destroy();
		this._requestParamsBuffer?.destroy();
		this._compactParamsBuffer?.destroy();
		this._allocationParamsBuffer?.destroy();
		this._dirtyParamsBuffer?.destroy();
		this._feedbackParamsBuffer?.destroy();
		this._pageTableBuffer = this._backend.createBuffer({
			size: Math.max(4, this._pageTable.byteLength),
			usage: BufferUsage.Storage | BufferUsage.CopyDst | BufferUsage.CopySrc,
			label: "WebGPUPagedShadowPageTable",
		});
		this._pageMetadataBuffer = this._backend.createBuffer({
			size: Math.max(4, this._pageMetadata.byteLength),
			usage: BufferUsage.Storage | BufferUsage.CopyDst,
			label: "WebGPUPagedShadowPageMetadata",
		});
		this._physicalDepthAtlas = this._backend.createTexture({
			width: physicalAtlasSize,
			height: physicalAtlasSize,
			format: TextureFormat.Depth32Float,
			usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
			label: "WebGPUPagedShadowPhysicalDepthAtlas",
		});
		this._pageRequestFlagsBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowPageRequestFlags",
			this._pageRequestFlags.byteLength
		);
		this._compactedRequestsBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowCompactedRequests",
			this._compactedRequests.byteLength
		);
		this._residencyStateBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowResidencyState",
			this._residencyState.byteLength
		);
		this._freeListBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowFreeList",
			this._freeList.byteLength
		);
		this._countersBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowCounters",
			this._counters.byteLength
		);
		this._dirtyPhysicalPagesBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowDirtyPhysicalPages",
			this._dirtyPhysicalPages.byteLength
		);
		this._feedbackFlagsBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowFeedbackFlags",
			this._feedbackFlags.byteLength
		);
		this._nextFeedbackFlagsBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowNextFeedbackFlags",
			this._nextFeedbackFlags.byteLength
		);
		this._layoutBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowLayouts",
			this._layoutData.byteLength
		);
		this._casterBoundsBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowCasterBounds",
			this._casterBoundsData.byteLength
		);
		this._cascadeViewProjectionBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowCascadeViewProjections",
			this._cascadeViewProjectionData.byteLength
		);
		this._requestParamsBuffer = this._createUniformBuffer(
			"WebGPUPagedShadowRequestParams",
			PAGE_REQUEST_PARAMS_UINTS * 4
		);
		this._compactParamsBuffer = this._createUniformBuffer(
			"WebGPUPagedShadowCompactParams",
			PAGE_REQUEST_PARAMS_UINTS * 4
		);
		this._allocationParamsBuffer = this._createUniformBuffer(
			"WebGPUPagedShadowAllocationParams",
			PAGE_ALLOC_PARAMS_UINTS * 4
		);
		this._dirtyParamsBuffer = this._createUniformBuffer(
			"WebGPUPagedShadowDirtyParams",
			PAGE_DIRTY_PARAMS_UINTS * 4
		);
		this._feedbackParamsBuffer = this._createUniformBuffer(
			"WebGPUPagedShadowFeedbackParams",
			PAGE_FEEDBACK_PARAMS_UINTS * 4
		);
		this._resourcesDirty = true;
		this._tableDirty = true;
		this._gpuResourcesDirty = true;
		this._gpuAllocationAuthoritative = false;
		this._computeBindGroups.clear();
	}

	private _allocatePhysicalPage(requestedKeys: ReadonlySet<string>): number {
		for (let index = 0; index < this._physicalPageCount; index++) {
			if (!this._physicalToKey[index]) {
				return index;
			}
		}
		let evictIndex = -1;
		let oldestFrame = Number.POSITIVE_INFINITY;
		for (let index = 0; index < this._physicalToKey.length; index++) {
			const key = this._physicalToKey[index];
			if (!key || requestedKeys.has(key)) {
				continue;
			}
			const page = this._residentPages.get(key);
			if (!page) {
				return index;
			}
			if (page.lastUsedFrame < oldestFrame) {
				oldestFrame = page.lastUsedFrame;
				evictIndex = index;
			}
		}
		if (evictIndex >= 0) {
			const key = this._physicalToKey[evictIndex];
			if (key) {
				this._residentPages.delete(key);
			}
			this._physicalToKey[evictIndex] = null;
		}
		return evictIndex;
	}

	private _createResidentPage(
		request: WebGPUPagedShadowPageRequest,
		physicalPageIndex: number
	): WebGPUPagedShadowResidentPage {
		const viewportX = (physicalPageIndex % this._physicalGridSize) * this._pageSize;
		const viewportY =
			Math.floor(physicalPageIndex / this._physicalGridSize) * this._pageSize;
		return {
			key: request.key,
			request,
			physicalPageIndex,
			dirty: true,
			lastUsedFrame: this._frameId,
			viewProjection: createPagedShadowPageViewProjection(request),
			viewportX,
			viewportY,
			viewportSize: this._pageSize,
		};
	}

	private _evictExpiredPages(
		requestedKeys: ReadonlySet<string>,
		cacheFrames: number
	): void {
		if (cacheFrames < 0) {
			return;
		}
		const minFrame = this._frameId - cacheFrames;
		for (const [key, page] of this._residentPages) {
			if (requestedKeys.has(key) || page.lastUsedFrame >= minFrame) {
				continue;
			}
			this._physicalToKey[page.physicalPageIndex] = null;
			this._residentPages.delete(key);
		}
	}

	private _rebuildPageTable(): void {
		this._pageTable.fill(WEBGPU_PAGED_SHADOW_NON_RESIDENT);
		this._pageMetadata.fill(0);
		this._residencyState.fill(WEBGPU_PAGED_SHADOW_NON_RESIDENT);
		this._dirtyPhysicalPages.fill(0);
		let dirtyIndex = 0;
		for (const page of this._residentPages.values()) {
			this._pageTable[page.request.pageTableIndex] = page.physicalPageIndex;
			const base = page.physicalPageIndex * PAGE_METADATA_UINTS;
			this._pageMetadata[base] = page.request.pageTableIndex >>> 0;
			this._pageMetadata[base + 1] = page.request.cascadeIndex >>> 0;
			this._pageMetadata[base + 2] = page.request.pageX >>> 0;
			this._pageMetadata[base + 3] = page.request.pageY >>> 0;
			this._pageMetadata[base + 4] = page.dirty ? 1 : 0;
			this._pageMetadata[base + 5] = this._frameId >>> 0;
			this._pageMetadata[base + 6] = page.lastUsedFrame >>> 0;
			this._pageMetadata[base + 7] = 0;
			const residencyBase = page.physicalPageIndex * PAGE_RESIDENCY_STATE_UINTS;
			this._residencyState[residencyBase] = page.request.pageTableIndex >>> 0;
			this._residencyState[residencyBase + 1] = page.lastUsedFrame >>> 0;
			this._residencyState[residencyBase + 2] = 1;
			this._residencyState[residencyBase + 3] = page.dirty ? 1 : 0;
			this._residencyState[residencyBase + 4] = page.request.cascadeIndex >>> 0;
			this._residencyState[residencyBase + 5] = page.request.pageX >>> 0;
			this._residencyState[residencyBase + 6] = page.request.pageY >>> 0;
			this._residencyState[residencyBase + 7] = 0;
			if (page.dirty && dirtyIndex < this._physicalPageCount) {
				const dirtyBase = dirtyIndex * DIRTY_PHYSICAL_PAGE_RECORD_UINTS;
				this._dirtyPhysicalPages[dirtyBase] = page.physicalPageIndex >>> 0;
				this._dirtyPhysicalPages[dirtyBase + 1] =
					page.request.pageTableIndex >>> 0;
				this._dirtyPhysicalPages[dirtyBase + 2] =
					page.request.cascadeIndex >>> 0;
				this._dirtyPhysicalPages[dirtyBase + 3] = page.request.pageX >>> 0;
				this._dirtyPhysicalPages[dirtyBase + 4] = page.request.pageY >>> 0;
				this._dirtyPhysicalPages[dirtyBase + 5] = page.viewportX >>> 0;
				this._dirtyPhysicalPages[dirtyBase + 6] = page.viewportY >>> 0;
				this._dirtyPhysicalPages[dirtyBase + 7] = page.viewportSize >>> 0;
				dirtyIndex++;
			}
		}
		this._tableDirty = true;
		this._gpuResourcesDirty = true;
	}

	private _writePageBuffersIfNeeded(): void {
		if (!this._pageTableBuffer || !this._pageMetadataBuffer) {
			return;
		}
		if (!this._tableDirty && !this._resourcesDirty && !this._gpuResourcesDirty) {
			return;
		}
		if (this._tableDirty || this._resourcesDirty) {
			if (!this._gpuAllocationAuthoritative) {
				this._backend.writeBuffer(this._pageTableBuffer, this._pageTable);
			}
			this._backend.writeBuffer(this._pageMetadataBuffer, this._pageMetadata);
		}
		if (this._gpuResourcesDirty) {
			this._writeGpuStateBuffers();
		}
		this._tableDirty = false;
		this._resourcesDirty = false;
		this._gpuResourcesDirty = false;
	}

	private _createStorageBuffer(label: string, size: number): IRenderBuffer {
		return this._backend.createBuffer({
			size: Math.max(4, size),
			usage: BufferUsage.Storage | BufferUsage.CopyDst | BufferUsage.CopySrc,
			label,
		});
	}

	private _createUniformBuffer(label: string, size: number): IRenderBuffer {
		return this._backend.createBuffer({
			size: Math.max(16, size),
			usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			label,
		});
	}

	private _updateGpuFrameInputs(
		request: WebGPUPagedShadowFrameRequest,
		layouts: readonly WebGPUPagedShadowRenderSetLayout[]
	): void {
		this._layoutData.fill(0);
		this._cascadeViewProjectionData.fill(0);
		for (let layoutIndex = 0; layoutIndex < layouts.length; layoutIndex++) {
			const layout = layouts[layoutIndex];
			const layoutOffset = layoutIndex * PAGE_LAYOUT_UINTS;
			this._layoutData[layoutOffset] = layout.pageTableBase >>> 0;
			this._layoutData[layoutOffset + 1] = layout.pageTableCascadeStride >>> 0;
			this._layoutData[layoutOffset + 2] = layout.metadata.pageGridSize >>> 0;
			this._layoutData[layoutOffset + 3] = layout.cascadeCount >>> 0;
			this._layoutData[layoutOffset + 4] = Math.max(0, 4 - layoutIndex) >>> 0;
			for (let cascadeIndex = 0; cascadeIndex < layout.cascadeCount; cascadeIndex++) {
				const slice = layout.renderSet.slices[cascadeIndex];
				const matrix = slice?.shadowMap.viewProjectionMatrix;
				if (!matrix) {
					continue;
				}
				this._setMatrixInFloatArray(
					matrix,
					this._cascadeViewProjectionData,
					(layoutIndex * 4 + cascadeIndex) * 16
				);
			}
		}

		const casterCount = request.shadowCasterPackets.length;
		if (this._casterBoundsData.length < Math.max(1, casterCount) * 4) {
			this._casterCapacity = Math.max(1, casterCount);
			this._casterBoundsData = new Float32Array(this._casterCapacity * 4);
			this._casterBoundsBuffer?.destroy();
			this._casterBoundsBuffer = this._createStorageBuffer(
				"WebGPUPagedShadowCasterBounds",
				this._casterBoundsData.byteLength
			);
			this._computeBindGroups.clear();
		}
		this._casterBoundsData.fill(0);
		for (let index = 0; index < casterCount; index++) {
			const bounds = request.shadowCasterPackets[index].worldBounds;
			const offset = index * 4;
			this._casterBoundsData[offset] = bounds.center.x;
			this._casterBoundsData[offset + 1] = bounds.center.y;
			this._casterBoundsData[offset + 2] = bounds.center.z;
			this._casterBoundsData[offset + 3] = Math.max(0, bounds.radius);
		}

		this._backend.writeBuffer(
			this._requestParamsBuffer!,
			new Uint32Array([
				this._pageTable.length,
				casterCount,
				layouts.length,
				this._frameId,
			])
		);
		this._backend.writeBuffer(
			this._compactParamsBuffer!,
			new Uint32Array([
				this._pageTable.length,
				this._requestBufferCapacity,
				layouts[0]?.metadata.pageGridSize ?? 1,
				0,
			])
		);
		this._backend.writeBuffer(
			this._allocationParamsBuffer!,
			new Uint32Array([
				this._frameId,
				this._requestBufferCapacity,
				this._physicalPageCount,
				resolveMaxPagesPerFrame(request.renderSets),
				resolveMaxCacheFrames(request.renderSets),
				this._pageTable.length,
				0,
				0,
			])
		);
		this._backend.writeBuffer(
			this._dirtyParamsBuffer!,
			new Uint32Array([
				this._physicalPageCount,
				this._physicalPageCount,
				this._physicalGridSize,
				this._pageSize,
			])
		);
		this._backend.writeBuffer(
			this._feedbackParamsBuffer!,
			new Uint32Array([
				this._pageTable.length,
				Math.max(1, request.context.attachments.width | 0),
				Math.max(1, request.context.attachments.height | 0),
				0,
			])
		);
		this._backend.writeBuffer(this._layoutBuffer!, this._layoutData);
		this._backend.writeBuffer(this._casterBoundsBuffer!, this._casterBoundsData);
		this._backend.writeBuffer(
			this._cascadeViewProjectionBuffer!,
			this._cascadeViewProjectionData
		);
		this._gpuResourcesDirty = true;
	}

	private _resetGpuRequestBuffers(): void {
		this._pageRequestFlags.fill(0);
		this._compactedRequests.fill(0);
		this._counters.fill(0);
		this._dirtyPhysicalPages.fill(0);
		if (!this._pageRequestFlagsBuffer || !this._compactedRequestsBuffer || !this._countersBuffer) {
			return;
		}
		this._backend.writeBuffer(this._pageRequestFlagsBuffer, this._pageRequestFlags);
		this._backend.writeBuffer(this._compactedRequestsBuffer, this._compactedRequests);
		this._backend.writeBuffer(this._countersBuffer, this._counters);
		if (this._dirtyPhysicalPagesBuffer) {
			this._backend.writeBuffer(
				this._dirtyPhysicalPagesBuffer,
				this._dirtyPhysicalPages
			);
		}
	}

	private _writeGpuStateBuffers(): void {
		const writes: Array<[IRenderBuffer | null, Uint32Array | Float32Array]> = [
			[this._pageRequestFlagsBuffer, this._pageRequestFlags],
			[this._compactedRequestsBuffer, this._compactedRequests],
			[this._residencyStateBuffer, this._residencyState],
			[this._freeListBuffer, this._freeList],
			[this._countersBuffer, this._counters],
			[this._dirtyPhysicalPagesBuffer, this._dirtyPhysicalPages],
			[this._feedbackFlagsBuffer, this._feedbackFlags],
			[this._nextFeedbackFlagsBuffer, this._nextFeedbackFlags],
		];
		for (const [buffer, data] of writes) {
			if (buffer) {
				this._backend.writeBuffer(buffer, data as BufferSource);
			}
		}
	}

	private async _recordComputePass(
		encoder: ICommandEncoder | null,
		shaderPart: string,
		label: string,
		resources: ReadonlyArray<IRenderBuffer | null>,
		x: number,
		y: number,
		z: number
	): Promise<void> {
		if (!encoder || resources.some((resource) => !resource)) {
			return;
		}
		const backend = this._backend as WebGPUBackend & {
			createBindingGroup?: (desc: {
				pipeline?: IComputePipeline;
				layoutIndex?: number;
				entries: Array<{ binding: number; resource: IRenderBuffer }>;
				label?: string;
			}) => IBindingGroup;
			createComputePipeline?: (desc: {
				label: string;
				compute: { module: IShaderModule; entryPoint: string };
			}) => Promise<IComputePipeline>;
			createShaderModule?: (desc: {
				label: string;
				code: string;
				sourceMap?: unknown;
				language: "wgsl";
				stage: "compute";
				entryPoint: "csMain";
				sourceKind: "shadow";
			}) => Promise<IShaderModule>;
		};
		if (
			typeof backend.createBindingGroup !== "function" ||
			typeof backend.createComputePipeline !== "function" ||
			typeof backend.createShaderModule !== "function"
		) {
			return;
		}
		const pipeline = await this._getComputePipeline(shaderPart, label);
		if (!pipeline) {
			return;
		}
		const bindGroupKey = `${shaderPart}:${resources.map((resource) => resource?.size ?? 0).join(":")}`;
		let bindGroup = this._computeBindGroups.get(bindGroupKey);
		if (!bindGroup) {
			bindGroup = backend.createBindingGroup({
				pipeline,
				layoutIndex: 0,
				label: `${label}BindGroup`,
				entries: resources.map((resource, binding) => ({
					binding,
					resource: resource!,
				})),
			});
			this._computeBindGroups.set(bindGroupKey, bindGroup);
		}
		encoder.beginComputePass({ label });
		encoder.setComputePipeline(pipeline);
		encoder.setBindingGroup(0, bindGroup);
		encoder.dispatchWorkgroups(x, y, z);
		encoder.endComputePass();
	}

	private async _getComputePipeline(
		shaderPart: string,
		label: string
	): Promise<IComputePipeline | null> {
		const cached = this._computePipelines.get(shaderPart);
		if (cached) {
			return cached;
		}
		const backend = this._backend as WebGPUBackend & {
			createComputePipeline?: (desc: {
				label: string;
				compute: { module: IShaderModule; entryPoint: string };
			}) => Promise<IComputePipeline>;
			createShaderModule?: (desc: {
				label: string;
				code: string;
				sourceMap?: unknown;
				language: "wgsl";
				stage: "compute";
				entryPoint: "csMain";
				sourceKind: "shadow";
			}) => Promise<IShaderModule>;
		};
		if (
			typeof backend.createShaderModule !== "function" ||
			typeof backend.createComputePipeline !== "function"
		) {
			return null;
		}
		let module = this._computeShaderModules.get(shaderPart);
		if (!module) {
			const composite = await ShaderSource.load(
				`webgpu.shadow.${shaderPart}.composite` as Parameters<
					typeof ShaderSource.load
				>[0]
			) as { code: string; sourceMap?: unknown };
			module = await backend.createShaderModule({
				label: `${label}Shader`,
				code: composite.code,
				sourceMap: composite.sourceMap,
				language: "wgsl",
				stage: "compute",
				entryPoint: "csMain",
				sourceKind: "shadow",
			});
			this._computeShaderModules.set(shaderPart, module);
		}
		const pipeline = await backend.createComputePipeline({
			label: `${label}Pipeline`,
			compute: { module, entryPoint: "csMain" },
		});
		this._computePipelines.set(shaderPart, pipeline);
		return pipeline;
	}

	private _swapFeedbackBuffers(): void {
		const feedbackFlags = this._feedbackFlags;
		this._feedbackFlags = this._nextFeedbackFlags;
		this._nextFeedbackFlags = feedbackFlags;
		const feedbackBuffer = this._feedbackFlagsBuffer;
		this._feedbackFlagsBuffer = this._nextFeedbackFlagsBuffer;
		this._nextFeedbackFlagsBuffer = feedbackBuffer;
		this._nextFeedbackFlags.fill(0);
		if (this._nextFeedbackFlagsBuffer) {
			this._backend.writeBuffer(
				this._nextFeedbackFlagsBuffer,
				this._nextFeedbackFlags
			);
		}
		this._computeBindGroups.clear();
	}

	private _setMatrixInFloatArray(
		matrix: Matrix4,
		target: Float32Array,
		offset: number
	): void {
		const elements = matrix.elements;
		target[offset] = elements[0][0];
		target[offset + 1] = elements[1][0];
		target[offset + 2] = elements[2][0];
		target[offset + 3] = elements[3][0];
		target[offset + 4] = elements[0][1];
		target[offset + 5] = elements[1][1];
		target[offset + 6] = elements[2][1];
		target[offset + 7] = elements[3][1];
		target[offset + 8] = elements[0][2];
		target[offset + 9] = elements[1][2];
		target[offset + 10] = elements[2][2];
		target[offset + 11] = elements[3][2];
		target[offset + 12] = elements[0][3];
		target[offset + 13] = elements[1][3];
		target[offset + 14] = elements[2][3];
		target[offset + 15] = elements[3][3];
	}

	private _getFallbackResources(): WebGPUPagedShadowResources {
		if (!this._fallbackPageTableBuffer) {
			this._fallbackPageTableBuffer = this._backend.createBuffer({
				size: 4,
				usage: BufferUsage.Storage | BufferUsage.CopyDst,
				label: "WebGPUPagedShadowFallbackPageTable",
			});
			this._backend.writeBuffer(
				this._fallbackPageTableBuffer,
				new Uint32Array([WEBGPU_PAGED_SHADOW_NON_RESIDENT])
			);
		}
		if (!this._fallbackMetadataBuffer) {
			this._fallbackMetadataBuffer = this._backend.createBuffer({
				size: PAGE_METADATA_UINTS * 4,
				usage: BufferUsage.Storage | BufferUsage.CopyDst,
				label: "WebGPUPagedShadowFallbackMetadata",
			});
			this._backend.writeBuffer(
				this._fallbackMetadataBuffer,
				new Uint32Array(PAGE_METADATA_UINTS)
			);
		}
		if (!this._fallbackDepthAtlas) {
			this._fallbackDepthAtlas = this._backend.createTexture({
				width: DEFAULT_FALLBACK_PAGE_SIZE,
				height: DEFAULT_FALLBACK_PAGE_SIZE,
				format: TextureFormat.Depth32Float,
				usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
				label: "WebGPUPagedShadowFallbackDepthAtlas",
			});
		}
		this._fallbackPageRequestFlags ??= this._createFallbackBuffer(
			"WebGPUPagedShadowFallbackPageRequestFlags",
			4
		);
		this._fallbackCompactedRequests ??= this._createFallbackBuffer(
			"WebGPUPagedShadowFallbackCompactedRequests",
			PAGE_REQUEST_RECORD_UINTS * 4
		);
		this._fallbackResidencyState ??= this._createFallbackBuffer(
			"WebGPUPagedShadowFallbackResidencyState",
			PAGE_RESIDENCY_STATE_UINTS * 4
		);
		this._fallbackFreeList ??= this._createFallbackBuffer(
			"WebGPUPagedShadowFallbackFreeList",
			4
		);
		this._fallbackCounters ??= this._createFallbackBuffer(
			"WebGPUPagedShadowFallbackCounters",
			16
		);
		this._fallbackDirtyPhysicalPages ??= this._createFallbackBuffer(
			"WebGPUPagedShadowFallbackDirtyPhysicalPages",
			DIRTY_PHYSICAL_PAGE_RECORD_UINTS * 4
		);
		this._fallbackFeedbackFlags ??= this._createFallbackBuffer(
			"WebGPUPagedShadowFallbackFeedbackFlags",
			4
		);
		this._fallbackNextFeedbackFlags ??= this._createFallbackBuffer(
			"WebGPUPagedShadowFallbackNextFeedbackFlags",
			4
		);
		return {
			pageTable: this._fallbackPageTableBuffer,
			physicalDepthAtlas: this._fallbackDepthAtlas,
			physicalTransmittanceAtlas: null,
			pageMetadataBuffer: this._fallbackMetadataBuffer,
			pageRequestFlags: this._fallbackPageRequestFlags,
			compactedRequests: this._fallbackCompactedRequests,
			residencyState: this._fallbackResidencyState,
			freeList: this._fallbackFreeList,
			counters: this._fallbackCounters,
			dirtyPhysicalPages: this._fallbackDirtyPhysicalPages,
			feedbackFlags: this._fallbackFeedbackFlags,
			nextFeedbackFlags: this._fallbackNextFeedbackFlags,
			pageSize: DEFAULT_FALLBACK_PAGE_SIZE,
			physicalGridSize: 1,
			physicalAtlasSize: DEFAULT_FALLBACK_PAGE_SIZE,
		};
	}

	private _createFallbackBuffer(label: string, size: number): IRenderBuffer {
		const buffer = this._backend.createBuffer({
			size: Math.max(4, size),
			usage: BufferUsage.Storage | BufferUsage.CopyDst,
			label,
		});
		this._backend.writeBuffer(buffer, new Uint32Array(Math.max(1, size / 4)));
		return buffer;
	}
}

export function collectWebGPUPagedShadowPageRequests(
	request: WebGPUPagedShadowFrameRequest,
	layouts: readonly WebGPUPagedShadowRenderSetLayout[]
): WebGPUPagedShadowPageRequest[] {
	const requests = new Map<string, WebGPUPagedShadowPageRequest>();
	for (const [light, renderSet] of request.renderSets) {
		if (light.type !== LightType.Directional) {
			continue;
		}
		const layout = layouts.find((entry) => entry.renderSet === renderSet);
		if (!layout) {
			continue;
		}
		for (let cascadeIndex = 0; cascadeIndex < layout.cascadeCount; cascadeIndex++) {
			const slice = renderSet.slices[cascadeIndex];
			const viewProjection = slice?.shadowMap.viewProjectionMatrix;
			if (!viewProjection) {
				continue;
			}
			for (const packet of request.shadowCasterPackets) {
				addRequestsForPacketBounds(
					requests,
					light,
					renderSet,
					layout,
					cascadeIndex,
					viewProjection,
					packet.worldBounds.center,
					packet.worldBounds.radius
				);
			}
		}
	}
	return Array.from(requests.values()).sort((left, right) => {
		if (left.cascadeIndex !== right.cascadeIndex) {
			return left.cascadeIndex - right.cascadeIndex;
		}
		if (left.priority !== right.priority) {
			return right.priority - left.priority;
		}
		return left.key.localeCompare(right.key);
	});
}

function addRequestsForPacketBounds(
	requests: Map<string, WebGPUPagedShadowPageRequest>,
	light: ShadowCastingLight,
	renderSet: ShadowRenderSet,
	layout: WebGPUPagedShadowRenderSetLayout,
	cascadeIndex: number,
	viewProjection: Matrix4,
	center: IVector3,
	radius: number
): void {
	const projected = projectSphereToShadowUvBounds(viewProjection, center, radius);
	if (!projected) {
		return;
	}
	const gridSize = Math.max(1, layout.metadata.pageGridSize | 0);
	const minPageX = clampInt(Math.floor(projected.minU * gridSize), 0, gridSize - 1);
	const maxPageX = clampInt(Math.floor(projected.maxU * gridSize), 0, gridSize - 1);
	const minPageY = clampInt(Math.floor(projected.minV * gridSize), 0, gridSize - 1);
	const maxPageY = clampInt(Math.floor(projected.maxV * gridSize), 0, gridSize - 1);
	for (let pageY = minPageY; pageY <= maxPageY; pageY++) {
		for (let pageX = minPageX; pageX <= maxPageX; pageX++) {
			const pageTableIndex =
				layout.pageTableBase +
				cascadeIndex * layout.pageTableCascadeStride +
				pageY * gridSize +
				pageX;
			const key = `${light.id}:${renderSet.configSignature}:${cascadeIndex}:${pageX}:${pageY}`;
			if (requests.has(key)) {
				continue;
			}
			requests.set(key, {
				key,
				lightId: light.id,
				shadowMapId: `${renderSet.configSignature}`,
				cascadeIndex,
				viewProjection,
				pageX,
				pageY,
				pageGridSize: gridSize,
				pageTableIndex,
				priority: Math.max(0, 4 - cascadeIndex),
			});
		}
	}
}

function projectSphereToShadowUvBounds(
	viewProjection: Matrix4,
	center: IVector3,
	radius: number
): { minU: number; minV: number; maxU: number; maxV: number } | null {
	const safeRadius = Math.max(0, Number.isFinite(radius) ? radius : 0);
	const corners = [
		{ x: center.x - safeRadius, y: center.y - safeRadius, z: center.z - safeRadius },
		{ x: center.x + safeRadius, y: center.y - safeRadius, z: center.z - safeRadius },
		{ x: center.x - safeRadius, y: center.y + safeRadius, z: center.z - safeRadius },
		{ x: center.x + safeRadius, y: center.y + safeRadius, z: center.z - safeRadius },
		{ x: center.x - safeRadius, y: center.y - safeRadius, z: center.z + safeRadius },
		{ x: center.x + safeRadius, y: center.y - safeRadius, z: center.z + safeRadius },
		{ x: center.x - safeRadius, y: center.y + safeRadius, z: center.z + safeRadius },
		{ x: center.x + safeRadius, y: center.y + safeRadius, z: center.z + safeRadius },
	];
	let minU = 1;
	let minV = 1;
	let maxU = 0;
	let maxV = 0;
	let hasProjectedPoint = false;
	for (const corner of corners) {
		const clip = Matrix4.transformPoint(viewProjection, corner);
		const w = typeof clip.w === "number" ? clip.w : 1;
		if (Math.abs(w) <= 1e-6) {
			continue;
		}
		const invW = 1 / w;
		const ndcX = clip.x * invW;
		const ndcY = clip.y * invW;
		const ndcZ = clip.z * invW;
		if (ndcZ < -1 || ndcZ > 1) {
			continue;
		}
		const u = ndcX * 0.5 + 0.5;
		const v = 0.5 - ndcY * 0.5;
		minU = Math.min(minU, u);
		minV = Math.min(minV, v);
		maxU = Math.max(maxU, u);
		maxV = Math.max(maxV, v);
		hasProjectedPoint = true;
	}
	if (!hasProjectedPoint) {
		return null;
	}
	if (maxU < 0 || minU > 1 || maxV < 0 || minV > 1) {
		return null;
	}
	return {
		minU: Math.max(0, minU),
		minV: Math.max(0, minV),
		maxU: Math.min(1, maxU),
		maxV: Math.min(1, maxV),
	};
}

function createPagedShadowPageViewProjection(
	request: WebGPUPagedShadowPageRequest
): Matrix4 {
	const grid = Math.max(1, request.pageGridSize);
	const scale = grid;
	const offsetX = grid - request.pageX * 2 - 1;
	const offsetY = request.pageY * 2 + 1 - grid;
	const crop = new Matrix4([
		[scale, 0, 0, offsetX],
		[0, scale, 0, offsetY],
		[0, 0, 1, 0],
		[0, 0, 0, 1],
	]);
	const source = findRequestViewProjection(request);
	return source ? Matrix4.multiply(crop, source) : crop;
}

function findRequestViewProjection(
	request: WebGPUPagedShadowPageRequest
): Matrix4 | null {
	return request.viewProjection ?? null;
}

function resolveMaxPagesPerFrame(
	renderSets: ReadonlyMap<ShadowCastingLight, ShadowRenderSet>
): number {
	let maxPages = 1;
	for (const renderSet of renderSets.values()) {
		const paged = renderSet.layout.paged;
		if (!paged) {
			continue;
		}
		maxPages = Math.max(maxPages, paged.maxPagesPerFrame | 0);
	}
	return Math.max(1, maxPages);
}

function resolveMaxCacheFrames(
	renderSets: ReadonlyMap<ShadowCastingLight, ShadowRenderSet>
): number {
	let cacheFrames = 0;
	for (const renderSet of renderSets.values()) {
		const paged = renderSet.layout.paged;
		if (!paged) {
			continue;
		}
		cacheFrames = Math.max(cacheFrames, paged.cacheFrames | 0);
	}
	return Math.max(0, cacheFrames);
}

function clampInt(value: number, min: number, max: number): number {
	return Math.max(min, Math.min(max, value | 0));
}

function createNonResidentUint32Array(length: number): Uint32Array {
	const array = new Uint32Array(Math.max(1, length));
	array.fill(WEBGPU_PAGED_SHADOW_NON_RESIDENT);
	return array;
}
