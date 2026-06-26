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
	type BindingResource,
	type IBindingGroup,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type IShaderModule,
} from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
import type { WebGPUShadowPass } from "./WebGPUShadowPass";
import { tryGetNativeWebGPUCommandEncoder } from "./WebGPUCommandEncoder";
import {
	getWebGPUBuffer,
	tryGetWebGPUBuffer,
} from "./WebGPUResourceAccess";

export const WEBGPU_PAGED_SHADOW_NON_RESIDENT = 0xffffffff;

const PAGE_METADATA_UINTS = 8;
const PAGE_REQUEST_RECORD_UINTS = 8;
const PAGE_RESIDENCY_STATE_UINTS = 8;
const DIRTY_PHYSICAL_PAGE_RECORD_UINTS = 8;
const PAGE_LAYOUT_UINTS = 8;
const PAGE_REQUEST_PARAMS_UINTS = 8;
const PAGE_ALLOC_PARAMS_UINTS = 8;
const PAGE_DIRTY_PARAMS_UINTS = 4;
const PAGE_DRAW_PARAMS_UINTS = 8;
const PAGE_FEEDBACK_PARAMS_UINTS = 8;
const DRAW_INDIRECT_UINTS = 5;
const SHADOW_INSTANCE_DATA_UINTS = 12;
const DEFAULT_FALLBACK_PAGE_SIZE = 1;
const DRAW_INSTANCE_INITIAL_PAGES_PER_CASTER = 4;
const DRAW_COUNTER_READBACK_UINTS = 8;
const WEBGPU_MAP_MODE_READ =
	(globalThis as { GPUMapMode?: { READ?: number } }).GPUMapMode?.READ ??
	0x0001;

export interface WebGPUPagedShadowFrameRequest {
	context: FrameContext;
	encoder: ICommandEncoder | null;
	renderSets: ReadonlyMap<ShadowCastingLight, ShadowRenderSet>;
	shadowCasterPackets: readonly DrawPacket[];
	shadowTransmitterPackets: readonly DrawPacket[];
	feedbackDepthTexture?: IRenderTexture | null;
	feedbackMotionDepthTexture?: IRenderTexture | null;
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
	drawMvpBuffer: IRenderBuffer;
	drawInstanceMetaBuffer: IRenderBuffer;
	drawTransmittanceBuffer: IRenderBuffer;
	drawIndirectArgsBuffer: IRenderBuffer;
	drawCandidateWorldMatrices: IRenderBuffer;
	pageSize: number;
	physicalGridSize: number;
	physicalAtlasSize: number;
	drawCandidateCount: number;
	drawInstanceCapacity: number;
	physicalPageCount: number;
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
	pageTableLength: number;
	physicalPageCount: number;
	pageSize: number;
	physicalGridSize: number;
	requestBufferCapacity: number;
	dirtyBufferCapacity: number;
	feedbackFlagCount: number;
	layoutCapacity: number;
	casterCapacity: number;
	drawCandidateCapacity: number;
	drawCandidateCount: number;
	drawInstanceCapacity: number;
	hasCounterBuffer: boolean;
	gpuAuthoritative: boolean;
}

export interface WebGPUPagedShadowRenderSetLayout {
	renderSet: ShadowRenderSet;
	metadata: PagedShadowLayoutMetadata;
	pageTableBase: number;
	pageTableCascadeStride: number;
	cascadeCount: number;
}

interface FrameCasterSnapshot {
	centerX: number;
	centerY: number;
	centerZ: number;
	radius: number;
}

interface PendingDrawCounterReadback {
	frameId: number;
	buffer: GPUBuffer;
	byteLength: number;
	queued: boolean;
	done: boolean;
}

/**
 * WebGPU paged shadow runtime.
 *
 * The WebGPU path keeps residency, dirty-page compaction, and page-table
 * updates GPU authoritative. CPU work is limited to frame-local layout,
 * cascade, caster bounds, world matrices, and indirect-argument seed uploads.
 */
export class WebGPUPagedShadowRuntime {
	private _backend: WebGPUBackend;
	private _shadowPass: WebGPUShadowPass;
	private _frameId = 0;
	private _lastRequest: WebGPUPagedShadowFrameRequest | null = null;
	private _preparedContext: FrameContext | null = null;
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
	private _fallbackDrawMvpBuffer: IRenderBuffer | null = null;
	private _fallbackDrawMetaBuffer: IRenderBuffer | null = null;
	private _fallbackDrawTransmittanceBuffer: IRenderBuffer | null = null;
	private _fallbackDrawIndirectArgsBuffer: IRenderBuffer | null = null;
	private _fallbackDrawCandidateWorldMatrices: IRenderBuffer | null = null;
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
	private _drawWorldMatrixBuffer: IRenderBuffer | null = null;
	private _drawMvpBuffer: IRenderBuffer | null = null;
	private _drawInstanceMetaBuffer: IRenderBuffer | null = null;
	private _drawTransmittanceBuffer: IRenderBuffer | null = null;
	private _drawIndirectArgsBuffer: IRenderBuffer | null = null;
	private _requestParamsBuffer: IRenderBuffer | null = null;
	private _compactParamsBuffer: IRenderBuffer | null = null;
	private _allocationParamsBuffer: IRenderBuffer | null = null;
	private _dirtyParamsBuffer: IRenderBuffer | null = null;
	private _drawParamsBuffer: IRenderBuffer | null = null;
	private _feedbackParamsBuffer: IRenderBuffer | null = null;
	private _feedbackCameraBuffer: IRenderBuffer | null = null;
	private _pageTableLength = 1;
	private _pageRequestFlags = new Uint32Array(1);
	private _compactedRequests = new Uint32Array(PAGE_REQUEST_RECORD_UINTS);
	private _residencyState = createNonResidentUint32Array(PAGE_RESIDENCY_STATE_UINTS);
	private _freeList = new Uint32Array([0]);
	private _counters = new Uint32Array(8);
	private _dirtyPhysicalPages = new Uint32Array(DIRTY_PHYSICAL_PAGE_RECORD_UINTS);
	private _feedbackFlags = new Uint32Array(1);
	private _nextFeedbackFlags = new Uint32Array(1);
	private _layoutData = new Uint32Array(PAGE_LAYOUT_UINTS);
	private _casterBoundsData = new Float32Array(4);
	private _cascadeViewProjectionData = new Float32Array(16);
	private _drawWorldMatrixData = new Float32Array(16);
	private _drawIndirectArgsData = new Uint32Array(DRAW_INDIRECT_UINTS);
	private _feedbackCameraData = new Float32Array(16);
	private _layouts: WebGPUPagedShadowRenderSetLayout[] = [];
	private _requestBufferCapacity = 1;
	private _pageTableBufferLength = 1;
	private _layoutCapacity = 1;
	private _casterCapacity = 1;
	private _cascadeCapacity = 1;
	private _drawCandidateCapacity = 1;
	private _drawCandidateCount = 0;
	private _drawInstanceCapacity = 1;
	private _drawInstanceCapacityPressure = 0;
	private _computeShaderModules = new Map<string, IShaderModule>();
	private _computePipelines = new Map<string, IComputePipeline>();
	private _computeBindGroups = new Map<string, IBindingGroup>();
	private _physicalPageCount = 1;
	private _pageSize = DEFAULT_FALLBACK_PAGE_SIZE;
	private _physicalGridSize = 1;
	private _physicalAtlasSize = DEFAULT_FALLBACK_PAGE_SIZE;
	private _previousCasterBounds = new Map<string, FrameCasterSnapshot>();
	private _pendingDrawCounterReadbacks: PendingDrawCounterReadback[] = [];

	public constructor(backend: WebGPUBackend, shadowPass: WebGPUShadowPass) {
		this._backend = backend;
		this._shadowPass = shadowPass;
	}

	/**
	 * @internal WebGPU frame graph preparation hook.
	 */
	public prepareFrame(request: WebGPUPagedShadowFrameRequest): void {
		this._lastRequest = request;
		this._scheduleQueuedDrawCounterReadbacks();
		this._collectCompletedDrawCounterReadbacks();
		if (this._preparedContext === request.context) {
			return;
		}
		this._preparedContext = request.context;
		this._frameId++;
		const layouts = this._resolvePagedRenderSetLayouts(request);
		this._layouts = layouts;
		this._prepareResourceShape(request, layouts);
		for (const layout of layouts) {
			layout.metadata.physicalAtlasSize = this._physicalAtlasSize;
			layout.metadata.physicalGridSize = this._physicalGridSize;
			layout.metadata.physicalPageSize = this._pageSize;
		}
		this._updateGpuFrameInputs(request, layouts);
	}

	/**
	 * @internal WebGPU page request compute pass hook.
	 */
	public async recordPageMarkPass(
		request: WebGPUPagedShadowFrameRequest
	): Promise<void> {
		this._lastRequest = request;
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
			Math.max(1, Math.ceil(Math.max(this._pageTableLength, this._casterBoundsData.length / 4) / 64)),
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
			Math.max(1, Math.ceil(this._pageTableLength / 64)),
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
				this._pageMetadataBuffer,
				this._freeListBuffer,
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
				this._pageTableBuffer,
				this._layoutBuffer,
				this._cascadeViewProjectionBuffer,
				request.feedbackDepthTexture,
				this._feedbackCameraBuffer,
			],
			Math.max(1, Math.ceil(Math.max(1, request.context.attachments.width) / 64)),
			Math.max(1, Math.ceil(Math.max(1, request.context.attachments.height) / 64)),
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
		this._resetGpuDrawCounters();
		await this._recordComputePass(
			request.encoder,
			"pagedShadowDrawBuild",
			"WebGPUPagedShadowDrawBuild",
			[
				this._drawParamsBuffer,
				this._dirtyPhysicalPagesBuffer,
				this._countersBuffer,
				this._casterBoundsBuffer,
				this._drawWorldMatrixBuffer,
				this._cascadeViewProjectionBuffer,
				this._drawMvpBuffer,
				this._drawInstanceMetaBuffer,
				this._drawTransmittanceBuffer,
				this._drawIndirectArgsBuffer,
			],
			Math.max(1, Math.ceil(this._drawCandidateCount / 64)),
			1,
			1
		);
		await this._shadowPass.renderPagedDepthIndirect(
			request.context,
			this.getResources(),
			request.encoder,
			request.shadowCasterPackets
		);
		this._queueDrawCounterReadback(request);
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
			!this._nextFeedbackFlagsBuffer ||
			!this._drawMvpBuffer ||
			!this._drawInstanceMetaBuffer ||
			!this._drawTransmittanceBuffer ||
			!this._drawIndirectArgsBuffer ||
			!this._drawWorldMatrixBuffer
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
			drawMvpBuffer: this._drawMvpBuffer,
			drawInstanceMetaBuffer: this._drawInstanceMetaBuffer,
			drawTransmittanceBuffer: this._drawTransmittanceBuffer,
			drawIndirectArgsBuffer: this._drawIndirectArgsBuffer,
			drawCandidateWorldMatrices: this._drawWorldMatrixBuffer,
			pageSize: this._pageSize,
			physicalGridSize: this._physicalGridSize,
			physicalAtlasSize: this._physicalAtlasSize,
			drawCandidateCount: this._drawCandidateCount,
			drawInstanceCapacity: this._drawInstanceCapacity,
			physicalPageCount: this._physicalPageCount,
		};
	}

	public getDebugState(): WebGPUPagedShadowDebugState {
		return {
			frameId: this._frameId,
			pageTableLength: this._pageTableLength,
			physicalPageCount: this._physicalPageCount,
			pageSize: this._pageSize,
			physicalGridSize: this._physicalGridSize,
			requestBufferCapacity: this._requestBufferCapacity,
			dirtyBufferCapacity: this._physicalPageCount,
			feedbackFlagCount: this._feedbackFlags.length,
			layoutCapacity: this._layoutCapacity,
			casterCapacity: this._casterCapacity,
			drawCandidateCapacity: this._drawCandidateCapacity,
			drawCandidateCount: this._drawCandidateCount,
			drawInstanceCapacity: this._drawInstanceCapacity,
			hasCounterBuffer: !!this._countersBuffer,
			gpuAuthoritative: true,
		};
	}

	/**
	 * @internal WebGPU runtime lifecycle hook.
	 */
	public destroy(): void {
		this._lastRequest = null;
		this._preparedContext = null;
		for (const pending of this._pendingDrawCounterReadbacks) {
			pending.buffer.destroy();
		}
		this._pendingDrawCounterReadbacks = [];
		for (const resource of [
			this._pageTableBuffer,
			this._pageMetadataBuffer,
			this._physicalDepthAtlas,
			this._pageRequestFlagsBuffer,
			this._compactedRequestsBuffer,
			this._residencyStateBuffer,
			this._freeListBuffer,
			this._countersBuffer,
			this._dirtyPhysicalPagesBuffer,
			this._feedbackFlagsBuffer,
			this._nextFeedbackFlagsBuffer,
			this._layoutBuffer,
			this._casterBoundsBuffer,
			this._cascadeViewProjectionBuffer,
			this._drawWorldMatrixBuffer,
			this._drawMvpBuffer,
			this._drawInstanceMetaBuffer,
			this._drawTransmittanceBuffer,
			this._drawIndirectArgsBuffer,
			this._requestParamsBuffer,
			this._compactParamsBuffer,
			this._allocationParamsBuffer,
			this._dirtyParamsBuffer,
			this._drawParamsBuffer,
			this._feedbackParamsBuffer,
			this._fallbackPageTableBuffer,
			this._fallbackMetadataBuffer,
			this._fallbackDepthAtlas,
			this._fallbackPageRequestFlags,
			this._fallbackCompactedRequests,
			this._fallbackResidencyState,
			this._fallbackFreeList,
			this._fallbackCounters,
			this._fallbackDirtyPhysicalPages,
			this._fallbackFeedbackFlags,
			this._fallbackNextFeedbackFlags,
			this._fallbackDrawMvpBuffer,
			this._fallbackDrawMetaBuffer,
			this._fallbackDrawTransmittanceBuffer,
			this._fallbackDrawIndirectArgsBuffer,
			this._fallbackDrawCandidateWorldMatrices,
		]) {
			resource?.destroy();
		}
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
		this._drawWorldMatrixBuffer = null;
		this._drawMvpBuffer = null;
		this._drawInstanceMetaBuffer = null;
		this._drawTransmittanceBuffer = null;
		this._drawIndirectArgsBuffer = null;
		this._requestParamsBuffer = null;
		this._compactParamsBuffer = null;
		this._allocationParamsBuffer = null;
		this._dirtyParamsBuffer = null;
		this._drawParamsBuffer = null;
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
		this._fallbackDrawMvpBuffer = null;
		this._fallbackDrawMetaBuffer = null;
		this._fallbackDrawTransmittanceBuffer = null;
		this._fallbackDrawIndirectArgsBuffer = null;
		this._fallbackDrawCandidateWorldMatrices = null;
		this._layouts = [];
		this._previousCasterBounds.clear();
		this._computeShaderModules.clear();
		this._computePipelines.clear();
		this._computeBindGroups.clear();
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
		this._pageTableLength = Math.max(1, pageTableCursor);
		return layouts;
	}

	private _prepareResourceShape(
		request: WebGPUPagedShadowFrameRequest,
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
			this._pageTableLength,
			...layouts.map((layout) => layout.metadata.maxPagesPerFrame | 0)
		);
		const layoutCapacity = Math.max(1, layouts.length);
		const currentCasterCount = request.shadowCasterPackets.length;
		const removedCasterCount = countRemovedCasterSnapshots(
			this._previousCasterBounds,
			request.shadowCasterPackets
		);
		const casterCapacityRequirement = Math.max(
			1,
			currentCasterCount + removedCasterCount
		);
		const cascadeCapacity = Math.max(1, layoutCapacity * 4);
		const chunkedCasterCount = Math.ceil(currentCasterCount / 32) * 32;
		const drawCandidateCapacity = Math.max(32, chunkedCasterCount);
		const drawInstanceCapacity = Math.max(
			32,
			Math.max(this._casterCapacity, casterCapacityRequirement) *
				DRAW_INSTANCE_INITIAL_PAGES_PER_CASTER,
			this._drawInstanceCapacityPressure
		);
		let resourcesChanged = false;
		const physicalReset = this._ensurePhysicalCacheResources(
			pageSize,
			physicalPageCount,
			physicalGridSize,
			physicalAtlasSize
		);
		resourcesChanged =
			this._ensureVirtualPageResources(requestBufferCapacity, physicalReset) ||
			resourcesChanged;
		resourcesChanged =
			this._ensureFrameInputResources(
				layoutCapacity,
				casterCapacityRequirement,
				cascadeCapacity,
				drawCandidateCapacity
			) || resourcesChanged;
		resourcesChanged =
			this._ensureDrawInstanceResources(drawInstanceCapacity) || resourcesChanged;
		resourcesChanged = this._ensureFixedFrameResources() || resourcesChanged;
		resourcesChanged = physicalReset || resourcesChanged;
		if (resourcesChanged) {
			this._computeBindGroups.clear();
		}
	}

	private _ensurePhysicalCacheResources(
		pageSize: number,
		physicalPageCount: number,
		physicalGridSize: number,
		physicalAtlasSize: number
	): boolean {
		const needsReset =
			this._pageSize !== pageSize ||
			this._physicalPageCount !== physicalPageCount ||
			this._physicalGridSize !== physicalGridSize ||
			this._physicalAtlasSize !== physicalAtlasSize ||
			!this._pageMetadataBuffer ||
			!this._physicalDepthAtlas ||
			!this._residencyStateBuffer ||
			!this._freeListBuffer ||
			!this._dirtyPhysicalPagesBuffer;
		if (!needsReset) {
			return false;
		}
		this._pageSize = pageSize;
		this._physicalPageCount = physicalPageCount;
		this._physicalGridSize = physicalGridSize;
		this._physicalAtlasSize = physicalAtlasSize;
		this._residencyState = createNonResidentUint32Array(
			physicalPageCount * PAGE_RESIDENCY_STATE_UINTS
		);
		this._freeList = new Uint32Array(physicalPageCount);
		for (let index = 0; index < physicalPageCount; index++) {
			this._freeList[index] = index;
		}
		this._dirtyPhysicalPages = new Uint32Array(
			physicalPageCount * DIRTY_PHYSICAL_PAGE_RECORD_UINTS
		);
		this._destroyBuffer(this._pageMetadataBuffer);
		this._destroyTexture(this._physicalDepthAtlas);
		this._destroyBuffer(this._residencyStateBuffer);
		this._destroyBuffer(this._freeListBuffer);
		this._destroyBuffer(this._dirtyPhysicalPagesBuffer);
		this._pageMetadataBuffer = this._backend.createBuffer({
			size: Math.max(4, physicalPageCount * PAGE_METADATA_UINTS * 4),
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
		this._residencyStateBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowResidencyState",
			this._residencyState.byteLength
		);
		this._freeListBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowFreeList",
			this._freeList.byteLength
		);
		this._dirtyPhysicalPagesBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowDirtyPhysicalPages",
			this._dirtyPhysicalPages.byteLength
		);
		this._ensureCounterBuffer();
		this._initializePhysicalCacheBuffers();
		return true;
	}

	private _ensureVirtualPageResources(
		requestBufferCapacityRequirement: number,
		resetPageTable: boolean
	): boolean {
		let changed = false;
		const pageTableLengthChanged =
			this._pageTableBufferLength !== this._pageTableLength ||
			!this._pageTableBuffer ||
			this._pageTableBuffer.size < this._pageTableLength * 4 ||
			!this._pageRequestFlagsBuffer ||
			!this._feedbackFlagsBuffer ||
			!this._nextFeedbackFlagsBuffer;
		const resetPhysicalForPageTable = pageTableLengthChanged && !resetPageTable;
		if (pageTableLengthChanged) {
			this._pageTableBufferLength = this._pageTableLength;
			this._pageRequestFlags = new Uint32Array(
				Math.max(1, this._pageTableLength)
			);
			this._feedbackFlags = new Uint32Array(Math.max(1, this._pageTableLength));
			this._nextFeedbackFlags = new Uint32Array(
				Math.max(1, this._pageTableLength)
			);
			this._destroyBuffer(this._pageTableBuffer);
			this._destroyBuffer(this._pageRequestFlagsBuffer);
			this._destroyBuffer(this._feedbackFlagsBuffer);
			this._destroyBuffer(this._nextFeedbackFlagsBuffer);
			this._pageTableBuffer = this._backend.createBuffer({
				size: Math.max(4, this._pageTableLength * 4),
				usage: BufferUsage.Storage | BufferUsage.CopyDst | BufferUsage.CopySrc,
				label: "WebGPUPagedShadowPageTable",
			});
			this._pageRequestFlagsBuffer = this._createStorageBuffer(
				"WebGPUPagedShadowPageRequestFlags",
				this._pageRequestFlags.byteLength
			);
			this._feedbackFlagsBuffer = this._createStorageBuffer(
				"WebGPUPagedShadowFeedbackFlags",
				this._feedbackFlags.byteLength
			);
			this._nextFeedbackFlagsBuffer = this._createStorageBuffer(
				"WebGPUPagedShadowNextFeedbackFlags",
				this._nextFeedbackFlags.byteLength
			);
			changed = true;
			resetPageTable = true;
		}
		if (
			this._requestBufferCapacity < requestBufferCapacityRequirement ||
			!this._compactedRequestsBuffer
		) {
			this._requestBufferCapacity = growCapacity(
				this._requestBufferCapacity,
				requestBufferCapacityRequirement
			);
			this._compactedRequests = new Uint32Array(
				this._requestBufferCapacity * PAGE_REQUEST_RECORD_UINTS
			);
			this._destroyBuffer(this._compactedRequestsBuffer);
			this._compactedRequestsBuffer = this._createStorageBuffer(
				"WebGPUPagedShadowCompactedRequests",
				this._compactedRequests.byteLength
			);
			this._backend.writeBuffer(
				this._compactedRequestsBuffer,
				this._compactedRequests
			);
			changed = true;
		}
		if (resetPageTable) {
			this._initializePageTableBuffers();
			if (resetPhysicalForPageTable) {
				this._initializePhysicalCacheBuffers();
			}
		}
		return changed;
	}

	private _ensureFrameInputResources(
		layoutCapacityRequirement: number,
		casterCapacityRequirement: number,
		cascadeCapacityRequirement: number,
		drawCandidateCapacityRequirement: number
	): boolean {
		let changed = false;
		if (
			this._layoutCapacity < layoutCapacityRequirement ||
			!this._layoutBuffer
		) {
			this._layoutCapacity = growCapacity(
				this._layoutCapacity,
				layoutCapacityRequirement
			);
			this._layoutData = new Uint32Array(
				this._layoutCapacity * PAGE_LAYOUT_UINTS
			);
			this._destroyBuffer(this._layoutBuffer);
			this._layoutBuffer = this._createStorageBuffer(
				"WebGPUPagedShadowLayouts",
				this._layoutData.byteLength
			);
			changed = true;
		}
		if (
			this._casterCapacity < casterCapacityRequirement ||
			!this._casterBoundsBuffer
		) {
			this._casterCapacity = growCapacity(
				this._casterCapacity,
				casterCapacityRequirement
			);
			this._casterBoundsData = new Float32Array(this._casterCapacity * 4);
			this._destroyBuffer(this._casterBoundsBuffer);
			this._casterBoundsBuffer = this._createStorageBuffer(
				"WebGPUPagedShadowCasterBounds",
				this._casterBoundsData.byteLength
			);
			changed = true;
		}
		if (
			this._cascadeCapacity < cascadeCapacityRequirement ||
			!this._cascadeViewProjectionBuffer
		) {
			this._cascadeCapacity = growCapacity(
				this._cascadeCapacity,
				cascadeCapacityRequirement
			);
			this._cascadeViewProjectionData = new Float32Array(
				this._cascadeCapacity * 16
			);
			this._destroyBuffer(this._cascadeViewProjectionBuffer);
			this._cascadeViewProjectionBuffer = this._createStorageBuffer(
				"WebGPUPagedShadowCascadeViewProjections",
				this._cascadeViewProjectionData.byteLength
			);
			changed = true;
		}
		if (
			this._drawCandidateCapacity < drawCandidateCapacityRequirement ||
			!this._drawWorldMatrixBuffer ||
			!this._drawIndirectArgsBuffer
		) {
			this._drawCandidateCapacity = growCapacity(
				this._drawCandidateCapacity,
				drawCandidateCapacityRequirement
			);
			this._drawWorldMatrixData = new Float32Array(
				this._drawCandidateCapacity * 16
			);
			this._drawIndirectArgsData = new Uint32Array(
				this._drawCandidateCapacity * DRAW_INDIRECT_UINTS
			);
			this._destroyBuffer(this._drawWorldMatrixBuffer);
			this._destroyBuffer(this._drawIndirectArgsBuffer);
			this._drawWorldMatrixBuffer = this._createStorageBuffer(
				"WebGPUPagedShadowDrawWorldMatrices",
				this._drawWorldMatrixData.byteLength
			);
			this._drawIndirectArgsBuffer = this._backend.createBuffer({
				size: Math.max(4, this._drawIndirectArgsData.byteLength),
				usage: BufferUsage.Storage | BufferUsage.CopyDst | BufferUsage.Indirect,
				label: "WebGPUPagedShadowDrawIndirectArgs",
			});
			changed = true;
		}
		return changed;
	}

	private _ensureDrawInstanceResources(
		drawInstanceCapacityRequirement: number
	): boolean {
		if (
			this._drawInstanceCapacity >= drawInstanceCapacityRequirement &&
			this._drawMvpBuffer &&
			this._drawInstanceMetaBuffer &&
			this._drawTransmittanceBuffer
		) {
			return false;
		}
		this._drawInstanceCapacity = growCapacity(
			this._drawInstanceCapacity,
			drawInstanceCapacityRequirement
		);
		this._drawInstanceCapacityPressure = 0;
		this._destroyBuffer(this._drawMvpBuffer);
		this._destroyBuffer(this._drawInstanceMetaBuffer);
		this._destroyBuffer(this._drawTransmittanceBuffer);
		this._drawMvpBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowDrawMvp",
			this._drawInstanceCapacity * 16 * 4
		);
		this._drawInstanceMetaBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowDrawInstanceMeta",
			this._drawInstanceCapacity * SHADOW_INSTANCE_DATA_UINTS * 4
		);
		this._drawTransmittanceBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowDrawTransmittance",
			this._drawInstanceCapacity * 4 * 4
		);
		return true;
	}

	private _ensureFixedFrameResources(): boolean {
		let changed = false;
		changed = this._ensureCounterBuffer() || changed;
		if (!this._requestParamsBuffer) {
			this._requestParamsBuffer = this._createUniformBuffer(
				"WebGPUPagedShadowRequestParams",
				PAGE_REQUEST_PARAMS_UINTS * 4
			);
			changed = true;
		}
		if (!this._compactParamsBuffer) {
			this._compactParamsBuffer = this._createUniformBuffer(
				"WebGPUPagedShadowCompactParams",
				PAGE_REQUEST_PARAMS_UINTS * 4
			);
			changed = true;
		}
		if (!this._allocationParamsBuffer) {
			this._allocationParamsBuffer = this._createUniformBuffer(
				"WebGPUPagedShadowAllocationParams",
				PAGE_ALLOC_PARAMS_UINTS * 4
			);
			changed = true;
		}
		if (!this._dirtyParamsBuffer) {
			this._dirtyParamsBuffer = this._createUniformBuffer(
				"WebGPUPagedShadowDirtyParams",
				PAGE_DIRTY_PARAMS_UINTS * 4
			);
			changed = true;
		}
		if (!this._drawParamsBuffer) {
			this._drawParamsBuffer = this._createUniformBuffer(
				"WebGPUPagedShadowDrawParams",
				PAGE_DRAW_PARAMS_UINTS * 4
			);
			changed = true;
		}
		if (!this._feedbackParamsBuffer) {
			this._feedbackParamsBuffer = this._createUniformBuffer(
				"WebGPUPagedShadowFeedbackParams",
				PAGE_FEEDBACK_PARAMS_UINTS * 4
			);
			changed = true;
		}
		if (!this._feedbackCameraBuffer) {
			this._feedbackCameraData = new Float32Array(16);
			this._feedbackCameraBuffer = this._createStorageBuffer(
				"WebGPUPagedShadowFeedbackCamera",
				this._feedbackCameraData.byteLength
			);
			changed = true;
		}
		return changed;
	}

	private _ensureCounterBuffer(): boolean {
		if (this._countersBuffer) {
			return false;
		}
		this._counters = new Uint32Array(8);
		this._countersBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowCounters",
			this._counters.byteLength
		);
		this._backend.writeBuffer(this._countersBuffer, this._counters);
		return true;
	}

	private _initializePageTableBuffers(): void {
		if (this._pageTableBuffer) {
			const pageTable = new Uint32Array(this._pageTableLength);
			pageTable.fill(WEBGPU_PAGED_SHADOW_NON_RESIDENT);
			this._backend.writeBuffer(this._pageTableBuffer, pageTable);
		}
		for (const [buffer, data] of [
			[this._pageRequestFlagsBuffer, this._pageRequestFlags],
			[this._feedbackFlagsBuffer, this._feedbackFlags],
			[this._nextFeedbackFlagsBuffer, this._nextFeedbackFlags],
		] as Array<[IRenderBuffer | null, Uint32Array]>) {
			if (buffer) {
				data.fill(0);
				this._backend.writeBuffer(buffer, data as Uint32Array<ArrayBuffer>);
			}
		}
	}

	private _initializePhysicalCacheBuffers(): void {
		this._counters[0] = 0;
		this._counters[1] = 0;
		this._counters[2] = this._physicalPageCount;
		this._counters[3] = 0;
		this._counters[4] = 0;
		if (this._pageTableBuffer) {
			const pageTable = new Uint32Array(this._pageTableLength);
			pageTable.fill(WEBGPU_PAGED_SHADOW_NON_RESIDENT);
			this._backend.writeBuffer(this._pageTableBuffer, pageTable);
		}
		if (this._pageMetadataBuffer) {
			this._backend.writeBuffer(
				this._pageMetadataBuffer,
				new Uint32Array(this._physicalPageCount * PAGE_METADATA_UINTS)
			);
		}
		for (const [buffer, data] of [
			[this._residencyStateBuffer, this._residencyState],
			[this._freeListBuffer, this._freeList],
			[this._countersBuffer, this._counters],
			[this._dirtyPhysicalPagesBuffer, this._dirtyPhysicalPages],
		] as Array<[IRenderBuffer | null, Uint32Array]>) {
			if (buffer) {
				this._backend.writeBuffer(buffer, data as Uint32Array<ArrayBuffer>);
			}
		}
	}

	private _destroyBuffer(buffer: IRenderBuffer | null): void {
		buffer?.destroy();
	}

	private _destroyTexture(texture: IRenderTexture | null): void {
		texture?.destroy();
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
			this._layoutData[layoutOffset + 5] =
				layout.metadata.feedbackMode === "screen-feedback" ? 1 : 0;
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

		this._drawCandidateCount = request.shadowCasterPackets.length;
		this._drawWorldMatrixData.fill(0);
		this._drawIndirectArgsData.fill(0);
		this._casterBoundsData.fill(0);
		this._feedbackCameraData.fill(0);
		const inverseViewProjection = Matrix4.inverse(
			request.context.camera.viewProjectionMatrix
		);
		if (inverseViewProjection) {
			this._setMatrixInFloatArray(
				inverseViewProjection,
				this._feedbackCameraData,
				0
			);
		}
		const casterCount = this._writeCasterBoundsWithTombstones(
			request.shadowCasterPackets
		);
		for (let index = 0; index < request.shadowCasterPackets.length; index++) {
			const packet = request.shadowCasterPackets[index];
			this._setMatrixInFloatArray(
				packet.worldMatrix,
				this._drawWorldMatrixData,
				index * 16
			);
			const indirectOffset = index * DRAW_INDIRECT_UINTS;
			this._drawIndirectArgsData[indirectOffset] =
				Math.max(0, packet.geometry.indices.length | 0) >>> 0;
			this._drawIndirectArgsData[indirectOffset + 1] = 0;
			this._drawIndirectArgsData[indirectOffset + 2] = 0;
			this._drawIndirectArgsData[indirectOffset + 3] = 0;
			this._drawIndirectArgsData[indirectOffset + 4] =
				(index * this._physicalPageCount) >>> 0;
		}

		const conservativeWarmup =
			this._frameId <= 1 || layouts.some((layout) =>
				layout.metadata.feedbackMode !== "screen-feedback"
			) ? 1 : 0;
		this._backend.writeBuffer(
			this._requestParamsBuffer!,
			new Uint32Array([
				this._pageTableLength,
				casterCount,
				layouts.length,
				this._frameId,
				conservativeWarmup,
				this._feedbackFlags.length,
				0,
				0,
			])
		);
		this._backend.writeBuffer(
			this._compactParamsBuffer!,
			new Uint32Array([
				this._pageTableLength,
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
				this._pageTableLength,
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
			this._drawParamsBuffer!,
			new Uint32Array([
				this._drawCandidateCount,
				this._physicalPageCount,
				this._physicalPageCount,
				this._pageSize,
				this._physicalGridSize,
				this._drawInstanceCapacity,
				this._frameId,
				0,
			])
		);
		this._backend.writeBuffer(
			this._feedbackParamsBuffer!,
			new Uint32Array([
				this._pageTableLength,
				Math.max(1, request.context.attachments.width | 0),
				Math.max(1, request.context.attachments.height | 0),
				layouts.length,
				this._frameId,
				this._feedbackFlags.length,
				0,
				0,
			])
		);
		this._backend.writeBuffer(this._layoutBuffer!, this._layoutData);
		this._backend.writeBuffer(this._casterBoundsBuffer!, this._casterBoundsData);
		this._backend.writeBuffer(
			this._cascadeViewProjectionBuffer!,
			this._cascadeViewProjectionData
		);
		this._backend.writeBuffer(
			this._feedbackCameraBuffer!,
			this._feedbackCameraData
		);
		this._backend.writeBuffer(
			this._drawWorldMatrixBuffer!,
			this._drawWorldMatrixData
		);
		this._backend.writeBuffer(
			this._drawIndirectArgsBuffer!,
			this._drawIndirectArgsData
		);
	}

	private _writeCasterBoundsWithTombstones(
		packets: readonly DrawPacket[]
	): number {
		const currentIds = new Set<string>();
		let cursor = 0;
		for (const packet of packets) {
			currentIds.add(packet.id);
			const bounds = packet.worldBounds;
			this._writeCasterBounds(cursor++, {
				centerX: bounds.center.x,
				centerY: bounds.center.y,
				centerZ: bounds.center.z,
				radius: Math.max(0, bounds.radius),
			});
		}
		for (const [id, bounds] of this._previousCasterBounds) {
			if (currentIds.has(id) || cursor >= this._casterCapacity) {
				continue;
			}
			this._writeCasterBounds(cursor++, bounds);
		}
		this._previousCasterBounds.clear();
		for (const packet of packets) {
			const bounds = packet.worldBounds;
			this._previousCasterBounds.set(packet.id, {
				centerX: bounds.center.x,
				centerY: bounds.center.y,
				centerZ: bounds.center.z,
				radius: Math.max(0, bounds.radius),
			});
		}
		return cursor;
	}

	private _writeCasterBounds(index: number, bounds: FrameCasterSnapshot): void {
		const offset = index * 4;
		this._casterBoundsData[offset] = bounds.centerX;
		this._casterBoundsData[offset + 1] = bounds.centerY;
		this._casterBoundsData[offset + 2] = bounds.centerZ;
		this._casterBoundsData[offset + 3] = bounds.radius;
	}

	private _resetGpuRequestBuffers(): void {
		this._pageRequestFlags.fill(0);
		this._compactedRequests.fill(0);
		this._counters[0] = 0;
		this._counters[1] = 0;
		this._dirtyPhysicalPages.fill(0);
		const nativeEncoder = tryGetNativeWebGPUCommandEncoder(this._lastRequest?.encoder);
		if (nativeEncoder && this._pageRequestFlagsBuffer) {
			nativeEncoder.clearBuffer(
				getWebGPUBuffer(this._pageRequestFlagsBuffer),
				0,
				this._pageRequestFlags.byteLength
			);
			if (this._compactedRequestsBuffer) {
				nativeEncoder.clearBuffer(
					getWebGPUBuffer(this._compactedRequestsBuffer),
					0,
					this._compactedRequests.byteLength
				);
			}
			if (this._dirtyPhysicalPagesBuffer) {
				nativeEncoder.clearBuffer(
					getWebGPUBuffer(this._dirtyPhysicalPagesBuffer),
					0,
					this._dirtyPhysicalPages.byteLength
				);
			}
		} else {
			for (const [buffer, data] of [
				[this._pageRequestFlagsBuffer, this._pageRequestFlags],
				[this._compactedRequestsBuffer, this._compactedRequests],
				[this._dirtyPhysicalPagesBuffer, this._dirtyPhysicalPages],
			] as Array<[IRenderBuffer | null, Uint32Array]>) {
				if (buffer) {
					this._backend.writeBuffer(buffer, data as Uint32Array<ArrayBuffer>);
				}
			}
		}
		if (this._countersBuffer) {
			this._backend.writeBuffer(
				this._countersBuffer,
				new Uint32Array(this._counters.buffer, 0, 2)
			);
		}
	}

	private _resetGpuDrawCounters(): void {
		this._counters[3] = 0;
		this._counters[4] = 0;
		if (this._countersBuffer) {
			this._backend.writeBuffer(
				this._countersBuffer,
				new Uint32Array(this._counters.buffer, 3 * 4, 2),
				3 * 4
			);
		}
	}

	private _queueDrawCounterReadback(
		request: WebGPUPagedShadowFrameRequest
	): void {
		if (!this._countersBuffer) {
			return;
		}
		const nativeEncoder = tryGetNativeWebGPUCommandEncoder(request.encoder);
		const sourceBuffer = tryGetWebGPUBuffer(this._countersBuffer);
		const backend = this._backend as WebGPUBackend & {
			device?: {
				createBuffer?: (descriptor: {
					label: string;
					size: number;
					usage: number;
				}) => GPUBuffer;
			};
		};
		const bufferUsage = (globalThis as {
			GPUBufferUsage?: { COPY_DST?: number; MAP_READ?: number };
		}).GPUBufferUsage;
		if (
			!nativeEncoder ||
			!sourceBuffer ||
			typeof backend.device?.createBuffer !== "function" ||
			typeof bufferUsage?.COPY_DST !== "number" ||
			typeof bufferUsage?.MAP_READ !== "number"
		) {
			return;
		}
		const byteLength = DRAW_COUNTER_READBACK_UINTS * 4;
		const readback = backend.device.createBuffer({
			label: "WebGPUPagedShadowDrawCounterReadback",
			size: byteLength,
			usage: bufferUsage.COPY_DST | bufferUsage.MAP_READ,
		});
		nativeEncoder.copyBufferToBuffer(sourceBuffer, 0, readback, 0, byteLength);
		this._pendingDrawCounterReadbacks.push({
			frameId: this._frameId,
			buffer: readback,
			byteLength,
			queued: false,
			done: false,
		});
	}

	private _scheduleQueuedDrawCounterReadbacks(): void {
		for (const pending of this._pendingDrawCounterReadbacks) {
			if (pending.queued || pending.done) {
				continue;
			}
			pending.queued = true;
			void pending.buffer
				.mapAsync(WEBGPU_MAP_MODE_READ, 0, pending.byteLength)
				.then(() => {
					const mapped = pending.buffer.getMappedRange(0, pending.byteLength);
					this._applyDrawCounterReadback(new Uint32Array(mapped.slice(0)));
					pending.done = true;
				})
				.catch(() => {
					pending.done = true;
				})
				.finally(() => {
					try {
						pending.buffer.unmap();
					} catch {
						// The buffer may already be unmapped after a failed map.
					}
					pending.buffer.destroy();
				});
		}
	}

	private _applyDrawCounterReadback(data: Uint32Array): void {
		const allocatedInstances = data[3] ?? 0;
		const overflowInstances = data[4] ?? 0;
		if (overflowInstances <= 0) {
			return;
		}
		const requiredCapacity = Math.max(
			32,
			this._casterCapacity * DRAW_INSTANCE_INITIAL_PAGES_PER_CASTER,
			allocatedInstances,
			this._drawInstanceCapacity + overflowInstances
		);
		this._drawInstanceCapacityPressure = Math.max(
			this._drawInstanceCapacityPressure,
			requiredCapacity
		);
	}

	private _collectCompletedDrawCounterReadbacks(): void {
		this._pendingDrawCounterReadbacks =
			this._pendingDrawCounterReadbacks.filter((pending) => !pending.done);
	}

	private async _recordComputePass(
		encoder: ICommandEncoder | null,
		shaderPart: string,
		label: string,
		resources: ReadonlyArray<BindingResource | null>,
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
				entries: Array<{ binding: number; resource: BindingResource }>;
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
		const bindGroupKey = `${shaderPart}:${resources.map(getResourceSizeSignature).join(":")}`;
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
			32
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
		this._fallbackDrawMvpBuffer ??= this._createFallbackBuffer(
			"WebGPUPagedShadowFallbackDrawMvp",
			16 * 4
		);
		this._fallbackDrawMetaBuffer ??= this._createFallbackBuffer(
			"WebGPUPagedShadowFallbackDrawMeta",
			SHADOW_INSTANCE_DATA_UINTS * 4
		);
		this._fallbackDrawTransmittanceBuffer ??= this._createFallbackBuffer(
			"WebGPUPagedShadowFallbackDrawTransmittance",
			4 * 4
		);
		this._fallbackDrawIndirectArgsBuffer ??= this._createFallbackBuffer(
			"WebGPUPagedShadowFallbackDrawIndirectArgs",
			DRAW_INDIRECT_UINTS * 4,
			BufferUsage.Storage | BufferUsage.CopyDst | BufferUsage.Indirect
		);
		this._fallbackDrawCandidateWorldMatrices ??= this._createFallbackBuffer(
			"WebGPUPagedShadowFallbackDrawWorldMatrices",
			16 * 4
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
			drawMvpBuffer: this._fallbackDrawMvpBuffer,
			drawInstanceMetaBuffer: this._fallbackDrawMetaBuffer,
			drawTransmittanceBuffer: this._fallbackDrawTransmittanceBuffer,
			drawIndirectArgsBuffer: this._fallbackDrawIndirectArgsBuffer,
			drawCandidateWorldMatrices: this._fallbackDrawCandidateWorldMatrices,
			pageSize: DEFAULT_FALLBACK_PAGE_SIZE,
			physicalGridSize: 1,
			physicalAtlasSize: DEFAULT_FALLBACK_PAGE_SIZE,
			drawCandidateCount: 0,
			drawInstanceCapacity: 1,
			physicalPageCount: 1,
		};
	}

	private _createFallbackBuffer(
		label: string,
		size: number,
		usage: BufferUsage = BufferUsage.Storage | BufferUsage.CopyDst
	): IRenderBuffer {
		const buffer = this._backend.createBuffer({
			size: Math.max(4, size),
			usage,
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

function createNonResidentUint32Array(length: number): Uint32Array {
	const data = new Uint32Array(Math.max(1, length));
	data.fill(WEBGPU_PAGED_SHADOW_NON_RESIDENT);
	return data;
}

function growCapacity(current: number, required: number): number {
	return Math.max(1, required, Math.ceil(Math.max(1, current) * 1.5));
}

function clampInt(value: number, min: number, max: number): number {
	if (!Number.isFinite(value)) {
		return min;
	}
	return Math.max(min, Math.min(max, value | 0));
}

function countRemovedCasterSnapshots(
	previous: ReadonlyMap<string, FrameCasterSnapshot>,
	packets: readonly DrawPacket[]
): number {
	if (previous.size <= 0) {
		return 0;
	}
	const currentIds = new Set(packets.map((packet) => packet.id));
	let count = 0;
	for (const id of previous.keys()) {
		if (!currentIds.has(id)) {
			count++;
		}
	}
	return count;
}

function getResourceSizeSignature(resource: BindingResource | null): string {
	if (!resource || typeof resource !== "object") {
		return "0";
	}
	const sized = resource as { size?: unknown; width?: unknown; height?: unknown };
	return `${sized.size ?? 0}:${sized.width ?? 0}:${sized.height ?? 0}`;
}
