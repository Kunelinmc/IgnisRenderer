import { LightType, type ShadowCastingLight } from "../../lights";
import type { PreparedPagedShadowSettings } from "../../lights/shadows/types";
import type { PreparedShadowLight, ShadowFramePlan } from "../../lights/shadows/ShadowFramePlan";
import { Matrix4 } from "../../maths/Matrix4";
import type { IVector3 } from "../../maths/types";
import type {
	DrawPacket,
	FrameContext,
} from "../../pipeline/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import { ShaderSource } from "../../shaders/ShaderSource";
import { TextureFormat } from "../../core/TextureFormat";
import {
	BufferUsage,
	TextureUsage,
	type BindingResource,
	type IBindingGroup,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type IShaderModule,
} from "../types";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type { WebGPUResourceManager } from "./WebGPUResourceManager";
import type { WebGPUShadowCasterRenderer } from "./WebGPUShadowCasterRenderer";
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
const PAGE_ADDRESS_UINTS = 8;
const PAGE_REQUEST_PARAMS_UINTS = 8;
const PAGE_ALLOC_PARAMS_UINTS = 8;
const PAGE_DIRTY_PARAMS_UINTS = 4;
const PAGE_DRAW_PARAMS_UINTS = 8;
const PAGE_FEEDBACK_PARAMS_UINTS = 8;
const DRAW_INDIRECT_UINTS = 5;
const CLEAR_DRAW_INDIRECT_UINTS = 4;
const SHADOW_INSTANCE_DATA_UINTS = 12;
const DEFAULT_FALLBACK_PAGE_SIZE = 1;
const DRAW_INSTANCE_INITIAL_PAGES_PER_CASTER = 4;
const DRAW_COUNTER_READBACK_UINTS = 8;
const DRAW_COUNTER_READBACK_RING_SIZE = 4;
const DIRTY_GRID_CELL_COUNT = 64;
const DIRTY_GRID_OFFSET_COUNT = DIRTY_GRID_CELL_COUNT + 1;
const DIRTY_PAGE_UV_RANGE_FLOATS = 4;
const WEBGPU_MAP_MODE_READ =
	(globalThis as { GPUMapMode?: { READ?: number } }).GPUMapMode?.READ ??
	0x0001;

export interface WebGPUPagedShadowFrameRequest {
	context: FrameContext;
	encoder: ICommandEncoder | null;
	shadowPlan: ShadowFramePlan;
	shadowCasterPackets: readonly DrawPacket[];
	shadowTransmitterPackets: readonly DrawPacket[];
	feedbackDepthTexture?: IRenderTexture | null;
	feedbackMotionDepthTexture?: IRenderTexture | null;
}

export interface WebGPUPagedShadowSamplingResources {
	pageTableTexture: IRenderTexture;
	physicalDepthAtlas: IRenderTexture;
}

export interface WebGPUPagedShadowIndirectRenderResources {
	physicalDepthAtlas: IRenderTexture;
	dirtyPhysicalPages: IRenderBuffer;
	drawMvpBuffer: IRenderBuffer;
	drawInstanceMetaBuffer: IRenderBuffer;
	drawTransmittanceBuffer: IRenderBuffer;
	drawIndirectArgsBuffer: IRenderBuffer;
	clearDrawIndirectArgsBuffer: IRenderBuffer;
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

export interface WebGPUPagedShadowLayout {
	prepared: PreparedShadowLight;
	metadata: Readonly<PreparedPagedShadowSettings>;
	pageTableBase: number;
	pageTableCascadeStride: number;
	cascadeCount: number;
}

interface FrameCasterSnapshot {
	centerX: number;
	centerY: number;
	centerZ: number;
	radius: number;
	worldMatrix: Matrix4;
}

type DrawCounterReadbackSlotState = "idle" | "copied" | "mapping";

interface DrawCounterReadbackSlot {
	frameId: number;
	buffer: GPUBuffer;
	byteLength: number;
	state: DrawCounterReadbackSlotState;
}

/**
 * WebGPU paged shadow technique.
 *
 * The WebGPU path keeps residency, dirty-page compaction, and page-table
 * updates GPU authoritative. CPU work is limited to frame-local layout,
 * cascade, caster bounds, world matrices, and indirect-argument seed uploads.
 */
export class WebGPUPagedShadowTechnique {
	private _backend: WebGPUDeviceResourceHost;
	private _resourceManager: WebGPUResourceManager;
	private _casterRenderer: WebGPUShadowCasterRenderer;
	private _frameId = 0;
	private _lastRequest: WebGPUPagedShadowFrameRequest | null = null;
	private _preparedContext: FrameContext | null = null;
	private _pageTableBuffer: IRenderBuffer | null = null;
	private _pageTableTexture: IRenderTexture | null = null;
	private _pageMetadataBuffer: IRenderBuffer | null = null;
	private _physicalDepthAtlas: IRenderTexture | null = null;
	private _fallbackPageTableTexture: IRenderTexture | null = null;
	private _fallbackDepthAtlas: IRenderTexture | null = null;
	private _pageRequestFlagsBuffer: IRenderBuffer | null = null;
	private _pageAddressBuffer: IRenderBuffer | null = null;
	private _compactedRequestsBuffer: IRenderBuffer | null = null;
	private _residencyStateBuffer: IRenderBuffer | null = null;
	private _freeListBuffer: IRenderBuffer | null = null;
	private _countersBuffer: IRenderBuffer | null = null;
	private _dirtyPhysicalPagesBuffer: IRenderBuffer | null = null;
	private _dirtyGridCountsBuffer: IRenderBuffer | null = null;
	private _dirtyGridOffsetsBuffer: IRenderBuffer | null = null;
	private _dirtyGridIndicesBuffer: IRenderBuffer | null = null;
	private _dirtyPageUvRangesBuffer: IRenderBuffer | null = null;
	private _feedbackFlagsBuffer: IRenderBuffer | null = null;
	private _nextFeedbackFlagsBuffer: IRenderBuffer | null = null;
	private _layoutBuffer: IRenderBuffer | null = null;
	private _casterBoundsBuffer: IRenderBuffer | null = null;
	private _casterStatesBuffer: IRenderBuffer | null = null;
	private _cascadeViewProjectionBuffer: IRenderBuffer | null = null;
	private _drawWorldMatrixBuffer: IRenderBuffer | null = null;
	private _drawMvpBuffer: IRenderBuffer | null = null;
	private _drawInstanceMetaBuffer: IRenderBuffer | null = null;
	private _drawTransmittanceBuffer: IRenderBuffer | null = null;
	private _drawIndirectArgsBuffer: IRenderBuffer | null = null;
	private _clearDrawIndirectArgsBuffer: IRenderBuffer | null = null;
	private _requestParamsBuffer: IRenderBuffer | null = null;
	private _pageTableCopyParamsBuffer: IRenderBuffer | null = null;
	private _compactParamsBuffer: IRenderBuffer | null = null;
	private _allocationParamsBuffer: IRenderBuffer | null = null;
	private _dirtyParamsBuffer: IRenderBuffer | null = null;
	private _drawParamsBuffer: IRenderBuffer | null = null;
	private _feedbackParamsBuffer: IRenderBuffer | null = null;
	private _feedbackCameraBuffer: IRenderBuffer | null = null;
	private _pageTableLength = 1;
	private _pageRequestFlags = new Uint32Array(1);
	private _pageAddressData = new Uint32Array(PAGE_ADDRESS_UINTS);
	private _pageAddressSignature = "";
	private _compactedRequests = new Uint32Array(PAGE_REQUEST_RECORD_UINTS);
	private _residencyState = createNonResidentUint32Array(PAGE_RESIDENCY_STATE_UINTS);
	private _freeList = new Uint32Array([0]);
	private _counters = new Uint32Array(8);
	private _dirtyPhysicalPages = new Uint32Array(DIRTY_PHYSICAL_PAGE_RECORD_UINTS);
	private _feedbackFlags = new Uint32Array(1);
	private _nextFeedbackFlags = new Uint32Array(1);
	private _layoutData = new Uint32Array(PAGE_LAYOUT_UINTS);
	private _casterBoundsData = new Float32Array(4);
	private _casterStatesData = new Uint32Array(4);
	private _cascadeViewProjectionData = new Float32Array(16);
	private _drawWorldMatrixData = new Float32Array(16);
	private _drawIndirectArgsData = new Uint32Array(DRAW_INDIRECT_UINTS);
	private _clearDrawIndirectArgsData = new Uint32Array([6, 0, 0, 0]);
	private _feedbackCameraData = new Float32Array(16);
	private _requestParamsData = new Uint32Array(PAGE_REQUEST_PARAMS_UINTS);
	private _compactParamsData = new Uint32Array(PAGE_REQUEST_PARAMS_UINTS);
	private _allocationParamsData = new Uint32Array(PAGE_ALLOC_PARAMS_UINTS);
	private _dirtyParamsData = new Uint32Array(PAGE_DIRTY_PARAMS_UINTS);
	private _drawParamsData = new Uint32Array(PAGE_DRAW_PARAMS_UINTS);
	private _feedbackParamsData = new Uint32Array(PAGE_FEEDBACK_PARAMS_UINTS);
	private _pageTableCopyParamsData = new Uint32Array(4);
	private _layouts: WebGPUPagedShadowLayout[] = [];
	private _requestBufferCapacity = 1;
	private _pageTableBufferLength = 1;
	private _layoutCapacity = 1;
	private _casterCapacity = 1;
	private _cascadeCapacity = 1;
	private _drawCandidateCapacity = 1;
	private _drawCandidateCount = 0;
	private _casterWorkCount = 0;
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
	private _previousCascadeViewProjectionData: Float32Array | null = null;
	private _projectionsChanged = false;
	private _drawCounterReadbackSlots: DrawCounterReadbackSlot[] = [];
	private _drawCounterReadbackCursor = 0;

	constructor(
		backend: WebGPUDeviceResourceHost,
		resourceManager: WebGPUResourceManager,
		casterRenderer: WebGPUShadowCasterRenderer,
	) {
		this._backend = backend;
		this._resourceManager = resourceManager;
		this._casterRenderer = casterRenderer;
	}

	/**
	 * @internal WebGPU frame graph preparation hook.
	 */
	public prepareFrame(request: WebGPUPagedShadowFrameRequest): void {
		this._lastRequest = request;
		this._scheduleQueuedDrawCounterReadbacks();
		if (this._preparedContext === request.context) {
			return;
		}
		this._preparedContext = request.context;
		this._frameId++;
		const layouts = this._resolvePagedRenderSetLayouts(request);
		this._layouts = layouts;
		this._prepareResourceShape(request, layouts);
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
				this._casterStatesBuffer,
			],
			Math.max(1, Math.ceil(Math.max(this._pageTableLength, this._casterWorkCount) / 64)),
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
				this._pageAddressBuffer,
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
	 * @internal WebGPU page table copy pass hook.
	 */
	public async recordPageTableCopyPass(
		request: WebGPUPagedShadowFrameRequest
	): Promise<void> {
		this._lastRequest = request;
		if (!this._pageTableCopyParamsBuffer || !this._pageTableBuffer || !this._pageTableTexture) {
			return;
		}

		let maxPageGridSize = 1;
		for (const layout of this._layouts) {
			maxPageGridSize = Math.max(maxPageGridSize, layout.metadata.pageGridSize | 0);
		}

		this._pageTableCopyParamsData[0] = this._pageTableLength;
		this._pageTableCopyParamsData[1] = maxPageGridSize;
		this._pageTableCopyParamsData[2] = 0;
		this._pageTableCopyParamsData[3] = 0;
		this._backend.writeBuffer(
			this._pageTableCopyParamsBuffer,
			this._pageTableCopyParamsData
		);

		await this._recordComputePass(
			request.encoder,
			"pagedShadowPageTableCopy",
			"WebGPUPagedShadowPageTableCopy",
			[
				this._pageTableCopyParamsBuffer,
				this._pageTableBuffer,
				this._pageTableTexture,
			],
			Math.max(1, Math.ceil(this._pageTableLength / 64)),
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
			"pagedShadowDirtyGridBuild",
			"WebGPUPagedShadowDirtyGridBuild",
			[
				this._drawParamsBuffer,
				this._dirtyPhysicalPagesBuffer,
				this._countersBuffer,
				this._dirtyGridCountsBuffer,
				this._dirtyGridOffsetsBuffer,
				this._dirtyGridIndicesBuffer,
				this._dirtyPageUvRangesBuffer,
				this._clearDrawIndirectArgsBuffer,
			],
			1,
			1,
			1
		);
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
				this._dirtyGridCountsBuffer,
				this._dirtyGridOffsetsBuffer,
				this._dirtyGridIndicesBuffer,
				this._dirtyPageUvRangesBuffer,
			],
			Math.max(1, Math.ceil(this._drawCandidateCount / 64)),
			1,
			1
		);
		const resources = this.getIndirectRenderResources();
		if (!resources) {
			return;
		}
		await this._casterRenderer.renderPagedDepthIndirect(
			request.context,
			resources,
			request.encoder,
			request.shadowCasterPackets
		);
		this._queueDrawCounterReadback(request);
	}

	/**
	 * @internal WebGPU resource query hook for scene shadow samplers.
	 */
	public getSamplingResources(): WebGPUPagedShadowSamplingResources {
		if (!this._pageTableTexture || !this._physicalDepthAtlas) {
			return this._getFallbackSamplingResources();
		}
		return {
			pageTableTexture: this._pageTableTexture,
			physicalDepthAtlas: this._physicalDepthAtlas,
		};
	}

	/**
	 * @internal WebGPU resource query hook for GPU-driven depth rendering.
	 */
	public getIndirectRenderResources(): WebGPUPagedShadowIndirectRenderResources | null {
		if (
			!this._physicalDepthAtlas ||
			!this._dirtyPhysicalPagesBuffer ||
			!this._drawMvpBuffer ||
			!this._drawInstanceMetaBuffer ||
			!this._drawTransmittanceBuffer ||
			!this._drawIndirectArgsBuffer ||
			!this._clearDrawIndirectArgsBuffer
		) {
			return null;
		}
		return {
			physicalDepthAtlas: this._physicalDepthAtlas,
			dirtyPhysicalPages: this._dirtyPhysicalPagesBuffer,
			drawMvpBuffer: this._drawMvpBuffer,
			drawInstanceMetaBuffer: this._drawInstanceMetaBuffer,
			drawTransmittanceBuffer: this._drawTransmittanceBuffer,
			drawIndirectArgsBuffer: this._drawIndirectArgsBuffer,
			clearDrawIndirectArgsBuffer: this._clearDrawIndirectArgsBuffer,
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
		this._destroyDrawCounterReadbackSlots();
		for (const resource of [
			this._pageTableBuffer,
			this._pageTableTexture,
			this._pageMetadataBuffer,
			this._physicalDepthAtlas,
			this._pageRequestFlagsBuffer,
			this._pageAddressBuffer,
			this._compactedRequestsBuffer,
			this._residencyStateBuffer,
			this._freeListBuffer,
			this._countersBuffer,
			this._dirtyPhysicalPagesBuffer,
			this._dirtyGridCountsBuffer,
			this._dirtyGridOffsetsBuffer,
			this._dirtyGridIndicesBuffer,
			this._dirtyPageUvRangesBuffer,
			this._feedbackFlagsBuffer,
			this._nextFeedbackFlagsBuffer,
			this._layoutBuffer,
			this._casterBoundsBuffer,
			this._casterStatesBuffer,
			this._cascadeViewProjectionBuffer,
			this._drawWorldMatrixBuffer,
			this._drawMvpBuffer,
			this._drawInstanceMetaBuffer,
			this._drawTransmittanceBuffer,
			this._drawIndirectArgsBuffer,
			this._clearDrawIndirectArgsBuffer,
			this._requestParamsBuffer,
			this._pageTableCopyParamsBuffer,
			this._compactParamsBuffer,
			this._allocationParamsBuffer,
			this._dirtyParamsBuffer,
			this._drawParamsBuffer,
			this._feedbackParamsBuffer,
			this._fallbackPageTableTexture,
			this._fallbackDepthAtlas,
		]) {
			resource?.destroy();
		}
		this._pageTableBuffer = null;
		this._pageTableTexture = null;
		this._pageMetadataBuffer = null;
		this._physicalDepthAtlas = null;
		this._pageRequestFlagsBuffer = null;
		this._pageAddressBuffer = null;
		this._pageAddressSignature = "";
		this._compactedRequestsBuffer = null;
		this._residencyStateBuffer = null;
		this._freeListBuffer = null;
		this._countersBuffer = null;
		this._dirtyPhysicalPagesBuffer = null;
		this._dirtyGridCountsBuffer = null;
		this._dirtyGridOffsetsBuffer = null;
		this._dirtyGridIndicesBuffer = null;
		this._dirtyPageUvRangesBuffer = null;
		this._feedbackFlagsBuffer = null;
		this._nextFeedbackFlagsBuffer = null;
		this._layoutBuffer = null;
		this._casterBoundsBuffer = null;
		this._casterStatesBuffer = null;
		this._cascadeViewProjectionBuffer = null;
		this._drawWorldMatrixBuffer = null;
		this._drawMvpBuffer = null;
		this._drawInstanceMetaBuffer = null;
		this._drawTransmittanceBuffer = null;
		this._drawIndirectArgsBuffer = null;
		this._clearDrawIndirectArgsBuffer = null;
		this._requestParamsBuffer = null;
		this._pageTableCopyParamsBuffer = null;
		this._compactParamsBuffer = null;
		this._allocationParamsBuffer = null;
		this._dirtyParamsBuffer = null;
		this._drawParamsBuffer = null;
		this._feedbackParamsBuffer = null;
		this._fallbackPageTableTexture = null;
		this._fallbackDepthAtlas = null;
		this._layouts = [];
		this._previousCasterBounds.clear();
		this._previousCascadeViewProjectionData = null;
		for (const bindGroup of this._computeBindGroups.values()) {
			const destroyFn = (bindGroup as { destroy?: () => void } | null)?.destroy;
			destroyFn?.();
		}
		this._computeBindGroups.clear();
		this._computeShaderModules.clear();
		this._computePipelines.clear();
	}

	private _resolvePagedRenderSetLayouts(
		request: WebGPUPagedShadowFrameRequest
	): WebGPUPagedShadowLayout[] {
		const layouts: WebGPUPagedShadowLayout[] = [];
		let pageTableCursor = 0;
		for (const prepared of request.shadowPlan?.lights ?? []) {
			const light = prepared.light;
			if (
				light.type !== LightType.Directional ||
				prepared.storage !== "paged" ||
				!prepared.pagedSettings
			) {
				continue;
			}
			const metadata = prepared.pagedSettings;
			const pageGridSize = Math.max(1, metadata.pageGridSize | 0);
			const cascadeCount = Math.max(1, Math.min(prepared.slices.length, 4));
			const pageTableCascadeStride = pageGridSize * pageGridSize;
			layouts.push({
				prepared,
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
		layouts: readonly WebGPUPagedShadowLayout[]
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
		let maxPageGridSize = 1;
		for (const layout of layouts) {
			maxPageGridSize = Math.max(maxPageGridSize, layout.metadata.pageGridSize | 0);
		}
		resourcesChanged =
			this._ensureVirtualPageResources(requestBufferCapacity, physicalReset, maxPageGridSize) ||
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
			for (const bindGroup of this._computeBindGroups.values()) {
				const destroyFn = (bindGroup as { destroy?: () => void } | null)?.destroy;
				destroyFn?.();
			}
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
			!this._dirtyPhysicalPagesBuffer ||
			!this._dirtyGridCountsBuffer ||
			!this._dirtyGridOffsetsBuffer ||
			!this._dirtyGridIndicesBuffer ||
			!this._dirtyPageUvRangesBuffer;
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
		this._destroyBuffer(this._dirtyGridCountsBuffer);
		this._destroyBuffer(this._dirtyGridOffsetsBuffer);
		this._destroyBuffer(this._dirtyGridIndicesBuffer);
		this._destroyBuffer(this._dirtyPageUvRangesBuffer);
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
		this._dirtyGridCountsBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowDirtyGridCounts",
			DIRTY_GRID_CELL_COUNT * 4
		);
		this._dirtyGridOffsetsBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowDirtyGridOffsets",
			DIRTY_GRID_OFFSET_COUNT * 4
		);
		this._dirtyGridIndicesBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowDirtyGridIndices",
			Math.max(1, physicalPageCount) * 4
		);
		this._dirtyPageUvRangesBuffer = this._createStorageBuffer(
			"WebGPUPagedShadowDirtyPageUvRanges",
			Math.max(1, physicalPageCount) * DIRTY_PAGE_UV_RANGE_FLOATS * 4
		);
		this._ensureCounterBuffer();
		this._initializePhysicalCacheBuffers();
		return true;
	}

	private _ensureVirtualPageResources(
		requestBufferCapacityRequirement: number,
		resetPageTable: boolean,
		maxPageGridSize: number
	): boolean {
		let changed = false;
		const pageTableHeight = Math.max(1, Math.ceil(this._pageTableLength / maxPageGridSize));
		const pageTableLengthChanged =
			this._pageTableBufferLength !== this._pageTableLength ||
			!this._pageTableBuffer ||
			this._pageTableBuffer.size < this._pageTableLength * 4 ||
			!this._pageRequestFlagsBuffer ||
			!this._pageAddressBuffer ||
			!this._feedbackFlagsBuffer ||
			!this._nextFeedbackFlagsBuffer ||
			!this._pageTableTexture ||
			this._pageTableTexture.width !== maxPageGridSize ||
			this._pageTableTexture.height !== pageTableHeight ||
			!this._pageTableCopyParamsBuffer;
		const resetPhysicalForPageTable = pageTableLengthChanged && !resetPageTable;
		if (pageTableLengthChanged) {
			this._pageTableBufferLength = this._pageTableLength;
			this._pageRequestFlags = new Uint32Array(
				Math.max(1, this._pageTableLength)
			);
			this._pageAddressData = new Uint32Array(
				Math.max(1, this._pageTableLength) * PAGE_ADDRESS_UINTS
			);
			this._pageAddressSignature = "";
			this._feedbackFlags = new Uint32Array(Math.max(1, this._pageTableLength));
			this._nextFeedbackFlags = new Uint32Array(
				Math.max(1, this._pageTableLength)
			);
			this._destroyBuffer(this._pageTableBuffer);
			this._destroyBuffer(this._pageRequestFlagsBuffer);
			this._destroyBuffer(this._pageAddressBuffer);
			this._destroyBuffer(this._feedbackFlagsBuffer);
			this._destroyBuffer(this._nextFeedbackFlagsBuffer);
			this._destroyTexture(this._pageTableTexture);
			this._destroyBuffer(this._pageTableCopyParamsBuffer);
			this._pageTableBuffer = this._backend.createBuffer({
				size: Math.max(4, this._pageTableLength * 4),
				usage: BufferUsage.Storage | BufferUsage.CopyDst | BufferUsage.CopySrc,
				label: "WebGPUPagedShadowPageTable",
			});
			this._pageRequestFlagsBuffer = this._createStorageBuffer(
				"WebGPUPagedShadowPageRequestFlags",
				this._pageRequestFlags.byteLength
			);
			this._pageAddressBuffer = this._createStorageBuffer(
				"WebGPUPagedShadowPageAddresses",
				this._pageAddressData.byteLength
			);
			this._feedbackFlagsBuffer = this._createStorageBuffer(
				"WebGPUPagedShadowFeedbackFlags",
				this._feedbackFlags.byteLength
			);
			this._nextFeedbackFlagsBuffer = this._createStorageBuffer(
				"WebGPUPagedShadowNextFeedbackFlags",
				this._nextFeedbackFlags.byteLength
			);
			this._pageTableTexture = this._backend.createTexture({
				width: maxPageGridSize,
				height: pageTableHeight,
				format: TextureFormat.R32Uint,
				usage: TextureUsage.TextureBinding | TextureUsage.StorageBinding,
				label: "WebGPUPagedShadowPageTableTexture",
			});
			this._pageTableCopyParamsBuffer = this._createUniformBuffer(
				"WebGPUPagedShadowPageTableCopyParams",
				16
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
			!this._casterBoundsBuffer ||
			!this._casterStatesBuffer
		) {
			this._casterCapacity = growCapacity(
				this._casterCapacity,
				casterCapacityRequirement
			);
			this._casterBoundsData = new Float32Array(this._casterCapacity * 4);
			this._casterStatesData = new Uint32Array(this._casterCapacity * 4);
			this._destroyBuffer(this._casterBoundsBuffer);
			this._destroyBuffer(this._casterStatesBuffer);
			this._casterBoundsBuffer = this._createStorageBuffer(
				"WebGPUPagedShadowCasterBounds",
				this._casterBoundsData.byteLength
			);
			this._casterStatesBuffer = this._createStorageBuffer(
				"WebGPUPagedShadowCasterStates",
				this._casterStatesData.byteLength
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
		if (!this._clearDrawIndirectArgsBuffer) {
			this._clearDrawIndirectArgsData = new Uint32Array([6, 0, 0, 0]);
			this._clearDrawIndirectArgsBuffer = this._backend.createBuffer({
				size: CLEAR_DRAW_INDIRECT_UINTS * 4,
				usage: BufferUsage.Storage | BufferUsage.CopyDst | BufferUsage.Indirect,
				label: "WebGPUPagedShadowClearDrawIndirectArgs",
			});
			this._backend.writeBuffer(
				this._clearDrawIndirectArgsBuffer,
				this._clearDrawIndirectArgsData
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
		this._backend.writeBuffer(
			this._dirtyGridCountsBuffer!,
			new Uint32Array(DIRTY_GRID_CELL_COUNT)
		);
		this._backend.writeBuffer(
			this._dirtyGridOffsetsBuffer!,
			new Uint32Array(DIRTY_GRID_OFFSET_COUNT)
		);
		this._backend.writeBuffer(
			this._dirtyGridIndicesBuffer!,
			new Uint32Array(Math.max(1, this._physicalPageCount))
		);
		this._backend.writeBuffer(
			this._dirtyPageUvRangesBuffer!,
			new Float32Array(
				Math.max(1, this._physicalPageCount) * DIRTY_PAGE_UV_RANGE_FLOATS
			)
		);
		if (this._clearDrawIndirectArgsBuffer) {
			this._clearDrawIndirectArgsData[0] = 6;
			this._clearDrawIndirectArgsData[1] = 0;
			this._clearDrawIndirectArgsData[2] = 0;
			this._clearDrawIndirectArgsData[3] = 0;
			this._backend.writeBuffer(
				this._clearDrawIndirectArgsBuffer,
				this._clearDrawIndirectArgsData
			);
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

	private _writePageAddressData(
		layouts: readonly WebGPUPagedShadowLayout[]
	): boolean {
		const signature = [
			this._pageTableLength,
			...layouts.map((layout) =>
				[
					layout.pageTableBase,
					layout.pageTableCascadeStride,
					layout.metadata.pageGridSize | 0,
					layout.cascadeCount,
				].join(":")
			),
		].join("|");
		if (signature === this._pageAddressSignature) {
			return false;
		}
		this._pageAddressSignature = signature;
		activeUint32Span(
			this._pageAddressData,
			Math.max(PAGE_ADDRESS_UINTS, this._pageTableLength * PAGE_ADDRESS_UINTS)
		).fill(0);
		for (let layoutIndex = 0; layoutIndex < layouts.length; layoutIndex++) {
			const layout = layouts[layoutIndex];
			const gridSize = Math.max(1, layout.metadata.pageGridSize | 0);
			const priorityBase = Math.max(0, 4 - layoutIndex);
			for (
				let cascadeIndex = 0;
				cascadeIndex < layout.cascadeCount;
				cascadeIndex++
			) {
				const matrixIndex = layoutIndex * 4 + cascadeIndex;
				const cascadeBase =
					layout.pageTableBase + cascadeIndex * layout.pageTableCascadeStride;
				const priority =
					priorityBase +
					(layout.cascadeCount - Math.min(cascadeIndex, layout.cascadeCount - 1));
				for (let pageY = 0; pageY < gridSize; pageY++) {
					for (let pageX = 0; pageX < gridSize; pageX++) {
						const tableIndex = cascadeBase + pageY * gridSize + pageX;
						if (tableIndex >= this._pageTableLength) {
							continue;
						}
						const addressOffset = tableIndex * PAGE_ADDRESS_UINTS;
						this._pageAddressData[addressOffset] = matrixIndex >>> 0;
						this._pageAddressData[addressOffset + 1] = pageX >>> 0;
						this._pageAddressData[addressOffset + 2] = pageY >>> 0;
						this._pageAddressData[addressOffset + 3] = gridSize >>> 0;
						this._pageAddressData[addressOffset + 4] = priority >>> 0;
						this._pageAddressData[addressOffset + 5] = 1;
					}
				}
			}
		}
		return true;
	}

	private _updateGpuFrameInputs(
		request: WebGPUPagedShadowFrameRequest,
		layouts: readonly WebGPUPagedShadowLayout[]
	): void {
		const layoutWriteCount = Math.max(
			PAGE_LAYOUT_UINTS,
			layouts.length * PAGE_LAYOUT_UINTS
		);
		const cascadeWriteCount = Math.max(16, layouts.length * 4 * 16);
		activeUint32Span(this._layoutData, layoutWriteCount).fill(0);
		activeFloat32Span(this._cascadeViewProjectionData, cascadeWriteCount).fill(0);
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
				const matrix = layout.prepared.slices[cascadeIndex]?.viewProjection;
				if (!matrix) continue;
				this._setMatrixInFloatArray(
					matrix,
					this._cascadeViewProjectionData,
					(layoutIndex * 4 + cascadeIndex) * 16
				);
			}
		}
		const pageAddressDataChanged = this._writePageAddressData(layouts);

		let projectionsChanged = false;
		if (this._previousCascadeViewProjectionData) {
			if (this._previousCascadeViewProjectionData.length !== cascadeWriteCount) {
				projectionsChanged = true;
			} else {
				for (let i = 0; i < cascadeWriteCount; i++) {
					if (Math.abs(this._cascadeViewProjectionData[i] - this._previousCascadeViewProjectionData[i]) > 1e-5) {
						projectionsChanged = true;
						break;
					}
				}
			}
		} else {
			projectionsChanged = true;
		}

		if (projectionsChanged) {
			if (!this._previousCascadeViewProjectionData || this._previousCascadeViewProjectionData.length !== cascadeWriteCount) {
				this._previousCascadeViewProjectionData = new Float32Array(cascadeWriteCount);
			}
			this._previousCascadeViewProjectionData.set(
				this._cascadeViewProjectionData.subarray(0, cascadeWriteCount)
			);
		}
		this._projectionsChanged = projectionsChanged;

		this._drawCandidateCount = request.shadowCasterPackets.length;
		this._feedbackCameraData.fill(0);
		const inverseViewProjection = Matrix4.inverse(
			request.context.viewCamera.viewProjectionMatrix
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
		this._casterWorkCount = casterCount;
		const drawWorldWriteCount = Math.max(16, this._drawCandidateCount * 16);
		const drawIndirectWriteCount = Math.max(
			DRAW_INDIRECT_UINTS,
			this._drawCandidateCount * DRAW_INDIRECT_UINTS
		);
		activeFloat32Span(this._drawWorldMatrixData, drawWorldWriteCount).fill(0);
		activeUint32Span(this._drawIndirectArgsData, drawIndirectWriteCount).fill(0);
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
		const maxPagesPerFrame = resolveMaxPagesPerFrame(request.shadowPlan);
		this._requestParamsData[0] = this._pageTableLength;
		this._requestParamsData[1] = casterCount;
		this._requestParamsData[2] = layouts.length;
		this._requestParamsData[3] = this._frameId;
		this._requestParamsData[4] = conservativeWarmup;
		this._requestParamsData[5] = this._feedbackFlags.length;
		this._requestParamsData[6] = 0;
		this._requestParamsData[7] = 0;
		this._compactParamsData[0] = this._pageTableLength;
		this._compactParamsData[1] = this._requestBufferCapacity;
		this._compactParamsData[2] = layouts.length;
		this._compactParamsData[3] = 0;
		this._allocationParamsData[0] = this._frameId;
		this._allocationParamsData[1] = this._requestBufferCapacity;
		this._allocationParamsData[2] = this._physicalPageCount;
		this._allocationParamsData[3] = maxPagesPerFrame;
		this._allocationParamsData[4] = resolveMaxCacheFrames(request.shadowPlan);
		this._allocationParamsData[5] = this._pageTableLength;
		this._allocationParamsData[6] = this._projectionsChanged ? 1 : 0;
		this._allocationParamsData[7] = resolveResidencyScanLimit(
			this._physicalPageCount,
			maxPagesPerFrame
		);
		this._dirtyParamsData[0] = this._physicalPageCount;
		this._dirtyParamsData[1] = this._physicalPageCount;
		this._dirtyParamsData[2] = this._physicalGridSize;
		this._dirtyParamsData[3] = this._pageSize;
		this._drawParamsData[0] = this._drawCandidateCount;
		this._drawParamsData[1] = this._physicalPageCount;
		this._drawParamsData[2] = this._physicalPageCount;
		this._drawParamsData[3] = this._pageSize;
		this._drawParamsData[4] = this._physicalGridSize;
		this._drawParamsData[5] = this._drawInstanceCapacity;
		this._drawParamsData[6] = this._frameId;
		this._drawParamsData[7] = 0;
		this._feedbackParamsData[0] = this._pageTableLength;
		this._feedbackParamsData[1] = Math.max(1, request.context.attachments.width | 0);
		this._feedbackParamsData[2] = Math.max(1, request.context.attachments.height | 0);
		this._feedbackParamsData[3] = layouts.length;
		this._feedbackParamsData[4] = this._frameId;
		this._feedbackParamsData[5] = this._feedbackFlags.length;
		this._feedbackParamsData[6] = 0;
		this._feedbackParamsData[7] = 0;
		this._backend.writeBuffer(this._requestParamsBuffer!, this._requestParamsData);
		this._backend.writeBuffer(this._compactParamsBuffer!, this._compactParamsData);
		this._backend.writeBuffer(this._allocationParamsBuffer!, this._allocationParamsData);
		this._backend.writeBuffer(this._dirtyParamsBuffer!, this._dirtyParamsData);
		this._backend.writeBuffer(this._drawParamsBuffer!, this._drawParamsData);
		this._backend.writeBuffer(this._feedbackParamsBuffer!, this._feedbackParamsData);
		this._backend.writeBuffer(
			this._layoutBuffer!,
			activeUint32Span(this._layoutData, layoutWriteCount)
		);
		if (pageAddressDataChanged) {
			this._backend.writeBuffer(
				this._pageAddressBuffer!,
				activeUint32Span(
					this._pageAddressData,
					Math.max(PAGE_ADDRESS_UINTS, this._pageTableLength * PAGE_ADDRESS_UINTS)
				)
			);
		}
		this._backend.writeBuffer(
			this._casterBoundsBuffer!,
			activeFloat32Span(this._casterBoundsData, Math.max(4, casterCount * 4))
		);
		this._backend.writeBuffer(
			this._casterStatesBuffer!,
			activeUint32Span(this._casterStatesData, Math.max(4, casterCount * 4))
		);
		this._backend.writeBuffer(
			this._cascadeViewProjectionBuffer!,
			activeFloat32Span(this._cascadeViewProjectionData, cascadeWriteCount)
		);
		this._backend.writeBuffer(
			this._feedbackCameraBuffer!,
			this._feedbackCameraData
		);
		this._backend.writeBuffer(
			this._drawWorldMatrixBuffer!,
			activeFloat32Span(this._drawWorldMatrixData, drawWorldWriteCount)
		);
		this._backend.writeBuffer(
			this._drawIndirectArgsBuffer!,
			activeUint32Span(this._drawIndirectArgsData, drawIndirectWriteCount)
		);
	}

	private _writeCasterBoundsWithTombstones(
		packets: readonly DrawPacket[]
	): number {
		const currentIds = new Set<string>();
		const movedIds = new Set<string>();
		let cursor = 0;

		// 1. Write current casters and check dirty states
		for (const packet of packets) {
			currentIds.add(packet.id);
			const bounds = packet.worldBounds;
			const prev = this._previousCasterBounds.get(packet.id);

			let isDirty = true;
			if (prev) {
				const matricesMatch = matrix4Equals(packet.worldMatrix, prev.worldMatrix);
				if (matricesMatch) {
					isDirty = false;
				} else {
					movedIds.add(packet.id);
				}
			}

			this._writeCasterBounds(cursor, {
				centerX: bounds.center.x,
				centerY: bounds.center.y,
				centerZ: bounds.center.z,
				radius: Math.max(0, bounds.radius),
				worldMatrix: packet.worldMatrix,
			});

			this._writeCasterState(cursor, isDirty ? 1 : 0);
			cursor++;
		}

		// 2. Write tombstones for removed or moved casters
		for (const [id, prevBounds] of this._previousCasterBounds) {
			const isRemoved = !currentIds.has(id);
			const hasMoved = movedIds.has(id);
			if ((isRemoved || hasMoved) && cursor < this._casterCapacity) {
				this._writeCasterBounds(cursor, prevBounds);
				this._writeCasterState(cursor, 1);
				cursor++;
			}
		}

		// 3. Cache current bounds for the next frame
		this._previousCasterBounds.clear();
		for (const packet of packets) {
			const bounds = packet.worldBounds;
			this._previousCasterBounds.set(packet.id, {
				centerX: bounds.center.x,
				centerY: bounds.center.y,
				centerZ: bounds.center.z,
				radius: Math.max(0, bounds.radius),
				worldMatrix: packet.worldMatrix,
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

	private _writeCasterState(index: number, state: number): void {
		const offset = index * 4;
		this._casterStatesData[offset] = state;
		this._casterStatesData[offset + 1] = 0;
		this._casterStatesData[offset + 2] = 0;
		this._casterStatesData[offset + 3] = 0;
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
		this._clearDrawIndirectArgsData[0] = 6;
		this._clearDrawIndirectArgsData[1] = 0;
		this._clearDrawIndirectArgsData[2] = 0;
		this._clearDrawIndirectArgsData[3] = 0;
		if (this._countersBuffer) {
			this._backend.writeBuffer(
				this._countersBuffer,
				new Uint32Array(this._counters.buffer, 3 * 4, 2),
				3 * 4
			);
		}
		if (this._clearDrawIndirectArgsBuffer) {
			this._backend.writeBuffer(
				this._clearDrawIndirectArgsBuffer,
				this._clearDrawIndirectArgsData
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
		const backend = this._backend as WebGPUDeviceResourceHost & {
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
		const slots = this._ensureDrawCounterReadbackSlots(
			backend.device,
			bufferUsage.COPY_DST | bufferUsage.MAP_READ,
			byteLength
		);
		if (slots.length <= 0) {
			return;
		}
		let slot: DrawCounterReadbackSlot | null = null;
		for (let i = 0; i < slots.length; i++) {
			const index = (this._drawCounterReadbackCursor + i) % slots.length;
			if (slots[index].state === "idle") {
				slot = slots[index];
				this._drawCounterReadbackCursor = (index + 1) % slots.length;
				break;
			}
		}
		if (!slot) {
			return;
		}
		nativeEncoder.copyBufferToBuffer(
			sourceBuffer,
			0,
			slot.buffer,
			0,
			byteLength
		);
		slot.frameId = this._frameId;
		slot.state = "copied";
	}

	private _scheduleQueuedDrawCounterReadbacks(): void {
		for (const slot of this._drawCounterReadbackSlots) {
			if (slot.state !== "copied") {
				continue;
			}
			slot.state = "mapping";
			void slot.buffer
				.mapAsync(WEBGPU_MAP_MODE_READ, 0, slot.byteLength)
				.then(() => {
					const mapped = slot.buffer.getMappedRange(0, slot.byteLength);
					this._applyDrawCounterReadback(new Uint32Array(mapped.slice(0)));
				})
				.catch(() => {
					// Failed readbacks are non-fatal; the next idle slot can retry.
				})
				.finally(() => {
					try {
						slot.buffer.unmap();
					} catch {
						// The buffer may already be unmapped after a failed map.
					}
					slot.state = "idle";
				});
		}
	}

	private _ensureDrawCounterReadbackSlots(
		device: {
			createBuffer: (descriptor: {
				label: string;
				size: number;
				usage: number;
			}) => GPUBuffer;
		},
		usage: number,
		byteLength: number
	): readonly DrawCounterReadbackSlot[] {
		if (
			this._drawCounterReadbackSlots.length > 0 &&
			this._drawCounterReadbackSlots.every((slot) => slot.byteLength === byteLength)
		) {
			return this._drawCounterReadbackSlots;
		}
		this._destroyDrawCounterReadbackSlots();
		for (let index = 0; index < DRAW_COUNTER_READBACK_RING_SIZE; index++) {
			this._drawCounterReadbackSlots.push({
				frameId: 0,
				buffer: device.createBuffer({
					label: `WebGPUPagedShadowDrawCounterReadback${index}`,
					size: byteLength,
					usage,
				}),
				byteLength,
				state: "idle",
			});
		}
		this._drawCounterReadbackCursor = 0;
		return this._drawCounterReadbackSlots;
	}

	private _destroyDrawCounterReadbackSlots(): void {
		for (const slot of this._drawCounterReadbackSlots) {
			slot.buffer.destroy();
		}
		this._drawCounterReadbackSlots = [];
		this._drawCounterReadbackCursor = 0;
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
		const backend = this._backend as WebGPUDeviceResourceHost & {
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
		const backend = this._backend as WebGPUDeviceResourceHost & {
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
				`webgpu.shadow.${shaderPart}` as Parameters<
					typeof ShaderSource.load
				>[0]
			);
			if (composite.kind !== "module") {
				throw new Error(`WebGPU shadow source "${shaderPart}" is not a module.`);
			}
			module = await backend.createShaderModule({
				label: `${label}Shader`,
				code: composite.source.code,
				sourceMap: composite.source.sourceMap,
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

	private _getFallbackSamplingResources(): WebGPUPagedShadowSamplingResources {
		if (!this._fallbackDepthAtlas) {
			this._fallbackDepthAtlas = this._backend.createTexture({
				width: DEFAULT_FALLBACK_PAGE_SIZE,
				height: DEFAULT_FALLBACK_PAGE_SIZE,
				format: TextureFormat.Depth32Float,
				usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
				label: "WebGPUPagedShadowFallbackDepthAtlas",
			});
		}
		if (!this._fallbackPageTableTexture) {
			this._fallbackPageTableTexture = this._backend.createTexture({
				width: 1,
				height: 1,
				format: TextureFormat.R32Uint,
				usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
				label: "WebGPUPagedShadowFallbackPageTableTexture",
			});
			this._resourceManager.writeTexture(
				this._fallbackPageTableTexture,
				new Uint32Array([WEBGPU_PAGED_SHADOW_NON_RESIDENT]),
				{},
				{ width: 1, height: 1 }
			);
		}
		return {
			pageTableTexture: this._fallbackPageTableTexture,
			physicalDepthAtlas: this._fallbackDepthAtlas,
		};
	}
}

export function collectWebGPUPagedShadowPageRequests(
	request: WebGPUPagedShadowFrameRequest,
	layouts: readonly WebGPUPagedShadowLayout[]
): WebGPUPagedShadowPageRequest[] {
	const requests = new Map<string, WebGPUPagedShadowPageRequest>();
	for (const prepared of request.shadowPlan?.lights ?? []) {
		const light = prepared.light;
		if (light.type !== LightType.Directional) {
			continue;
		}
		const layout = layouts.find((entry) => entry.prepared === prepared);
		if (!layout) {
			continue;
		}
		for (let cascadeIndex = 0; cascadeIndex < layout.cascadeCount; cascadeIndex++) {
			const viewProjection = prepared.slices[cascadeIndex]?.viewProjection;
			if (!viewProjection) continue;
			for (const packet of request.shadowCasterPackets) {
				addRequestsForPacketBounds(
					requests,
					light,
					prepared,
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
	prepared: PreparedShadowLight,
	layout: WebGPUPagedShadowLayout,
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
			const key = `${light.id}:${prepared.definition.id}:${cascadeIndex}:${pageX}:${pageY}`;
			if (requests.has(key)) {
				continue;
			}
			requests.set(key, {
				key,
				lightId: light.id,
				shadowMapId: prepared.definition.id,
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
	plan: ShadowFramePlan | undefined
): number {
	let maxPages = 1;
	for (const prepared of plan?.lights ?? []) {
		const paged = prepared.pagedSettings;
		if (!paged) {
			continue;
		}
		maxPages = Math.max(maxPages, paged.maxPagesPerFrame | 0);
	}
	return Math.max(1, maxPages);
}

function resolveMaxCacheFrames(
	plan: ShadowFramePlan | undefined
): number {
	let cacheFrames = 0;
	for (const prepared of plan?.lights ?? []) {
		const paged = prepared.pagedSettings;
		if (!paged) {
			continue;
		}
		cacheFrames = Math.max(cacheFrames, paged.cacheFrames | 0);
	}
	return Math.max(0, cacheFrames);
}

function resolveResidencyScanLimit(
	physicalPageCount: number,
	maxPagesPerFrame: number
): number {
	return Math.max(
		1,
		Math.min(
			physicalPageCount | 0,
			Math.max(64, (maxPagesPerFrame | 0) * 8)
		)
	);
}

function activeUint32Span(
	data: Uint32Array,
	elementCount: number
): Uint32Array<ArrayBuffer> {
	return data.subarray(0, Math.max(1, elementCount)) as Uint32Array<ArrayBuffer>;
}

function activeFloat32Span(
	data: Float32Array,
	elementCount: number
): Float32Array<ArrayBuffer> {
	return data.subarray(0, Math.max(1, elementCount)) as Float32Array<ArrayBuffer>;
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

const nextUniqueIdMap = new WeakMap<object, number>();
let nextUniqueIdVal = 1;

function getUniqueObjectId(obj: object): number {
	let id = nextUniqueIdMap.get(obj);
	if (id === undefined) {
		id = nextUniqueIdVal++;
		nextUniqueIdMap.set(obj, id);
	}
	return id;
}

function getResourceSizeSignature(resource: BindingResource | null): string {
	if (!resource || typeof resource !== "object") {
		return "0";
	}
	const id = getUniqueObjectId(resource);
	const sized = resource as { size?: unknown; width?: unknown; height?: unknown };
	return `${id}:${sized.size ?? 0}:${sized.width ?? 0}:${sized.height ?? 0}`;
}

function matrix4Equals(a: Matrix4, b: Matrix4): boolean {
	const ae = a.elements;
	const be = b.elements;
	for (let r = 0; r < 4; r++) {
		for (let c = 0; c < 4; c++) {
			if (Math.abs(ae[r][c] - be[r][c]) > 1e-5) {
				return false;
			}
		}
	}
	return true;
}
