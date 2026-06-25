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
import {
	BufferUsage,
	TextureFormat,
	TextureUsage,
	type IRenderBuffer,
	type IRenderTexture,
} from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
import type { WebGPUShadowPass } from "./WebGPUShadowPass";

export const WEBGPU_PAGED_SHADOW_NON_RESIDENT = 0xffffffff;
const PAGE_METADATA_UINTS = 8;
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
	private _pageTable = new Uint32Array([WEBGPU_PAGED_SHADOW_NON_RESIDENT]);
	private _pageMetadata = new Uint32Array(PAGE_METADATA_UINTS);
	private _requests: WebGPUPagedShadowPageRequest[] = [];
	private _residentPages = new Map<string, WebGPUPagedShadowResidentPage>();
	private _physicalToKey: Array<string | null> = [];
	private _physicalPageCount = 1;
	private _pageSize = DEFAULT_FALLBACK_PAGE_SIZE;
	private _physicalGridSize = 1;
	private _physicalAtlasSize = DEFAULT_FALLBACK_PAGE_SIZE;
	private _resourcesDirty = true;
	private _tableDirty = true;

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
		this._tableDirty = true;
	}

	/**
	 * @internal CPU page request pass hook.
	 */
	public recordPageMarkPass(
		request: WebGPUPagedShadowFrameRequest
	): void | Promise<void> {
		this._lastRequest = request;
	}

	/**
	 * @internal CPU page allocation pass hook.
	 */
	public recordPageAllocationPass(
		request: WebGPUPagedShadowFrameRequest
	): void | Promise<void> {
		this._lastRequest = request;
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
			residentPages,
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
		if (!this._pageTableBuffer || !this._pageMetadataBuffer || !this._physicalDepthAtlas) {
			return this._getFallbackResources();
		}
		return {
			pageTable: this._pageTableBuffer,
			physicalDepthAtlas: this._physicalDepthAtlas,
			physicalTransmittanceAtlas: null,
			pageMetadataBuffer: this._pageMetadataBuffer,
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
		this._fallbackPageTableBuffer?.destroy();
		this._fallbackMetadataBuffer?.destroy();
		this._fallbackDepthAtlas?.destroy();
		this._pageTableBuffer = null;
		this._pageMetadataBuffer = null;
		this._physicalDepthAtlas = null;
		this._fallbackPageTableBuffer = null;
		this._fallbackMetadataBuffer = null;
		this._fallbackDepthAtlas = null;
		this._requests = [];
		this._residentPages.clear();
		this._physicalToKey = [];
		this._pageTable = new Uint32Array([WEBGPU_PAGED_SHADOW_NON_RESIDENT]);
		this._pageMetadata = new Uint32Array(PAGE_METADATA_UINTS);
		this._resourcesDirty = true;
		this._tableDirty = true;
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
		if (
			this._pageSize === pageSize &&
			this._physicalPageCount === physicalPageCount &&
			this._physicalGridSize === physicalGridSize &&
			this._physicalAtlasSize === physicalAtlasSize &&
			this._pageTableBuffer &&
			this._pageTableBuffer.size >= Math.max(4, this._pageTable.byteLength) &&
			this._pageMetadataBuffer &&
			this._physicalDepthAtlas
		) {
			return;
		}

		this._pageSize = pageSize;
		this._physicalPageCount = physicalPageCount;
		this._physicalGridSize = physicalGridSize;
		this._physicalAtlasSize = physicalAtlasSize;
		this._residentPages.clear();
		this._physicalToKey = new Array(physicalPageCount).fill(null);
		this._pageMetadata = new Uint32Array(
			Math.max(1, physicalPageCount) * PAGE_METADATA_UINTS
		);
		this._pageTableBuffer?.destroy();
		this._pageMetadataBuffer?.destroy();
		this._physicalDepthAtlas?.destroy();
		this._pageTableBuffer = this._backend.createBuffer({
			size: Math.max(4, this._pageTable.byteLength),
			usage: BufferUsage.Storage | BufferUsage.CopyDst,
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
		this._resourcesDirty = true;
		this._tableDirty = true;
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
		}
		this._tableDirty = true;
	}

	private _writePageBuffersIfNeeded(): void {
		if (!this._pageTableBuffer || !this._pageMetadataBuffer) {
			return;
		}
		if (!this._tableDirty && !this._resourcesDirty) {
			return;
		}
		this._backend.writeBuffer(this._pageTableBuffer, this._pageTable);
		this._backend.writeBuffer(this._pageMetadataBuffer, this._pageMetadata);
		this._tableDirty = false;
		this._resourcesDirty = false;
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
		return {
			pageTable: this._fallbackPageTableBuffer,
			physicalDepthAtlas: this._fallbackDepthAtlas,
			physicalTransmittanceAtlas: null,
			pageMetadataBuffer: this._fallbackMetadataBuffer,
			pageSize: DEFAULT_FALLBACK_PAGE_SIZE,
			physicalGridSize: 1,
			physicalAtlasSize: DEFAULT_FALLBACK_PAGE_SIZE,
		};
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
