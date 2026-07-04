import { Platform } from "../../../foundation/Platform";
import { AlphaMode } from "../../../materials/Material";
import { Projector } from "../Projector";
import type { DrawPacket, FrameContext } from "../../../pipeline/types";
import type {
	ProjectedFace,
	ProjectedVertex,
} from "../../../core/types";
import type { Rasterizer, RasterizerContext } from "../Rasterizer";
import { createSoftwareShadowSampler, getSoftwareShadowRuntimeMap } from "./SoftwareShadowPass";
import type { WorkerLike } from "../../../workers/types";
import { globalWorkerScheduler } from "../../../workers/WorkerScheduler";
import { DEFAULT_WORKER_TRANSPORT_PLUGINS } from "../../../workers/transports";
import { type SoftwareRasterMode, type SoftwareTileOptions } from "../types";
import { DEFAULT_SOFTWARE_TILE_SIZE, DEFAULT_SOFTWARE_RASTER_MODE } from "../constants";
import { Logger } from "../../../foundation/Logger";
import type { SoftwarePassLike } from "./types";
import type {
	SoftwareRasterTileBounds,
	SoftwareRasterWorkerTaskPayload,
	SoftwareRasterWorkerTaskResult,
} from "../workers/softwareRasterWorkerProtocol";
import {
	SOFTWARE_TAA_RENDER_STATE_KEY,
} from "../../../postprocess/passes/TemporalAntiAliasingPass";

interface TileTriangleWorkItem {
	index: number;
	pts: [ProjectedVertex, ProjectedVertex, ProjectedVertex];
	face: ProjectedFace;
}

interface TileClipRect {
	minX: number;
	minY: number;
	maxX: number;
	maxY: number;
}

interface SoftwareMainRasterExecutorLike extends SoftwarePassLike<
	[FrameContext, DrawPacket[], boolean],
	Promise<void>
> {
	getActiveMode(): SoftwareRasterMode;
	destroy(): void;
}

export interface SoftwareMainPassOptions {
	mode?: SoftwareRasterMode;
	tile?: SoftwareTileOptions;
	enableEarlyZPrepass?: boolean;
}

function resolvePreparedSceneEnvironment(
	scene: FrameContext["scene"]
): {
	backgroundEnabled: boolean;
	lightingEnabled: boolean;
	backgroundTexture: any;
	iblTexture: any;
	backgroundStrength: number;
	backgroundTintLinear: { r: number; g: number; b: number };
	backgroundExposure: number;
} {
	const environment = (scene as { environment?: unknown }).environment as
		| {
				backgroundEnabled?: boolean;
				lightingEnabled?: boolean;
				backgroundTexture?: unknown;
				iblTexture?: unknown;
				backgroundStrength?: number;
				backgroundTintLinear?: { r?: number; g?: number; b?: number };
				backgroundExposure?: number;
		  }
		| undefined;
	return {
		backgroundEnabled: environment?.backgroundEnabled ?? true,
		lightingEnabled: environment?.lightingEnabled ?? true,
		backgroundTexture:
			(environment?.backgroundTexture as any | null | undefined) ?? null,
		iblTexture: (environment?.iblTexture as any | null | undefined) ?? null,
		backgroundStrength:
			typeof environment?.backgroundStrength === "number" ?
				environment.backgroundStrength
			:	1,
		backgroundTintLinear: {
			r:
				typeof environment?.backgroundTintLinear?.r === "number" ?
					environment.backgroundTintLinear.r
				:	1,
			g:
				typeof environment?.backgroundTintLinear?.g === "number" ?
					environment.backgroundTintLinear.g
				:	1,
			b:
				typeof environment?.backgroundTintLinear?.b === "number" ?
					environment.backgroundTintLinear.b
				:	1,
		},
		backgroundExposure:
			typeof environment?.backgroundExposure === "number" ?
				environment.backgroundExposure
			:	1,
	};
}

function createRasterizerContext(context: FrameContext): RasterizerContext {
	const runtimeMap = getSoftwareShadowRuntimeMap(context.transient);
	const sampleShadow = createSoftwareShadowSampler(
		context.shadowMaps,
		runtimeMap,
		{ camera: context.viewCamera }
	);
	const environment = resolvePreparedSceneEnvironment(context.scene);

	return {
		width: context.attachments.width,
		height: context.attachments.height,
		depthBuffer: context.attachments.depthBuffer!,
		normalBuffer: context.attachments.normalBuffer,
		motionBuffer: context.attachments.motionBuffer,
		taa: context.transient.get(SOFTWARE_TAA_RENDER_STATE_KEY),
		camera: {
			position: context.viewCamera.getWorldPosition(),
			viewMatrix: context.viewCamera.viewMatrix,
		},
		lights: context.scene.lights,
		shadowMaps: context.shadowMaps,
		sampleShadow,
		shAmbientCoeffs: context.shAmbientCoeffs,
		environmentSpecularTexture:
			environment.lightingEnabled ?
				environment.iblTexture
			:	null,
		enableLighting: context.features.enableLighting,
		enableSH: context.features.enableSH,
		enableShadows: context.features.enableShadows,
	};
}

function collectProjectedTriangles(
	context: FrameContext,
	packets: DrawPacket[],
	transparent: boolean,
	dirtyRects: TileClipRect[] | null = null
): TileTriangleWorkItem[] {
	const triangles: TileTriangleWorkItem[] = [];
	const frameWidth = context.attachments.width;
	const frameHeight = context.attachments.height;

	for (const packet of packets) {
		const faces = Projector.projectPacket(packet, context);
		if (transparent) {
			faces.sort((left, right) => right.depthInfo.avg - left.depthInfo.avg);
		}

		for (const face of faces) {
			const projected = face.projected;
			for (let i = 1; i < projected.length - 1; i++) {
				const triangle: TileTriangleWorkItem = {
					index: triangles.length,
					pts: [projected[0], projected[i], projected[i + 1]],
					face,
				};
				if (
					dirtyRects &&
					dirtyRects.length > 0 &&
					!triangleIntersectsAnyDirtyRect(
						triangle,
						frameWidth,
						frameHeight,
						dirtyRects
					)
				) {
					continue;
				}
				triangles.push(triangle);
			}
		}
	}

	return triangles;
}

function computeTileBounds(
	triangle: TileTriangleWorkItem,
	width: number,
	height: number,
	tileSize: number
): SoftwareRasterTileBounds {
	const p0 = triangle.pts[0];
	const p1 = triangle.pts[1];
	const p2 = triangle.pts[2];

	const minProjectedX = Math.min(p0.x, p1.x, p2.x);
	const maxProjectedX = Math.max(p0.x, p1.x, p2.x);
	const minProjectedY = Math.min(p0.y, p1.y, p2.y);
	const maxProjectedY = Math.max(p0.y, p1.y, p2.y);

	const minPixelX = Math.max(0, Math.ceil(minProjectedX - 0.5));
	const maxPixelX = Math.min(width - 1, Math.floor(maxProjectedX - 0.5));
	const minPixelY = Math.max(0, Math.ceil(minProjectedY - 0.5));
	const maxPixelY = Math.min(height - 1, Math.floor(maxProjectedY - 0.5));

	if (
		minPixelX > maxPixelX ||
		minPixelY > maxPixelY ||
		width <= 0 ||
		height <= 0
	) {
		return {
			minTileX: 1,
			minTileY: 1,
			maxTileX: 0,
			maxTileY: 0,
		};
	}

	return {
		minTileX: Math.floor(minPixelX / tileSize),
		minTileY: Math.floor(minPixelY / tileSize),
		maxTileX: Math.floor(maxPixelX / tileSize),
		maxTileY: Math.floor(maxPixelY / tileSize),
	};
}

function mergeBins(
	results: SoftwareRasterWorkerTaskResult[]
): Map<number, number[]> {
	const merged = new Map<number, number[]>();

	for (const result of results) {
		if (result.type !== "bin-main-pass") continue;
		for (const bin of result.bins) {
			let bucket = merged.get(bin.tileIndex);
			if (!bucket) {
				bucket = [];
				merged.set(bin.tileIndex, bucket);
			}
			for (const triangleIndex of bin.triangleIndices) {
				bucket.push(triangleIndex);
			}
		}
	}

	for (const bucket of merged.values()) {
		bucket.sort((left, right) => left - right);
	}

	return merged;
}

function createTileClipRect(
	tileIndex: number,
	tileSize: number,
	tileColumns: number,
	width: number,
	height: number
): TileClipRect {
	const tileX = tileIndex % tileColumns;
	const tileY = Math.floor(tileIndex / tileColumns);
	const minX = tileX * tileSize;
	const minY = tileY * tileSize;
	const maxX = Math.min(width - 1, minX + tileSize - 1);
	const maxY = Math.min(height - 1, minY + tileSize - 1);
	return {
		minX,
		minY,
		maxX,
		maxY,
	};
}

function resolveDirtyClipRects(context: FrameContext): TileClipRect[] {
	const width = Math.max(1, context.attachments.width);
	const height = Math.max(1, context.attachments.height);
	const incremental = context.incremental;
	if (
		!incremental.enabled ||
		incremental.forceFullFrame ||
		incremental.dirtyRects.length === 0
	) {
		return [{
			minX: 0,
			minY: 0,
			maxX: width - 1,
			maxY: height - 1,
		}];
	}

	const dirtyRects: TileClipRect[] = [];
	for (const rect of incremental.dirtyRects) {
		const minX = Math.max(0, Math.floor(rect.x));
		const minY = Math.max(0, Math.floor(rect.y));
		const maxX = Math.min(width - 1, Math.ceil(rect.x + rect.width) - 1);
		const maxY = Math.min(height - 1, Math.ceil(rect.y + rect.height) - 1);
		if (minX > maxX || minY > maxY) {
			continue;
		}
		dirtyRects.push({
			minX,
			minY,
			maxX,
			maxY,
		});
	}
	return dirtyRects;
}

function clipRectsIntersect(left: TileClipRect, right: TileClipRect): boolean {
	return !(
		left.maxX < right.minX ||
		left.minX > right.maxX ||
		left.maxY < right.minY ||
		left.minY > right.maxY
	);
}

function intersectsAnyDirtyRect(
	rect: TileClipRect,
	dirtyRects: TileClipRect[]
): boolean {
	for (const dirtyRect of dirtyRects) {
		if (clipRectsIntersect(rect, dirtyRect)) {
			return true;
		}
	}
	return false;
}

function triangleIntersectsAnyDirtyRect(
	triangle: TileTriangleWorkItem,
	width: number,
	height: number,
	dirtyRects: TileClipRect[]
): boolean {
	const p0 = triangle.pts[0];
	const p1 = triangle.pts[1];
	const p2 = triangle.pts[2];
	const minX = Math.max(0, Math.ceil(Math.min(p0.x, p1.x, p2.x) - 0.5));
	const maxX = Math.min(width - 1, Math.floor(Math.max(p0.x, p1.x, p2.x) - 0.5));
	const minY = Math.max(0, Math.ceil(Math.min(p0.y, p1.y, p2.y) - 0.5));
	const maxY = Math.min(height - 1, Math.floor(Math.max(p0.y, p1.y, p2.y) - 0.5));
	if (minX > maxX || minY > maxY) {
		return false;
	}
	return intersectsAnyDirtyRect(
		{
			minX,
			minY,
			maxX,
			maxY,
		},
		dirtyRects
	);
}

function shouldRunEarlyDepthPrepass(
	transparent: boolean,
	enabled: boolean
): boolean {
	return !transparent && enabled;
}

function isMaskTriangle(triangle: TileTriangleWorkItem): boolean {
	return triangle.face.material?.alphaMode === AlphaMode.Mask;
}

function shouldSkipEarlyDepthPrepassTriangle(
	triangle: TileTriangleWorkItem
): boolean {
	const material = triangle.face.material;
	return (
		isMaskTriangle(triangle) ||
		(!!material && !material.depthWrite)
	);
}

function prepareEarlyDepthBuffer(
	previous: Float32Array | null,
	context: FrameContext,
	dirtyRects: TileClipRect[]
): Float32Array {
	const width = Math.max(1, context.attachments.width | 0);
	const height = Math.max(1, context.attachments.height | 0);
	const size = width * height;
	const depthBuffer = context.attachments.depthBuffer!;
	const next =
		previous && previous.length === size ? previous : new Float32Array(size);
	next.set(depthBuffer);

	for (const rect of dirtyRects) {
		const minX = Math.max(0, Math.floor(rect.minX));
		const minY = Math.max(0, Math.floor(rect.minY));
		const maxX = Math.min(width - 1, Math.floor(rect.maxX));
		const maxY = Math.min(height - 1, Math.floor(rect.maxY));
		if (minX > maxX || minY > maxY) continue;

		for (let y = minY; y <= maxY; y++) {
			const rowStart = y * width;
			for (let x = minX; x <= maxX; x++) {
				next[rowStart + x] = Infinity;
			}
		}
	}

	return next;
}

class ScanlineMainRasterExecutor implements SoftwareMainRasterExecutorLike {
	private _rasterizer: Rasterizer;
	private _enableEarlyZPrepass: boolean;
	private _earlyDepthBuffer: Float32Array | null = null;

	public constructor(
		rasterizer: Rasterizer,
		enableEarlyZPrepass: boolean = true
	) {
		this._rasterizer = rasterizer;
		this._enableEarlyZPrepass = enableEarlyZPrepass;
	}

	public async render(
		context: FrameContext,
		packets: DrawPacket[],
		transparent: boolean
	): Promise<void> {
		const dirtyRects = resolveDirtyClipRects(context);
		if (dirtyRects.length === 0) {
			return;
		}
		const triangles = collectProjectedTriangles(
			context,
			packets,
			transparent,
			dirtyRects
		);
		if (triangles.length === 0) {
			return;
		}
		const rasterizerContext = createRasterizerContext(context);
		if (shouldRunEarlyDepthPrepass(transparent, this._enableEarlyZPrepass)) {
			this._earlyDepthBuffer = prepareEarlyDepthBuffer(
				this._earlyDepthBuffer,
				context,
				dirtyRects
			);
			rasterizerContext.earlyDepthBuffer = this._earlyDepthBuffer;
			for (const triangle of triangles) {
				if (shouldSkipEarlyDepthPrepassTriangle(triangle)) continue;
				this._rasterizer.drawCameraDepthTriangle(
					triangle.pts,
					rasterizerContext
				);
			}
		}
		for (const triangle of triangles) {
			this._rasterizer.drawTriangle(
				triangle.pts,
				triangle.face,
				context.attachments.pixels!,
				rasterizerContext,
				transparent
			);
		}
	}

	public getActiveMode(): SoftwareRasterMode {
		return "scanline";
	}

	public destroy(): void {
		this._earlyDepthBuffer = null;
	}
}

class TileMainRasterExecutor implements SoftwareMainRasterExecutorLike {
	private _rasterizer: Rasterizer;
	private _scanlineFallback: ScanlineMainRasterExecutor;
	private _tileSize: number;
	private _workerCount: number;
	private _scheduler: typeof globalWorkerScheduler;
	private _poolId: string;
	private _defaultTimeoutMs: number;
	private _enableEarlyZPrepass: boolean;
	private _earlyDepthBuffer: Float32Array | null = null;
	private _poolOwned = false;
	private _poolReady = false;
	private _failedOverToScanline = false;

	public constructor(
		rasterizer: Rasterizer,
		tileOptions: SoftwareTileOptions = {},
		enableEarlyZPrepass: boolean = true
	) {
		this._rasterizer = rasterizer;
		this._scanlineFallback = new ScanlineMainRasterExecutor(
			rasterizer,
			enableEarlyZPrepass
		);
		this._tileSize = Math.max(
			1,
			Math.floor(tileOptions.tileSize ?? DEFAULT_SOFTWARE_TILE_SIZE)
		);
		this._workerCount = Math.max(
			1,
			Math.floor(
				tileOptions.workerCount ?? Platform.getHardwareConcurrency(4)
			)
		);
		this._scheduler = tileOptions.scheduler ?? globalWorkerScheduler;
		this._poolId =
			tileOptions.poolId ??
			`software-main-raster-${Math.random().toString(36).slice(2)}`;
		this._defaultTimeoutMs = Math.max(
			0,
			Math.floor(tileOptions.defaultTimeoutMs ?? 0)
		);
		this._enableEarlyZPrepass = enableEarlyZPrepass;
	}

	public async render(
		context: FrameContext,
		packets: DrawPacket[],
		transparent: boolean
	): Promise<void> {
		if (this._failedOverToScanline) {
			await this._scanlineFallback.render(context, packets, transparent);
			return;
		}

		if (!this._ensureWorkerPool()) {
			await this._scanlineFallback.render(context, packets, transparent);
			return;
		}

		const dirtyRects = resolveDirtyClipRects(context);
		if (dirtyRects.length === 0) {
			return;
		}
		const triangles = collectProjectedTriangles(
			context,
			packets,
			transparent,
			dirtyRects
		);
		if (triangles.length === 0) return;

		const width = context.attachments.width;
		const height = context.attachments.height;
		const tileColumns = Math.max(1, Math.ceil(width / this._tileSize));
		const tileRows = Math.max(1, Math.ceil(height / this._tileSize));
		const triangleBounds = triangles.map((triangle) =>
			computeTileBounds(triangle, width, height, this._tileSize)
		);

		let bins: Map<number, number[]>;
		try {
			const results = await this._dispatchTileBinningTasks(
				width,
				height,
				triangleBounds
			);
			bins = mergeBins(results);
		} catch (error) {
			this._fallbackToScanline(
				"software-raster-worker-task-failed",
				`Software tile raster worker tasks failed. Falling back to scanline rasterizer. ${String(error)}`
			);
			await this._scanlineFallback.render(context, packets, transparent);
			return;
		}

		const sortedTileIndices = [...bins.keys()].sort((left, right) => left - right);
		const pixels = context.attachments.pixels!;
		const baseContext = createRasterizerContext(context);
		if (shouldRunEarlyDepthPrepass(transparent, this._enableEarlyZPrepass)) {
			this._earlyDepthBuffer = prepareEarlyDepthBuffer(
				this._earlyDepthBuffer,
				context,
				dirtyRects
			);
			baseContext.earlyDepthBuffer = this._earlyDepthBuffer;
			for (const tileIndex of sortedTileIndices) {
				const triangleIndices = bins.get(tileIndex);
				if (!triangleIndices || triangleIndices.length === 0) continue;

				const clipRect = createTileClipRect(
					tileIndex,
					this._tileSize,
					tileColumns,
					width,
					height
				);
				if (
					clipRect.minX > clipRect.maxX ||
					clipRect.minY > clipRect.maxY ||
					tileIndex < 0 ||
					tileIndex >= tileColumns * tileRows ||
					!intersectsAnyDirtyRect(clipRect, dirtyRects)
				) {
					continue;
				}

				const tileContext: RasterizerContext = {
					...baseContext,
					clipRect,
				};

				for (const triangleIndex of triangleIndices) {
					const triangle = triangles[triangleIndex];
					if (!triangle || shouldSkipEarlyDepthPrepassTriangle(triangle)) {
						continue;
					}
					this._rasterizer.drawCameraDepthTriangle(
						triangle.pts,
						tileContext
					);
				}
			}
		}

		for (const tileIndex of sortedTileIndices) {
			const triangleIndices = bins.get(tileIndex);
			if (!triangleIndices || triangleIndices.length === 0) continue;

			const clipRect = createTileClipRect(
				tileIndex,
				this._tileSize,
				tileColumns,
				width,
				height
			);
			if (
				clipRect.minX > clipRect.maxX ||
				clipRect.minY > clipRect.maxY ||
				tileIndex < 0 ||
				tileIndex >= tileColumns * tileRows ||
				!intersectsAnyDirtyRect(clipRect, dirtyRects)
			) {
				continue;
			}

			const tileContext: RasterizerContext = {
				...baseContext,
				clipRect,
			};

			for (const triangleIndex of triangleIndices) {
				const triangle = triangles[triangleIndex];
				if (!triangle) continue;
				this._rasterizer.drawTriangle(
					triangle.pts,
					triangle.face,
					pixels,
					tileContext,
					transparent
				);
			}
		}
	}

	public getActiveMode(): SoftwareRasterMode {
		return this._failedOverToScanline ? "scanline" : "tile";
	}

	public destroy(): void {
		if (this._poolOwned) {
			try {
				this._scheduler.unregisterPool(this._poolId);
			} catch {
				// Ignore cleanup failures during backend shutdown.
			}
		}
		this._poolOwned = false;
		this._poolReady = false;
		this._earlyDepthBuffer = null;
	}

	private _ensureWorkerPool(): boolean {
		if (this._failedOverToScanline) return false;
		if (this._poolReady) return true;

		if (!Platform.hasWorker()) {
			this._fallbackToScanline(
				"software-raster-worker-unavailable",
				"Software tile rasterizer requested workers, but Worker API is unavailable. Falling back to scanline rasterizer."
			);
			return false;
		}

		if (this._scheduler.hasPool(this._poolId)) {
			this._poolReady = true;
			return true;
		}

		try {
			this._scheduler.registerPool({
				id: this._poolId,
				size: this._workerCount,
				createWorker: (workerIndex, poolId) =>
					this._createWorker(workerIndex, poolId),
				transportPlugins: DEFAULT_WORKER_TRANSPORT_PLUGINS,
				runtimeCapabilities: {
					sharedArrayBuffer: Platform.supportsSharedArrayBufferTransport(),
					crossOriginIsolated: Platform.isCrossOriginIsolated(globalThis, false),
				},
				defaultTimeoutMs: this._defaultTimeoutMs,
			});
			this._poolOwned = true;
			this._poolReady = true;
			return true;
		} catch (error) {
			this._fallbackToScanline(
				"software-raster-worker-pool-init-failed",
				`Software tile raster worker pool initialization failed. Falling back to scanline rasterizer. ${String(error)}`
			);
			return false;
		}
	}

	private _fallbackToScanline(key: string, message: string): void {
		this._failedOverToScanline = true;
		this._poolReady = false;
		if (this._poolOwned) {
			try {
				this._scheduler.unregisterPool(this._poolId);
			} catch {
				// Ignore pool cleanup failures during failover.
			}
		}
		this._poolOwned = false;
		Logger.warn(`[${key}] ${message}`, {
			scope: "SoftwareMainPass",
			onceKey: key,
		});
	}

	private _createWorker(workerIndex: number, poolId: string): WorkerLike {
		if (typeof Worker !== "function") {
			throw new Error(
				`Worker constructor is unavailable for pool "${poolId}" (worker #${workerIndex})`
			);
		}

		return new Worker(
			new URL("../workers/softwareRaster.worker.ts", import.meta.url),
			{
				type: "module",
			}
		) as unknown as WorkerLike;
	}

	private async _dispatchTileBinningTasks(
		width: number,
		height: number,
		triangleBounds: SoftwareRasterTileBounds[]
	): Promise<SoftwareRasterWorkerTaskResult[]> {
		const triangleCount = triangleBounds.length;
		if (triangleCount <= 0) return [];

		const chunkCount = Math.max(
			1,
			Math.min(this._workerCount, triangleCount)
		);
		const chunkSize = Math.max(1, Math.ceil(triangleCount / chunkCount));
		const tasks: Promise<SoftwareRasterWorkerTaskResult>[] = [];

		for (
			let startIndex = 0;
			startIndex < triangleCount;
			startIndex += chunkSize
		) {
			const endIndex = Math.min(triangleCount, startIndex + chunkSize);
			const payload: SoftwareRasterWorkerTaskPayload = {
				type: "bin-main-pass",
				width,
				height,
				tileSize: this._tileSize,
				triangleBounds,
				startIndex,
				endIndex,
			};

			tasks.push(
				this._scheduler.schedule<
					SoftwareRasterWorkerTaskResult,
					SoftwareRasterWorkerTaskPayload
				>(this._poolId, payload, {
					timeoutMs: this._defaultTimeoutMs,
				})
			);
		}

		return Promise.all(tasks);
	}
}

export class SoftwareMainPass implements SoftwarePassLike<
	[FrameContext, DrawPacket[], boolean],
	Promise<void>
> {
	private _executor: SoftwareMainRasterExecutorLike;

	public constructor(rasterizer: Rasterizer, options: SoftwareMainPassOptions = {}) {
		const enableEarlyZPrepass = options.enableEarlyZPrepass !== false;
		const mode = options.mode ?? DEFAULT_SOFTWARE_RASTER_MODE;
		if (mode === "tile") {
			this._executor = new TileMainRasterExecutor(
				rasterizer,
				options.tile,
				enableEarlyZPrepass
			);
		} else {
			this._executor = new ScanlineMainRasterExecutor(
				rasterizer,
				enableEarlyZPrepass
			);
		}
	}

	public async render(
		context: FrameContext,
		packets: DrawPacket[],
		transparent: boolean,
	): Promise<void> {
		await this._executor.render(context, packets, transparent);
	}

	public getActiveMode(): SoftwareRasterMode {
		return this._executor.getActiveMode();
	}

	public destroy(): void {
		this._executor.destroy();
	}
}
