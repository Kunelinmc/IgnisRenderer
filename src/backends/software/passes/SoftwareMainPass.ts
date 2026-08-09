import { AlphaMode } from "../../../materials/Material";
import { Projector } from "../Projector";
import type {
	DecalPacket,
	DrawPacket,
} from "../../../pipeline/types";
import type {
	ProjectedFace,
	ProjectedVertex,
} from "../../../core/types";
import type { Rasterizer } from "../Rasterizer";
import type { SoftwarePassLike } from "./types";
import type { SoftwarePassContext } from "../SoftwareFrameServices";
import type {
	SoftwareClipRegion,
	SoftwareFrameView,
} from "../SoftwareFrameView";
import { createSoftwareRasterizerContext } from "../SoftwareRasterContextFactory";

interface ProjectedTriangleWorkItem {
	pts: [ProjectedVertex, ProjectedVertex, ProjectedVertex];
	face: ProjectedFace;
	decalPackets: readonly DecalPacket[];
}

interface RasterClipRect {
	minX: number;
	minY: number;
	maxX: number;
	maxY: number;
}

export interface SoftwareMainPassOptions {
	enableEarlyZPrepass?: boolean;
}

function collectProjectedTriangles(
	frame: SoftwareFrameView,
	packets: DrawPacket[],
	transparent: boolean,
	dirtyRects: RasterClipRect[] | null = null
): ProjectedTriangleWorkItem[] {
	const triangles: ProjectedTriangleWorkItem[] = [];
	const frameWidth = frame.attachments.width;
	const frameHeight = frame.attachments.height;

	for (const packet of packets) {
		const decalPackets = transparent ?
				EMPTY_DECAL_PACKETS
			:	collectPacketDecals(packet, frame.scene.decalPackets);
		const faces = Projector.projectPacket(packet, frame);
		if (transparent) {
			faces.sort((left, right) => right.depthInfo.avg - left.depthInfo.avg);
		}

		for (const face of faces) {
			const projected = face.projected;
			for (let i = 1; i < projected.length - 1; i++) {
				const triangle: ProjectedTriangleWorkItem = {
					pts: [projected[0], projected[i], projected[i + 1]],
					face,
					decalPackets,
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

const EMPTY_DECAL_PACKETS: readonly DecalPacket[] = [];

function collectPacketDecals(
	packet: DrawPacket,
	decals: readonly DecalPacket[]
): readonly DecalPacket[] {
	let result: DecalPacket[] | null = null;
	for (const decal of decals) {
		if (
			(packet.meshInstance.renderLayers & decal.receiverLayerMask) === 0 ||
			!boundingSpheresIntersect(packet, decal)
		) {
			continue;
		}
		(result ??= []).push(decal);
	}
	return result ?? EMPTY_DECAL_PACKETS;
}

function boundingSpheresIntersect(
	packet: DrawPacket,
	decal: DecalPacket
): boolean {
	const dx = packet.worldBounds.center.x - decal.worldBounds.center.x;
	const dy = packet.worldBounds.center.y - decal.worldBounds.center.y;
	const dz = packet.worldBounds.center.z - decal.worldBounds.center.z;
	const radius = packet.worldBounds.radius + decal.worldBounds.radius;
	return dx * dx + dy * dy + dz * dz <= radius * radius;
}

function resolveDirtyClipRects(frame: SoftwareFrameView): RasterClipRect[] {
	return frame.clipRegions.map((region: SoftwareClipRegion) => ({
		minX: region.minX,
		minY: region.minY,
		maxX: region.maxXExclusive - 1,
		maxY: region.maxYExclusive - 1,
	}));
}

function clipRectsIntersect(left: RasterClipRect, right: RasterClipRect): boolean {
	return !(
		left.maxX < right.minX ||
		left.minX > right.maxX ||
		left.maxY < right.minY ||
		left.minY > right.maxY
	);
}

function intersectsAnyDirtyRect(
	rect: RasterClipRect,
	dirtyRects: RasterClipRect[]
): boolean {
	for (const dirtyRect of dirtyRects) {
		if (clipRectsIntersect(rect, dirtyRect)) {
			return true;
		}
	}
	return false;
}

function triangleIntersectsAnyDirtyRect(
	triangle: ProjectedTriangleWorkItem,
	width: number,
	height: number,
	dirtyRects: RasterClipRect[]
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

function isMaskTriangle(triangle: ProjectedTriangleWorkItem): boolean {
	return triangle.face.material?.alphaMode === AlphaMode.Mask;
}

function shouldSkipEarlyDepthPrepassTriangle(
	triangle: ProjectedTriangleWorkItem
): boolean {
	const material = triangle.face.material;
	return (
		isMaskTriangle(triangle) ||
		(!!material && !material.depthWrite)
	);
}

function prepareEarlyDepthBuffer(
	previous: Float32Array | null,
	frame: SoftwareFrameView,
	dirtyRects: RasterClipRect[]
): Float32Array {
	const width = frame.attachments.width;
	const height = frame.attachments.height;
	const size = width * height;
	const depthBuffer = frame.attachments.depthBuffer;
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

export class SoftwareMainPass implements SoftwarePassLike<
	[SoftwarePassContext, DrawPacket[], boolean],
	Promise<void>
> {
	private _rasterizer: Rasterizer;
	private _enableEarlyZPrepass: boolean;
	private _earlyDepthBuffer: Float32Array | null = null;

	public constructor(rasterizer: Rasterizer, options: SoftwareMainPassOptions = {}) {
		this._rasterizer = rasterizer;
		this._enableEarlyZPrepass = options.enableEarlyZPrepass !== false;
	}

	public async render(
		context: SoftwarePassContext,
		packets: DrawPacket[],
		transparent: boolean
	): Promise<void> {
		const frame = context.frame;
		const dirtyRects = resolveDirtyClipRects(frame);
		if (dirtyRects.length === 0) {
			return;
		}
		const triangles = collectProjectedTriangles(
			frame,
			packets,
			transparent,
			dirtyRects
		);
		if (triangles.length === 0) {
			return;
		}
		const rasterizerContext = createSoftwareRasterizerContext(context);
		const runEarlyDepthPrepass = shouldRunEarlyDepthPrepass(
			transparent,
			this._enableEarlyZPrepass,
		);
		if (runEarlyDepthPrepass) {
			this._earlyDepthBuffer = prepareEarlyDepthBuffer(
				this._earlyDepthBuffer,
				frame,
				dirtyRects
			);
			rasterizerContext.earlyDepthBuffer = this._earlyDepthBuffer;
		}
		for (const dirtyRect of dirtyRects) {
			rasterizerContext.clipRect = dirtyRect;
			if (runEarlyDepthPrepass) {
				for (const triangle of triangles) {
					if (shouldSkipEarlyDepthPrepassTriangle(triangle)) continue;
					this._rasterizer.drawCameraDepthTriangle(
						triangle.pts,
						rasterizerContext,
					);
				}
			}
			for (const triangle of triangles) {
				const program = this._rasterizer.prepareFragmentProgram(
					triangle.face,
					rasterizerContext,
					transparent,
					triangle.decalPackets,
				);
				this._rasterizer.drawTriangle(
					triangle.pts,
					triangle.face,
					frame.attachments.pixels,
					rasterizerContext,
					program,
					transparent,
				);
			}
		}
		rasterizerContext.clipRect = null;
	}

	public destroy(): void {
		this._earlyDepthBuffer = null;
	}
}
