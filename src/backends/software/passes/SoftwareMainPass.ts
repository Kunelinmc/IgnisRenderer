import { AlphaMode } from "../../../materials/Material";
import { Projector } from "../Projector";
import type {
	DecalPacket,
	DrawPacket,
	FrameContext,
} from "../../../pipeline/types";
import type {
	ProjectedFace,
	ProjectedVertex,
} from "../../../core/types";
import type { Rasterizer, RasterizerContext } from "../Rasterizer";
import { createSoftwareShadowSampler, getSoftwareShadowRuntimeMap } from "./SoftwareShadowPass";
import type { SoftwarePassLike } from "./types";
import {
	SOFTWARE_TAA_RENDER_STATE_KEY,
} from "../../../postprocess/passes/TemporalAntiAliasingPass";

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
	dirtyRects: RasterClipRect[] | null = null
): ProjectedTriangleWorkItem[] {
	const triangles: ProjectedTriangleWorkItem[] = [];
	const frameWidth = context.attachments.width;
	const frameHeight = context.attachments.height;

	for (const packet of packets) {
		const decalPackets = transparent ?
				EMPTY_DECAL_PACKETS
			:	collectPacketDecals(packet, context.scene.decalPackets);
		const faces = Projector.projectPacket(packet, context);
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

function resolveDirtyClipRects(context: FrameContext): RasterClipRect[] {
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

	const dirtyRects: RasterClipRect[] = [];
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
	context: FrameContext,
	dirtyRects: RasterClipRect[]
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

export class SoftwareMainPass implements SoftwarePassLike<
	[FrameContext, DrawPacket[], boolean],
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
			const program = this._rasterizer.prepareFragmentProgram(
				triangle.face,
				rasterizerContext,
				transparent,
				triangle.decalPackets
			);
			this._rasterizer.drawTriangle(
				triangle.pts,
				triangle.face,
				context.attachments.pixels!,
				rasterizerContext,
				program,
				transparent
			);
		}
	}

	public destroy(): void {
		this._earlyDepthBuffer = null;
	}
}
