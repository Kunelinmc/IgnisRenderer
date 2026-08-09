import type { IVertex, ProjectedVertex } from "../../core/types";
import { Matrix4 } from "../../maths/Matrix4";
import { Vector3 } from "../../maths/Vector3";
import type { DrawPacket } from "../../pipeline/types";
import { Projector } from "./Projector";
import type { Rasterizer } from "./Rasterizer";
import type { SoftwareFrameView } from "./SoftwareFrameView";
import type { PreparedShadowSlice } from "../../lights/shadows/ShadowFramePlan";
import type { SoftwareShadowRenderTarget } from "./SoftwareShadowContracts";
import { SoftwareShadowConstants } from "./SoftwareShadowConstants";

interface ClipVertex {
	x: number;
	y: number;
	z: number;
	w: number;
	u: number;
	v: number;
}

/** @internal Rasterizes caster and transmitter geometry into CPU shadow targets. */
export class SoftwareShadowRasterPass {
	private readonly _mvpMatrix = Matrix4.identity();
	private readonly _lightDirModel = new Vector3();
	private readonly _projectedVertsPool: ProjectedVertex[] = [];
	private readonly _projectedVertsView: ProjectedVertex[] = [];
	private readonly _clipInputPool: ClipVertex[] = [];
	private readonly _clipVertsPool: ClipVertex[] = [];
	private _clipPoolCursor = 0;
	private readonly _clipScratchA: ClipVertex[] = [];
	private readonly _clipScratchB: ClipVertex[] = [];

	public constructor(private readonly _rasterizer: Rasterizer) {
		for (let index = 0; index < 4; index++) {
			this._projectedVertsPool.push({
				x: 0,
				y: 0,
				z: 0,
				w: 0,
				world: { x: 0, y: 0, z: 0 },
			});
		}
	}

	public renderSlice(
		frame: SoftwareFrameView,
		slice: PreparedShadowSlice,
		target: SoftwareShadowRenderTarget,
		casters: readonly DrawPacket[],
		transmitters: readonly DrawPacket[],
	): void {
		for (const packet of casters) {
			Matrix4.multiply(slice.viewProjection, packet.worldMatrix, this._mvpMatrix);
			const inverse = Matrix4.inverse3x3(packet.worldMatrix);
			if (!inverse) continue;
			Matrix4.transformNormal(inverse, slice.lightDirection, this._lightDirModel);
			for (const face of Projector.getPacketFacesWithFrame(packet, frame)) {
				const dot = Vector3.dot(
					face.normal ?? Vector3.calculateNormal(face.vertices),
					this._lightDirModel,
				);
				if (!packet.material.doubleSided && dot > 0) continue;
				const projected = this._projectFace(face.vertices, slice.resolution);
				if (projected) {
					this._rasterizer.drawDepthTriangle(projected, target, packet.material);
				}
			}
		}

		for (const packet of transmitters) {
			Matrix4.multiply(slice.viewProjection, packet.worldMatrix, this._mvpMatrix);
			for (const face of Projector.getPacketFacesWithFrame(packet, frame)) {
				const projected = this._projectFace(face.vertices, slice.resolution);
				if (!projected) continue;
				this._rasterizer.drawTransmissionTriangle(
					projected,
					{
						...face,
						projected,
						center: packet.worldBounds.center,
						depthInfo: { min: 0, max: 0, avg: 0 },
					},
					target,
				);
			}
		}
	}

	private _allocClipVertex(
		x: number,
		y: number,
		z: number,
		w: number,
		uCoord: number = 0,
		vCoord: number = 0,
	): ClipVertex {
		let clipVert = this._clipVertsPool[this._clipPoolCursor];
		if (!clipVert) {
			clipVert = { x: 0, y: 0, z: 0, w: 0, u: 0, v: 0 };
			this._clipVertsPool.push(clipVert);
		}

		clipVert.x = x;
		clipVert.y = y;
		clipVert.z = z;
		clipVert.w = w;
		clipVert.u = uCoord;
		clipVert.v = vCoord;
		this._clipPoolCursor++;
		return clipVert;
	}

	private _clipDistance(vertex: ClipVertex, plane: number): number {
		switch (plane) {
			case SoftwareShadowConstants.CLIP_PLANE_MIN_W:
				return vertex.w - SoftwareShadowConstants.MIN_CLIP_W;
			case SoftwareShadowConstants.CLIP_PLANE_LEFT:
				return vertex.x + vertex.w;
			case SoftwareShadowConstants.CLIP_PLANE_RIGHT:
				return -vertex.x + vertex.w;
			case SoftwareShadowConstants.CLIP_PLANE_BOTTOM:
				return vertex.y + vertex.w;
			case SoftwareShadowConstants.CLIP_PLANE_TOP:
				return -vertex.y + vertex.w;
			case SoftwareShadowConstants.CLIP_PLANE_NEAR:
				return vertex.z + vertex.w;
			case SoftwareShadowConstants.CLIP_PLANE_FAR:
				return -vertex.z + vertex.w;
			default:
				return -1;
		}
	}

	private _clipAgainstPlane(input: ClipVertex[], output: ClipVertex[], plane: number): void {
		output.length = 0;
		if (input.length === 0) return;

		let previous = input[input.length - 1];
		let previousDistance = this._clipDistance(previous, plane);
		let previousInside = previousDistance >= 0;

		for (let i = 0; i < input.length; i++) {
			const current = input[i];
			const currentDistance = this._clipDistance(current, plane);
			const currentInside = currentDistance >= 0;

			if (currentInside !== previousInside) {
				const denominator = previousDistance - currentDistance;
				const t =
					Math.abs(denominator) > SoftwareShadowConstants.CLIP_EPSILON
						? previousDistance / denominator
						: 0;
				output.push(
					this._allocClipVertex(
						previous.x + (current.x - previous.x) * t,
						previous.y + (current.y - previous.y) * t,
						previous.z + (current.z - previous.z) * t,
						previous.w + (current.w - previous.w) * t,
						previous.u + (current.u - previous.u) * t,
						previous.v + (current.v - previous.v) * t,
					),
				);
			}

			if (currentInside) {
				output.push(
					this._allocClipVertex(
						current.x,
						current.y,
						current.z,
						current.w,
						current.u,
						current.v,
					),
				);
			}

			previous = current;
			previousDistance = currentDistance;
			previousInside = currentInside;
		}
	}

	private _clipToLightFrustum(input: ClipVertex[], count: number): ClipVertex[] {
		this._clipPoolCursor = 0;
		this._clipScratchA.length = 0;
		this._clipScratchB.length = 0;

		for (let i = 0; i < count; i++) {
			const vertex = input[i];
			this._clipScratchA.push(
				this._allocClipVertex(vertex.x, vertex.y, vertex.z, vertex.w, vertex.u, vertex.v),
			);
		}

		let inPolygon = this._clipScratchA;
		let outPolygon = this._clipScratchB;

		for (let plane = 0; plane < SoftwareShadowConstants.CLIP_PLANE_COUNT; plane++) {
			this._clipAgainstPlane(inPolygon, outPolygon, plane);
			if (outPolygon.length < 3) return outPolygon;
			const temp = inPolygon;
			inPolygon = outPolygon;
			outPolygon = temp;
		}

		return inPolygon;
	}

	private _projectFace(vertices: IVertex[], shadowMapSize: number): ProjectedVertex[] | null {
		const count = vertices.length;

		while (this._projectedVertsPool.length < count) {
			this._projectedVertsPool.push({
				x: 0,
				y: 0,
				z: 0,
				w: 0,
				world: { x: 0, y: 0, z: 0 },
			});
		}
		while (this._clipInputPool.length < count) {
			this._clipInputPool.push({
				x: 0,
				y: 0,
				z: 0,
				w: 0,
				u: 0,
				v: 0,
			});
		}

		let allInside = true;
		let initialOutCodes = -1;

		for (let i = 0; i < count; i++) {
			const vertex = vertices[i];
			const projected = Matrix4.transformPoint(this._mvpMatrix, vertex);
			const clipVertex = this._clipInputPool[i];
			clipVertex.x = projected.x;
			clipVertex.y = projected.y;
			clipVertex.z = projected.z;
			clipVertex.w = projected.w;
			clipVertex.u = vertex.u ?? 0;
			clipVertex.v = vertex.v ?? 0;

			let code = 0;
			if (clipVertex.w < SoftwareShadowConstants.MIN_CLIP_W) code |= 1;
			if (clipVertex.x < -clipVertex.w) code |= 2;
			if (clipVertex.x > clipVertex.w) code |= 4;
			if (clipVertex.y < -clipVertex.w) code |= 8;
			if (clipVertex.y > clipVertex.w) code |= 16;
			if (clipVertex.z < -clipVertex.w) code |= 32;
			if (clipVertex.z > clipVertex.w) code |= 64;

			if (code !== 0) allInside = false;
			if (initialOutCodes === -1) {
				initialOutCodes = code;
			} else {
				initialOutCodes &= code;
			}
		}

		if (initialOutCodes !== 0) return null;

		let clippedVertices: ClipVertex[];
		let clippedCount: number;

		if (allInside) {
			clippedVertices = this._clipInputPool;
			clippedCount = count;
		} else {
			const result = this._clipToLightFrustum(this._clipInputPool, count);
			clippedVertices = result;
			clippedCount = result.length;
			if (clippedCount < 3) return null;
		}

		while (this._projectedVertsPool.length < clippedCount) {
			this._projectedVertsPool.push({
				x: 0,
				y: 0,
				z: 0,
				w: 0,
				world: { x: 0, y: 0, z: 0 },
			});
		}

		const activeVertices = this._projectedVertsPool;
		const projectedView = this._projectedVertsView;
		for (let i = 0; i < clippedCount; i++) {
			const clipVertex = clippedVertices[i];
			const outputVertex = activeVertices[i];
			const invW = 1 / clipVertex.w;
			outputVertex.x = (clipVertex.x * invW * 0.5 + 0.5) * shadowMapSize;
			outputVertex.y = (0.5 - clipVertex.y * invW * 0.5) * shadowMapSize;
			outputVertex.z = clipVertex.z * invW;
			outputVertex.w = invW;
			outputVertex.u = clipVertex.u;
			outputVertex.v = clipVertex.v;
			projectedView[i] = outputVertex;
		}

		projectedView.length = clippedCount;
		return projectedView;
	}
}
