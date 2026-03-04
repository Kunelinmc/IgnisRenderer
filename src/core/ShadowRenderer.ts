import { Vector3 } from "../maths/Vector3";
import { Matrix4 } from "../maths/Matrix4";
import { isShadowCastingLight } from "../lights";
import { ShadowMap } from "../utils/ShadowMapping";
import { ShadowConstants } from "./constants";
import { Projector } from "./software/Projector";
import type { Renderer } from "./Renderer";
import type { IVertex, ProjectedVertex } from "./types";
import type { Rasterizer } from "./software/Rasterizer";
import type {
	FrameContext,
	PreparedScene,
	ResolvedFeatureState,
} from "./pipeline/types";

interface ClipVertex {
	x: number;
	y: number;
	z: number;
	w: number;
	u: number;
	v: number;
}

export class ShadowRenderer {
	private _renderer: Renderer;
	private _rasterizer: Rasterizer;
	private _mvpMatrix = Matrix4.identity();
	private _lightDirModel = new Vector3();
	private _projectedVertsPool: ProjectedVertex[] = [];
	private _clipInputPool: ClipVertex[] = [];
	private _clipVertsPool: ClipVertex[] = [];
	private _clipPoolCursor = 0;
	private _clipScratchA: ClipVertex[] = [];
	private _clipScratchB: ClipVertex[] = [];

	constructor(renderer: Renderer, rasterizer: Rasterizer) {
		this._renderer = renderer;
		this._rasterizer = rasterizer;
		for (let i = 0; i < 4; i++) {
			this._projectedVertsPool.push({
				x: 0,
				y: 0,
				z: 0,
				w: 0,
				world: { x: 0, y: 0, z: 0 },
			});
		}
	}

	public render(context: FrameContext): void {
		const features = context.features;
		if (!features.enableShadows) return;

		const frame = context.scene;
		const shadowMaps = context.shadowMaps;
		const shadowLights = frame.lights.filter(isShadowCastingLight);

		if (shadowLights.length === 0) {
			shadowMaps.clear();
			return;
		}

		for (const [light] of shadowMaps) {
			if (!shadowLights.includes(light)) {
				shadowMaps.delete(light);
			}
		}

		const worldMatrix = context.worldMatrix;
		for (const shadowLight of shadowLights) {
			let shadowMap = shadowMaps.get(shadowLight);
			if (!shadowMap) {
				shadowMap = new ShadowMap();
				shadowMaps.set(shadowLight, shadowMap);
			}

			shadowMap.setLightCamera(shadowLight, frame.sceneBounds, worldMatrix);
			shadowMap.clear();

			const vpMatrix = shadowMap.viewProjectionMatrix;
			if (!vpMatrix) continue;

			const lightDir = shadowMap.latestLightDir;
			const shadowMapSize = shadowMap.size;

			for (const packet of frame.shadowCasterPackets) {
				Matrix4.multiply(vpMatrix, packet.worldMatrix, this._mvpMatrix);
				const inv3x3 = Matrix4.inverse3x3(packet.worldMatrix);
				if (!inv3x3) continue;

				Matrix4.transformNormal(inv3x3, lightDir, this._lightDirModel);

				for (const face of Projector.getPacketFaces(packet)) {
					const dot = Vector3.dot(
						face.normal ?? Vector3.calculateNormal(face.vertices),
						this._lightDirModel
					);
					if (!packet.material.doubleSided && dot > 0) continue;

					const projected = this._projectFace(face.vertices, shadowMapSize);
					if (!projected) continue;

					this._rasterizer.drawDepthTriangle(
						projected,
						shadowMap,
						packet.material
					);
				}
			}

			for (const packet of frame.shadowTransmitterPackets) {
				Matrix4.multiply(vpMatrix, packet.worldMatrix, this._mvpMatrix);

				for (const face of Projector.getPacketFaces(packet)) {
					const projected = this._projectFace(face.vertices, shadowMapSize);
					if (!projected) continue;

					this._rasterizer.drawTransmissionTriangle(
						projected,
						{
							...face,
							projected,
							center: packet.worldBounds.center,
							depthInfo: { min: 0, max: 0, avg: 0 },
						},
						shadowMap
					);
				}
			}
		}
	}

	private _allocClipVertex(
		x: number,
		y: number,
		z: number,
		w: number,
		uCoord: number = 0,
		vCoord: number = 0
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
			case ShadowConstants.CLIP_PLANE_MIN_W:
				return vertex.w - ShadowConstants.MIN_CLIP_W;
			case ShadowConstants.CLIP_PLANE_LEFT:
				return vertex.x + vertex.w;
			case ShadowConstants.CLIP_PLANE_RIGHT:
				return -vertex.x + vertex.w;
			case ShadowConstants.CLIP_PLANE_BOTTOM:
				return vertex.y + vertex.w;
			case ShadowConstants.CLIP_PLANE_TOP:
				return -vertex.y + vertex.w;
			case ShadowConstants.CLIP_PLANE_NEAR:
				return vertex.z + vertex.w;
			case ShadowConstants.CLIP_PLANE_FAR:
				return -vertex.z + vertex.w;
			default:
				return -1;
		}
	}

	private _clipAgainstPlane(
		input: ClipVertex[],
		output: ClipVertex[],
		plane: number
	): void {
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
					Math.abs(denominator) > ShadowConstants.CLIP_EPSILON
						? previousDistance / denominator
						: 0;
				output.push(
					this._allocClipVertex(
						previous.x + (current.x - previous.x) * t,
						previous.y + (current.y - previous.y) * t,
						previous.z + (current.z - previous.z) * t,
						previous.w + (current.w - previous.w) * t,
						previous.u + (current.u - previous.u) * t,
						previous.v + (current.v - previous.v) * t
					)
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
						current.v
					)
				);
			}

			previous = current;
			previousDistance = currentDistance;
			previousInside = currentInside;
		}
	}

	private _clipToLightFrustum(
		input: ClipVertex[],
		count: number
	): ClipVertex[] {
		this._clipPoolCursor = 0;
		this._clipScratchA.length = 0;
		this._clipScratchB.length = 0;

		for (let i = 0; i < count; i++) {
			const vertex = input[i];
			this._clipScratchA.push(
				this._allocClipVertex(
					vertex.x,
					vertex.y,
					vertex.z,
					vertex.w,
					vertex.u,
					vertex.v
				)
			);
		}

		let inPolygon = this._clipScratchA;
		let outPolygon = this._clipScratchB;

		for (let plane = 0; plane < ShadowConstants.CLIP_PLANE_COUNT; plane++) {
			this._clipAgainstPlane(inPolygon, outPolygon, plane);
			if (outPolygon.length < 3) return outPolygon;
			const temp = inPolygon;
			inPolygon = outPolygon;
			outPolygon = temp;
		}

		return inPolygon;
	}

	private _projectFace(
		vertices: IVertex[],
		shadowMapSize: number
	): ProjectedVertex[] | null {
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
			if (clipVertex.w < ShadowConstants.MIN_CLIP_W) code |= 1;
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
		}

		return activeVertices.slice(0, clippedCount);
	}
}
