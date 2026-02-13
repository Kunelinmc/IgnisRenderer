import { Matrix4 } from "../maths/Matrix4";
import { Vector3 } from "../maths/Vector3";
import { Projector } from "./Projector";
import { isShadowCastingLight } from "../lights";
import type { Renderer } from "./Renderer";
import type { ProjectedVertex } from "./types";
import { ShadowMap } from "../utils/ShadowMapping";
import { ShadowConstants } from "./Constants";

interface ClipVertex {
	x: number;
	y: number;
	z: number;
	w: number;
}

export class ShadowRenderer {
	private _renderer: Renderer;

	// Cache matrices to avoid allocation per frame/model
	private _mvpMatrix = Matrix4.identity();
	private _normalMatrix = Matrix4.identity();
	private _tempVec3 = new Vector3();

	// Vertex pool to reduce object creation during projection
	private _projectedVertsPool: ProjectedVertex[] = [];
	private _clipInputPool: ClipVertex[] = [];
	private _clipVertsPool: ClipVertex[] = [];
	private _clipPoolCursor = 0;
	private _clipScratchA: ClipVertex[] = [];
	private _clipScratchB: ClipVertex[] = [];

	constructor(renderer: Renderer) {
		this._renderer = renderer;
		// Initialize pool with some capacity
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

	private _allocClipVertex(
		x: number,
		y: number,
		z: number,
		w: number
	): ClipVertex {
		let v = this._clipVertsPool[this._clipPoolCursor];
		if (!v) {
			v = { x: 0, y: 0, z: 0, w: 0 };
			this._clipVertsPool.push(v);
		}

		v.x = x;
		v.y = y;
		v.z = z;
		v.w = w;
		this._clipPoolCursor++;
		return v;
	}

	private _clipDistance(v: ClipVertex, plane: number): number {
		switch (plane) {
			case ShadowConstants.CLIP_PLANE_MIN_W:
				return v.w - ShadowConstants.MIN_CLIP_W;
			case ShadowConstants.CLIP_PLANE_LEFT:
				return v.x + v.w;
			case ShadowConstants.CLIP_PLANE_RIGHT:
				return -v.x + v.w;
			case ShadowConstants.CLIP_PLANE_BOTTOM:
				return v.y + v.w;
			case ShadowConstants.CLIP_PLANE_TOP:
				return -v.y + v.w;
			case ShadowConstants.CLIP_PLANE_NEAR:
				return v.z + v.w;
			case ShadowConstants.CLIP_PLANE_FAR:
				return -v.z + v.w;
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

		let prev = input[input.length - 1];
		let prevDist = this._clipDistance(prev, plane);
		let prevInside = prevDist >= 0;

		for (let i = 0; i < input.length; i++) {
			const curr = input[i];
			const currDist = this._clipDistance(curr, plane);
			const currInside = currDist >= 0;

			if (currInside !== prevInside) {
				const denom = prevDist - currDist;
				const t =
					Math.abs(denom) > ShadowConstants.CLIP_EPSILON ? prevDist / denom : 0;
				output.push(
					this._allocClipVertex(
						prev.x + (curr.x - prev.x) * t,
						prev.y + (curr.y - prev.y) * t,
						prev.z + (curr.z - prev.z) * t,
						prev.w + (curr.w - prev.w) * t
					)
				);
			}

			if (currInside) {
				output.push(this._allocClipVertex(curr.x, curr.y, curr.z, curr.w));
			}

			prev = curr;
			prevDist = currDist;
			prevInside = currInside;
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
			const v = input[i];
			this._clipScratchA.push(this._allocClipVertex(v.x, v.y, v.z, v.w));
		}

		let inPoly = this._clipScratchA;
		let outPoly = this._clipScratchB;

		for (let plane = 0; plane < ShadowConstants.CLIP_PLANE_COUNT; plane++) {
			this._clipAgainstPlane(inPoly, outPoly, plane);
			if (outPoly.length < 3) {
				return outPoly;
			}

			const tmp = inPoly;
			inPoly = outPoly;
			outPoly = tmp;
		}

		return inPoly;
	}

	/**
	 * Renders the shadow pass for the scene.
	 */
	public render(): void {
		const renderer = this._renderer;
		if (!renderer.params.enableShadows) return;

		const shadowLights = renderer.scene.lights.filter(isShadowCastingLight);
		if (shadowLights.length === 0) {
			renderer.shadowMaps.clear();
			return;
		}

		// Clean up shadow maps for lights that no longer exist
		for (const [light] of renderer.shadowMaps) {
			if (!shadowLights.includes(light)) {
				renderer.shadowMaps.delete(light);
			}
		}

		const bounds = renderer.scene.getBounds();
		const worldMatrix = renderer.params.worldMatrix;

		for (const shadowLight of shadowLights) {
			let shadowMap = renderer.shadowMaps.get(shadowLight);
			if (!shadowMap) {
				shadowMap = new ShadowMap();
				renderer.shadowMaps.set(shadowLight, shadowMap);
			}

			shadowMap.setLightCamera(shadowLight, bounds, worldMatrix);
			shadowMap.clear();

			const vpMatrix = shadowMap.viewProjectionMatrix;
			if (!vpMatrix) continue;

			const lightDir = shadowMap.latestLightDir;
			const shadowMapSize = shadowMap.size;

			for (const model of renderer.scene.models) {
				const modelMatrix = Projector.getModelMatrix(model);

				// MVP = VP * Model
				Matrix4.multiply(vpMatrix, modelMatrix, this._mvpMatrix);
				Matrix4.normalMatrix(modelMatrix, this._normalMatrix);

				for (const face of model.faces) {
					const worldNormal =
						face.normal ?? Vector3.calculateNormal(face.vertices);

					Matrix4.transformNormal(
						this._normalMatrix,
						worldNormal,
						this._tempVec3
					).normalize();

					const dot = Vector3.dot(this._tempVec3, lightDir);
					const material = face.material;
					const isDoubleSided = material?.doubleSided;

					if (!isDoubleSided && dot > 0) continue;

					// Ensure pool is large enough
					while (this._projectedVertsPool.length < face.vertices.length) {
						this._projectedVertsPool.push({
							x: 0,
							y: 0,
							z: 0,
							w: 0,
							world: { x: 0, y: 0, z: 0 },
						});
					}
					while (this._clipInputPool.length < face.vertices.length) {
						this._clipInputPool.push({
							x: 0,
							y: 0,
							z: 0,
							w: 0,
						});
					}

					const count = face.vertices.length;
					for (let i = 0; i < count; i++) {
						const v = face.vertices[i];
						const p = Matrix4.transformPoint(this._mvpMatrix, v);
						const clipV = this._clipInputPool[i];
						clipV.x = p.x;
						clipV.y = p.y;
						clipV.z = p.z;
						clipV.w = p.w;
					}

					const clippedVerts = this._clipToLightFrustum(
						this._clipInputPool,
						count
					);
					const clippedCount = clippedVerts.length;
					if (clippedCount < 3) continue;

					while (this._projectedVertsPool.length < clippedCount) {
						this._projectedVertsPool.push({
							x: 0,
							y: 0,
							z: 0,
							w: 0,
							world: { x: 0, y: 0, z: 0 },
						});
					}

					const activeVerts = this._projectedVertsPool;
					for (let i = 0; i < clippedCount; i++) {
						const clipV = clippedVerts[i];
						const outV = activeVerts[i];
						const invW = 1 / clipV.w;
						outV.x = (clipV.x * invW * 0.5 + 0.5) * shadowMapSize;
						outV.y = (0.5 - clipV.y * invW * 0.5) * shadowMapSize;
						outV.z = clipV.z * invW;
						outV.w = invW;
					}

					for (let i = 1; i < clippedCount - 1; i++) {
						renderer.rasterizer.drawDepthTriangle(
							[activeVerts[0], activeVerts[i], activeVerts[i + 1]],
							shadowMap
						);
					}
				}
			}
		}
	}
}
