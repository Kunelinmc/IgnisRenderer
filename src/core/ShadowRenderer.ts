import { Matrix4 } from "../maths/Matrix4";
import { Vector3 } from "../maths/Vector3";
import { Projector } from "./Projector";
import { isShadowCastingLight } from "../lights";
import { ShadowMap } from "../utils/ShadowMapping";
import { ShadowConstants } from "./Constants";
import type { Renderer } from "./Renderer";
import type { ProjectedVertex } from "./types";

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

	private _mvpMatrix = Matrix4.identity();
	private _lightDirModel = new Vector3();

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
						prev.w + (curr.w - prev.w) * t,
						prev.u + (curr.u - prev.u) * t,
						prev.v + (curr.v - prev.v) * t
					)
				);
			}

			if (currInside) {
				output.push(
					this._allocClipVertex(curr.x, curr.y, curr.z, curr.w, curr.u, curr.v)
				);
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
			this._clipScratchA.push(
				this._allocClipVertex(v.x, v.y, v.z, v.w, v.u, v.v)
			);
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

			// Pass 1: Opaque objects (Depth Map)
			for (const model of renderer.scene.models) {
				// Optimization 1: Cull model against light frustum
				if (!this._isModelInFrustum(model, vpMatrix)) continue;

				const modelMatrix = Projector.getModelMatrix(model);
				Matrix4.multiply(vpMatrix, modelMatrix, this._mvpMatrix);

				// Optimization 2: Model-space lighting
				// Get model-space light direction to avoid transforming normals
				const inv3x3 = Matrix4.inverse3x3(modelMatrix);
				if (!inv3x3) continue;

				// L_model = transformNormal(transpose(normalMatrix), L_world)
				// transpose(normalMatrix) = transpose(transpose(inv3x3)) = inv3x3
				Matrix4.transformNormal(inv3x3, lightDir, this._lightDirModel);

				for (const face of model.faces) {
					const material = face.material;
					const alphaMode = material?.alphaMode;
					if (alphaMode === "BLEND") continue;

					// Cache face normal
					if (!face.normal) {
						face.normal = Vector3.calculateNormal(face.vertices);
					}

					const dot = Vector3.dot(face.normal, this._lightDirModel);
					const isDoubleSided = material?.doubleSided;

					if (!isDoubleSided && dot > 0) continue;

					const projected = this._projectFace(face, shadowMapSize);
					if (!projected) continue;

					for (let i = 1; i < projected.length - 1; i++) {
						renderer.rasterizer.drawDepthTriangle(
							[projected[0], projected[i], projected[i + 1]],
							shadowMap,
							material
						);
					}
				}
			}

			// Pass 2: Transparent objects (Transmission Map/Colored Shadows)
			for (const model of renderer.scene.models) {
				if (!this._isModelInFrustum(model, vpMatrix)) continue;

				const modelMatrix = Projector.getModelMatrix(model);
				Matrix4.multiply(vpMatrix, modelMatrix, this._mvpMatrix);

				for (const face of model.faces) {
					const material = face.material;
					if (material?.alphaMode !== "BLEND") continue;

					const projected = this._projectFace(face, shadowMapSize);
					if (!projected) continue;

					for (let i = 1; i < projected.length - 1; i++) {
						renderer.rasterizer.drawTransmissionTriangle(
							[projected[0], projected[i], projected[i + 1]],
							face as any,
							shadowMap
						);
					}
				}
			}
		}
	}

	private _isModelInFrustum(model: any, vpMatrix: Matrix4): boolean {
		if (!model.boundingBox) return true;

		// Simple AABB vs Frustum check using clip codes for the 8 corners
		const box = model.boundingBox;
		const corners = [
			{ x: box.min.x, y: box.min.y, z: box.min.z },
			{ x: box.max.x, y: box.min.y, z: box.min.z },
			{ x: box.min.x, y: box.max.y, z: box.min.z },
			{ x: box.max.x, y: box.max.y, z: box.min.z },
			{ x: box.min.x, y: box.min.y, z: box.max.z },
			{ x: box.max.x, y: box.min.y, z: box.max.z },
			{ x: box.min.x, y: box.max.y, z: box.max.z },
			{ x: box.max.x, y: box.max.y, z: box.max.z },
		];

		let initialOutCodes = -1;
		for (const corner of corners) {
			const p = Matrix4.transformPoint(vpMatrix, corner);
			let code = 0;
			if (p.w < ShadowConstants.MIN_CLIP_W) code |= 1;
			if (p.x < -p.w) code |= 2;
			if (p.x > p.w) code |= 4;
			if (p.y < -p.w) code |= 8;
			if (p.y > p.w) code |= 16;
			if (p.z < -p.w) code |= 32;
			if (p.z > p.w) code |= 64;

			if (code === 0) return true; // At least one corner is inside
			initialOutCodes &= code;
		}

		// If initialOutCodes is non-zero, all corners are on the outside of at least one common plane
		return initialOutCodes === 0;
	}

	private _projectFace(
		face: any,
		shadowMapSize: number
	): ProjectedVertex[] | null {
		const count = face.vertices.length;

		// Ensure pool is large enough
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
		let combinedOutCodes = 0;
		let initialOutCodes = -1; // All ones

		for (let i = 0; i < count; i++) {
			const v = face.vertices[i];
			const p = Matrix4.transformPoint(this._mvpMatrix, v);
			const clipV = this._clipInputPool[i];
			clipV.x = p.x;
			clipV.y = p.y;
			clipV.z = p.z;
			clipV.w = p.w;
			clipV.u = v.u ?? 0;
			clipV.v = v.v ?? 0;

			// Compute clip codes
			let code = 0;
			if (clipV.w < ShadowConstants.MIN_CLIP_W) code |= 1; // W
			if (clipV.x < -clipV.w) code |= 2; // Left
			if (clipV.x > clipV.w) code |= 4; // Right
			if (clipV.y < -clipV.w) code |= 8; // Bottom
			if (clipV.y > clipV.w) code |= 16; // Top
			if (clipV.z < -clipV.w) code |= 32; // Near
			if (clipV.z > clipV.w) code |= 64; // Far

			if (code !== 0) allInside = false;
			combinedOutCodes |= code;
			if (initialOutCodes === -1) initialOutCodes = code;
			else initialOutCodes &= code;
		}

		// Trivial rejection: all vertices are outside at least one common plane
		if (initialOutCodes !== 0) return null;

		let clippedVerts: ClipVertex[];
		let clippedCount: number;

		if (allInside) {
			// Trivial acceptance: all vertices are inside
			clippedVerts = this._clipInputPool;
			clippedCount = count;
		} else {
			// Need clipping
			const result = this._clipToLightFrustum(this._clipInputPool, count);
			clippedVerts = result;
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

		const activeVerts = this._projectedVertsPool;
		for (let i = 0; i < clippedCount; i++) {
			const clipV = clippedVerts[i];
			const outV = activeVerts[i];
			const invW = 1 / clipV.w;
			outV.x = (clipV.x * invW * 0.5 + 0.5) * shadowMapSize;
			outV.y = (0.5 - clipV.y * invW * 0.5) * shadowMapSize;
			outV.z = clipV.z * invW;
			outV.w = invW;
			outV.u = clipV.u;
			outV.v = clipV.v;
		}

		// return a view of the pool
		return activeVerts.slice(0, clippedCount);
	}
}
