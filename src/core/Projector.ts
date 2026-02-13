import { Matrix4 } from "../maths/Matrix4";
import { Vector3 } from "../maths/Vector3";
import type { IModel, IVertex, ProjectedVertex, ProjectedFace } from "./types";
import type { Renderer } from "./Renderer";

export class Projector {
	/**
	 * Projects a model's faces into screen space, including clipping and culling.
	 * @param model - The model to project.
	 * @param renderer - The renderer instance (for camera and canvas info).
	 */
	public static projectModel(
		model: IModel,
		renderer: Renderer,
		flipCulling: boolean = false,
		overrideSize?: { width: number; height: number }
	): void {
		const targetWidth = overrideSize?.width ?? renderer.canvas.width;
		const targetHeight = overrideSize?.height ?? renderer.canvas.height;

		const modelMatrix = Projector.getModelMatrix(model);
		const viewMatrix = renderer.camera.viewMatrix;
		const projectionMatrix = renderer.camera.projectionMatrix;

		// Compute normal matrix for correct normal transformation with non-uniform scaling
		const normalMat = Matrix4.normalMatrix(modelMatrix);

		const worldCenter = Matrix4.transformPoint(
			modelMatrix,
			model.boundingSphere.center
		);
		const projectedFaces: ProjectedFace[] = [];
		const interpolateUV = (
			a: number | undefined,
			b: number | undefined,
			t: number
		): number | undefined => {
			if (a === undefined && b === undefined) return undefined;
			const from = a ?? b ?? 0;
			const to = b ?? a ?? 0;
			return from + (to - from) * t;
		};

		const interpolateNormal = (
			a: IVertex["normal"],
			b: IVertex["normal"],
			t: number
		): IVertex["normal"] => {
			if (a && b) {
				return Vector3.normalize({
					x: a.x + (b.x - a.x) * t,
					y: a.y + (b.y - a.y) * t,
					z: a.z + (b.z - a.z) * t,
				});
			}
			return a || b || null;
		};

		for (const face of model.faces) {
			const worldVerts: IVertex[] = [];
			const viewVerts: IVertex[] = [];
			const faceVertices = face.vertices;
			for (let i = 0; i < faceVertices.length; i++) {
				const v = faceVertices[i];
				const pWorld = Matrix4.transformPoint(modelMatrix, v);
				const nWorld =
					v.normal ?
						Vector3.normalize(Matrix4.transformNormal(normalMat, v.normal))
					:	null;
				const worldV: IVertex = {
					x: pWorld.x,
					y: pWorld.y,
					z: pWorld.z,
					u: v.u,
					v: v.v,
					normal: nWorld,
				};
				worldVerts.push(worldV);

				const pView = Matrix4.transformPoint(viewMatrix, worldV);
				viewVerts.push({
					x: pView.x,
					y: pView.y,
					z: pView.z,
					u: v.u,
					v: v.v,
					normal: nWorld,
				});
			}

			// 1. Near plane clipping (in View Space)
			const nearZ = -renderer.camera.near;
			let clippedVerts: { view: IVertex; world: IVertex }[] = [];

			for (let i = 0; i < viewVerts.length; i++) {
				const v1 = viewVerts[i];
				const w1 = worldVerts[i];
				const nextIdx = (i + 1) % viewVerts.length;
				const v2 = viewVerts[nextIdx];
				const w2 = worldVerts[nextIdx];

				const in1 = v1.z <= nearZ;
				const in2 = v2.z <= nearZ;

				if (in1) {
					if (in2) {
						clippedVerts.push({ view: v2, world: w2 });
					} else {
						const t = (nearZ - v1.z) / (v2.z - v1.z);
						clippedVerts.push({
							view: {
								x: v1.x + (v2.x - v1.x) * t,
								y: v1.y + (v2.y - v1.y) * t,
								z: nearZ,
								u: interpolateUV(v1.u, v2.u, t),
								v: interpolateUV(v1.v, v2.v, t),
								normal: interpolateNormal(v1.normal, v2.normal, t),
							},
							world: {
								x: w1.x + (w2.x - w1.x) * t,
								y: w1.y + (w2.y - w1.y) * t,
								z: w1.z + (w2.z - w1.z) * t,
								u: interpolateUV(w1.u, w2.u, t),
								v: interpolateUV(w1.v, w2.v, t),
								normal: interpolateNormal(w1.normal, w2.normal, t),
							},
						});
					}
				} else if (in2) {
					const t = (nearZ - v1.z) / (v2.z - v1.z);
					clippedVerts.push({
						view: {
							x: v1.x + (v2.x - v1.x) * t,
							y: v1.y + (v2.y - v1.y) * t,
							z: nearZ,
							u: interpolateUV(v1.u, v2.u, t),
							v: interpolateUV(v1.v, v2.v, t),
							normal: interpolateNormal(v1.normal, v2.normal, t),
						},
						world: {
							x: w1.x + (w2.x - w1.x) * t,
							y: w1.y + (w2.y - w1.y) * t,
							z: w1.z + (w2.z - w1.z) * t,
							u: interpolateUV(w1.u, w2.u, t),
							v: interpolateUV(w1.v, w2.v, t),
							normal: interpolateNormal(w1.normal, w2.normal, t),
						},
					});
					clippedVerts.push({ view: v2, world: w2 });
				}
			}

			if (clippedVerts.length < 3) continue;

			// 2. Backface culling in camera space (Corrected for perspective)
			const cullNormal = Vector3.calculateNormal(
				clippedVerts.map((v) => v.view)
			);

			const v0 = clippedVerts[0].view;
			const dot =
				cullNormal.x * v0.x + cullNormal.y * v0.y + cullNormal.z * v0.z;
			const material = face.material;
			const isDoubleSided = material?.doubleSided || face.doubleSided;

			if (!isDoubleSided) {
				if (flipCulling ? dot < 0 : dot > 0) continue;
			}

			// 3. Project remaining clipped vertices
			const projectedVerts: ProjectedVertex[] = [];
			for (const v of clippedVerts) {
				const p = Matrix4.transformPoint(projectionMatrix, v.view);
				const w = p.w || 1e-6;
				const ndcX = p.x / w;
				const ndcY = p.y / w;
				const ndcZ = p.z / w;

				projectedVerts.push({
					x: (ndcX * 0.5 + 0.5) * targetWidth,
					y: (0.5 - ndcY * 0.5) * targetHeight,
					z: ndcZ,
					w: 1 / w,
					u: v.view.u,
					v: v.view.v,
					normal: v.view.normal,
					world: v.world,
				});
			}

			let minDepth = Infinity;
			let maxDepth = -Infinity;
			let sumDepth = 0;
			const cvLen = clippedVerts.length;
			for (let i = 0; i < cvLen; i++) {
				const d = -clippedVerts[i].view.z;
				if (d < minDepth) minDepth = d;
				if (d > maxDepth) maxDepth = d;
				sumDepth += d;
			}

			const faceDepthInfo = {
				min: minDepth,
				max: maxDepth,
				avg: sumDepth / cvLen,
			};

			// Face center in world space
			let center = { x: 0, y: 0, z: 0 };
			for (let i = 0; i < cvLen; i++) {
				const w = clippedVerts[i].world;
				center.x += w.x;
				center.y += w.y;
				center.z += w.z;
			}
			center.x /= cvLen;
			center.y /= cvLen;
			center.z /= cvLen;

			const transformedFaceNormal =
				face.normal ?
					Vector3.normalize(Matrix4.transformNormal(normalMat, face.normal))
				:	Vector3.calculateNormal(worldVerts);

			const projectedFace: ProjectedFace = {
				...face,
				material,
				projected: projectedVerts,
				center,
				normal: transformedFaceNormal,
				depthInfo: faceDepthInfo,
				modelDepth: -Matrix4.transformPoint(viewMatrix, worldCenter).z,
			};
			projectedFaces.push(projectedFace);
		}

		model.projectedFaces = projectedFaces;
	}

	/**
	 * Computes the world matrix for a model.
	 * @param model
	 * @returns {number[][]}
	 */
	public static getModelMatrix(model: IModel): Matrix4 {
		const transform = model.transform;
		const scaleMatrix = [
			[transform.scale.x, 0, 0, 0],
			[0, transform.scale.y, 0, 0],
			[0, 0, transform.scale.z, 0],
			[0, 0, 0, 1],
		];
		const rotationMatrix = Matrix4.rotationFromEuler(
			transform.rotation.x,
			transform.rotation.y,
			transform.rotation.z
		);
		const translationMatrix = [
			[1, 0, 0, transform.position.x],
			[0, 1, 0, transform.position.y],
			[0, 0, 1, transform.position.z],
			[0, 0, 0, 1],
		];
		return Matrix4.multiply(
			new Matrix4(translationMatrix),
			Matrix4.multiply(rotationMatrix, new Matrix4(scaleMatrix))
		);
	}
}
