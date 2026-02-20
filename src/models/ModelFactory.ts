import { SimpleModel, ModelFace, ModelVertex } from "./SimpleModel";
import type { IVector3 } from "../maths/types";
import type { Material } from "../materials";

export class ModelFactory {
	/**
	 * Creates a box-shaped model.
	 * @param {IVector3} base - The center position of the box.
	 * @param {number} width - The width of the box along the X-axis.
	 * @param {number} depth - The depth of the box along the Z-axis.
	 * @param {number} height - The height of the box along the Y-axis.
	 * @param {Material} [material=null] - The material to apply to the box.
	 * @returns {SimpleModel} A new SimpleModel instance representing a box.
	 */
	public static createBox(
		base: IVector3,
		width: number,
		depth: number,
		height: number,
		material: Material | null = null
	): SimpleModel {
		const w2 = width / 2;
		const h2 = height / 2;
		const d2 = depth / 2;

		const vertices: IVector3[] = [
			{ x: -w2, y: -h2, z: -d2 },
			{ x: w2, y: -h2, z: -d2 },
			{ x: w2, y: -h2, z: d2 },
			{ x: -w2, y: -h2, z: d2 },
			{ x: -w2, y: h2, z: -d2 },
			{ x: w2, y: h2, z: -d2 },
			{ x: w2, y: h2, z: d2 },
			{ x: -w2, y: h2, z: d2 },
		];

		const faceSpecs = [
			{
				indices: [0, 1, 2, 3],
				normal: { x: 0, y: -1, z: 0 },
				uv: [0, 1, 1, 0, 0, 0, 1, 1],
			}, // Bottom
			{
				indices: [4, 7, 6, 5],
				normal: { x: 0, y: 1, z: 0 },
				uv: [0, 0, 1, 1, 0, 1, 1, 0],
			}, // Top
			{
				indices: [0, 4, 5, 1],
				normal: { x: 0, y: 0, z: -1 },
				uv: [0, 0, 1, 1, 0, 1, 1, 0],
			}, // Front
			{
				indices: [3, 2, 6, 7],
				normal: { x: 0, y: 0, z: 1 },
				uv: [0, 1, 1, 0, 0, 0, 1, 1],
			}, // Back
			{
				indices: [0, 3, 7, 4],
				normal: { x: -1, y: 0, z: 0 },
				uv: [0, 1, 1, 0, 0, 0, 1, 1],
			}, // Left
			{
				indices: [1, 5, 6, 2],
				normal: { x: 1, y: 0, z: 0 },
				uv: [0, 1, 1, 0, 0, 0, 1, 1],
			}, // Right
		];

		const faces: ModelFace[] = faceSpecs.map((spec) => ({
			vertices: spec.indices.map(
				(vIdx, i): ModelVertex => ({
					...vertices[vIdx],
					u: spec.uv[i],
					v: spec.uv[i + 4],
					normal: { ...spec.normal }, // Provide per-vertex normal to prevent smoothing
				})
			),
			normal: spec.normal,
			material,
		}));

		const model = new SimpleModel(faces);
		model.transform.position.copy(base);
		return model;
	}

	/**
	 * Creates a sphere-shaped model.
	 * @param {IVector3} base - The center position of the sphere.
	 * @param {number} radius - The radius of the sphere.
	 * @param {number} [segments=24] - The number of horizontal segments.
	 * @param {number} [rings=12] - The number of vertical rings.
	 * @param {Material} [material=null] - The material to apply to the sphere.
	 * @returns {SimpleModel} A new SimpleModel instance representing a sphere.
	 */
	public static createSphere(
		base: IVector3,
		radius: number,
		segments: number = 24,
		rings: number = 12,
		material: Material | null = null
	): SimpleModel {
		const vertices: IVector3[][] = [];
		for (let r = 0; r <= rings; r++) {
			const phi = (r / rings) * Math.PI;
			const row: IVector3[] = [];
			for (let s = 0; s <= segments; s++) {
				const theta = (s / segments) * Math.PI * 2;

				let x: number, y: number, z: number;
				if (r === 0) {
					x = 0;
					y = radius;
					z = 0;
				} else if (r === rings) {
					x = 0;
					y = -radius;
					z = 0;
				} else {
					x = radius * Math.sin(phi) * Math.cos(theta);
					y = radius * Math.cos(phi);
					z = radius * Math.sin(phi) * Math.sin(theta);
				}

				// Final rounding/clamping to avoid precision cracks
				if (s === segments) {
					// Ensure the seam is exactly closed
					const start = row[0];
					row.push({ ...start });
				} else {
					// Precision clamping
					if (Math.abs(x) < 1e-10) x = 0;
					if (Math.abs(z) < 1e-10) z = 0;
					row.push({ x, y, z });
				}
			}
			vertices.push(row);
		}

		const faces: ModelFace[] = [];
		for (let r = 0; r < rings; r++) {
			for (let s = 0; s < segments; s++) {
				const v1: ModelVertex = {
					...vertices[r][s],
					u: 1 - s / segments,
					v: r / rings,
				};
				const v2: ModelVertex = {
					...vertices[r][s + 1],
					u: 1 - (s + 1) / segments,
					v: r / rings,
				};
				const v3: ModelVertex = {
					...vertices[r + 1][s + 1],
					u: 1 - (s + 1) / segments,
					v: (r + 1) / rings,
				};
				const v4: ModelVertex = {
					...vertices[r + 1][s],
					u: 1 - s / segments,
					v: (r + 1) / rings,
				};

				// At the poles, we only need triangles
				if (r === 0) {
					// Top pole: v1 is the pole, v3 and v4 are on the first ring
					faces.push({
						vertices: [
							{ ...v1, normal: { x: 0, y: 1, z: 0 } },
							{
								...v3,
								normal: {
									x: v3.x / radius,
									y: v3.y / radius,
									z: v3.z / radius,
								},
							},
							{
								...v4,
								normal: {
									x: v4.x / radius,
									y: v4.y / radius,
									z: v4.z / radius,
								},
							},
						],
						normal: { x: 0, y: 1, z: 0 },
						material,
					});
				} else if (r === rings - 1) {
					// Bottom pole: v3 is the pole, v1 and v2 are on the last ring
					faces.push({
						vertices: [
							{
								...v1,
								normal: {
									x: v1.x / radius,
									y: v1.y / radius,
									z: v1.z / radius,
								},
							},
							{
								...v2,
								normal: {
									x: v2.x / radius,
									y: v2.y / radius,
									z: v2.z / radius,
								},
							},
							{ ...v3, normal: { x: 0, y: -1, z: 0 } },
						],
						normal: { x: 0, y: -1, z: 0 },
						material,
					});
				} else {
					// Standard quad (will be split into two triangles by the renderer)
					const avgX = (v1.x + v2.x + v3.x + v4.x) / 4;
					const avgY = (v1.y + v2.y + v3.y + v4.y) / 4;
					const avgZ = (v1.z + v2.z + v3.z + v4.z) / 4;
					const len = Math.hypot(avgX, avgY, avgZ) || 1;

					faces.push({
						vertices: [
							{
								...v1,
								normal: {
									x: v1.x / radius,
									y: v1.y / radius,
									z: v1.z / radius,
								},
							},
							{
								...v2,
								normal: {
									x: v2.x / radius,
									y: v2.y / radius,
									z: v2.z / radius,
								},
							},
							{
								...v3,
								normal: {
									x: v3.x / radius,
									y: v3.y / radius,
									z: v3.z / radius,
								},
							},
							{
								...v4,
								normal: {
									x: v4.x / radius,
									y: v4.y / radius,
									z: v4.z / radius,
								},
							},
						],
						normal: {
							x: avgX / len,
							y: avgY / len,
							z: avgZ / len,
						},
						material,
					});
				}
			}
		}

		const model = new SimpleModel(faces);
		model.transform.position.copy(base);
		return model;
	}

	/**
	 * Creates a cylinder-shaped model.
	 * @param {IVector3} base - The center position of the cylinder.
	 * @param {number} radius - The radius of the cylinder.
	 * @param {number} height - The height of the cylinder.
	 * @param {number} [segments=16] - The number of vertical segments around the circumference.
	 * @param {Material} [material=null] - The material to apply to the cylinder.
	 * @returns {SimpleModel} A new SimpleModel instance representing a cylinder.
	 */
	public static createCylinder(
		base: IVector3,
		radius: number,
		height: number,
		segments: number = 16,
		material: Material | null = null
	): SimpleModel {
		const bottom: IVector3[] = [];
		const top: IVector3[] = [];
		const h2 = height / 2;

		for (let i = 0; i < segments; i++) {
			const theta = (i / segments) * Math.PI * 2;
			bottom.push({
				x: Math.cos(theta) * radius,
				y: -h2,
				z: Math.sin(theta) * radius,
			});
			top.push({
				x: Math.cos(theta) * radius,
				y: h2,
				z: Math.sin(theta) * radius,
			});
		}

		const faces: ModelFace[] = [];
		for (let i = 0; i < segments; i++) {
			const next = (i + 1) % segments;
			const u1 = i / segments;
			const u2 = (i + 1) / segments;

			// Bottom cap
			faces.push({
				vertices: [
					{ ...bottom[i], u: u1, v: 0, normal: { x: 0, y: -1, z: 0 } },
					{ ...bottom[next], u: u2, v: 0, normal: { x: 0, y: -1, z: 0 } },
					{ x: 0, y: -h2, z: 0, u: 0.5, v: 0, normal: { x: 0, y: -1, z: 0 } },
				],
				normal: { x: 0, y: -1, z: 0 },
				material,
			});
			// Top cap
			faces.push({
				vertices: [
					{ ...top[i], u: u1, v: 1, normal: { x: 0, y: 1, z: 0 } },
					{ x: 0, y: h2, z: 0, u: 0.5, v: 1, normal: { x: 0, y: 1, z: 0 } },
					{ ...top[next], u: u2, v: 1, normal: { x: 0, y: 1, z: 0 } },
				],
				normal: { x: 0, y: 1, z: 0 },
				material,
			});
			// Side
			const cos1 = Math.cos((i / segments) * 2 * Math.PI);
			const sin1 = Math.sin((i / segments) * 2 * Math.PI);
			const cos2 = Math.cos(((i + 1) / segments) * 2 * Math.PI);
			const sin2 = Math.sin(((i + 1) / segments) * 2 * Math.PI);

			faces.push({
				vertices: [
					{ ...bottom[i], u: u1, v: 0, normal: { x: cos1, y: 0, z: sin1 } },
					{ ...top[i], u: u1, v: 1, normal: { x: cos1, y: 0, z: sin1 } },
					{ ...top[next], u: u2, v: 1, normal: { x: cos2, y: 0, z: sin2 } },
					{ ...bottom[next], u: u2, v: 0, normal: { x: cos2, y: 0, z: sin2 } },
				],
				normal: {
					x: Math.cos(((i + 0.5) / segments) * 2 * Math.PI),
					y: 0,
					z: Math.sin(((i + 0.5) / segments) * 2 * Math.PI),
				},
				material,
			});
		}

		const model = new SimpleModel(faces);
		model.transform.position.copy(base);
		return model;
	}

	/**
	 * Creates a plane-shaped model.
	 * @param {IVector3} base - The center position of the plane.
	 * @param {number} width - The width of the plane.
	 * @param {number} height - The height of the plane.
	 * @param {Material} [material=null] - The material to apply to the plane.
	 * @returns {SimpleModel} A new SimpleModel instance representing a plane.
	 */
	public static createPlane(
		base: IVector3,
		width: number,
		height: number,
		material: Material | null = null
	): SimpleModel {
		const w2 = width / 2;
		const h2 = height / 2;

		const faces: ModelFace[] = [
			{
				vertices: [
					{
						x: -w2,
						y: 0,
						z: -h2,
						u: 0,
						v: 0,
						normal: { x: 0, y: 1, z: 0 },
					},
					{
						x: w2,
						y: 0,
						z: -h2,
						u: 1,
						v: 0,
						normal: { x: 0, y: 1, z: 0 },
					},
					{
						x: w2,
						y: 0,
						z: h2,
						u: 1,
						v: 1,
						normal: { x: 0, y: 1, z: 0 },
					},
					{
						x: -w2,
						y: 0,
						z: h2,
						u: 0,
						v: 1,
						normal: { x: 0, y: 1, z: 0 },
					},
				],
				normal: { x: 0, y: 1, z: 0 },
				material,
			},
		];
		const model = new SimpleModel(faces);
		model.transform.position.copy(base);
		return model;
	}
	/**
	 * Creates a torus-shaped model.
	 * @param {IVector3} base - The center position of the torus.
	 * @param {number} radius - The radius of the torus (distance from center of torus to center of tube).
	 * @param {number} tubeRadius - The radius of the tube.
	 * @param {number} [radialSegments=16] - The number of segments around the tube.
	 * @param {number} [tubularSegments=32] - The number of segments along the torus.
	 * @param {Material} [material=null] - The material to apply to the torus.
	 * @returns {SimpleModel} A new SimpleModel instance representing a torus.
	 */
	public static createTorus(
		base: IVector3,
		radius: number,
		tubeRadius: number,
		radialSegments: number = 16,
		tubularSegments: number = 32,
		material: Material | null = null
	): SimpleModel {
		const vertices: IVector3[][] = [];
		const normals: IVector3[][] = [];
		const uvs: { u: number; v: number }[][] = [];

		for (let j = 0; j <= radialSegments; j++) {
			vertices.push([]);
			normals.push([]);
			uvs.push([]);
			for (let i = 0; i <= tubularSegments; i++) {
				const u = (i / tubularSegments) * Math.PI * 2;
				const v = (j / radialSegments) * Math.PI * 2;

				const x = (radius + tubeRadius * Math.cos(v)) * Math.cos(u);
				const z = (radius + tubeRadius * Math.cos(v)) * Math.sin(u);
				const y = tubeRadius * Math.sin(v);

				vertices[j].push({ x, y, z });

				const nx = Math.cos(v) * Math.cos(u);
				const nz = Math.cos(v) * Math.sin(u);
				const ny = Math.sin(v);

				normals[j].push({ x: nx, y: ny, z: nz });
				uvs[j].push({ u: i / tubularSegments, v: j / radialSegments });
			}
		}

		const faces: ModelFace[] = [];
		for (let j = 1; j <= radialSegments; j++) {
			for (let i = 1; i <= tubularSegments; i++) {
				const a = i - 1;
				const b = i;
				const c = j - 1;
				const d = j;

				const v1: ModelVertex = {
					...vertices[c][a],
					u: uvs[c][a].u,
					v: uvs[c][a].v,
					normal: normals[c][a],
				};
				const v2: ModelVertex = {
					...vertices[c][b],
					u: uvs[c][b].u,
					v: uvs[c][b].v,
					normal: normals[c][b],
				};
				const v3: ModelVertex = {
					...vertices[d][b],
					u: uvs[d][b].u,
					v: uvs[d][b].v,
					normal: normals[d][b],
				};
				const v4: ModelVertex = {
					...vertices[d][a],
					u: uvs[d][a].u,
					v: uvs[d][a].v,
					normal: normals[d][a],
				};

				// Calculate face normal
				const faceNormal = {
					x: (v1.normal.x + v2.normal.x + v3.normal.x + v4.normal.x) / 4,
					y: (v1.normal.y + v2.normal.y + v3.normal.y + v4.normal.y) / 4,
					z: (v1.normal.z + v2.normal.z + v3.normal.z + v4.normal.z) / 4,
				};
				const len = Math.hypot(faceNormal.x, faceNormal.y, faceNormal.z) || 1;
				faceNormal.x /= len;
				faceNormal.y /= len;
				faceNormal.z /= len;

				faces.push({
					vertices: [v1, v4, v3, v2],
					normal: faceNormal,
					material,
				});
			}
		}

		const model = new SimpleModel(faces);
		model.transform.position.copy(base);
		return model;
	}

	/**
	 * Creates a tube-shaped model (hollow cylinder).
	 * @param {IVector3} base - The center position of the tube.
	 * @param {number} innerRadius - The inner radius of the tube.
	 * @param {number} outerRadius - The outer radius of the tube.
	 * @param {number} height - The height of the tube.
	 * @param {number} [segments=16] - The number of segments around the tube.
	 * @param {Material} [material=null] - The material to apply to the tube.
	 * @returns {SimpleModel} A new SimpleModel instance representing a tube.
	 */
	public static createTube(
		base: IVector3,
		innerRadius: number,
		outerRadius: number,
		height: number,
		segments: number = 16,
		material: Material | null = null
	): SimpleModel {
		const h2 = height / 2;
		const faces: ModelFace[] = [];

		for (let i = 0; i < segments; i++) {
			const u1 = i / segments;
			const u2 = (i + 1) / segments;
			const theta1 = u1 * Math.PI * 2;
			const theta2 = u2 * Math.PI * 2;

			const cos1 = Math.cos(theta1);
			const sin1 = Math.sin(theta1);
			const cos2 = Math.cos(theta2);
			const sin2 = Math.sin(theta2);

			const pOuter1Bottom = {
				x: cos1 * outerRadius,
				y: -h2,
				z: sin1 * outerRadius,
			};
			const pOuter1Top = {
				x: cos1 * outerRadius,
				y: h2,
				z: sin1 * outerRadius,
			};
			const pOuter2Bottom = {
				x: cos2 * outerRadius,
				y: -h2,
				z: sin2 * outerRadius,
			};
			const pOuter2Top = {
				x: cos2 * outerRadius,
				y: h2,
				z: sin2 * outerRadius,
			};

			const pInner1Bottom = {
				x: cos1 * innerRadius,
				y: -h2,
				z: sin1 * innerRadius,
			};
			const pInner1Top = {
				x: cos1 * innerRadius,
				y: h2,
				z: sin1 * innerRadius,
			};
			const pInner2Bottom = {
				x: cos2 * innerRadius,
				y: -h2,
				z: sin2 * innerRadius,
			};
			const pInner2Top = {
				x: cos2 * innerRadius,
				y: h2,
				z: sin2 * innerRadius,
			};

			const nOuter1 = { x: cos1, y: 0, z: sin1 };
			const nOuter2 = { x: cos2, y: 0, z: sin2 };
			const nInner1 = { x: -cos1, y: 0, z: -sin1 };
			const nInner2 = { x: -cos2, y: 0, z: -sin2 };

			const nOuterFace = {
				x: Math.cos((u1 + u2) * Math.PI),
				y: 0,
				z: Math.sin((u1 + u2) * Math.PI),
			};
			const nInnerFace = { x: -nOuterFace.x, y: 0, z: -nOuterFace.z };

			// Outer face
			faces.push({
				vertices: [
					{ ...pOuter1Bottom, u: u1, v: 0, normal: nOuter1 },
					{ ...pOuter1Top, u: u1, v: 1, normal: nOuter1 },
					{ ...pOuter2Top, u: u2, v: 1, normal: nOuter2 },
					{ ...pOuter2Bottom, u: u2, v: 0, normal: nOuter2 },
				],
				normal: nOuterFace,
				material,
			});

			// Inner face
			faces.push({
				vertices: [
					{ ...pInner2Bottom, u: u2, v: 0, normal: nInner2 },
					{ ...pInner2Top, u: u2, v: 1, normal: nInner2 },
					{ ...pInner1Top, u: u1, v: 1, normal: nInner1 },
					{ ...pInner1Bottom, u: u1, v: 0, normal: nInner1 },
				],
				normal: nInnerFace,
				material,
			});

			// Top face
			faces.push({
				vertices: [
					{ ...pInner1Top, u: u1, v: 0, normal: { x: 0, y: 1, z: 0 } },
					{ ...pInner2Top, u: u2, v: 0, normal: { x: 0, y: 1, z: 0 } },
					{ ...pOuter2Top, u: u2, v: 1, normal: { x: 0, y: 1, z: 0 } },
					{ ...pOuter1Top, u: u1, v: 1, normal: { x: 0, y: 1, z: 0 } },
				],
				normal: { x: 0, y: 1, z: 0 },
				material,
			});

			// Bottom face
			faces.push({
				vertices: [
					{ ...pOuter1Bottom, u: u1, v: 1, normal: { x: 0, y: -1, z: 0 } },
					{ ...pOuter2Bottom, u: u2, v: 1, normal: { x: 0, y: -1, z: 0 } },
					{ ...pInner2Bottom, u: u2, v: 0, normal: { x: 0, y: -1, z: 0 } },
					{ ...pInner1Bottom, u: u1, v: 0, normal: { x: 0, y: -1, z: 0 } },
				],
				normal: { x: 0, y: -1, z: 0 },
				material,
			});
		}

		const model = new SimpleModel(faces);
		model.transform.position.copy(base);
		return model;
	}

	/**
	 * Creates a cone-shaped model.
	 * @param {IVector3} base - The center position of the cone.
	 * @param {number} radius - The radius of the base of the cone.
	 * @param {number} height - The height of the cone.
	 * @param {number} [segments=16] - The number of vertical segments around the circumference.
	 * @param {Material} [material=null] - The material to apply to the cone.
	 * @returns {SimpleModel} A new SimpleModel instance representing a cone.
	 */
	public static createCone(
		base: IVector3,
		radius: number,
		height: number,
		segments: number = 16,
		material: Material | null = null
	): SimpleModel {
		const h2 = height / 2;
		const bottom: IVector3[] = [];
		const topVec = { x: 0, y: h2, z: 0 };

		for (let i = 0; i < segments; i++) {
			const theta = (i / segments) * Math.PI * 2;
			bottom.push({
				x: Math.cos(theta) * radius,
				y: -h2,
				z: Math.sin(theta) * radius,
			});
		}

		const faces: ModelFace[] = [];
		const ny = radius;
		const nLenBase = height;

		for (let i = 0; i < segments; i++) {
			const next = (i + 1) % segments;
			const u1 = i / segments;
			const u2 = (i + 1) / segments;

			// Bottom cap
			faces.push({
				vertices: [
					{ ...bottom[i], u: u1, v: 0, normal: { x: 0, y: -1, z: 0 } },
					{ ...bottom[next], u: u2, v: 0, normal: { x: 0, y: -1, z: 0 } },
					{ x: 0, y: -h2, z: 0, u: 0.5, v: 0, normal: { x: 0, y: -1, z: 0 } },
				],
				normal: { x: 0, y: -1, z: 0 },
				material,
			});

			// Side
			const theta1 = u1 * Math.PI * 2;
			const theta2 = u2 * Math.PI * 2;

			const n1 = {
				x: nLenBase * Math.cos(theta1),
				y: ny,
				z: nLenBase * Math.sin(theta1),
			};
			const n1Len = Math.hypot(n1.x, n1.y, n1.z);
			n1.x /= n1Len;
			n1.y /= n1Len;
			n1.z /= n1Len;

			const n2 = {
				x: nLenBase * Math.cos(theta2),
				y: ny,
				z: nLenBase * Math.sin(theta2),
			};
			const n2Len = Math.hypot(n2.x, n2.y, n2.z);
			n2.x /= n2Len;
			n2.y /= n2Len;
			n2.z /= n2Len;

			const thetaMid = ((i + 0.5) / segments) * Math.PI * 2;
			const nFace = {
				x: nLenBase * Math.cos(thetaMid),
				y: ny,
				z: nLenBase * Math.sin(thetaMid),
			};
			const nFaceLen = Math.hypot(nFace.x, nFace.y, nFace.z);
			nFace.x /= nFaceLen;
			nFace.y /= nFaceLen;
			nFace.z /= nFaceLen;

			faces.push({
				vertices: [
					{ ...bottom[i], u: u1, v: 0, normal: n1 },
					{ ...topVec, u: (u1 + u2) / 2, v: 1, normal: n1 },
					{ ...bottom[next], u: u2, v: 0, normal: n2 },
				],
				normal: nFace,
				material,
			});
		}

		const model = new SimpleModel(faces);
		model.transform.position.copy(base);
		return model;
	}
}
