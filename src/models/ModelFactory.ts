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
}
