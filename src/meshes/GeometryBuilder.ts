import { Material } from "../materials/Material";
import { IdGenerator } from "../foundation/IdGenerator";
import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../core/types";
import type {
	BoundingBox,
	BoundingSphere,
	IPrimitive,
	IPrimitiveGeometry,
	IVertex,
} from "../core/types";
import type { RGBA } from "../foundation/Color";
import type { IVector3 } from "../maths/types";

export interface GeometryFaceVertex extends IVertex {
	[key: string]: unknown;
}

export interface GeometryFace {
	vertices: GeometryFaceVertex[];
	material?: Material | null;
	color?: RGBA;
	normal?: IVector3;
	doubleSided?: boolean;
	[key: string]: unknown;
}

interface PrimitiveFaceGroup {
	material: Material;
	faces: GeometryFace[];
}

export class GeometryBuilder {
	public static buildPrimitivesFromFaces(faces: GeometryFace[]): IPrimitive[] {
		if (faces.length === 0) {
			return [];
		}

		const normalizedFaces = this._ensureVertexNormals(faces);
		const groups = this._groupFacesByMaterial(normalizedFaces);

		return groups.map((group) => this._createPrimitiveFromFaces(group));
	}

	public static getVertexCount(geometry: IPrimitiveGeometry): number {
		return (geometry.positions.length / 3) | 0;
	}

	public static computeBoundingBox(geometry: IPrimitiveGeometry): BoundingBox {
		const positions = geometry.positions;
		if (positions.length === 0) {
			return {
				min: { x: 0, y: 0, z: 0 },
				max: { x: 0, y: 0, z: 0 },
			};
		}

		let minX = Infinity;
		let minY = Infinity;
		let minZ = Infinity;
		let maxX = -Infinity;
		let maxY = -Infinity;
		let maxZ = -Infinity;

		for (let i = 0; i < positions.length; i += 3) {
			const x = positions[i];
			const y = positions[i + 1];
			const z = positions[i + 2];

			if (x < minX) minX = x;
			if (y < minY) minY = y;
			if (z < minZ) minZ = z;
			if (x > maxX) maxX = x;
			if (y > maxY) maxY = y;
			if (z > maxZ) maxZ = z;
		}

		return {
			min: { x: minX, y: minY, z: minZ },
			max: { x: maxX, y: maxY, z: maxZ },
		};
	}

	public static computeBoundingSphere(
		geometry: IPrimitiveGeometry,
		box?: BoundingBox
	): BoundingSphere {
		const bounds = box ?? this.computeBoundingBox(geometry);
		const center = {
			x: (bounds.min.x + bounds.max.x) / 2,
			y: (bounds.min.y + bounds.max.y) / 2,
			z: (bounds.min.z + bounds.max.z) / 2,
		};

		let maxDistSq = 0;
		const positions = geometry.positions;
		for (let i = 0; i < positions.length; i += 3) {
			const dx = positions[i] - center.x;
			const dy = positions[i + 1] - center.y;
			const dz = positions[i + 2] - center.z;
			const distSq = dx * dx + dy * dy + dz * dz;
			if (distSq > maxDistSq) {
				maxDistSq = distSq;
			}
		}

		return {
			center,
			radius: Math.sqrt(maxDistSq),
		};
	}

	public static computeModelBoundingBox(primitives: IPrimitive[]): BoundingBox {
		if (primitives.length === 0) {
			return {
				min: { x: 0, y: 0, z: 0 },
				max: { x: 0, y: 0, z: 0 },
			};
		}

		let minX = Infinity;
		let minY = Infinity;
		let minZ = Infinity;
		let maxX = -Infinity;
		let maxY = -Infinity;
		let maxZ = -Infinity;

		for (const primitive of primitives) {
			const box = primitive.boundingBox;
			if (box.min.x < minX) minX = box.min.x;
			if (box.min.y < minY) minY = box.min.y;
			if (box.min.z < minZ) minZ = box.min.z;
			if (box.max.x > maxX) maxX = box.max.x;
			if (box.max.y > maxY) maxY = box.max.y;
			if (box.max.z > maxZ) maxZ = box.max.z;
		}

		return {
			min: { x: minX, y: minY, z: minZ },
			max: { x: maxX, y: maxY, z: maxZ },
		};
	}

	public static computeModelBoundingSphere(
		primitives: IPrimitive[],
		box?: BoundingBox
	): BoundingSphere {
		if (primitives.length === 0) {
			return { center: { x: 0, y: 0, z: 0 }, radius: 0 };
		}

		const bounds = box ?? this.computeModelBoundingBox(primitives);
		const center = {
			x: (bounds.min.x + bounds.max.x) / 2,
			y: (bounds.min.y + bounds.max.y) / 2,
			z: (bounds.min.z + bounds.max.z) / 2,
		};

		let maxDistSq = 0;
		for (const primitive of primitives) {
			const positions = primitive.geometry.positions;
			for (let i = 0; i < positions.length; i += 3) {
				const dx = positions[i] - center.x;
				const dy = positions[i + 1] - center.y;
				const dz = positions[i + 2] - center.z;
				const distSq = dx * dx + dy * dy + dz * dz;
				if (distSq > maxDistSq) {
					maxDistSq = distSq;
				}
			}
		}

		return {
			center,
			radius: Math.sqrt(maxDistSq),
		};
	}

	public static createVerticesForTriangle(
		primitive: IPrimitive,
		triangleIndex: number,
		overrides?: Partial<IPrimitiveGeometry>
	): IVertex[] {
		const geometry = mergeGeometry(primitive.geometry, overrides);
		const indices = geometry.indices;
		const baseIndex = triangleIndex * 3;

		return [
			this._createVertex(geometry, indices[baseIndex]),
			this._createVertex(geometry, indices[baseIndex + 1]),
			this._createVertex(geometry, indices[baseIndex + 2]),
		];
	}

	private static _createPrimitiveFromFaces(
		group: PrimitiveFaceGroup
	): IPrimitive {
		const positions: number[] = [];
		const normals: number[] = [];
		const tangents: number[] = [];
		const uv0: number[] = [];
		const uv1: number[] = [];
		const colors: number[] = [];
		const indices: number[] = [];

		let hasNormals = false;
		let hasTangents = false;
		let hasUv0 = false;
		let hasUv1 = false;
		let hasColors = false;
		let vertexIndex = 0;

		for (const face of group.faces) {
			if (face.vertices.length < 3) continue;

			for (let i = 1; i < face.vertices.length - 1; i++) {
				const triangle = [
					face.vertices[0],
					face.vertices[i],
					face.vertices[i + 1],
				];

				for (const vertex of triangle) {
					positions.push(vertex.x, vertex.y, vertex.z);

					if (vertex.normal) {
						hasNormals = true;
						normals.push(vertex.normal.x, vertex.normal.y, vertex.normal.z);
					} else {
						normals.push(0, 0, 0);
					}

					if (vertex.tangent) {
						hasTangents = true;
						tangents.push(
							vertex.tangent.x,
							vertex.tangent.y,
							vertex.tangent.z,
							vertex.tangent.w
						);
					} else {
						tangents.push(0, 0, 0, 0);
					}

					if (vertex.u !== undefined || vertex.v !== undefined) {
						hasUv0 = true;
					}
					uv0.push(vertex.u ?? 0, vertex.v ?? 0);

					if (vertex.u2 !== undefined || vertex.v2 !== undefined) {
						hasUv1 = true;
					}
					uv1.push(vertex.u2 ?? 0, vertex.v2 ?? 0);

					if (vertex.color) {
						hasColors = true;
						colors.push(
							vertex.color.r / 255,
							vertex.color.g / 255,
							vertex.color.b / 255,
							vertex.color.a ?? 1
						);
					} else {
						colors.push(1, 1, 1, 1);
					}

					indices.push(vertexIndex++);
				}
			}
		}

		const geometry: IPrimitiveGeometry = {
			positions: new Float32Array(positions),
			indices: new Uint32Array(indices),
			normals: hasNormals ? new Float32Array(normals) : null,
			tangents: hasTangents ? new Float32Array(tangents) : null,
			uv0: hasUv0 ? new Float32Array(uv0) : null,
			uv1: hasUv1 ? new Float32Array(uv1) : null,
			colors: hasColors ? new Float32Array(colors) : null,
		};

		const boundingBox = this.computeBoundingBox(geometry);
		const boundingSphere = this.computeBoundingSphere(geometry, boundingBox);

		return {
			id: IdGenerator.nextId("primitive"),
			geometry,
			geometryVersion: 0,
			topology: DEFAULT_PRIMITIVE_DRAW_TOPOLOGY,
			material: group.material,
			boundingBox,
			boundingSphere,
			visible: true,
			castShadows: true,
			receiveShadows: true,
		};
	}

	private static _groupFacesByMaterial(
		faces: GeometryFace[]
	): PrimitiveFaceGroup[] {
		const groups = new Map<Material, GeometryFace[]>();

		for (const face of faces) {
			const material = face.material ?? new Material();
			let bucket = groups.get(material);
			if (!bucket) {
				bucket = [];
				groups.set(material, bucket);
			}
			bucket.push(face);
		}

		return Array.from(groups.entries()).map(([material, groupedFaces]) => ({
			material,
			faces: groupedFaces,
		}));
	}

	private static _ensureVertexNormals(faces: GeometryFace[]): GeometryFace[] {
		if (faces.length === 0) {
			return faces;
		}

		let totalVerts = 0;
		let normalCount = 0;
		for (const face of faces) {
			for (const vertex of face.vertices) {
				totalVerts++;
				if (vertex.normal) {
					normalCount++;
				}
			}
		}

		if (normalCount > totalVerts * 0.8) {
			return faces;
		}

		const vertexNormals = new Map<string, IVector3>();

		for (const face of faces) {
			const faceNormal =
				face.normal ?? this._calculateFaceNormal(face.vertices);
			face.normal = faceNormal;

			for (const vertex of face.vertices) {
				const key = this._vertexKey(vertex);
				let sum = vertexNormals.get(key);
				if (!sum) {
					sum = { x: 0, y: 0, z: 0 };
					vertexNormals.set(key, sum);
				}
				sum.x += faceNormal.x;
				sum.y += faceNormal.y;
				sum.z += faceNormal.z;
			}
		}

		for (const face of faces) {
			for (const vertex of face.vertices) {
				const key = this._vertexKey(vertex);
				const sum = vertexNormals.get(key);
				if (!sum) continue;

				const length = Math.hypot(sum.x, sum.y, sum.z) || 1e-6;
				vertex.normal = {
					x: sum.x / length,
					y: sum.y / length,
					z: sum.z / length,
				};
			}
		}

		return faces;
	}

	private static _calculateFaceNormal(
		vertices: GeometryFaceVertex[]
	): IVector3 {
		const v0 = vertices[0];
		const v1 = vertices[1];
		const v2 = vertices[2];
		const ux = v1.x - v0.x;
		const uy = v1.y - v0.y;
		const uz = v1.z - v0.z;
		const vx = v2.x - v0.x;
		const vy = v2.y - v0.y;
		const vz = v2.z - v0.z;
		const nx = uy * vz - uz * vy;
		const ny = uz * vx - ux * vz;
		const nz = ux * vy - uy * vx;
		const length = Math.hypot(nx, ny, nz) || 1e-6;

		return {
			x: nx / length,
			y: ny / length,
			z: nz / length,
		};
	}

	private static _vertexKey(vertex: GeometryFaceVertex): string {
		return `${vertex.x.toFixed(5)},${vertex.y.toFixed(5)},${vertex.z.toFixed(5)}`;
	}

	private static _createVertex(
		geometry: IPrimitiveGeometry,
		index: number
	): IVertex {
		const positionOffset = index * 3;
		const uvOffset = index * 2;
		const tangentOffset = index * 4;
		const colorOffset = index * 4;

		const vertex: IVertex = {
			x: geometry.positions[positionOffset],
			y: geometry.positions[positionOffset + 1],
			z: geometry.positions[positionOffset + 2],
		};

		if (geometry.normals) {
			vertex.normal = {
				x: geometry.normals[positionOffset],
				y: geometry.normals[positionOffset + 1],
				z: geometry.normals[positionOffset + 2],
			};
		}

		if (geometry.tangents) {
			vertex.tangent = {
				x: geometry.tangents[tangentOffset],
				y: geometry.tangents[tangentOffset + 1],
				z: geometry.tangents[tangentOffset + 2],
				w: geometry.tangents[tangentOffset + 3],
			};
		}

		if (geometry.uv0) {
			vertex.u = geometry.uv0[uvOffset];
			vertex.v = geometry.uv0[uvOffset + 1];
		}

		if (geometry.uv1) {
			vertex.u2 = geometry.uv1[uvOffset];
			vertex.v2 = geometry.uv1[uvOffset + 1];
		}

		if (geometry.colors) {
			vertex.color = {
				r: Math.round(geometry.colors[colorOffset] * 255),
				g: Math.round(geometry.colors[colorOffset + 1] * 255),
				b: Math.round(geometry.colors[colorOffset + 2] * 255),
				a: geometry.colors[colorOffset + 3],
			};
		}

		if (geometry.joints0) {
			const base = index * 4;
			vertex.joints0 = [
				geometry.joints0[base],
				geometry.joints0[base + 1],
				geometry.joints0[base + 2],
				geometry.joints0[base + 3],
			];
		}

		if (geometry.weights0) {
			const base = index * 4;
			vertex.weights0 = [
				geometry.weights0[base],
				geometry.weights0[base + 1],
				geometry.weights0[base + 2],
				geometry.weights0[base + 3],
			];
		}

		if (geometry.joints1) {
			const base = index * 4;
			vertex.joints1 = [
				geometry.joints1[base],
				geometry.joints1[base + 1],
				geometry.joints1[base + 2],
				geometry.joints1[base + 3],
			];
		}

		if (geometry.weights1) {
			const base = index * 4;
			vertex.weights1 = [
				geometry.weights1[base],
				geometry.weights1[base + 1],
				geometry.weights1[base + 2],
				geometry.weights1[base + 3],
			];
		}

		return vertex;
	}
}

function mergeGeometry(
	base: IPrimitiveGeometry,
	override?: Partial<IPrimitiveGeometry>
): IPrimitiveGeometry {
	if (!override) return base;
	return {
		positions: override.positions ?? base.positions,
		normals: override.normals ?? base.normals,
		tangents: override.tangents ?? base.tangents,
		uv0: override.uv0 ?? base.uv0,
		uv1: override.uv1 ?? base.uv1,
		colors: override.colors ?? base.colors,
		joints0: override.joints0 ?? base.joints0,
		weights0: override.weights0 ?? base.weights0,
		joints1: override.joints1 ?? base.joints1,
		weights1: override.weights1 ?? base.weights1,
		morphTargets: override.morphTargets ?? base.morphTargets,
		indices: override.indices ?? base.indices,
	};
}
