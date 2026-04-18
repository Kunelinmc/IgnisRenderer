import type { BoundingBox } from "../core/types";
import type { Node } from "../core/Node";
import { MeshInstance } from "../meshes";
import type {
	CollisionGeometryTriangles,
	ICollisionGeometryProvider,
} from "./types";

export interface TriangleBVHNode {
	bounds: BoundingBox;
	start: number;
	end: number;
	left: TriangleBVHNode | null;
	right: TriangleBVHNode | null;
}

export interface TriangleBVHCacheEntry {
	meshId: string;
	geometryKey: string;
	vertices: Float32Array;
	indices: Uint32Array;
	triangleOrder: Uint32Array;
	root: TriangleBVHNode | null;
	bounds: BoundingBox;
	triangleCount: number;
}

const TRIANGLE_BVH_LEAF_SIZE = 8;

export class TriangleBVHCache {
	private _entryByGeometryKey = new Map<string, TriangleBVHCacheEntry>();
	private _geometryKeyByMeshId = new Map<string, string>();

	public get(node: Node): TriangleBVHCacheEntry | null {
		if (!(node instanceof MeshInstance)) return null;
		const key = this._geometryKeyByMeshId.get(node.mesh.id);
		if (!key) return null;
		return this._entryByGeometryKey.get(key) ?? null;
	}

	public getOrCreate(
		node: Node,
		geometryProvider: ICollisionGeometryProvider
	): TriangleBVHCacheEntry | null {
		if (!(node instanceof MeshInstance)) return null;
		const triangles = geometryProvider.getTriangles(node, {
			space: "local",
			useCache: true,
		});
		if (!triangles) return null;

		const previousKey = this._geometryKeyByMeshId.get(node.mesh.id);
		if (previousKey && previousKey !== triangles.geometryKey) {
			this._entryByGeometryKey.delete(previousKey);
		}
		this._geometryKeyByMeshId.set(node.mesh.id, triangles.geometryKey);

		const cached = this._entryByGeometryKey.get(triangles.geometryKey);
		if (cached) return cached;

		const entry = buildTriangleBVHEntry(node.mesh.id, triangles);
		this._entryByGeometryKey.set(triangles.geometryKey, entry);
		return entry;
	}

	public invalidateNode(node: Node): void {
		if (!(node instanceof MeshInstance)) return;
		const key = this._geometryKeyByMeshId.get(node.mesh.id);
		if (!key) return;
		this._geometryKeyByMeshId.delete(node.mesh.id);
		this._entryByGeometryKey.delete(key);
	}

	public clear(): void {
		this._entryByGeometryKey.clear();
		this._geometryKeyByMeshId.clear();
	}
}

function buildTriangleBVHEntry(
	meshId: string,
	triangles: CollisionGeometryTriangles
): TriangleBVHCacheEntry {
	const vertices = triangles.vertices;
	const indices = triangles.indices;
	const triangleCount = Math.floor(indices.length / 3);
	const triangleOrder = new Uint32Array(triangleCount);
	const boundsByTriangle = new Array<BoundingBox>(triangleCount);

	for (let triangle = 0; triangle < triangleCount; triangle++) {
		triangleOrder[triangle] = triangle;
		boundsByTriangle[triangle] = computeTriangleBounds(vertices, indices, triangle);
	}

	const root =
		triangleCount > 0 ?
			buildNode(boundsByTriangle, triangleOrder, 0, triangleCount)
		:	null;
	const bounds = root?.bounds ?? {
		min: { x: 0, y: 0, z: 0 },
		max: { x: 0, y: 0, z: 0 },
	};

	return {
		meshId,
		geometryKey: triangles.geometryKey,
		vertices,
		indices,
		triangleOrder,
		root,
		bounds,
		triangleCount,
	};
}

function buildNode(
	boundsByTriangle: BoundingBox[],
	triangleOrder: Uint32Array,
	start: number,
	end: number
): TriangleBVHNode {
	const bounds = computeBoundsFromRange(boundsByTriangle, triangleOrder, start, end);
	const count = end - start;
	if (count <= TRIANGLE_BVH_LEAF_SIZE) {
		return {
			bounds,
			start,
			end,
			left: null,
			right: null,
		};
	}

	const axis = resolveSplitAxis(boundsByTriangle, triangleOrder, start, end);
	sortTriangleOrderByAxis(boundsByTriangle, triangleOrder, start, end, axis);
	const middle = start + Math.floor(count * 0.5);

	return {
		bounds,
		start,
		end,
		left: buildNode(boundsByTriangle, triangleOrder, start, middle),
		right: buildNode(boundsByTriangle, triangleOrder, middle, end),
	};
}

function computeTriangleBounds(
	vertices: Float32Array,
	indices: Uint32Array,
	triangleIndex: number
): BoundingBox {
	const base = triangleIndex * 3;
	const index0 = indices[base] * 3;
	const index1 = indices[base + 1] * 3;
	const index2 = indices[base + 2] * 3;

	const x0 = vertices[index0];
	const y0 = vertices[index0 + 1];
	const z0 = vertices[index0 + 2];
	const x1 = vertices[index1];
	const y1 = vertices[index1 + 1];
	const z1 = vertices[index1 + 2];
	const x2 = vertices[index2];
	const y2 = vertices[index2 + 1];
	const z2 = vertices[index2 + 2];

	return {
		min: {
			x: Math.min(x0, x1, x2),
			y: Math.min(y0, y1, y2),
			z: Math.min(z0, z1, z2),
		},
		max: {
			x: Math.max(x0, x1, x2),
			y: Math.max(y0, y1, y2),
			z: Math.max(z0, z1, z2),
		},
	};
}

function computeBoundsFromRange(
	boundsByTriangle: BoundingBox[],
	triangleOrder: Uint32Array,
	start: number,
	end: number
): BoundingBox {
	let minX = Infinity;
	let minY = Infinity;
	let minZ = Infinity;
	let maxX = -Infinity;
	let maxY = -Infinity;
	let maxZ = -Infinity;

	for (let i = start; i < end; i++) {
		const bounds = boundsByTriangle[triangleOrder[i]];
		if (bounds.min.x < minX) minX = bounds.min.x;
		if (bounds.min.y < minY) minY = bounds.min.y;
		if (bounds.min.z < minZ) minZ = bounds.min.z;
		if (bounds.max.x > maxX) maxX = bounds.max.x;
		if (bounds.max.y > maxY) maxY = bounds.max.y;
		if (bounds.max.z > maxZ) maxZ = bounds.max.z;
	}

	return {
		min: { x: minX, y: minY, z: minZ },
		max: { x: maxX, y: maxY, z: maxZ },
	};
}

function resolveSplitAxis(
	boundsByTriangle: BoundingBox[],
	triangleOrder: Uint32Array,
	start: number,
	end: number
): 0 | 1 | 2 {
	let minCx = Infinity;
	let minCy = Infinity;
	let minCz = Infinity;
	let maxCx = -Infinity;
	let maxCy = -Infinity;
	let maxCz = -Infinity;

	for (let i = start; i < end; i++) {
		const bounds = boundsByTriangle[triangleOrder[i]];
		const cx = (bounds.min.x + bounds.max.x) * 0.5;
		const cy = (bounds.min.y + bounds.max.y) * 0.5;
		const cz = (bounds.min.z + bounds.max.z) * 0.5;
		if (cx < minCx) minCx = cx;
		if (cy < minCy) minCy = cy;
		if (cz < minCz) minCz = cz;
		if (cx > maxCx) maxCx = cx;
		if (cy > maxCy) maxCy = cy;
		if (cz > maxCz) maxCz = cz;
	}

	const extentX = maxCx - minCx;
	const extentY = maxCy - minCy;
	const extentZ = maxCz - minCz;
	if (extentX >= extentY && extentX >= extentZ) return 0;
	if (extentY >= extentX && extentY >= extentZ) return 1;
	return 2;
}

function sortTriangleOrderByAxis(
	boundsByTriangle: BoundingBox[],
	triangleOrder: Uint32Array,
	start: number,
	end: number,
	axis: 0 | 1 | 2
): void {
	const range = Array.from(triangleOrder.slice(start, end));
	range.sort((left, right) => {
		const lc = readCentroidAxis(boundsByTriangle[left], axis);
		const rc = readCentroidAxis(boundsByTriangle[right], axis);
		return lc - rc;
	});
	for (let i = start; i < end; i++) {
		triangleOrder[i] = range[i - start];
	}
}

function readCentroidAxis(bounds: BoundingBox, axis: 0 | 1 | 2): number {
	if (axis === 0) return (bounds.min.x + bounds.max.x) * 0.5;
	if (axis === 1) return (bounds.min.y + bounds.max.y) * 0.5;
	return (bounds.min.z + bounds.max.z) * 0.5;
}
