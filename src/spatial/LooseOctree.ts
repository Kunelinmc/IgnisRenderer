import type { BoundingBox } from "../core/types";
import type { Frustum } from "../maths/Frustum";
import type { MeshInstance } from "../meshes";
import type {
	SpatialIndex3D,
	SpatialQueryOptions,
	SpatialRayHit,
	SpatialRayQueryOptions,
} from "./types";

interface LooseOctreeNode {
	centerX: number;
	centerY: number;
	centerZ: number;
	halfSize: number;
	objects: MeshInstance[];
	objectBounds: BoundingBox[];
	children: Array<LooseOctreeNode | null> | null;
}

interface LooseOctreeEntry {
	node: LooseOctreeNode;
	bounds: BoundingBox;
}

interface ChildPlacement {
	index: number;
	centerX: number;
	centerY: number;
	centerZ: number;
}

interface PendingBuildEntry {
	meshInstance: MeshInstance;
	bounds: BoundingBox;
}

export interface LooseOctreeOptions {
	leafCapacity?: number;
	maxDepth?: number;
	looseness?: number;
}

const DEFAULT_LEAF_CAPACITY = 16;
const DEFAULT_MAX_DEPTH = 8;
const DEFAULT_LOOSENESS = 1.5;
const MIN_HALF_SIZE = 1e-3;

const FRUSTUM_OUTSIDE = -1;
const FRUSTUM_INTERSECT = 0;
const FRUSTUM_INSIDE = 1;

export class LooseOctree implements SpatialIndex3D {
	private _root: LooseOctreeNode | null = null;
	private _entriesByMeshInstance = new Map<MeshInstance, LooseOctreeEntry>();
	private readonly _leafCapacity: number;
	private readonly _maxDepth: number;
	private readonly _looseness: number;

	constructor(
		meshInstances: MeshInstance[] = [],
		options: LooseOctreeOptions = {}
	) {
		this._leafCapacity = resolveLeafCapacity(options.leafCapacity);
		this._maxDepth = resolveMaxDepth(options.maxDepth);
		this._looseness = resolveLooseness(options.looseness);
		this.rebuild(meshInstances);
	}

	public get size(): number {
		return this._entriesByMeshInstance.size;
	}

	public get dirty(): boolean {
		return false;
	}

	public setMeshInstances(meshInstances: MeshInstance[]): void {
		this.rebuild(meshInstances);
	}

	public markDirty(meshInstance?: MeshInstance): void {
		if (!meshInstance) {
			this.rebuild();
			return;
		}
		if (!this._entriesByMeshInstance.has(meshInstance)) return;
		this.upsert(meshInstance);
	}

	public upsert(meshInstance: MeshInstance): void {
		this._detachEntry(meshInstance);
		const bounds = meshInstance.getWorldBoundingBox();
		this._insertEntry(meshInstance, bounds);
	}

	public remove(meshInstance: MeshInstance): boolean {
		return this._detachEntry(meshInstance);
	}

	public rebuild(meshInstances?: MeshInstance[]): void {
		const source =
			meshInstances ?? Array.from(this._entriesByMeshInstance.keys());
		this._entriesByMeshInstance.clear();
		this._root = null;
		if (source.length === 0) return;

		const pending: PendingBuildEntry[] = new Array<PendingBuildEntry>(
			source.length
		);
		let minX = Infinity;
		let minY = Infinity;
		let minZ = Infinity;
		let maxX = -Infinity;
		let maxY = -Infinity;
		let maxZ = -Infinity;

		for (let i = 0; i < source.length; i++) {
			const meshInstance = source[i];
			const bounds = meshInstance.getWorldBoundingBox();
			pending[i] = {
				meshInstance,
				bounds,
			};
			if (bounds.min.x < minX) minX = bounds.min.x;
			if (bounds.min.y < minY) minY = bounds.min.y;
			if (bounds.min.z < minZ) minZ = bounds.min.z;
			if (bounds.max.x > maxX) maxX = bounds.max.x;
			if (bounds.max.y > maxY) maxY = bounds.max.y;
			if (bounds.max.z > maxZ) maxZ = bounds.max.z;
		}

		this._root = createRootNode({
			min: { x: minX, y: minY, z: minZ },
			max: { x: maxX, y: maxY, z: maxZ },
		});

		for (const entry of pending) {
			this._insertEntry(entry.meshInstance, entry.bounds);
		}
	}

	public queryFrustum(
		frustum: Frustum,
		options?: SpatialQueryOptions
	): MeshInstance[] {
		if (!this._root) return [];
		const maxResults = resolveMaxResults(options?.maxResults);
		if (maxResults <= 0) return [];
		const includeInvisible = options?.includeInvisible === true;
		const result: MeshInstance[] = [];
		this._queryNodeFrustum(
			this._root,
			frustum,
			includeInvisible,
			maxResults,
			result
		);
		return result;
	}

	public queryRay(
		origin: { x: number; y: number; z: number },
		direction: { x: number; y: number; z: number },
		options?: SpatialRayQueryOptions
	): MeshInstance[] {
		return this.queryRayDetailed(origin, direction, options).map(
			(hit) => hit.meshInstance
		);
	}

	public queryRayDetailed(
		origin: { x: number; y: number; z: number },
		direction: { x: number; y: number; z: number },
		options?: SpatialRayQueryOptions
	): SpatialRayHit[] {
		if (!this._root) return [];
		const maxResults = resolveMaxResults(options?.maxResults);
		if (maxResults <= 0) return [];
		const maxDistance = resolveMaxDistance(options?.maxDistance);
		if (maxDistance <= 0) return [];

		const directionLength = Math.hypot(direction.x, direction.y, direction.z);
		if (!(directionLength > 1e-8)) {
			throw new Error("LooseOctree.queryRay direction must be non-zero");
		}

		const invDirectionLength = 1 / directionLength;
		const normalizedDirection = {
			x: direction.x * invDirectionLength,
			y: direction.y * invDirectionLength,
			z: direction.z * invDirectionLength,
		};
		const includeInvisible = options?.includeInvisible === true;

		const stack: LooseOctreeNode[] = [this._root];
		const hits: SpatialRayHit[] = [];

		while (stack.length > 0) {
			const node = stack.pop();
			if (!node) continue;

			const looseHalfSize = node.halfSize * this._looseness;
			const nodeMinX = node.centerX - looseHalfSize;
			const nodeMinY = node.centerY - looseHalfSize;
			const nodeMinZ = node.centerZ - looseHalfSize;
			const nodeMaxX = node.centerX + looseHalfSize;
			const nodeMaxY = node.centerY + looseHalfSize;
			const nodeMaxZ = node.centerZ + looseHalfSize;
			const nodeDistance = intersectRayAABB(
				origin,
				normalizedDirection,
				maxDistance,
				nodeMinX,
				nodeMinY,
				nodeMinZ,
				nodeMaxX,
				nodeMaxY,
				nodeMaxZ
			);
			if (nodeDistance === null) continue;

			for (let i = 0; i < node.objects.length; i++) {
				const meshInstance = node.objects[i];
				if (!includeInvisible && meshInstance.visible === false) {
					continue;
				}
				const bounds = node.objectBounds[i];
				const distance = intersectRayAABB(
					origin,
					normalizedDirection,
					maxDistance,
					bounds.min.x,
					bounds.min.y,
					bounds.min.z,
					bounds.max.x,
					bounds.max.y,
					bounds.max.z
				);
				if (distance === null) continue;
				hits.push({
					meshInstance,
					distance,
				});
			}

			if (!node.children) continue;
			for (const child of node.children) {
				if (!child) continue;
				stack.push(child);
			}
		}

		if (hits.length === 0) return [];
		hits.sort(compareRayHits);
		if (hits.length > maxResults) {
			return hits.slice(0, maxResults);
		}
		return hits;
	}

	private _insertEntry(meshInstance: MeshInstance, bounds: BoundingBox): void {
		if (!this._root) {
			this._root = createRootNode(bounds);
		}
		this._expandRootToFit(bounds);
		if (!this._root) return;
		const node = this._insertIntoNode(this._root, meshInstance, bounds, 0);
		this._entriesByMeshInstance.set(meshInstance, {
			node,
			bounds,
		});
	}

	private _insertIntoNode(
		node: LooseOctreeNode,
		meshInstance: MeshInstance,
		bounds: BoundingBox,
		depth: number
	): LooseOctreeNode {
		if (depth < this._maxDepth) {
			if (!node.children && node.objects.length >= this._leafCapacity) {
				node.children = createChildrenArray();
				this._redistributeNodeObjects(node, depth);
			}

			const childPlacement = resolveChildPlacement(node, bounds);
			if (childPlacement) {
				const child = this._ensureChild(node, childPlacement);
				return this._insertIntoNode(child, meshInstance, bounds, depth + 1);
			}
		}

		node.objects.push(meshInstance);
		node.objectBounds.push(bounds);
		return node;
	}

	private _redistributeNodeObjects(
		node: LooseOctreeNode,
		depth: number
	): void {
		if (!node.children || depth >= this._maxDepth) return;
		let index = 0;
		while (index < node.objects.length) {
			const meshInstance = node.objects[index];
			const bounds = node.objectBounds[index];
			const childPlacement = resolveChildPlacement(node, bounds);
			if (!childPlacement) {
				index++;
				continue;
			}

			node.objects.splice(index, 1);
			node.objectBounds.splice(index, 1);
			const child = this._ensureChild(node, childPlacement);
			const insertedNode = this._insertIntoNode(
				child,
				meshInstance,
				bounds,
				depth + 1
			);
			const entry = this._entriesByMeshInstance.get(meshInstance);
			if (entry) {
				entry.node = insertedNode;
			}
		}
	}

	private _ensureChild(
		node: LooseOctreeNode,
		childPlacement: ChildPlacement
	): LooseOctreeNode {
		if (!node.children) {
			node.children = createChildrenArray();
		}
		let child = node.children[childPlacement.index];
		if (!child) {
			child = {
				centerX: childPlacement.centerX,
				centerY: childPlacement.centerY,
				centerZ: childPlacement.centerZ,
				halfSize: node.halfSize * 0.5,
				objects: [],
				objectBounds: [],
				children: null,
			};
			node.children[childPlacement.index] = child;
		}
		return child;
	}

	private _expandRootToFit(bounds: BoundingBox): void {
		while (
			this._root &&
			!containsBoundsInCube(
				this._root.centerX,
				this._root.centerY,
				this._root.centerZ,
				this._root.halfSize,
				bounds
			)
		) {
			const root = this._root;
			const centerX = (bounds.min.x + bounds.max.x) * 0.5;
			const centerY = (bounds.min.y + bounds.max.y) * 0.5;
			const centerZ = (bounds.min.z + bounds.max.z) * 0.5;
			const dirX = centerX >= root.centerX ? 1 : -1;
			const dirY = centerY >= root.centerY ? 1 : -1;
			const dirZ = centerZ >= root.centerZ ? 1 : -1;
			const newRoot = {
				centerX: root.centerX + dirX * root.halfSize,
				centerY: root.centerY + dirY * root.halfSize,
				centerZ: root.centerZ + dirZ * root.halfSize,
				halfSize: root.halfSize * 2,
				objects: [],
				objectBounds: [],
				children: createChildrenArray(),
			} satisfies LooseOctreeNode;
			const oldChildIndex = resolveChildIndexFromPoint(
				newRoot,
				root.centerX,
				root.centerY,
				root.centerZ
			);
			newRoot.children![oldChildIndex] = root;
			this._root = newRoot;
		}
	}

	private _detachEntry(meshInstance: MeshInstance): boolean {
		const entry = this._entriesByMeshInstance.get(meshInstance);
		if (!entry) return false;
		const node = entry.node;
		const index = node.objects.indexOf(meshInstance);
		if (index >= 0) {
			node.objects.splice(index, 1);
			node.objectBounds.splice(index, 1);
		}
		this._entriesByMeshInstance.delete(meshInstance);
		return true;
	}

	private _queryNodeFrustum(
		node: LooseOctreeNode,
		frustum: Frustum,
		includeInvisible: boolean,
		maxResults: number,
		result: MeshInstance[]
	): boolean {
		if (result.length >= maxResults) return true;
		const looseHalfSize = node.halfSize * this._looseness;
		const nodeMinX = node.centerX - looseHalfSize;
		const nodeMinY = node.centerY - looseHalfSize;
		const nodeMinZ = node.centerZ - looseHalfSize;
		const nodeMaxX = node.centerX + looseHalfSize;
		const nodeMaxY = node.centerY + looseHalfSize;
		const nodeMaxZ = node.centerZ + looseHalfSize;
		const nodeStatus = classifyAABBFrustum(
			frustum,
			nodeMinX,
			nodeMinY,
			nodeMinZ,
			nodeMaxX,
			nodeMaxY,
			nodeMaxZ
		);
		if (nodeStatus === FRUSTUM_OUTSIDE) {
			return false;
		}

		if (nodeStatus === FRUSTUM_INSIDE) {
			this._appendNodeSubtree(node, includeInvisible, maxResults, result);
			return result.length >= maxResults;
		}

		for (let i = 0; i < node.objects.length; i++) {
			if (result.length >= maxResults) return true;
			const meshInstance = node.objects[i];
			if (!includeInvisible && meshInstance.visible === false) {
				continue;
			}
			const bounds = node.objectBounds[i];
			if (
				classifyAABBFrustum(
					frustum,
					bounds.min.x,
					bounds.min.y,
					bounds.min.z,
					bounds.max.x,
					bounds.max.y,
					bounds.max.z
				) !== FRUSTUM_OUTSIDE
			) {
				result.push(meshInstance);
			}
		}

		if (!node.children) return result.length >= maxResults;
		for (const child of node.children) {
			if (!child) continue;
			if (
				this._queryNodeFrustum(
					child,
					frustum,
					includeInvisible,
					maxResults,
					result
				)
			) {
				return true;
			}
		}
		return result.length >= maxResults;
	}

	private _appendNodeSubtree(
		node: LooseOctreeNode,
		includeInvisible: boolean,
		maxResults: number,
		result: MeshInstance[]
	): boolean {
		for (const meshInstance of node.objects) {
			if (result.length >= maxResults) return true;
			if (!includeInvisible && meshInstance.visible === false) {
				continue;
			}
			result.push(meshInstance);
		}
		if (!node.children) {
			return result.length >= maxResults;
		}
		for (const child of node.children) {
			if (!child) continue;
			if (this._appendNodeSubtree(child, includeInvisible, maxResults, result)) {
				return true;
			}
		}
		return result.length >= maxResults;
	}
}

function resolveLeafCapacity(value: number | undefined): number {
	if (!Number.isFinite(value)) return DEFAULT_LEAF_CAPACITY;
	return Math.max(1, Math.floor(value!));
}

function resolveMaxDepth(value: number | undefined): number {
	if (!Number.isFinite(value)) return DEFAULT_MAX_DEPTH;
	return Math.max(0, Math.floor(value!));
}

function resolveLooseness(value: number | undefined): number {
	if (!Number.isFinite(value)) return DEFAULT_LOOSENESS;
	return Math.max(1, value!);
}

function resolveMaxResults(value: number | undefined): number {
	if (value === undefined) return Infinity;
	if (!Number.isFinite(value)) return Infinity;
	return Math.max(0, Math.floor(value));
}

function resolveMaxDistance(value: number | undefined): number {
	if (value === undefined) return Infinity;
	if (!Number.isFinite(value)) return Infinity;
	return Math.max(0, value);
}

function createRootNode(bounds: BoundingBox): LooseOctreeNode {
	const centerX = (bounds.min.x + bounds.max.x) * 0.5;
	const centerY = (bounds.min.y + bounds.max.y) * 0.5;
	const centerZ = (bounds.min.z + bounds.max.z) * 0.5;
	const sizeX = bounds.max.x - bounds.min.x;
	const sizeY = bounds.max.y - bounds.min.y;
	const sizeZ = bounds.max.z - bounds.min.z;
	const extent = Math.max(sizeX, sizeY, sizeZ);
	let halfSize = Math.max(MIN_HALF_SIZE, extent * 0.5);
	halfSize = roundUpPowerOfTwo(halfSize);
	return {
		centerX,
		centerY,
		centerZ,
		halfSize,
		objects: [],
		objectBounds: [],
		children: null,
	};
}

function roundUpPowerOfTwo(value: number): number {
	if (!(value > 0)) return MIN_HALF_SIZE;
	let result = MIN_HALF_SIZE;
	while (result < value) {
		result *= 2;
	}
	return result;
}

function createChildrenArray(): Array<LooseOctreeNode | null> {
	return [null, null, null, null, null, null, null, null];
}

function containsBoundsInCube(
	centerX: number,
	centerY: number,
	centerZ: number,
	halfSize: number,
	bounds: BoundingBox
): boolean {
	return (
		bounds.min.x >= centerX - halfSize &&
		bounds.max.x <= centerX + halfSize &&
		bounds.min.y >= centerY - halfSize &&
		bounds.max.y <= centerY + halfSize &&
		bounds.min.z >= centerZ - halfSize &&
		bounds.max.z <= centerZ + halfSize
	);
}

function resolveChildPlacement(
	node: LooseOctreeNode,
	bounds: BoundingBox
): ChildPlacement | null {
	const childHalf = node.halfSize * 0.5;
	if (!(childHalf > MIN_HALF_SIZE)) return null;
	const centerX = (bounds.min.x + bounds.max.x) * 0.5;
	const centerY = (bounds.min.y + bounds.max.y) * 0.5;
	const centerZ = (bounds.min.z + bounds.max.z) * 0.5;
	const childCenterX = node.centerX + (centerX >= node.centerX ? childHalf : -childHalf);
	const childCenterY = node.centerY + (centerY >= node.centerY ? childHalf : -childHalf);
	const childCenterZ = node.centerZ + (centerZ >= node.centerZ ? childHalf : -childHalf);

	if (
		!containsBoundsInCube(
			childCenterX,
			childCenterY,
			childCenterZ,
			childHalf,
			bounds
		)
	) {
		return null;
	}

	const index = resolveChildIndexFromPoint(node, centerX, centerY, centerZ);
	return {
		index,
		centerX: childCenterX,
		centerY: childCenterY,
		centerZ: childCenterZ,
	};
}

function resolveChildIndexFromPoint(
	node: LooseOctreeNode,
	x: number,
	y: number,
	z: number
): number {
	let index = 0;
	if (x >= node.centerX) index |= 1;
	if (y >= node.centerY) index |= 2;
	if (z >= node.centerZ) index |= 4;
	return index;
}

function classifyAABBFrustum(
	frustum: Frustum,
	minX: number,
	minY: number,
	minZ: number,
	maxX: number,
	maxY: number,
	maxZ: number
): number {
	let fullyInside = true;
	for (const plane of frustum.planes) {
		const nx = plane.normal.x;
		const ny = plane.normal.y;
		const nz = plane.normal.z;

		const px = nx >= 0 ? maxX : minX;
		const py = ny >= 0 ? maxY : minY;
		const pz = nz >= 0 ? maxZ : minZ;
		const positiveDistance = nx * px + ny * py + nz * pz + plane.constant;
		if (positiveDistance < 0) {
			return FRUSTUM_OUTSIDE;
		}

		const nxPoint = nx >= 0 ? minX : maxX;
		const nyPoint = ny >= 0 ? minY : maxY;
		const nzPoint = nz >= 0 ? minZ : maxZ;
		const negativeDistance =
			nx * nxPoint + ny * nyPoint + nz * nzPoint + plane.constant;
		if (negativeDistance < 0) {
			fullyInside = false;
		}
	}
	return fullyInside ? FRUSTUM_INSIDE : FRUSTUM_INTERSECT;
}

function intersectRayAABB(
	origin: { x: number; y: number; z: number },
	direction: { x: number; y: number; z: number },
	maxDistance: number,
	minX: number,
	minY: number,
	minZ: number,
	maxX: number,
	maxY: number,
	maxZ: number
): number | null {
	let tMin = 0;
	let tMax = maxDistance;

	const axisHit = (
		originValue: number,
		directionValue: number,
		minValue: number,
		maxValue: number
	): boolean => {
		if (Math.abs(directionValue) < 1e-10) {
			return originValue >= minValue && originValue <= maxValue;
		}

		const invDirection = 1 / directionValue;
		let t0 = (minValue - originValue) * invDirection;
		let t1 = (maxValue - originValue) * invDirection;
		if (t0 > t1) {
			const tmp = t0;
			t0 = t1;
			t1 = tmp;
		}
		tMin = Math.max(tMin, t0);
		tMax = Math.min(tMax, t1);
		return tMax >= tMin;
	};

	if (!axisHit(origin.x, direction.x, minX, maxX)) return null;
	if (!axisHit(origin.y, direction.y, minY, maxY)) return null;
	if (!axisHit(origin.z, direction.z, minZ, maxZ)) return null;

	if (tMax < 0 || tMin > maxDistance) {
		return null;
	}
	if (tMin >= 0) return tMin;
	if (tMax >= 0) return 0;
	return null;
}

function compareRayHits(left: SpatialRayHit, right: SpatialRayHit): number {
	if (left.distance !== right.distance) {
		return left.distance - right.distance;
	}
	const leftEntity = left.meshInstance.entityId ?? Number.MAX_SAFE_INTEGER;
	const rightEntity = right.meshInstance.entityId ?? Number.MAX_SAFE_INTEGER;
	if (leftEntity !== rightEntity) {
		return leftEntity - rightEntity;
	}
	return left.meshInstance.id.localeCompare(right.meshInstance.id);
}
