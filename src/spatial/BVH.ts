import type { BoundingBox } from "../core/types";
import type { Frustum } from "../maths/Frustum";
import type { MeshInstance } from "../meshes";
import type {
	SpatialIndex3D,
	SpatialQueryOptions,
	SpatialRayHit,
	SpatialRayQueryOptions,
} from "./types";

export interface SpatialNode {
	bounds: BoundingBox;
	left?: SpatialNode;
	right?: SpatialNode;
	objects?: MeshInstance[];
	objectBounds?: BoundingBox[];
}

export type BVHQueryOptions = SpatialQueryOptions;
export type BVHRayQueryOptions = SpatialRayQueryOptions;
export type BVHRayHit = SpatialRayHit;

interface SpatialBuildEntry {
	meshInstance: MeshInstance;
	bounds: BoundingBox;
	centroidX: number;
	centroidY: number;
	centroidZ: number;
}

const DEFAULT_LEAF_SIZE = 8;
const DEGENERATE_AXIS_EPSILON = 1e-6;
const FRUSTUM_OUTSIDE = -1;
const FRUSTUM_INTERSECT = 0;
const FRUSTUM_INSIDE = 1;

export class BVH implements SpatialIndex3D {
	private _root: SpatialNode | null;
	private readonly _leafSize: number;
	private _entries: SpatialBuildEntry[];
	private _meshInstances: MeshInstance[];
	private _meshInstanceSet: Set<MeshInstance>;
	private _entryIndexByMeshInstance: Map<MeshInstance, number>;
	private _structureDirty: boolean;
	private _boundsDirtyMeshInstances: Set<MeshInstance>;

	constructor(
		meshInstances: MeshInstance[] = [],
		leafSize: number = DEFAULT_LEAF_SIZE
	) {
		this._root = null;
		this._leafSize = resolveLeafSize(leafSize);
		this._entries = [];
		this._meshInstances = [];
		this._meshInstanceSet = new Set();
		this._entryIndexByMeshInstance = new Map();
		this._structureDirty = false;
		this._boundsDirtyMeshInstances = new Set();
		this.rebuild(meshInstances);
	}

	public get root(): SpatialNode | null {
		return this._root;
	}

	public get size(): number {
		return this._meshInstances.length;
	}

	public get dirty(): boolean {
		return this._structureDirty || this._boundsDirtyMeshInstances.size > 0;
	}

	/**
	 * Replaces tracked mesh instances and marks BVH for lazy rebuild.
	 */
	public setMeshInstances(meshInstances: MeshInstance[]): void {
		this._meshInstances = meshInstances.slice();
		this._meshInstanceSet = new Set(this._meshInstances);
		this._structureDirty = true;
		this._boundsDirtyMeshInstances.clear();
	}

	/**
	 * Marks one tracked mesh (or all meshes) dirty so bounds are refreshed.
	 */
	public markDirty(meshInstance?: MeshInstance): void {
		if (!meshInstance) {
			for (const tracked of this._meshInstances) {
				this._boundsDirtyMeshInstances.add(tracked);
			}
			return;
		}
		if (this._meshInstanceSet.has(meshInstance)) {
			this._boundsDirtyMeshInstances.add(meshInstance);
		}
	}

	/**
	 * Adds a mesh instance, or refreshes a tracked one.
	 */
	public upsert(meshInstance: MeshInstance): void {
		if (!this._meshInstanceSet.has(meshInstance)) {
			this._meshInstanceSet.add(meshInstance);
			this._meshInstances.push(meshInstance);
			this._structureDirty = true;
			return;
		}
		this._boundsDirtyMeshInstances.add(meshInstance);
	}

	/**
	 * Removes a mesh instance from BVH tracking.
	 */
	public remove(meshInstance: MeshInstance): boolean {
		if (!this._meshInstanceSet.delete(meshInstance)) {
			return false;
		}
		const index = this._meshInstances.indexOf(meshInstance);
		if (index >= 0) {
			this._meshInstances.splice(index, 1);
		}
		this._boundsDirtyMeshInstances.delete(meshInstance);
		this._structureDirty = true;
		return true;
	}

	/**
	 * Rebuilds BVH nodes from current world-space mesh bounds.
	 */
	public rebuild(meshInstances?: MeshInstance[]): void {
		if (meshInstances) {
			this._meshInstances = meshInstances.slice();
			this._meshInstanceSet = new Set(this._meshInstances);
		}

		const count = this._meshInstances.length;
		if (count === 0) {
			this._entries.length = 0;
			this._entryIndexByMeshInstance.clear();
			this._root = null;
			this._structureDirty = false;
			this._boundsDirtyMeshInstances.clear();
			return;
		}

		this._entries = new Array<SpatialBuildEntry>(count);
		for (let index = 0; index < count; index++) {
			const meshInstance = this._meshInstances[index];
			const bounds = meshInstance.getWorldBoundingBox();
			this._entries[index] = createBuildEntry(meshInstance, bounds);
		}

		this._root = this._buildRange(0, this._entries.length);
		this._entryIndexByMeshInstance.clear();
		for (let index = 0; index < this._entries.length; index++) {
			this._entryIndexByMeshInstance.set(
				this._entries[index].meshInstance,
				index
			);
		}
		this._structureDirty = false;
		this._boundsDirtyMeshInstances.clear();
	}

	/**
	 * Returns mesh instances whose bounds overlap the given frustum.
	 */
	public queryFrustum(
		frustum: Frustum,
		options?: BVHQueryOptions
	): MeshInstance[] {
		this._ensureFresh();
		if (!this._root) return [];

		const maxResults = resolveMaxResults(options?.maxResults);
		const includeInvisible = options?.includeInvisible === true;
		const result: MeshInstance[] = [];

		this._queryNode(this._root, frustum, includeInvisible, maxResults, result);

		return result;
	}

	public queryRay(
		origin: { x: number; y: number; z: number },
		direction: { x: number; y: number; z: number },
		options?: BVHRayQueryOptions
	): MeshInstance[] {
		return this.queryRayDetailed(origin, direction, options).map(
			(hit) => hit.meshInstance
		);
	}

	public queryRayDetailed(
		origin: { x: number; y: number; z: number },
		direction: { x: number; y: number; z: number },
		options?: BVHRayQueryOptions
	): BVHRayHit[] {
		this._ensureFresh();
		if (!this._root) return [];

		const maxResults = resolveMaxResults(options?.maxResults);
		if (maxResults <= 0) return [];

		const maxDistance = resolveMaxDistance(options?.maxDistance);
		if (maxDistance <= 0) return [];

		const includeInvisible = options?.includeInvisible === true;
		const directionLength = Math.hypot(direction.x, direction.y, direction.z);
		if (!(directionLength > 1e-8)) {
			throw new Error("BVH.queryRay direction must be non-zero");
		}

		const invDirectionLength = 1 / directionLength;
		const normalizedDirection = {
			x: direction.x * invDirectionLength,
			y: direction.y * invDirectionLength,
			z: direction.z * invDirectionLength,
		};

		const stack: SpatialNode[] = [this._root];
		const hits: BVHRayHit[] = [];

		while (stack.length > 0) {
			const node = stack.pop();
			if (!node) continue;
			const nodeDistance = intersectRayAABB(
				origin,
				normalizedDirection,
				maxDistance,
				node.bounds.min,
				node.bounds.max
			);
			if (nodeDistance === null) continue;

			if (node.objects && node.objectBounds) {
				for (let index = 0; index < node.objects.length; index++) {
					const meshInstance = node.objects[index];
					if (!includeInvisible && meshInstance.visible === false) {
						continue;
					}
					const objectBounds = node.objectBounds[index];
					const distance = intersectRayAABB(
						origin,
						normalizedDirection,
						maxDistance,
						objectBounds.min,
						objectBounds.max
					);
					if (distance === null) continue;
					hits.push({
						meshInstance,
						distance,
					});
				}
				continue;
			}

			if (node.left) stack.push(node.left);
			if (node.right) stack.push(node.right);
		}

		if (hits.length === 0) {
			return [];
		}

		hits.sort((left, right) => {
			if (left.distance !== right.distance) {
				return left.distance - right.distance;
			}
			const leftEntity = left.meshInstance.entityId ?? Number.MAX_SAFE_INTEGER;
			const rightEntity =
				right.meshInstance.entityId ?? Number.MAX_SAFE_INTEGER;
			if (leftEntity !== rightEntity) {
				return leftEntity - rightEntity;
			}
			return left.meshInstance.id.localeCompare(right.meshInstance.id);
		});

		if (hits.length > maxResults) {
			return hits.slice(0, maxResults);
		}
		return hits;
	}

	private _ensureFresh(): void {
		if (this._structureDirty) {
			this.rebuild();
			return;
		}
		if (this._boundsDirtyMeshInstances.size === 0) return;
		this._refitDirtyBounds();
	}

	private _refitDirtyBounds(): void {
		if (!this._root || this._entries.length === 0) {
			this._boundsDirtyMeshInstances.clear();
			return;
		}

		for (const meshInstance of this._boundsDirtyMeshInstances) {
			const entryIndex = this._entryIndexByMeshInstance.get(meshInstance);
			if (entryIndex === undefined) continue;
			const entry = this._entries[entryIndex];
			const updatedBounds = meshInstance.getWorldBoundingBox();
			copyBoundingBoxValues(entry.bounds, updatedBounds);
			entry.centroidX = (entry.bounds.min.x + entry.bounds.max.x) * 0.5;
			entry.centroidY = (entry.bounds.min.y + entry.bounds.max.y) * 0.5;
			entry.centroidZ = (entry.bounds.min.z + entry.bounds.max.z) * 0.5;
		}

		this._refitNode(this._root);
		this._boundsDirtyMeshInstances.clear();
	}

	private _refitNode(node: SpatialNode): BoundingBox {
		if (node.objects && node.objectBounds) {
			unionBoundingBoxes(node.bounds, node.objectBounds);
			return node.bounds;
		}

		if (!node.left && !node.right) {
			return node.bounds;
		}

		if (!node.left) {
			const rightBounds = this._refitNode(node.right!);
			copyBoundingBoxValues(node.bounds, rightBounds);
			return node.bounds;
		}
		if (!node.right) {
			const leftBounds = this._refitNode(node.left);
			copyBoundingBoxValues(node.bounds, leftBounds);
			return node.bounds;
		}

		const leftBounds = this._refitNode(node.left);
		const rightBounds = this._refitNode(node.right);
		mergeBoundingBoxes(node.bounds, leftBounds, rightBounds);
		return node.bounds;
	}

	private _buildRange(start: number, end: number): SpatialNode | null {
		const count = end - start;
		if (count <= 0) return null;

		const stats = computeRangeStats(this._entries, start, end);
		if (count <= this._leafSize) {
			return createLeafNode(this._entries, start, end, stats.bounds);
		}

		const axis = resolveSplitAxisFromExtents(
			stats.centroidExtentX,
			stats.centroidExtentY,
			stats.centroidExtentZ
		);
		if (!axis) {
			const middle = start + (count >> 1);
			const left = this._buildRange(start, middle);
			const right = this._buildRange(middle, end);
			return createInnerNode(
				stats.bounds,
				left,
				right,
				this._entries,
				start,
				end
			);
		}

		const middle = start + (count >> 1);
		quickSelectByAxis(this._entries, start, end, middle, axis);

		const left = this._buildRange(start, middle);
		const right = this._buildRange(middle, end);
		return createInnerNode(
			stats.bounds,
			left,
			right,
			this._entries,
			start,
			end
		);
	}

	private _queryNode(
		node: SpatialNode,
		frustum: Frustum,
		includeInvisible: boolean,
		maxResults: number,
		result: MeshInstance[]
	): boolean {
		if (result.length >= maxResults) return true;

		const nodeFrustumStatus = classifyAABBFrustum(
			frustum,
			node.bounds.min,
			node.bounds.max
		);
		if (nodeFrustumStatus === FRUSTUM_OUTSIDE) {
			return false;
		}

		if (node.objects && node.objectBounds) {
			if (nodeFrustumStatus === FRUSTUM_INSIDE) {
				return this._appendLeafObjects(
					node.objects,
					includeInvisible,
					maxResults,
					result
				);
			}
			return this._appendLeafObjectsWithAABB(
				node.objects,
				node.objectBounds,
				frustum,
				includeInvisible,
				maxResults,
				result
			);
		}

		if (nodeFrustumStatus === FRUSTUM_INSIDE) {
			this._appendSubtree(node, includeInvisible, maxResults, result);
			return result.length >= maxResults;
		}

		if (
			node.left &&
			this._queryNode(node.left, frustum, includeInvisible, maxResults, result)
		) {
			return true;
		}

		if (
			node.right &&
			this._queryNode(node.right, frustum, includeInvisible, maxResults, result)
		) {
			return true;
		}

		return result.length >= maxResults;
	}

	private _appendSubtree(
		node: SpatialNode,
		includeInvisible: boolean,
		maxResults: number,
		result: MeshInstance[]
	): boolean {
		if (result.length >= maxResults) return true;

		if (node.objects) {
			return this._appendLeafObjects(
				node.objects,
				includeInvisible,
				maxResults,
				result
			);
		}

		if (
			node.left &&
			this._appendSubtree(node.left, includeInvisible, maxResults, result)
		) {
			return true;
		}

		if (
			node.right &&
			this._appendSubtree(node.right, includeInvisible, maxResults, result)
		) {
			return true;
		}

		return result.length >= maxResults;
	}

	private _appendLeafObjects(
		objects: MeshInstance[],
		includeInvisible: boolean,
		maxResults: number,
		result: MeshInstance[]
	): boolean {
		for (const meshInstance of objects) {
			if (result.length >= maxResults) return true;
			if (!includeInvisible && meshInstance.visible === false) {
				continue;
			}
			result.push(meshInstance);
		}
		return result.length >= maxResults;
	}

	private _appendLeafObjectsWithAABB(
		objects: MeshInstance[],
		bounds: BoundingBox[],
		frustum: Frustum,
		includeInvisible: boolean,
		maxResults: number,
		result: MeshInstance[]
	): boolean {
		const count = objects.length;
		for (let index = 0; index < count; index++) {
			if (result.length >= maxResults) return true;
			const meshInstance = objects[index];
			if (!includeInvisible && meshInstance.visible === false) {
				continue;
			}
			const objectBounds = bounds[index];
			if (
				classifyAABBFrustum(frustum, objectBounds.min, objectBounds.max) !==
				FRUSTUM_OUTSIDE
			) {
				result.push(meshInstance);
			}
		}
		return result.length >= maxResults;
	}
}

function resolveLeafSize(value: number): number {
	if (!Number.isFinite(value)) return DEFAULT_LEAF_SIZE;
	return Math.max(1, Math.floor(value));
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

function createBuildEntry(
	meshInstance: MeshInstance,
	bounds: BoundingBox
): SpatialBuildEntry {
	return {
		meshInstance,
		bounds,
		centroidX: (bounds.min.x + bounds.max.x) * 0.5,
		centroidY: (bounds.min.y + bounds.max.y) * 0.5,
		centroidZ: (bounds.min.z + bounds.max.z) * 0.5,
	};
}

function createLeafNode(
	entries: SpatialBuildEntry[],
	start: number,
	end: number,
	bounds: BoundingBox
): SpatialNode {
	const count = end - start;
	const objects = new Array<MeshInstance>(count);
	const objectBounds = new Array<BoundingBox>(count);

	for (let index = 0; index < count; index++) {
		const entry = entries[start + index];
		objects[index] = entry.meshInstance;
		objectBounds[index] = entry.bounds;
	}

	return {
		bounds,
		objects,
		objectBounds,
	};
}

function createInnerNode(
	bounds: BoundingBox,
	left: SpatialNode | null,
	right: SpatialNode | null,
	entries: SpatialBuildEntry[],
	start: number,
	end: number
): SpatialNode {
	if (!left && !right) {
		return createLeafNode(entries, start, end, bounds);
	}
	if (!left || !right) {
		const fallbackMiddle = start + ((end - start) >> 1);
		const fallbackLeft = left ?? createLeafNode(entries, start, fallbackMiddle, bounds);
		const fallbackRight =
			right ?? createLeafNode(entries, fallbackMiddle, end, bounds);
		return {
			bounds,
			left: fallbackLeft,
			right: fallbackRight,
		};
	}
	return {
		bounds,
		left,
		right,
	};
}

function resolveSplitAxisFromExtents(
	extentX: number,
	extentY: number,
	extentZ: number
): "x" | "y" | "z" | null {
	const maxExtent = Math.max(extentX, extentY, extentZ);
	if (!(maxExtent > DEGENERATE_AXIS_EPSILON)) {
		return null;
	}
	if (extentX >= extentY && extentX >= extentZ) return "x";
	if (extentY >= extentX && extentY >= extentZ) return "y";
	return "z";
}

function getEntryCentroid(entry: SpatialBuildEntry, axis: "x" | "y" | "z"): number {
	if (axis === "x") return entry.centroidX;
	if (axis === "y") return entry.centroidY;
	return entry.centroidZ;
}

function swapEntries(
	entries: SpatialBuildEntry[],
	leftIndex: number,
	rightIndex: number
): void {
	if (leftIndex === rightIndex) return;
	const tmp = entries[leftIndex];
	entries[leftIndex] = entries[rightIndex];
	entries[rightIndex] = tmp;
}

/**
 * 3-way quickselect partition to avoid full range sorting at each node.
 */
function quickSelectByAxis(
	entries: SpatialBuildEntry[],
	start: number,
	end: number,
	target: number,
	axis: "x" | "y" | "z"
): void {
	let left = start;
	let right = end - 1;

	while (left < right) {
		const pivotIndex = left + ((right - left) >> 1);
		const pivotValue = getEntryCentroid(entries[pivotIndex], axis);
		let lt = left;
		let gt = right;
		let index = left;

		while (index <= gt) {
			const value = getEntryCentroid(entries[index], axis);
			if (value < pivotValue) {
				swapEntries(entries, lt, index);
				lt++;
				index++;
			} else if (value > pivotValue) {
				swapEntries(entries, index, gt);
				gt--;
			} else {
				index++;
			}
		}

		if (target < lt) {
			right = lt - 1;
		} else if (target > gt) {
			left = gt + 1;
		} else {
			return;
		}
	}
}

function computeRangeStats(
	entries: SpatialBuildEntry[],
	start: number,
	end: number
): {
	bounds: BoundingBox;
	centroidExtentX: number;
	centroidExtentY: number;
	centroidExtentZ: number;
} {
	let minX = Infinity;
	let minY = Infinity;
	let minZ = Infinity;
	let maxX = -Infinity;
	let maxY = -Infinity;
	let maxZ = -Infinity;
	let centroidMinX = Infinity;
	let centroidMinY = Infinity;
	let centroidMinZ = Infinity;
	let centroidMaxX = -Infinity;
	let centroidMaxY = -Infinity;
	let centroidMaxZ = -Infinity;

	for (let index = start; index < end; index++) {
		const entry = entries[index];
		const min = entry.bounds.min;
		const max = entry.bounds.max;
		if (min.x < minX) minX = min.x;
		if (min.y < minY) minY = min.y;
		if (min.z < minZ) minZ = min.z;
		if (max.x > maxX) maxX = max.x;
		if (max.y > maxY) maxY = max.y;
		if (max.z > maxZ) maxZ = max.z;

		if (entry.centroidX < centroidMinX) centroidMinX = entry.centroidX;
		if (entry.centroidY < centroidMinY) centroidMinY = entry.centroidY;
		if (entry.centroidZ < centroidMinZ) centroidMinZ = entry.centroidZ;
		if (entry.centroidX > centroidMaxX) centroidMaxX = entry.centroidX;
		if (entry.centroidY > centroidMaxY) centroidMaxY = entry.centroidY;
		if (entry.centroidZ > centroidMaxZ) centroidMaxZ = entry.centroidZ;
	}

	return {
		bounds: {
			min: { x: minX, y: minY, z: minZ },
			max: { x: maxX, y: maxY, z: maxZ },
		},
		centroidExtentX: centroidMaxX - centroidMinX,
		centroidExtentY: centroidMaxY - centroidMinY,
		centroidExtentZ: centroidMaxZ - centroidMinZ,
	};
}

function copyBoundingBoxValues(target: BoundingBox, source: BoundingBox): void {
	target.min.x = source.min.x;
	target.min.y = source.min.y;
	target.min.z = source.min.z;
	target.max.x = source.max.x;
	target.max.y = source.max.y;
	target.max.z = source.max.z;
}

function mergeBoundingBoxes(
	target: BoundingBox,
	left: BoundingBox,
	right: BoundingBox
): void {
	target.min.x = Math.min(left.min.x, right.min.x);
	target.min.y = Math.min(left.min.y, right.min.y);
	target.min.z = Math.min(left.min.z, right.min.z);
	target.max.x = Math.max(left.max.x, right.max.x);
	target.max.y = Math.max(left.max.y, right.max.y);
	target.max.z = Math.max(left.max.z, right.max.z);
}

function unionBoundingBoxes(target: BoundingBox, bounds: BoundingBox[]): void {
	if (bounds.length === 0) return;
	let minX = Infinity;
	let minY = Infinity;
	let minZ = Infinity;
	let maxX = -Infinity;
	let maxY = -Infinity;
	let maxZ = -Infinity;
	for (const bound of bounds) {
		if (bound.min.x < minX) minX = bound.min.x;
		if (bound.min.y < minY) minY = bound.min.y;
		if (bound.min.z < minZ) minZ = bound.min.z;
		if (bound.max.x > maxX) maxX = bound.max.x;
		if (bound.max.y > maxY) maxY = bound.max.y;
		if (bound.max.z > maxZ) maxZ = bound.max.z;
	}
	target.min.x = minX;
	target.min.y = minY;
	target.min.z = minZ;
	target.max.x = maxX;
	target.max.y = maxY;
	target.max.z = maxZ;
}

function classifyAABBFrustum(
	frustum: Frustum,
	min: { x: number; y: number; z: number },
	max: { x: number; y: number; z: number }
): number {
	let fullyInside = true;

	for (const plane of frustum.planes) {
		const nx = plane.normal.x;
		const ny = plane.normal.y;
		const nz = plane.normal.z;

		const px = nx >= 0 ? max.x : min.x;
		const py = ny >= 0 ? max.y : min.y;
		const pz = nz >= 0 ? max.z : min.z;
		const positiveDistance = nx * px + ny * py + nz * pz + plane.constant;
		if (positiveDistance < 0) {
			return FRUSTUM_OUTSIDE;
		}

		const nxPoint = nx >= 0 ? min.x : max.x;
		const nyPoint = ny >= 0 ? min.y : max.y;
		const nzPoint = nz >= 0 ? min.z : max.z;
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
	min: { x: number; y: number; z: number },
	max: { x: number; y: number; z: number }
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

	if (!axisHit(origin.x, direction.x, min.x, max.x)) return null;
	if (!axisHit(origin.y, direction.y, min.y, max.y)) return null;
	if (!axisHit(origin.z, direction.z, min.z, max.z)) return null;

	if (tMax < 0 || tMin > maxDistance) {
		return null;
	}

	if (tMin >= 0) {
		return tMin;
	}
	if (tMax >= 0) {
		return 0;
	}
	return null;
}
