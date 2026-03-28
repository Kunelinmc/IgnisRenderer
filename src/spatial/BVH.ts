import type { BoundingBox } from "../core/types";
import type { Frustum } from "../maths/Frustum";
import type { MeshInstance } from "../meshes";

export interface SpatialNode {
	bounds: BoundingBox;
	left?: SpatialNode;
	right?: SpatialNode;
	objects?: MeshInstance[];
	objectBounds?: BoundingBox[];
}

export interface BVHQueryOptions {
	maxResults?: number;
	includeInvisible?: boolean;
}

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

export class BVH {
	private _root: SpatialNode | null;
	private readonly _leafSize: number;
	private _entries: SpatialBuildEntry[];
	private _meshInstances: MeshInstance[];
	private _meshInstanceSet: Set<MeshInstance>;
	private _dirty: boolean;

	constructor(
		meshInstances: MeshInstance[] = [],
		leafSize: number = DEFAULT_LEAF_SIZE
	) {
		this._root = null;
		this._leafSize = resolveLeafSize(leafSize);
		this._entries = [];
		this._meshInstances = [];
		this._meshInstanceSet = new Set();
		this._dirty = false;
		this.rebuild(meshInstances);
	}

	public get root(): SpatialNode | null {
		return this._root;
	}

	public get size(): number {
		return this._meshInstances.length;
	}

	public get dirty(): boolean {
		return this._dirty;
	}

	/**
	 * Replaces tracked mesh instances and marks BVH for lazy rebuild.
	 */
	public setMeshInstances(meshInstances: MeshInstance[]): void {
		this._meshInstances = meshInstances.slice();
		this._meshInstanceSet = new Set(this._meshInstances);
		this._dirty = true;
	}

	/**
	 * Marks one tracked mesh (or all meshes) dirty so bounds are refreshed.
	 */
	public markDirty(meshInstance?: MeshInstance): void {
		if (!meshInstance || this._meshInstanceSet.has(meshInstance)) {
			this._dirty = true;
		}
	}

	/**
	 * Adds a mesh instance, or refreshes a tracked one.
	 */
	public upsert(meshInstance: MeshInstance): void {
		if (!this._meshInstanceSet.has(meshInstance)) {
			this._meshInstanceSet.add(meshInstance);
			this._meshInstances.push(meshInstance);
		}
		this._dirty = true;
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
		this._dirty = true;
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
			this._root = null;
			this._dirty = false;
			return;
		}

		this._entries = new Array<SpatialBuildEntry>(count);
		for (let index = 0; index < count; index++) {
			const meshInstance = this._meshInstances[index];
			const bounds = meshInstance.getWorldBoundingBox();
			this._entries[index] = createBuildEntry(meshInstance, bounds);
		}

		this._root = this._buildRange(0, this._entries.length);
		this._dirty = false;
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

		this._queryNode(
			this._root,
			frustum,
			includeInvisible,
			maxResults,
			result
		);

		return result;
	}

	private _ensureFresh(): void {
		if (this._dirty) {
			this.rebuild();
		}
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
			return createInnerNode(stats.bounds, left, right, this._entries, start, end);
		}

		const middle = start + (count >> 1);
		quickSelectByAxis(this._entries, start, end, middle, axis);

		const left = this._buildRange(start, middle);
		const right = this._buildRange(middle, end);
		return createInnerNode(stats.bounds, left, right, this._entries, start, end);
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

		if (node.left && this._queryNode(node.left, frustum, includeInvisible, maxResults, result)) {
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

		if (node.left && this._appendSubtree(node.left, includeInvisible, maxResults, result)) {
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
				classifyAABBFrustum(
					frustum,
					objectBounds.min,
					objectBounds.max
				) !== FRUSTUM_OUTSIDE
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
