import type { BoundingBox } from "../core/types";
import type { Frustum } from "../maths/Frustum";
import type { MeshInstance } from "../meshes";
import type {
	SpatialIndex3D,
	SpatialBounds3D,
	SpatialQueryOptions,
	SpatialRayHit,
	SpatialRayQueryOptions,
} from "./types";

export interface SpatialNode {
	bounds: BoundingBox;
	parent: SpatialNode | null;
	subtreeObjectCount: number;
	surfaceArea: number;
	left?: SpatialNode;
	right?: SpatialNode;
	objects?: MeshInstance[];
	objectBounds?: BoundingBox[];
}

export type BVHQueryOptions = SpatialQueryOptions;
export type BVHRayQueryOptions = SpatialRayQueryOptions;
export type BVHRayHit = SpatialRayHit;
export type BVHBuildStrategy = "median" | "sah";

export interface BVHOptions {
	leafSize?: number;
	buildStrategy?: BVHBuildStrategy;
	rebuildDirtyRatio?: number;
	rebuildSurfaceAreaInflation?: number;
}

interface SpatialBuildEntry {
	meshInstance: MeshInstance;
	bounds: BoundingBox;
	centroidX: number;
	centroidY: number;
	centroidZ: number;
}

interface SpatialLeafLocator {
	leaf: SpatialNode;
	objectIndex: number;
	entry: SpatialBuildEntry;
}

interface SpatialRangeStats {
	bounds: BoundingBox;
	centroidMinX: number;
	centroidMinY: number;
	centroidMinZ: number;
	centroidMaxX: number;
	centroidMaxY: number;
	centroidMaxZ: number;
	centroidExtentX: number;
	centroidExtentY: number;
	centroidExtentZ: number;
}

interface BinnedSAHSplit {
	axis: "x" | "y" | "z";
	threshold: number;
}

interface SAHBucket {
	count: number;
	minX: number;
	minY: number;
	minZ: number;
	maxX: number;
	maxY: number;
	maxZ: number;
}

const DEFAULT_LEAF_SIZE = 8;
const DEFAULT_BUILD_STRATEGY: BVHBuildStrategy = "median";
const DEFAULT_REBUILD_DIRTY_RATIO = 0.15;
const DEFAULT_REBUILD_SURFACE_AREA_INFLATION = 2;
const SAH_BUCKET_COUNT = 8;
const DEGENERATE_AXIS_EPSILON = 1e-6;
const FRUSTUM_OUTSIDE = -1;
const FRUSTUM_INTERSECT = 0;
const FRUSTUM_INSIDE = 1;

export class BVH implements SpatialIndex3D {
	private _root: SpatialNode | null;
	private readonly _leafSize: number;
	private readonly _buildStrategy: BVHBuildStrategy;
	private readonly _rebuildDirtyRatio: number;
	private readonly _rebuildSurfaceAreaInflation: number;
	private _entries: SpatialBuildEntry[];
	private _meshInstances: MeshInstance[];
	private _meshInstanceSet: Set<MeshInstance>;
	private _leafLocatorByMeshInstance: Map<MeshInstance, SpatialLeafLocator>;
	private _structureDirty: boolean;
	private _boundsDirtyMeshInstances: Set<MeshInstance>;
	private _boundsScratch: BoundingBox;
	private _qualityCost = 0;
	private _qualityBaseline = 0;
	private _fullRebuildCount = 0;
	private _lastRefitNodeCount = 0;
	private readonly _queryNodeStack: SpatialNode[] = [];
	private readonly _queryStatusStack: number[] = [];
	private readonly _frustumPlaneData = new Float64Array(24);
	private readonly _rayNodeHeap: SpatialNode[] = [];
	private readonly _rayDistanceHeap: number[] = [];
	private readonly _dirtyLeafScratch = new Set<SpatialNode>();
	private readonly _refitCurrentScratch = new Set<SpatialNode>();
	private readonly _refitNextScratch = new Set<SpatialNode>();

	constructor(
		meshInstances: MeshInstance[] = [],
		options: number | BVHOptions = DEFAULT_LEAF_SIZE
	) {
		const resolvedOptions = resolveBVHOptions(options);
		this._root = null;
		this._leafSize = BVH._resolveLeafSize(resolvedOptions.leafSize);
		this._buildStrategy = resolvedOptions.buildStrategy;
		this._rebuildDirtyRatio = resolvedOptions.rebuildDirtyRatio;
		this._rebuildSurfaceAreaInflation =
			resolvedOptions.rebuildSurfaceAreaInflation;
		this._entries = [];
		this._meshInstances = [];
		this._meshInstanceSet = new Set();
		this._leafLocatorByMeshInstance = new Map();
		this._structureDirty = false;
		this._boundsDirtyMeshInstances = new Set();
		this._boundsScratch = createBoundingBox();
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
		this._fullRebuildCount++;
		this._lastRefitNodeCount = 0;
		if (meshInstances) {
			this._meshInstances = meshInstances.slice();
			this._meshInstanceSet = new Set(this._meshInstances);
		}

		const count = this._meshInstances.length;
		if (count === 0) {
			this._entries.length = 0;
			this._leafLocatorByMeshInstance.clear();
			this._root = null;
			this._qualityCost = 0;
			this._qualityBaseline = 0;
			this._structureDirty = false;
			this._boundsDirtyMeshInstances.clear();
			return;
		}

		this._entries = new Array<SpatialBuildEntry>(count);
		for (let index = 0; index < count; index++) {
			const meshInstance = this._meshInstances[index];
			const bounds = meshInstance.getOwnWorldBoundingBox();
			this._entries[index] = createBuildEntry(meshInstance, bounds);
		}

		this._leafLocatorByMeshInstance.clear();
		this._root = this._buildRange(0, this._entries.length);
		if (this._root) this._root.parent = null;
		this._qualityCost = computeTreeQualityCost(this._root);
		this._qualityBaseline = this._qualityCost;
		this._structureDirty = false;
		this._boundsDirtyMeshInstances.clear();
	}

	/**
	 * Returns mesh instances whose bounds overlap the given frustum.
	 */
	public queryFrustumInto(
		frustum: Frustum,
		out: MeshInstance[],
		options?: BVHQueryOptions
	): MeshInstance[] {
		out.length = 0;
		this._ensureFresh();
		if (!this._root) return out;

		const maxResults = resolveMaxResults(options?.maxResults);
		if (maxResults <= 0) return out;
		const includeInvisible = options?.includeInvisible === true;

		captureFrustumPlaneData(frustum, this._frustumPlaneData);
		this._queryFrustumIterative(
			this._root,
			this._frustumPlaneData,
			includeInvisible,
			maxResults,
			out
		);

		return out;
	}

	public queryFrustum(
		frustum: Frustum,
		options?: BVHQueryOptions
	): MeshInstance[] {
		return this.queryFrustumInto(frustum, [], options);
	}

	public queryBoundsInto(
		bounds: SpatialBounds3D,
		out: MeshInstance[],
		options?: BVHQueryOptions
	): MeshInstance[] {
		out.length = 0;
		this._ensureFresh();
		if (!this._root) return out;

		const maxResults = resolveMaxResults(options?.maxResults);
		if (maxResults <= 0) return out;
		const includeInvisible = options?.includeInvisible === true;
		this._queryBoundsIterative(
			this._root,
			bounds,
			includeInvisible,
			maxResults,
			out
		);
		return out;
	}

	public queryBounds(
		bounds: SpatialBounds3D,
		options?: BVHQueryOptions
	): MeshInstance[] {
		return this.queryBoundsInto(bounds, [], options);
	}

	public queryRay(
		origin: { x: number; y: number; z: number },
		direction: { x: number; y: number; z: number },
		options?: BVHRayQueryOptions
	): MeshInstance[] {
		const hits = this.queryRayDetailed(origin, direction, options);
		const result = new Array<MeshInstance>(hits.length);
		for (let i = 0; i < hits.length; i++) {
			result[i] = hits[i].meshInstance;
		}
		return result;
	}

	public queryRayDetailedInto(
		origin: { x: number; y: number; z: number },
		direction: { x: number; y: number; z: number },
		out: BVHRayHit[],
		options?: BVHRayQueryOptions
	): BVHRayHit[] {
		out.length = 0;
		const normalizedDirection = normalizeRayDirection(
			direction,
			"BVH.queryRayDetailedInto"
		);
		this._ensureFresh();
		if (!this._root) return out;

		const maxResults = resolveMaxResults(options?.maxResults);
		if (maxResults <= 0) return out;

		const maxDistance = resolveMaxDistance(options?.maxDistance);
		if (maxDistance <= 0) return out;

		const includeInvisible = options?.includeInvisible === true;
		if (maxResults === 1) {
			const hit = this._queryNearestRayHit(
				this._root,
				origin,
				normalizedDirection,
				maxDistance,
				includeInvisible
			);
			if (hit) {
				out.push(hit);
			}
			return out;
		}
		if (Number.isFinite(maxResults)) {
			this._queryTopKRayHits(
				this._root,
				origin,
				normalizedDirection,
				maxDistance,
				maxResults,
				includeInvisible,
				out
			);
			out.sort(compareRayHits);
			return out;
		}

		const stack = this._queryNodeStack;
		stack.length = 0;
		stack.push(this._root);

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
					out.push({
						meshInstance,
						distance,
					});
				}
				continue;
			}

			if (node.right) stack.push(node.right);
			if (node.left) stack.push(node.left);
		}

		if (out.length === 0) {
			return out;
		}

		out.sort(compareRayHits);

		if (out.length > maxResults) {
			out.length = maxResults;
		}
		return out;
	}

	public queryRayDetailed(
		origin: { x: number; y: number; z: number },
		direction: { x: number; y: number; z: number },
		options?: BVHRayQueryOptions
	): BVHRayHit[] {
		return this.queryRayDetailedInto(origin, direction, [], options);
	}

	private _queryNearestRayHit(
		root: SpatialNode,
		origin: { x: number; y: number; z: number },
		normalizedDirection: { x: number; y: number; z: number },
		maxDistance: number,
		includeInvisible: boolean
	): BVHRayHit | null {
		const rootDistance = intersectRayAABB(
			origin,
			normalizedDirection,
			maxDistance,
			root.bounds.min,
			root.bounds.max
		);
		if (rootDistance === null) return null;
		const nodes = this._rayNodeHeap;
		const distances = this._rayDistanceHeap;
		nodes.length = 0;
		distances.length = 0;
		pushRayNodeMinHeap(nodes, distances, root, rootDistance);

		let best: BVHRayHit | null = null;
		let bestDistance = maxDistance;
		while (nodes.length > 0) {
			const node = nodes[0];
			const nodeDistance = distances[0];
			popRayNodeMinHeap(nodes, distances);
			if (nodeDistance > bestDistance) break;

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
						bestDistance,
						objectBounds.min,
						objectBounds.max
					);
					if (distance === null) continue;
					const candidate = { meshInstance, distance };
					if (!best || compareRayHits(candidate, best) < 0) {
						best = candidate;
						bestDistance = distance;
					}
				}
				continue;
			}

			this._pushRayChild(
				node.left,
				origin,
				normalizedDirection,
				bestDistance,
				nodes,
				distances
			);
			this._pushRayChild(
				node.right,
				origin,
				normalizedDirection,
				bestDistance,
				nodes,
				distances
			);
		}
		return best;
	}

	private _queryTopKRayHits(
		root: SpatialNode,
		origin: { x: number; y: number; z: number },
		normalizedDirection: { x: number; y: number; z: number },
		maxDistance: number,
		maxResults: number,
		includeInvisible: boolean,
		out: BVHRayHit[]
	): void {
		const rootDistance = intersectRayAABB(
			origin,
			normalizedDirection,
			maxDistance,
			root.bounds.min,
			root.bounds.max
		);
		if (rootDistance === null) return;

		const nodes = this._rayNodeHeap;
		const distances = this._rayDistanceHeap;
		nodes.length = 0;
		distances.length = 0;
		pushRayNodeMinHeap(nodes, distances, root, rootDistance);
		let traversalMaxDistance = maxDistance;

		while (nodes.length > 0) {
			const node = nodes[0];
			const nodeDistance = distances[0];
			popRayNodeMinHeap(nodes, distances);
			if (nodeDistance > traversalMaxDistance) break;

			if (node.objects && node.objectBounds) {
				for (let index = 0; index < node.objects.length; index++) {
					const meshInstance = node.objects[index];
					if (!includeInvisible && meshInstance.visible === false) continue;
					const bounds = node.objectBounds[index];
					const distance = intersectRayAABB(
						origin,
						normalizedDirection,
						traversalMaxDistance,
						bounds.min,
						bounds.max
					);
					if (distance === null) continue;
					if (out.length < maxResults) {
						pushRayHitMaxHeap(out, { meshInstance, distance });
					} else if (
						compareRayCandidate(distance, meshInstance, out[0]) < 0
					) {
						out[0] = { meshInstance, distance };
						siftRayHitMaxHeapDown(out, 0);
					}
					if (out.length === maxResults) {
						traversalMaxDistance = Math.min(
							maxDistance,
							out[0].distance
						);
					}
				}
				continue;
			}

			this._pushRayChild(
				node.left,
				origin,
				normalizedDirection,
				traversalMaxDistance,
				nodes,
				distances
			);
			this._pushRayChild(
				node.right,
				origin,
				normalizedDirection,
				traversalMaxDistance,
				nodes,
				distances
			);
		}
	}

	private _pushRayChild(
		child: SpatialNode | undefined,
		origin: { x: number; y: number; z: number },
		direction: { x: number; y: number; z: number },
		maxDistance: number,
		nodes: SpatialNode[],
		distances: number[]
	): void {
		if (!child) return;
		const distance = intersectRayAABB(
			origin,
			direction,
			maxDistance,
			child.bounds.min,
			child.bounds.max
		);
		if (distance === null) return;
		pushRayNodeMinHeap(nodes, distances, child, distance);
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
		if (
			this._entries.length > this._leafSize * 2 &&
			this._boundsDirtyMeshInstances.size / this._entries.length >=
			this._rebuildDirtyRatio
		) {
			this.rebuild();
			return;
		}

		const dirtyLeaves = this._dirtyLeafScratch;
		dirtyLeaves.clear();
		this._lastRefitNodeCount = 0;
		for (const meshInstance of this._boundsDirtyMeshInstances) {
			const locator = this._leafLocatorByMeshInstance.get(meshInstance);
			if (!locator) continue;
			const entry = locator.entry;
			meshInstance.getOwnWorldBoundingBox(this._boundsScratch);
			copyBoundingBoxValues(entry.bounds, this._boundsScratch);
			entry.centroidX = (entry.bounds.min.x + entry.bounds.max.x) * 0.5;
			entry.centroidY = (entry.bounds.min.y + entry.bounds.max.y) * 0.5;
			entry.centroidZ = (entry.bounds.min.z + entry.bounds.max.z) * 0.5;
			dirtyLeaves.add(locator.leaf);
		}

		let currentLevel = this._refitCurrentScratch;
		let nextLevel = this._refitNextScratch;
		currentLevel.clear();
		nextLevel.clear();
		for (const leaf of dirtyLeaves) {
			this._lastRefitNodeCount++;
			if (this._refitLeaf(leaf) && leaf.parent) {
				currentLevel.add(leaf.parent);
			}
		}
		while (currentLevel.size > 0) {
			nextLevel.clear();
			for (const node of currentLevel) {
				this._lastRefitNodeCount++;
				if (this._refitInnerNode(node) && node.parent) {
					nextLevel.add(node.parent);
				}
			}
			const swap = currentLevel;
			currentLevel = nextLevel;
			nextLevel = swap;
		}

		if (
			this._qualityBaseline > 0 &&
			this._qualityCost / this._qualityBaseline >=
				this._rebuildSurfaceAreaInflation
		) {
			this.rebuild();
			return;
		}
		this._boundsDirtyMeshInstances.clear();
	}

	private _refitLeaf(node: SpatialNode): boolean {
		if (!node.objectBounds) return false;
		const bounds = node.bounds;
		const minX = bounds.min.x;
		const minY = bounds.min.y;
		const minZ = bounds.min.z;
		const maxX = bounds.max.x;
		const maxY = bounds.max.y;
		const maxZ = bounds.max.z;
		unionBoundingBoxes(node.bounds, node.objectBounds);
		return this._finishNodeRefit(node, minX, minY, minZ, maxX, maxY, maxZ);
	}

	private _refitInnerNode(node: SpatialNode): boolean {
		const bounds = node.bounds;
		const minX = bounds.min.x;
		const minY = bounds.min.y;
		const minZ = bounds.min.z;
		const maxX = bounds.max.x;
		const maxY = bounds.max.y;
		const maxZ = bounds.max.z;
		if (!node.left && !node.right) return false;
		if (!node.left) copyBoundingBoxValues(node.bounds, node.right!.bounds);
		else if (!node.right) copyBoundingBoxValues(node.bounds, node.left.bounds);
		else mergeBoundingBoxes(node.bounds, node.left.bounds, node.right.bounds);
		return this._finishNodeRefit(node, minX, minY, minZ, maxX, maxY, maxZ);
	}

	private _finishNodeRefit(
		node: SpatialNode,
		minX: number,
		minY: number,
		minZ: number,
		maxX: number,
		maxY: number,
		maxZ: number
	): boolean {
		const bounds = node.bounds;
		if (
			bounds.min.x === minX &&
			bounds.min.y === minY &&
			bounds.min.z === minZ &&
			bounds.max.x === maxX &&
			bounds.max.y === maxY &&
			bounds.max.z === maxZ
		) {
			return false;
		}
		const previousContribution = node.surfaceArea * node.subtreeObjectCount;
		node.surfaceArea = computeBoundingBoxSurfaceArea(node.bounds);
		this._qualityCost +=
			node.surfaceArea * node.subtreeObjectCount - previousContribution;
		return true;
	}

	private _buildRange(start: number, end: number): SpatialNode | null {
		const count = end - start;
		if (count <= 0) return null;

		const stats = computeRangeStats(this._entries, start, end);
		if (count <= this._leafSize) {
			return this._createLeafNode(start, end, stats.bounds);
		}

		const sahSplit =
			this._buildStrategy === "sah" ?
				resolveBinnedSAHSplit(this._entries, start, end, stats)
			:	null;
		if (sahSplit) {
			const middle = partitionEntriesByAxisThreshold(
				this._entries,
				start,
				end,
				sahSplit.axis,
				sahSplit.threshold
			);
			if (middle > start && middle < end) {
				const left = this._buildRange(start, middle);
				const right = this._buildRange(middle, end);
				return this._createInnerNode(
					stats.bounds,
					left,
					right,
					start,
					end
				);
			}
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
			return this._createInnerNode(
				stats.bounds,
				left,
				right,
				start,
				end
			);
		}

		const middle = start + (count >> 1);
		quickSelectByAxis(this._entries, start, end, middle, axis);

		const left = this._buildRange(start, middle);
		const right = this._buildRange(middle, end);
		return this._createInnerNode(
			stats.bounds,
			left,
			right,
			start,
			end
		);
	}

	private static _resolveLeafSize(value: number): number {
		if (!Number.isFinite(value)) return DEFAULT_LEAF_SIZE;
		return Math.max(1, Math.floor(value));
	}

	private _createLeafNode(
		start: number,
		end: number,
		bounds: BoundingBox
	): SpatialNode {
		const count = end - start;
		const objects = new Array<MeshInstance>(count);
		const objectBounds = new Array<BoundingBox>(count);

		for (let index = 0; index < count; index++) {
			const entry = this._entries[start + index];
			objects[index] = entry.meshInstance;
			objectBounds[index] = entry.bounds;
		}

		const node: SpatialNode = {
			bounds,
			parent: null,
			subtreeObjectCount: count,
			surfaceArea: computeBoundingBoxSurfaceArea(bounds),
			objects,
			objectBounds,
		};
		for (let index = 0; index < count; index++) {
			const entry = this._entries[start + index];
			this._leafLocatorByMeshInstance.set(entry.meshInstance, {
				leaf: node,
				objectIndex: index,
				entry,
			});
		}
		return node;
	}

	private _createInnerNode(
		bounds: BoundingBox,
		left: SpatialNode | null,
		right: SpatialNode | null,
		start: number,
		end: number
	): SpatialNode {
		if (!left && !right) {
			return this._createLeafNode(start, end, bounds);
		}
		if (!left || !right) {
			const fallbackMiddle = start + ((end - start) >> 1);
			const fallbackLeft =
				left ?? this._createLeafNode(start, fallbackMiddle, bounds);
			const fallbackRight =
				right ?? this._createLeafNode(fallbackMiddle, end, bounds);
			const node: SpatialNode = {
				bounds,
				parent: null,
				subtreeObjectCount:
					fallbackLeft.subtreeObjectCount + fallbackRight.subtreeObjectCount,
				surfaceArea: computeBoundingBoxSurfaceArea(bounds),
				left: fallbackLeft,
				right: fallbackRight,
			};
			fallbackLeft.parent = node;
			fallbackRight.parent = node;
			return node;
		}
		const node: SpatialNode = {
			bounds,
			parent: null,
			subtreeObjectCount:
				left.subtreeObjectCount + right.subtreeObjectCount,
			surfaceArea: computeBoundingBoxSurfaceArea(bounds),
			left,
			right,
		};
		left.parent = node;
		right.parent = node;
		return node;
	}

	private _queryFrustumIterative(
		root: SpatialNode,
		planeData: Float64Array,
		includeInvisible: boolean,
		maxResults: number,
		result: MeshInstance[]
	): void {
		const rootStatus = classifyAABBFrustumData(
			planeData,
			root.bounds.min,
			root.bounds.max
		);
		if (rootStatus === FRUSTUM_OUTSIDE) return;
		const nodes = this._queryNodeStack;
		const statuses = this._queryStatusStack;
		nodes.length = 0;
		statuses.length = 0;
		nodes.push(root);
		statuses.push(rootStatus);

		while (nodes.length > 0 && result.length < maxResults) {
			const node = nodes.pop()!;
			const status = statuses.pop()!;
			if (node.objects && node.objectBounds) {
				if (status === FRUSTUM_INSIDE) {
					this._appendLeafObjects(
						node.objects,
						includeInvisible,
						maxResults,
						result
					);
				} else {
					this._appendLeafObjectsWithAABB(
						node.objects,
						node.objectBounds,
						planeData,
						includeInvisible,
						maxResults,
						result
					);
				}
				continue;
			}

			this._pushFrustumChild(node.right, status, planeData, nodes, statuses);
			this._pushFrustumChild(node.left, status, planeData, nodes, statuses);
		}
	}

	private _pushFrustumChild(
		child: SpatialNode | undefined,
		parentStatus: number,
		planeData: Float64Array,
		nodes: SpatialNode[],
		statuses: number[]
	): void {
		if (!child) return;
		const status =
			parentStatus === FRUSTUM_INSIDE ?
				FRUSTUM_INSIDE
			:	classifyAABBFrustumData(
					planeData,
					child.bounds.min,
					child.bounds.max
				);
		if (status === FRUSTUM_OUTSIDE) return;
		nodes.push(child);
		statuses.push(status);
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
		planeData: Float64Array,
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
				classifyAABBFrustumData(
					planeData,
					objectBounds.min,
					objectBounds.max
				) !==
				FRUSTUM_OUTSIDE
			) {
				result.push(meshInstance);
			}
		}
		return result.length >= maxResults;
	}

	private _queryBoundsIterative(
		root: SpatialNode,
		queryBounds: {
			min: { x: number; y: number; z: number };
			max: { x: number; y: number; z: number };
		},
		includeInvisible: boolean,
		maxResults: number,
		result: MeshInstance[]
	): void {
		if (!intersectsAABB(root.bounds, queryBounds)) return;
		const nodes = this._queryNodeStack;
		const statuses = this._queryStatusStack;
		nodes.length = 0;
		statuses.length = 0;
		nodes.push(root);
		statuses.push(containsAABB(queryBounds, root.bounds) ? 1 : 0);
		while (nodes.length > 0 && result.length < maxResults) {
			const node = nodes.pop()!;
			const contained = statuses.pop()! === 1;
			if (node.objects && node.objectBounds) {
				if (contained) {
					this._appendLeafObjects(
						node.objects,
						includeInvisible,
						maxResults,
						result
					);
				} else {
					this._appendLeafObjectsWithBounds(
						node.objects,
						node.objectBounds,
						queryBounds,
						includeInvisible,
						maxResults,
						result
					);
				}
				continue;
			}
			this._pushBoundsChild(node.right, contained, queryBounds, nodes, statuses);
			this._pushBoundsChild(node.left, contained, queryBounds, nodes, statuses);
		}
	}

	private _pushBoundsChild(
		child: SpatialNode | undefined,
		parentContained: boolean,
		queryBounds: SpatialBounds3D,
		nodes: SpatialNode[],
		statuses: number[]
	): void {
		if (!child) return;
		if (!parentContained && !intersectsAABB(child.bounds, queryBounds)) return;
		nodes.push(child);
		statuses.push(
			parentContained || containsAABB(queryBounds, child.bounds) ? 1 : 0
		);
	}

	private _appendLeafObjectsWithBounds(
		objects: MeshInstance[],
		bounds: BoundingBox[],
		queryBounds: {
			min: { x: number; y: number; z: number };
			max: { x: number; y: number; z: number };
		},
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
			if (intersectsAABB(objectBounds, queryBounds)) {
				result.push(meshInstance);
			}
		}
		return result.length >= maxResults;
	}
}

function resolveBVHOptions(options: number | BVHOptions): Required<BVHOptions> {
	const source =
		typeof options === "number" ?
			{ leafSize: options }
		:	options;
	return {
		leafSize: source.leafSize ?? DEFAULT_LEAF_SIZE,
		buildStrategy:
			source.buildStrategy === "sah" ? "sah" : DEFAULT_BUILD_STRATEGY,
		rebuildDirtyRatio: resolvePositiveRatio(
			source.rebuildDirtyRatio,
			DEFAULT_REBUILD_DIRTY_RATIO
		),
		rebuildSurfaceAreaInflation: resolvePositiveNumber(
			source.rebuildSurfaceAreaInflation,
			DEFAULT_REBUILD_SURFACE_AREA_INFLATION
		),
	};
}

function resolvePositiveRatio(
	value: number | undefined,
	fallback: number
): number {
	if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
		return fallback;
	}
	return Math.min(1, value);
}

function resolvePositiveNumber(
	value: number | undefined,
	fallback: number
): number {
	if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
		return fallback;
	}
	return value;
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

function resolveBinnedSAHSplit(
	entries: SpatialBuildEntry[],
	start: number,
	end: number,
	stats: SpatialRangeStats
): BinnedSAHSplit | null {
	let best: BinnedSAHSplit | null = null;
	let bestCost = Infinity;
	const axes: Array<"x" | "y" | "z"> = ["x", "y", "z"];
	for (const axis of axes) {
		const centroidMin = getStatsCentroidMin(stats, axis);
		const centroidExtent = getStatsCentroidExtent(stats, axis);
		if (!(centroidExtent > DEGENERATE_AXIS_EPSILON)) continue;

		const buckets = createSAHBuckets();
		for (let index = start; index < end; index++) {
			const entry = entries[index];
			const bucketIndex = Math.max(
				0,
				Math.min(
					SAH_BUCKET_COUNT - 1,
					Math.floor(
						((getEntryCentroid(entry, axis) - centroidMin) /
							centroidExtent) *
							SAH_BUCKET_COUNT
					)
				)
			);
			expandSAHBucket(buckets[bucketIndex], entry.bounds);
		}

		const prefix = createSAHBuckets();
		const suffix = createSAHBuckets();
		for (let i = 0; i < SAH_BUCKET_COUNT; i++) {
			copySAHBucket(prefix[i], buckets[i]);
			if (i > 0) {
				mergeSAHBucket(prefix[i], prefix[i - 1]);
			}
		}
		for (let i = SAH_BUCKET_COUNT - 1; i >= 0; i--) {
			copySAHBucket(suffix[i], buckets[i]);
			if (i < SAH_BUCKET_COUNT - 1) {
				mergeSAHBucket(suffix[i], suffix[i + 1]);
			}
		}

		for (let split = 0; split < SAH_BUCKET_COUNT - 1; split++) {
			const left = prefix[split];
			const right = suffix[split + 1];
			if (left.count === 0 || right.count === 0) continue;
			const cost =
				left.count * computeSAHBucketSurfaceArea(left) +
				right.count * computeSAHBucketSurfaceArea(right);
			if (cost < bestCost) {
				bestCost = cost;
				best = {
					axis,
					threshold:
						centroidMin +
						centroidExtent * ((split + 1) / SAH_BUCKET_COUNT),
				};
			}
		}
	}
	return best;
}

function partitionEntriesByAxisThreshold(
	entries: SpatialBuildEntry[],
	start: number,
	end: number,
	axis: "x" | "y" | "z",
	threshold: number
): number {
	let left = start;
	let right = end - 1;
	while (left <= right) {
		while (
			left <= right &&
			getEntryCentroid(entries[left], axis) <= threshold
		) {
			left++;
		}
		while (
			left <= right &&
			getEntryCentroid(entries[right], axis) > threshold
		) {
			right--;
		}
		if (left <= right) {
			swapEntries(entries, left, right);
			left++;
			right--;
		}
	}
	return left;
}

function computeRangeStats(
	entries: SpatialBuildEntry[],
	start: number,
	end: number
): SpatialRangeStats {
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
		centroidMinX,
		centroidMinY,
		centroidMinZ,
		centroidMaxX,
		centroidMaxY,
		centroidMaxZ,
		centroidExtentX: centroidMaxX - centroidMinX,
		centroidExtentY: centroidMaxY - centroidMinY,
		centroidExtentZ: centroidMaxZ - centroidMinZ,
	};
}

function getStatsCentroidMin(
	stats: SpatialRangeStats,
	axis: "x" | "y" | "z"
): number {
	if (axis === "x") return stats.centroidMinX;
	if (axis === "y") return stats.centroidMinY;
	return stats.centroidMinZ;
}

function getStatsCentroidExtent(
	stats: SpatialRangeStats,
	axis: "x" | "y" | "z"
): number {
	if (axis === "x") return stats.centroidExtentX;
	if (axis === "y") return stats.centroidExtentY;
	return stats.centroidExtentZ;
}

function createSAHBuckets(): SAHBucket[] {
	const buckets = new Array<SAHBucket>(SAH_BUCKET_COUNT);
	for (let i = 0; i < SAH_BUCKET_COUNT; i++) {
		buckets[i] = createSAHBucket();
	}
	return buckets;
}

function createSAHBucket(): SAHBucket {
	return {
		count: 0,
		minX: Infinity,
		minY: Infinity,
		minZ: Infinity,
		maxX: -Infinity,
		maxY: -Infinity,
		maxZ: -Infinity,
	};
}

function expandSAHBucket(bucket: SAHBucket, bounds: BoundingBox): void {
	bucket.count++;
	if (bounds.min.x < bucket.minX) bucket.minX = bounds.min.x;
	if (bounds.min.y < bucket.minY) bucket.minY = bounds.min.y;
	if (bounds.min.z < bucket.minZ) bucket.minZ = bounds.min.z;
	if (bounds.max.x > bucket.maxX) bucket.maxX = bounds.max.x;
	if (bounds.max.y > bucket.maxY) bucket.maxY = bounds.max.y;
	if (bounds.max.z > bucket.maxZ) bucket.maxZ = bounds.max.z;
}

function copySAHBucket(target: SAHBucket, source: SAHBucket): void {
	target.count = source.count;
	target.minX = source.minX;
	target.minY = source.minY;
	target.minZ = source.minZ;
	target.maxX = source.maxX;
	target.maxY = source.maxY;
	target.maxZ = source.maxZ;
}

function mergeSAHBucket(target: SAHBucket, source: SAHBucket): void {
	if (source.count === 0) return;
	if (target.count === 0) {
		copySAHBucket(target, source);
		return;
	}
	target.count += source.count;
	if (source.minX < target.minX) target.minX = source.minX;
	if (source.minY < target.minY) target.minY = source.minY;
	if (source.minZ < target.minZ) target.minZ = source.minZ;
	if (source.maxX > target.maxX) target.maxX = source.maxX;
	if (source.maxY > target.maxY) target.maxY = source.maxY;
	if (source.maxZ > target.maxZ) target.maxZ = source.maxZ;
}

function computeSAHBucketSurfaceArea(bucket: SAHBucket): number {
	if (bucket.count === 0) return 0;
	return computeBoxSurfaceAreaValues(
		bucket.minX,
		bucket.minY,
		bucket.minZ,
		bucket.maxX,
		bucket.maxY,
		bucket.maxZ
	);
}

function createBoundingBox(): BoundingBox {
	return {
		min: { x: 0, y: 0, z: 0 },
		max: { x: 0, y: 0, z: 0 },
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

function computeTreeQualityCost(root: SpatialNode | null): number {
	if (!root) return 0;
	let cost = 0;
	const stack: SpatialNode[] = [root];
	while (stack.length > 0) {
		const node = stack.pop();
		if (!node) continue;
		cost += node.surfaceArea * node.subtreeObjectCount;
		if (node.left) stack.push(node.left);
		if (node.right) stack.push(node.right);
	}
	return cost;
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

function computeBoundingBoxSurfaceArea(bounds: BoundingBox): number {
	return computeBoxSurfaceAreaValues(
		bounds.min.x,
		bounds.min.y,
		bounds.min.z,
		bounds.max.x,
		bounds.max.y,
		bounds.max.z
	);
}

function computeBoxSurfaceAreaValues(
	minX: number,
	minY: number,
	minZ: number,
	maxX: number,
	maxY: number,
	maxZ: number
): number {
	const sizeX = Math.max(0, maxX - minX);
	const sizeY = Math.max(0, maxY - minY);
	const sizeZ = Math.max(0, maxZ - minZ);
	return 2 * (sizeX * sizeY + sizeX * sizeZ + sizeY * sizeZ);
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

function captureFrustumPlaneData(
	frustum: Frustum,
	target: Float64Array
): void {
	let offset = 0;
	for (const plane of frustum.planes) {
		target[offset++] = plane.normal.x;
		target[offset++] = plane.normal.y;
		target[offset++] = plane.normal.z;
		target[offset++] = plane.constant;
	}
}

function classifyAABBFrustumData(
	planeData: Float64Array,
	min: { x: number; y: number; z: number },
	max: { x: number; y: number; z: number }
): number {
	let fullyInside = true;

	for (let offset = 0; offset < 24; offset += 4) {
		const nx = planeData[offset];
		const ny = planeData[offset + 1];
		const nz = planeData[offset + 2];
		const constant = planeData[offset + 3];

		const px = nx >= 0 ? max.x : min.x;
		const py = ny >= 0 ? max.y : min.y;
		const pz = nz >= 0 ? max.z : min.z;
		const positiveDistance = nx * px + ny * py + nz * pz + constant;
		if (positiveDistance < 0) {
			return FRUSTUM_OUTSIDE;
		}

		const nxPoint = nx >= 0 ? min.x : max.x;
		const nyPoint = ny >= 0 ? min.y : max.y;
		const nzPoint = nz >= 0 ? min.z : max.z;
		const negativeDistance =
			nx * nxPoint + ny * nyPoint + nz * nzPoint + constant;
		if (negativeDistance < 0) {
			fullyInside = false;
		}
	}

	return fullyInside ? FRUSTUM_INSIDE : FRUSTUM_INTERSECT;
}

function normalizeRayDirection(
	direction: { x: number; y: number; z: number },
	label: string
): { x: number; y: number; z: number } {
	const directionLength = Math.hypot(direction.x, direction.y, direction.z);
	if (!(directionLength > 1e-8)) {
		throw new Error(`${label} direction must be non-zero`);
	}
	const invDirectionLength = 1 / directionLength;
	return {
		x: direction.x * invDirectionLength,
		y: direction.y * invDirectionLength,
		z: direction.z * invDirectionLength,
	};
}

function compareRayHits(left: SpatialRayHit, right: SpatialRayHit): number {
	if (left.distance !== right.distance) {
		return left.distance - right.distance;
	}
	return left.meshInstance.id.localeCompare(right.meshInstance.id);
}

function compareRayCandidate(
	distance: number,
	meshInstance: MeshInstance,
	right: SpatialRayHit
): number {
	if (distance !== right.distance) return distance - right.distance;
	return meshInstance.id.localeCompare(right.meshInstance.id);
}

function pushRayHitMaxHeap(heap: SpatialRayHit[], hit: SpatialRayHit): void {
	heap.push(hit);
	let index = heap.length - 1;
	while (index > 0) {
		const parent = (index - 1) >> 1;
		if (compareRayHits(heap[index], heap[parent]) <= 0) break;
		const swap = heap[index];
		heap[index] = heap[parent];
		heap[parent] = swap;
		index = parent;
	}
}

function siftRayHitMaxHeapDown(heap: SpatialRayHit[], start: number): void {
	let index = start;
	while (true) {
		const left = index * 2 + 1;
		if (left >= heap.length) return;
		const right = left + 1;
		let larger = left;
		if (
			right < heap.length &&
			compareRayHits(heap[right], heap[left]) > 0
		) {
			larger = right;
		}
		if (compareRayHits(heap[larger], heap[index]) <= 0) return;
		const swap = heap[index];
		heap[index] = heap[larger];
		heap[larger] = swap;
		index = larger;
	}
}

function pushRayNodeMinHeap(
	nodes: SpatialNode[],
	distances: number[],
	node: SpatialNode,
	distance: number
): void {
	nodes.push(node);
	distances.push(distance);
	let index = nodes.length - 1;
	while (index > 0) {
		const parent = (index - 1) >> 1;
		if (distances[parent] <= distance) break;
		nodes[index] = nodes[parent];
		distances[index] = distances[parent];
		index = parent;
	}
	nodes[index] = node;
	distances[index] = distance;
}

function popRayNodeMinHeap(
	nodes: SpatialNode[],
	distances: number[]
): void {
	const lastNode = nodes.pop();
	const lastDistance = distances.pop();
	if (!lastNode || lastDistance === undefined || nodes.length === 0) return;
	let index = 0;
	while (true) {
		const left = index * 2 + 1;
		if (left >= nodes.length) break;
		const right = left + 1;
		const smaller =
			right < nodes.length && distances[right] < distances[left] ? right : left;
		if (distances[smaller] >= lastDistance) break;
		nodes[index] = nodes[smaller];
		distances[index] = distances[smaller];
		index = smaller;
	}
	nodes[index] = lastNode;
	distances[index] = lastDistance;
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

function intersectsAABB(
	left: {
		min: { x: number; y: number; z: number };
		max: { x: number; y: number; z: number };
	},
	right: {
		min: { x: number; y: number; z: number };
		max: { x: number; y: number; z: number };
	}
): boolean {
	return !(
		left.max.x < right.min.x ||
		left.min.x > right.max.x ||
		left.max.y < right.min.y ||
		left.min.y > right.max.y ||
		left.max.z < right.min.z ||
		left.min.z > right.max.z
	);
}

function containsAABB(
	container: SpatialBounds3D,
	contained: BoundingBox
): boolean {
	return (
		container.min.x <= contained.min.x &&
		container.min.y <= contained.min.y &&
		container.min.z <= contained.min.z &&
		container.max.x >= contained.max.x &&
		container.max.y >= contained.max.y &&
		container.max.z >= contained.max.z
	);
}
