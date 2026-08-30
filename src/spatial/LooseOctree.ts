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

interface LooseOctreeNode {
	centerX: number;
	centerY: number;
	centerZ: number;
	halfSize: number;
	objects: MeshInstance[];
	objectBounds: BoundingBox[];
	children: Array<LooseOctreeNode | null> | null;
	parent: LooseOctreeNode | null;
	childIndex: number;
}

interface LooseOctreeEntry {
	node: LooseOctreeNode;
	bounds: BoundingBox;
	objectIndex: number;
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

interface InsertPlacement {
	node: LooseOctreeNode;
	objectIndex: number;
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
	private readonly _queryNodeStack: LooseOctreeNode[] = [];
	private readonly _queryStatusStack: number[] = [];
	private readonly _frustumPlaneData = new Float64Array(24);
	private readonly _rayNodeHeap: LooseOctreeNode[] = [];
	private readonly _rayDistanceHeap: number[] = [];

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
		const entry = this._entriesByMeshInstance.get(meshInstance);
		if (!entry) return;
		const bounds = meshInstance.getOwnWorldBoundingBox();
		if (containsBoundsInLooseNode(entry.node, bounds, this._looseness)) {
			copyBoundingBoxValues(entry.bounds, bounds);
			return;
		}
		this._detachEntry(meshInstance);
		this._insertEntry(meshInstance, bounds);
	}

	public upsert(meshInstance: MeshInstance): void {
		this._detachEntry(meshInstance);
		const bounds = meshInstance.getOwnWorldBoundingBox();
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
			const bounds = meshInstance.getOwnWorldBoundingBox();
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

	public queryFrustumInto(
		frustum: Frustum,
		out: MeshInstance[],
		options?: SpatialQueryOptions
	): MeshInstance[] {
		out.length = 0;
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
		options?: SpatialQueryOptions
	): MeshInstance[] {
		return this.queryFrustumInto(frustum, [], options);
	}

	public queryBoundsInto(
		bounds: SpatialBounds3D,
		out: MeshInstance[],
		options?: SpatialQueryOptions
	): MeshInstance[] {
		out.length = 0;
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
		options?: SpatialQueryOptions
	): MeshInstance[] {
		return this.queryBoundsInto(bounds, [], options);
	}

	public queryRay(
		origin: { x: number; y: number; z: number },
		direction: { x: number; y: number; z: number },
		options?: SpatialRayQueryOptions
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
		out: SpatialRayHit[],
		options?: SpatialRayQueryOptions
	): SpatialRayHit[] {
		out.length = 0;
		const normalizedDirection = normalizeRayDirection(
			direction,
			"LooseOctree.queryRayDetailedInto"
		);
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
			if (hit) out.push(hit);
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
				out.push({
					meshInstance,
					distance,
				});
			}

			if (!node.children) continue;
			for (let index = node.children.length - 1; index >= 0; index--) {
				const child = node.children[index];
				if (child) stack.push(child);
			}
		}

		if (out.length === 0) return out;
		out.sort(compareRayHits);
		if (out.length > maxResults) {
			out.length = maxResults;
		}
		return out;
	}

	public queryRayDetailed(
		origin: { x: number; y: number; z: number },
		direction: { x: number; y: number; z: number },
		options?: SpatialRayQueryOptions
	): SpatialRayHit[] {
		return this.queryRayDetailedInto(origin, direction, [], options);
	}

	private _queryNearestRayHit(
		root: LooseOctreeNode,
		origin: { x: number; y: number; z: number },
		normalizedDirection: { x: number; y: number; z: number },
		maxDistance: number,
		includeInvisible: boolean
	): SpatialRayHit | null {
		const rootDistance = intersectRayLooseNode(
			origin,
			normalizedDirection,
			maxDistance,
			root,
			this._looseness
		);
		if (rootDistance === null) return null;
		const nodes = this._rayNodeHeap;
		const distances = this._rayDistanceHeap;
		nodes.length = 0;
		distances.length = 0;
		pushLooseRayNodeMinHeap(nodes, distances, root, rootDistance);

		let best: SpatialRayHit | null = null;
		let bestDistance = maxDistance;
		while (nodes.length > 0) {
			const node = nodes[0];
			const nodeDistance = distances[0];
			popLooseRayNodeMinHeap(nodes, distances);
			if (nodeDistance > bestDistance) break;

			for (let i = 0; i < node.objects.length; i++) {
				const meshInstance = node.objects[i];
				if (!includeInvisible && meshInstance.visible === false) {
					continue;
				}
				const bounds = node.objectBounds[i];
				const distance = intersectRayAABB(
					origin,
					normalizedDirection,
					bestDistance,
					bounds.min.x,
					bounds.min.y,
					bounds.min.z,
					bounds.max.x,
					bounds.max.y,
					bounds.max.z
				);
				if (distance === null) continue;
				const candidate = { meshInstance, distance };
				if (!best || compareRayHits(candidate, best) < 0) {
					best = candidate;
					bestDistance = distance;
				}
			}

			if (!node.children) continue;
			for (const child of node.children) {
				if (!child) continue;
				const distance = intersectRayLooseNode(
					origin,
					normalizedDirection,
					bestDistance,
					child,
					this._looseness
				);
				if (distance === null) continue;
				pushLooseRayNodeMinHeap(nodes, distances, child, distance);
			}
		}
		return best;
	}

	private _queryTopKRayHits(
		root: LooseOctreeNode,
		origin: { x: number; y: number; z: number },
		direction: { x: number; y: number; z: number },
		maxDistance: number,
		maxResults: number,
		includeInvisible: boolean,
		out: SpatialRayHit[]
	): void {
		const rootDistance = intersectRayLooseNode(
			origin,
			direction,
			maxDistance,
			root,
			this._looseness
		);
		if (rootDistance === null) return;
		const nodes = this._rayNodeHeap;
		const distances = this._rayDistanceHeap;
		nodes.length = 0;
		distances.length = 0;
		pushLooseRayNodeMinHeap(nodes, distances, root, rootDistance);
		let traversalMaxDistance = maxDistance;

		while (nodes.length > 0) {
			const node = nodes[0];
			const nodeDistance = distances[0];
			popLooseRayNodeMinHeap(nodes, distances);
			if (nodeDistance > traversalMaxDistance) break;

			for (let i = 0; i < node.objects.length; i++) {
				const meshInstance = node.objects[i];
				if (!includeInvisible && meshInstance.visible === false) continue;
				const bounds = node.objectBounds[i];
				const distance = intersectRayAABB(
					origin,
					direction,
					traversalMaxDistance,
					bounds.min.x,
					bounds.min.y,
					bounds.min.z,
					bounds.max.x,
					bounds.max.y,
					bounds.max.z
				);
				if (distance === null) continue;
				if (out.length < maxResults) {
					pushLooseRayHitMaxHeap(out, { meshInstance, distance });
				} else if (
					compareLooseRayCandidate(distance, meshInstance, out[0]) < 0
				) {
					out[0] = { meshInstance, distance };
					siftLooseRayHitMaxHeapDown(out, 0);
				}
				if (out.length === maxResults) {
					traversalMaxDistance = Math.min(
						maxDistance,
						out[0].distance
					);
				}
			}

			if (!node.children) continue;
			for (const child of node.children) {
				if (!child) continue;
				const distance = intersectRayLooseNode(
					origin,
					direction,
					traversalMaxDistance,
					child,
					this._looseness
				);
				if (distance === null) continue;
				pushLooseRayNodeMinHeap(nodes, distances, child, distance);
			}
		}
	}

	private _insertEntry(meshInstance: MeshInstance, bounds: BoundingBox): void {
		if (!this._root) {
			this._root = createRootNode(bounds);
		}
		this._expandRootToFit(bounds);
		if (!this._root) return;
		const placement = this._insertIntoNode(this._root, meshInstance, bounds, 0);
		this._entriesByMeshInstance.set(meshInstance, {
			node: placement.node,
			bounds,
			objectIndex: placement.objectIndex,
		});
	}

	private _insertIntoNode(
		node: LooseOctreeNode,
		meshInstance: MeshInstance,
		bounds: BoundingBox,
		depth: number
	): InsertPlacement {
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

		const objectIndex = node.objects.length;
		node.objects.push(meshInstance);
		node.objectBounds.push(bounds);
		return { node, objectIndex };
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

			this._swapRemoveNodeObject(node, index);
			const child = this._ensureChild(node, childPlacement);
			const placement = this._insertIntoNode(
				child,
				meshInstance,
				bounds,
				depth + 1
			);
			const entry = this._entriesByMeshInstance.get(meshInstance);
			if (entry) {
				entry.node = placement.node;
				entry.objectIndex = placement.objectIndex;
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
				parent: node,
				childIndex: childPlacement.index,
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
				parent: null,
				childIndex: -1,
			} satisfies LooseOctreeNode;
			const oldChildIndex = resolveChildIndexFromPoint(
				newRoot,
				root.centerX,
				root.centerY,
				root.centerZ
			);
			root.parent = newRoot;
			root.childIndex = oldChildIndex;
			newRoot.children![oldChildIndex] = root;
			this._root = newRoot;
		}
	}

	private _detachEntry(meshInstance: MeshInstance): boolean {
		const entry = this._entriesByMeshInstance.get(meshInstance);
		if (!entry) return false;
		this._swapRemoveNodeObject(entry.node, entry.objectIndex);
		this._entriesByMeshInstance.delete(meshInstance);
		this._pruneEmptyAncestors(entry.node);
		return true;
	}

	private _swapRemoveNodeObject(node: LooseOctreeNode, index: number): void {
		const lastIndex = node.objects.length - 1;
		if (index < 0 || index > lastIndex) return;
		if (index !== lastIndex) {
			const movedMeshInstance = node.objects[lastIndex];
			node.objects[index] = movedMeshInstance;
			node.objectBounds[index] = node.objectBounds[lastIndex];
			const movedEntry =
				this._entriesByMeshInstance.get(movedMeshInstance);
			if (movedEntry) {
				movedEntry.node = node;
				movedEntry.objectIndex = index;
			}
		}
		node.objects.pop();
		node.objectBounds.pop();
	}

	private _pruneEmptyAncestors(node: LooseOctreeNode): void {
		let current: LooseOctreeNode | null = node;
		while (
			current &&
			current !== this._root &&
			current.objects.length === 0 &&
			isChildrenArrayEmpty(current.children)
		) {
			const parent = current.parent;
			if (!parent?.children) return;
			parent.children[current.childIndex] = null;
			if (isChildrenArrayEmpty(parent.children)) {
				parent.children = null;
			}
			current = parent;
		}
	}

	private _queryFrustumIterative(
		root: LooseOctreeNode,
		planeData: Float64Array,
		includeInvisible: boolean,
		maxResults: number,
		result: MeshInstance[]
	): void {
		const rootStatus = classifyLooseNodeFrustum(
			root,
			this._looseness,
			planeData
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
			for (let i = 0; i < node.objects.length; i++) {
				if (result.length >= maxResults) return;
				const meshInstance = node.objects[i];
				if (!includeInvisible && meshInstance.visible === false) continue;
				const bounds = node.objectBounds[i];
				if (
					status === FRUSTUM_INSIDE ||
					classifyAABBFrustumData(
						planeData,
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

			if (!node.children) continue;
			for (let i = node.children.length - 1; i >= 0; i--) {
				const child = node.children[i];
				if (!child) continue;
				const childStatus =
					status === FRUSTUM_INSIDE ?
						FRUSTUM_INSIDE
					:	classifyLooseNodeFrustum(
							child,
							this._looseness,
							planeData
						);
				if (childStatus === FRUSTUM_OUTSIDE) continue;
				nodes.push(child);
				statuses.push(childStatus);
			}
		}
	}

	private _queryBoundsIterative(
		root: LooseOctreeNode,
		queryBounds: SpatialBounds3D,
		includeInvisible: boolean,
		maxResults: number,
		result: MeshInstance[]
	): void {
		if (!looseNodeIntersectsBounds(root, this._looseness, queryBounds)) return;
		const nodes = this._queryNodeStack;
		const statuses = this._queryStatusStack;
		nodes.length = 0;
		statuses.length = 0;
		nodes.push(root);
		statuses.push(
			queryContainsLooseNode(queryBounds, root, this._looseness) ? 1 : 0
		);

		while (nodes.length > 0 && result.length < maxResults) {
			const node = nodes.pop()!;
			const contained = statuses.pop()! === 1;
			for (let i = 0; i < node.objects.length; i++) {
				if (result.length >= maxResults) return;
				const meshInstance = node.objects[i];
				if (!includeInvisible && meshInstance.visible === false) continue;
				if (contained || intersectsAABB(node.objectBounds[i], queryBounds)) {
					result.push(meshInstance);
				}
			}

			if (!node.children) continue;
			for (let i = node.children.length - 1; i >= 0; i--) {
				const child = node.children[i];
				if (!child) continue;
				if (
					!contained &&
					!looseNodeIntersectsBounds(child, this._looseness, queryBounds)
				) {
					continue;
				}
				nodes.push(child);
				statuses.push(
					contained ||
						queryContainsLooseNode(queryBounds, child, this._looseness) ?
						1
					:	0
				);
			}
		}
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
		parent: null,
		childIndex: -1,
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

function containsBoundsInLooseNode(
	node: LooseOctreeNode,
	bounds: BoundingBox,
	looseness: number
): boolean {
	return containsBoundsInCube(
		node.centerX,
		node.centerY,
		node.centerZ,
		node.halfSize * looseness,
		bounds
	);
}

function copyBoundingBoxValues(target: BoundingBox, source: BoundingBox): void {
	target.min.x = source.min.x;
	target.min.y = source.min.y;
	target.min.z = source.min.z;
	target.max.x = source.max.x;
	target.max.y = source.max.y;
	target.max.z = source.max.z;
}

function isChildrenArrayEmpty(
	children: Array<LooseOctreeNode | null> | null
): boolean {
	if (!children) return true;
	for (const child of children) {
		if (child) return false;
	}
	return true;
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
	const childCenterX =
		node.centerX + (centerX >= node.centerX ? childHalf : -childHalf);
	const childCenterY =
		node.centerY + (centerY >= node.centerY ? childHalf : -childHalf);
	const childCenterZ =
		node.centerZ + (centerZ >= node.centerZ ? childHalf : -childHalf);

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

function classifyLooseNodeFrustum(
	node: LooseOctreeNode,
	looseness: number,
	planeData: Float64Array
): number {
	const halfSize = node.halfSize * looseness;
	return classifyAABBFrustumData(
		planeData,
		node.centerX - halfSize,
		node.centerY - halfSize,
		node.centerZ - halfSize,
		node.centerX + halfSize,
		node.centerY + halfSize,
		node.centerZ + halfSize
	);
}

function classifyAABBFrustumData(
	planeData: Float64Array,
	minX: number,
	minY: number,
	minZ: number,
	maxX: number,
	maxY: number,
	maxZ: number
): number {
	let fullyInside = true;
	for (let offset = 0; offset < 24; offset += 4) {
		const nx = planeData[offset];
		const ny = planeData[offset + 1];
		const nz = planeData[offset + 2];
		const constant = planeData[offset + 3];
		const px = nx >= 0 ? maxX : minX;
		const py = ny >= 0 ? maxY : minY;
		const pz = nz >= 0 ? maxZ : minZ;
		if (nx * px + ny * py + nz * pz + constant < 0) {
			return FRUSTUM_OUTSIDE;
		}
		const negativeX = nx >= 0 ? minX : maxX;
		const negativeY = ny >= 0 ? minY : maxY;
		const negativeZ = nz >= 0 ? minZ : maxZ;
		if (
			nx * negativeX + ny * negativeY + nz * negativeZ + constant < 0
		) {
			fullyInside = false;
		}
	}
	return fullyInside ? FRUSTUM_INSIDE : FRUSTUM_INTERSECT;
}

function looseNodeIntersectsBounds(
	node: LooseOctreeNode,
	looseness: number,
	bounds: SpatialBounds3D
): boolean {
	const halfSize = node.halfSize * looseness;
	return intersectsAABBValues(
		node.centerX - halfSize,
		node.centerY - halfSize,
		node.centerZ - halfSize,
		node.centerX + halfSize,
		node.centerY + halfSize,
		node.centerZ + halfSize,
		bounds
	);
}

function queryContainsLooseNode(
	query: SpatialBounds3D,
	node: LooseOctreeNode,
	looseness: number
): boolean {
	const halfSize = node.halfSize * looseness;
	return (
		query.min.x <= node.centerX - halfSize &&
		query.min.y <= node.centerY - halfSize &&
		query.min.z <= node.centerZ - halfSize &&
		query.max.x >= node.centerX + halfSize &&
		query.max.y >= node.centerY + halfSize &&
		query.max.z >= node.centerZ + halfSize
	);
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

function intersectRayLooseNode(
	origin: { x: number; y: number; z: number },
	direction: { x: number; y: number; z: number },
	maxDistance: number,
	node: LooseOctreeNode,
	looseness: number
): number | null {
	const looseHalfSize = node.halfSize * looseness;
	return intersectRayAABB(
		origin,
		direction,
		maxDistance,
		node.centerX - looseHalfSize,
		node.centerY - looseHalfSize,
		node.centerZ - looseHalfSize,
		node.centerX + looseHalfSize,
		node.centerY + looseHalfSize,
		node.centerZ + looseHalfSize
	);
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

function intersectsAABBValues(
	minX: number,
	minY: number,
	minZ: number,
	maxX: number,
	maxY: number,
	maxZ: number,
	right: {
		min: { x: number; y: number; z: number };
		max: { x: number; y: number; z: number };
	}
): boolean {
	return !(
		maxX < right.min.x ||
		minX > right.max.x ||
		maxY < right.min.y ||
		minY > right.max.y ||
		maxZ < right.min.z ||
		minZ > right.max.z
	);
}

function compareRayHits(left: SpatialRayHit, right: SpatialRayHit): number {
	if (left.distance !== right.distance) {
		return left.distance - right.distance;
	}
	return left.meshInstance.id.localeCompare(right.meshInstance.id);
}

function compareLooseRayCandidate(
	distance: number,
	meshInstance: MeshInstance,
	right: SpatialRayHit
): number {
	if (distance !== right.distance) return distance - right.distance;
	return meshInstance.id.localeCompare(right.meshInstance.id);
}

function pushLooseRayHitMaxHeap(
	heap: SpatialRayHit[],
	hit: SpatialRayHit
): void {
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

function siftLooseRayHitMaxHeapDown(
	heap: SpatialRayHit[],
	start: number
): void {
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

function pushLooseRayNodeMinHeap(
	nodes: LooseOctreeNode[],
	distances: number[],
	node: LooseOctreeNode,
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

function popLooseRayNodeMinHeap(
	nodes: LooseOctreeNode[],
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
