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
		const bounds = meshInstance.getWorldBoundingBox();
		if (containsBoundsInLooseNode(entry.node, bounds, this._looseness)) {
			copyBoundingBoxValues(entry.bounds, bounds);
			return;
		}
		this._detachEntry(meshInstance);
		this._insertEntry(meshInstance, bounds);
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
		this._queryNodeFrustum(
			this._root,
			frustum,
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
		this._queryNodeBounds(
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

		const stack: LooseOctreeNode[] = [this._root];
		const shouldBoundHits = Number.isFinite(maxResults);
		let traversalMaxDistance = maxDistance;

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
				traversalMaxDistance,
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
					traversalMaxDistance,
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
				if (shouldBoundHits && out.length >= maxResults) {
					out.sort(compareRayHits);
					if (out.length > maxResults) {
						out.length = maxResults;
					}
					traversalMaxDistance = Math.min(
						traversalMaxDistance,
						out[out.length - 1].distance
					);
				}
			}

			if (!node.children) continue;
			for (const child of node.children) {
				if (!child) continue;
				stack.push(child);
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
		const stack: Array<{ node: LooseOctreeNode; distance: number }> = [];
		const rootDistance = intersectRayLooseNode(
			origin,
			normalizedDirection,
			maxDistance,
			root,
			this._looseness
		);
		if (rootDistance === null) return null;
		stack.push({ node: root, distance: rootDistance });

		let best: SpatialRayHit | null = null;
		let bestDistance = maxDistance;
		while (stack.length > 0) {
			const current = stack.pop();
			if (!current || current.distance > bestDistance) continue;
			const node = current.node;

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
				stack.push({ node: child, distance });
			}
			stack.sort((left, right) => right.distance - left.distance);
		}
		return best;
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

	private _queryNodeBounds(
		node: LooseOctreeNode,
		queryBounds: {
			min: { x: number; y: number; z: number };
			max: { x: number; y: number; z: number };
		},
		includeInvisible: boolean,
		maxResults: number,
		result: MeshInstance[]
	): boolean {
		if (result.length >= maxResults) return true;

		const looseHalfSize = node.halfSize * this._looseness;
		if (
			!intersectsAABBValues(
				node.centerX - looseHalfSize,
				node.centerY - looseHalfSize,
				node.centerZ - looseHalfSize,
				node.centerX + looseHalfSize,
				node.centerY + looseHalfSize,
				node.centerZ + looseHalfSize,
				queryBounds
			)
		) {
			return false;
		}

		for (let i = 0; i < node.objects.length; i++) {
			if (result.length >= maxResults) return true;
			const meshInstance = node.objects[i];
			if (!includeInvisible && meshInstance.visible === false) continue;
			if (intersectsAABB(node.objectBounds[i], queryBounds)) {
				result.push(meshInstance);
			}
		}

		if (!node.children) return result.length >= maxResults;
		for (const child of node.children) {
			if (!child) continue;
			if (
				this._queryNodeBounds(
					child,
					queryBounds,
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
	const leftEntity = left.meshInstance.entityId ?? Number.MAX_SAFE_INTEGER;
	const rightEntity = right.meshInstance.entityId ?? Number.MAX_SAFE_INTEGER;
	if (leftEntity !== rightEntity) {
		return leftEntity - rightEntity;
	}
	return left.meshInstance.id.localeCompare(right.meshInstance.id);
}
