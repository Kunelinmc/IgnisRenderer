import type { BoundingBox } from "../core/types";
import type { Frustum } from "../maths/Frustum";
import type { MeshInstance } from "../meshes";

export interface SpatialNode {
	bounds: BoundingBox;
	left?: SpatialNode;
	right?: SpatialNode;
	objects?: MeshInstance[];
}

interface SpatialBuildEntry {
	meshInstance: MeshInstance;
	bounds: BoundingBox;
	centroidX: number;
	centroidY: number;
	centroidZ: number;
}

const DEFAULT_LEAF_SIZE = 8;

export class BVH {
	private _root: SpatialNode | null;
	private readonly _leafSize: number;

	constructor(
		meshInstances: MeshInstance[] = [],
		leafSize: number = DEFAULT_LEAF_SIZE
	) {
		this._root = null;
		this._leafSize = resolveLeafSize(leafSize);
		this.rebuild(meshInstances);
	}

	public get root(): SpatialNode | null {
		return this._root;
	}

	/**
	 * Rebuilds BVH nodes from the current world-space mesh bounds.
	 */
	public rebuild(meshInstances: MeshInstance[]): void {
		if (meshInstances.length === 0) {
			this._root = null;
			return;
		}

		const entries: SpatialBuildEntry[] = meshInstances.map((meshInstance) => {
			const bounds = meshInstance.getWorldBoundingBox();
			return {
				meshInstance,
				bounds,
				centroidX: (bounds.min.x + bounds.max.x) * 0.5,
				centroidY: (bounds.min.y + bounds.max.y) * 0.5,
				centroidZ: (bounds.min.z + bounds.max.z) * 0.5,
			};
		});
		this._root = this._build(entries);
	}

	/**
	 * Returns mesh instances whose node bounds overlap the given frustum.
	 */
	public queryFrustum(frustum: Frustum): MeshInstance[] {
		if (!this._root) return [];
		const result: MeshInstance[] = [];
		this._query(this._root, frustum, result);
		return result;
	}

	private _build(entries: SpatialBuildEntry[]): SpatialNode | null {
		if (entries.length === 0) return null;

		const bounds = unionEntryBounds(entries);
		if (entries.length <= this._leafSize) {
			return {
				bounds,
				objects: entries.map((entry) => entry.meshInstance),
			};
		}

		const axis = resolveSplitAxis(bounds);
		const sorted = entries
			.slice()
			.sort(
				(left, right) =>
					getEntryCentroid(left, axis) - getEntryCentroid(right, axis)
			);
		const splitIndex = sorted.length >> 1;
		if (splitIndex <= 0 || splitIndex >= sorted.length) {
			return {
				bounds,
				objects: sorted.map((entry) => entry.meshInstance),
			};
		}

		const left = this._build(sorted.slice(0, splitIndex));
		const right = this._build(sorted.slice(splitIndex));
		if (!left || !right) {
			return {
				bounds,
				objects: sorted.map((entry) => entry.meshInstance),
			};
		}

		return {
			bounds,
			left,
			right,
		};
	}

	private _query(
		node: SpatialNode,
		frustum: Frustum,
		result: MeshInstance[]
	): void {
		if (!frustum.intersectsAABB(node.bounds.min, node.bounds.max)) {
			return;
		}

		if (node.objects?.length) {
			for (const meshInstance of node.objects) {
				const bounds = meshInstance.getWorldBoundingBox();
				if (frustum.intersectsAABB(bounds.min, bounds.max)) {
					result.push(meshInstance);
				}
			}
		}

		if (node.left) {
			this._query(node.left, frustum, result);
		}
		if (node.right) {
			this._query(node.right, frustum, result);
		}
	}
}

function resolveLeafSize(value: number): number {
	if (!Number.isFinite(value)) return DEFAULT_LEAF_SIZE;
	return Math.max(1, Math.floor(value));
}

function unionEntryBounds(entries: SpatialBuildEntry[]): BoundingBox {
	let minX = Infinity;
	let minY = Infinity;
	let minZ = Infinity;
	let maxX = -Infinity;
	let maxY = -Infinity;
	let maxZ = -Infinity;

	for (const entry of entries) {
		const { min, max } = entry.bounds;
		if (min.x < minX) minX = min.x;
		if (min.y < minY) minY = min.y;
		if (min.z < minZ) minZ = min.z;
		if (max.x > maxX) maxX = max.x;
		if (max.y > maxY) maxY = max.y;
		if (max.z > maxZ) maxZ = max.z;
	}

	return {
		min: { x: minX, y: minY, z: minZ },
		max: { x: maxX, y: maxY, z: maxZ },
	};
}

function resolveSplitAxis(bounds: BoundingBox): "x" | "y" | "z" {
	const extentX = bounds.max.x - bounds.min.x;
	const extentY = bounds.max.y - bounds.min.y;
	const extentZ = bounds.max.z - bounds.min.z;

	if (extentX >= extentY && extentX >= extentZ) return "x";
	if (extentY >= extentX && extentY >= extentZ) return "y";
	return "z";
}

function getEntryCentroid(
	entry: SpatialBuildEntry,
	axis: "x" | "y" | "z"
): number {
	if (axis === "x") return entry.centroidX;
	if (axis === "y") return entry.centroidY;
	return entry.centroidZ;
}
