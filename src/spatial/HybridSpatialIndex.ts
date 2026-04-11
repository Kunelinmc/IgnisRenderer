import type { Frustum } from "../maths/Frustum";
import type { MeshInstance } from "../meshes";
import { BVH } from "./BVH";
import { isDynamicSpatialMeshInstance } from "./classification";
import { LooseOctree, type LooseOctreeOptions } from "./LooseOctree";
import type {
	SpatialIndex3D,
	SpatialQueryOptions,
	SpatialRayHit,
	SpatialRayQueryOptions,
} from "./types";

type SpatialBucket = "static" | "dynamic";

export interface HybridSpatialIndexOptions {
	dynamicPredicate?: (meshInstance: MeshInstance) => boolean;
	staticBVHLeafSize?: number;
	dynamicOctree?: LooseOctreeOptions;
}

export class HybridSpatialIndex implements SpatialIndex3D {
	private readonly _staticBVH: BVH;
	private readonly _dynamicOctree: LooseOctree;
	private readonly _bucketByMeshInstance = new Map<MeshInstance, SpatialBucket>();
	private readonly _dynamicPredicate: (meshInstance: MeshInstance) => boolean;

	constructor(
		meshInstances: MeshInstance[] = [],
		options: HybridSpatialIndexOptions = {}
	) {
		this._dynamicPredicate =
			options.dynamicPredicate ?? isDynamicSpatialMeshInstance;
		this._staticBVH = new BVH([], options.staticBVHLeafSize);
		this._dynamicOctree = new LooseOctree([], options.dynamicOctree);
		this.rebuild(meshInstances);
	}

	public get size(): number {
		return this._bucketByMeshInstance.size;
	}

	public get dirty(): boolean {
		return this._staticBVH.dirty || this._dynamicOctree.dirty;
	}

	public setMeshInstances(meshInstances: MeshInstance[]): void {
		this.rebuild(meshInstances);
	}

	public markDirty(meshInstance?: MeshInstance): void {
		if (!meshInstance) {
			this._staticBVH.markDirty();
			this._dynamicOctree.markDirty();
			return;
		}

		const bucket = this._resolveBucket(meshInstance);
		const previousBucket = this._bucketByMeshInstance.get(meshInstance);
		if (!previousBucket) {
			this._upsertToBucket(meshInstance, bucket);
			this._bucketByMeshInstance.set(meshInstance, bucket);
			return;
		}

		if (previousBucket !== bucket) {
			this._removeFromBucket(meshInstance, previousBucket);
			this._upsertToBucket(meshInstance, bucket);
			this._bucketByMeshInstance.set(meshInstance, bucket);
			return;
		}

		if (bucket === "dynamic") {
			this._dynamicOctree.markDirty(meshInstance);
		} else {
			this._staticBVH.markDirty(meshInstance);
		}
	}

	public upsert(meshInstance: MeshInstance): void {
		const bucket = this._resolveBucket(meshInstance);
		const previousBucket = this._bucketByMeshInstance.get(meshInstance);
		if (previousBucket && previousBucket !== bucket) {
			this._removeFromBucket(meshInstance, previousBucket);
		}
		this._upsertToBucket(meshInstance, bucket);
		this._bucketByMeshInstance.set(meshInstance, bucket);
	}

	public remove(meshInstance: MeshInstance): boolean {
		const bucket = this._bucketByMeshInstance.get(meshInstance);
		if (!bucket) return false;
		this._bucketByMeshInstance.delete(meshInstance);
		this._removeFromBucket(meshInstance, bucket);
		return true;
	}

	public rebuild(meshInstances?: MeshInstance[]): void {
		const source =
			meshInstances ?? Array.from(this._bucketByMeshInstance.keys());
		const staticMeshInstances: MeshInstance[] = [];
		const dynamicMeshInstances: MeshInstance[] = [];
		this._bucketByMeshInstance.clear();
		for (const meshInstance of source) {
			const bucket = this._resolveBucket(meshInstance);
			this._bucketByMeshInstance.set(meshInstance, bucket);
			if (bucket === "dynamic") {
				dynamicMeshInstances.push(meshInstance);
			} else {
				staticMeshInstances.push(meshInstance);
			}
		}
		this._staticBVH.rebuild(staticMeshInstances);
		this._dynamicOctree.rebuild(dynamicMeshInstances);
	}

	public queryFrustum(
		frustum: Frustum,
		options?: SpatialQueryOptions
	): MeshInstance[] {
		const maxResults = resolveMaxResults(options?.maxResults);
		if (maxResults <= 0) return [];
		const resolvedOptions: SpatialQueryOptions = {
			includeInvisible: options?.includeInvisible,
		};

		const staticHits = this._staticBVH.queryFrustum(frustum, resolvedOptions);
		const dynamicHits = this._dynamicOctree.queryFrustum(frustum, resolvedOptions);
		if (dynamicHits.length === 0) {
			return staticHits.length > maxResults ?
					staticHits.slice(0, maxResults)
				:	staticHits;
		}
		if (staticHits.length === 0) {
			return dynamicHits.length > maxResults ?
					dynamicHits.slice(0, maxResults)
				:	dynamicHits;
		}

		const result: MeshInstance[] = [];
		const seen = new Set<MeshInstance>();
		appendUniqueMeshInstances(staticHits, seen, maxResults, result);
		if (result.length >= maxResults) return result;
		appendUniqueMeshInstances(dynamicHits, seen, maxResults, result);
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
		const maxResults = resolveMaxResults(options?.maxResults);
		if (maxResults <= 0) return [];
		const resolvedOptions: SpatialRayQueryOptions = {
			includeInvisible: options?.includeInvisible,
			maxDistance: options?.maxDistance,
		};

		const staticHits = this._staticBVH.queryRayDetailed(
			origin,
			direction,
			resolvedOptions
		);
		const dynamicHits = this._dynamicOctree.queryRayDetailed(
			origin,
			direction,
			resolvedOptions
		);
		if (dynamicHits.length === 0) {
			return staticHits.length > maxResults ?
					staticHits.slice(0, maxResults)
				:	staticHits;
		}
		if (staticHits.length === 0) {
			return dynamicHits.length > maxResults ?
					dynamicHits.slice(0, maxResults)
				:	dynamicHits;
		}

		const hitByMeshInstance = new Map<MeshInstance, SpatialRayHit>();
		for (const hit of staticHits) {
			hitByMeshInstance.set(hit.meshInstance, hit);
		}
		for (const hit of dynamicHits) {
			const previous = hitByMeshInstance.get(hit.meshInstance);
			if (!previous || hit.distance < previous.distance) {
				hitByMeshInstance.set(hit.meshInstance, hit);
			}
		}

		const result = Array.from(hitByMeshInstance.values());
		if (result.length === 0) return [];
		result.sort(compareRayHits);
		if (result.length > maxResults) {
			return result.slice(0, maxResults);
		}
		return result;
	}

	private _resolveBucket(meshInstance: MeshInstance): SpatialBucket {
		return this._dynamicPredicate(meshInstance) ? "dynamic" : "static";
	}

	private _upsertToBucket(
		meshInstance: MeshInstance,
		bucket: SpatialBucket
	): void {
		if (bucket === "dynamic") {
			this._dynamicOctree.upsert(meshInstance);
			return;
		}
		this._staticBVH.upsert(meshInstance);
	}

	private _removeFromBucket(
		meshInstance: MeshInstance,
		bucket: SpatialBucket
	): void {
		if (bucket === "dynamic") {
			this._dynamicOctree.remove(meshInstance);
			return;
		}
		this._staticBVH.remove(meshInstance);
	}
}

function resolveMaxResults(value: number | undefined): number {
	if (value === undefined) return Infinity;
	if (!Number.isFinite(value)) return Infinity;
	return Math.max(0, Math.floor(value));
}

function appendUniqueMeshInstances(
	source: MeshInstance[],
	seen: Set<MeshInstance>,
	maxResults: number,
	result: MeshInstance[]
): void {
	for (const meshInstance of source) {
		if (result.length >= maxResults) return;
		if (seen.has(meshInstance)) continue;
		seen.add(meshInstance);
		result.push(meshInstance);
	}
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
