import type { Frustum } from "../maths/Frustum";
import type { MeshInstance } from "../meshes";
import { BVH } from "./BVH";
import { isDynamicSpatialMeshInstance } from "./classification";
import { LooseOctree, type LooseOctreeOptions } from "./LooseOctree";
import type {
	SpatialIndex3D,
	SpatialBounds3D,
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
	private readonly _meshScratch: MeshInstance[] = [];
	private readonly _staticRayScratch: SpatialRayHit[] = [];
	private readonly _dynamicRayScratch: SpatialRayHit[] = [];

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

	public queryFrustumInto(
		frustum: Frustum,
		out: MeshInstance[],
		options?: SpatialQueryOptions
	): MeshInstance[] {
		out.length = 0;
		const maxResults = resolveMaxResults(options?.maxResults);
		if (maxResults <= 0) return out;
		const resolvedOptions: SpatialQueryOptions = {
			includeInvisible: options?.includeInvisible,
			maxResults,
		};

		this._staticBVH.queryFrustumInto(frustum, out, resolvedOptions);
		if (out.length >= maxResults) return out;
		this._meshScratch.length = 0;
		this._dynamicOctree.queryFrustumInto(
			frustum,
			this._meshScratch,
			{
				includeInvisible: options?.includeInvisible,
				maxResults: maxResults - out.length,
			}
		);
		appendMeshInstances(this._meshScratch, maxResults, out);
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
		const maxResults = resolveMaxResults(options?.maxResults);
		if (maxResults <= 0) return out;
		const resolvedOptions: SpatialQueryOptions = {
			includeInvisible: options?.includeInvisible,
			maxResults,
		};

		this._staticBVH.queryBoundsInto(bounds, out, resolvedOptions);
		if (out.length >= maxResults) return out;
		this._meshScratch.length = 0;
		this._dynamicOctree.queryBoundsInto(
			bounds,
			this._meshScratch,
			{
				includeInvisible: options?.includeInvisible,
				maxResults: maxResults - out.length,
			}
		);
		appendMeshInstances(this._meshScratch, maxResults, out);
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
		const maxResults = resolveMaxResults(options?.maxResults);
		if (maxResults <= 0) return out;
		const baseOptions: SpatialRayQueryOptions = {
			includeInvisible: options?.includeInvisible,
			maxDistance: options?.maxDistance,
			maxResults,
		};

		this._staticRayScratch.length = 0;
		this._dynamicRayScratch.length = 0;
		this._staticBVH.queryRayDetailedInto(
			origin,
			direction,
			this._staticRayScratch,
			baseOptions
		);
		if (maxResults === 1) {
			const staticNearest = this._staticRayScratch[0] ?? null;
			const dynamicMaxDistance =
				staticNearest?.distance ?? options?.maxDistance;
			this._dynamicOctree.queryRayDetailedInto(
				origin,
				direction,
				this._dynamicRayScratch,
				{
					includeInvisible: options?.includeInvisible,
					maxDistance: dynamicMaxDistance,
					maxResults: 1,
				}
			);
			const dynamicNearest = this._dynamicRayScratch[0] ?? null;
			const best =
				!staticNearest ? dynamicNearest
				: !dynamicNearest ? staticNearest
				: compareRayHits(staticNearest, dynamicNearest) <= 0 ?
					staticNearest
				:	dynamicNearest;
			if (best) out.push(best);
			return out;
		}

		this._dynamicOctree.queryRayDetailedInto(
			origin,
			direction,
			this._dynamicRayScratch,
			baseOptions
		);
		mergeSortedRayHits(
			this._staticRayScratch,
			this._dynamicRayScratch,
			maxResults,
			out
		);
		return out;
	}

	public queryRayDetailed(
		origin: { x: number; y: number; z: number },
		direction: { x: number; y: number; z: number },
		options?: SpatialRayQueryOptions
	): SpatialRayHit[] {
		return this.queryRayDetailedInto(origin, direction, [], options);
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

function appendMeshInstances(
	source: MeshInstance[],
	maxResults: number,
	result: MeshInstance[]
): void {
	for (const meshInstance of source) {
		if (result.length >= maxResults) return;
		result.push(meshInstance);
	}
}

function mergeSortedRayHits(
	left: SpatialRayHit[],
	right: SpatialRayHit[],
	maxResults: number,
	result: SpatialRayHit[]
): void {
	let leftIndex = 0;
	let rightIndex = 0;
	while (
		result.length < maxResults &&
		(leftIndex < left.length || rightIndex < right.length)
	) {
		if (leftIndex >= left.length) {
			result.push(right[rightIndex++]);
			continue;
		}
		if (rightIndex >= right.length) {
			result.push(left[leftIndex++]);
			continue;
		}
		if (compareRayHits(left[leftIndex], right[rightIndex]) <= 0) {
			result.push(left[leftIndex++]);
		} else {
			result.push(right[rightIndex++]);
		}
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
