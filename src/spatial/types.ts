import type { Frustum } from "../maths/Frustum";
import type { MeshInstance } from "../meshes";

export interface SpatialBounds3D {
	min: { x: number; y: number; z: number };
	max: { x: number; y: number; z: number };
}

export interface SpatialQueryOptions {
	maxResults?: number;
	includeInvisible?: boolean;
}

export interface SpatialRayQueryOptions extends SpatialQueryOptions {
	maxDistance?: number;
}

export interface SpatialRayHit {
	meshInstance: MeshInstance;
	distance: number;
}

export interface SpatialIndex3D {
	readonly size: number;
	readonly dirty: boolean;
	setMeshInstances(meshInstances: MeshInstance[]): void;
	markDirty(meshInstance?: MeshInstance): void;
	upsert(meshInstance: MeshInstance): void;
	remove(meshInstance: MeshInstance): boolean;
	rebuild(meshInstances?: MeshInstance[]): void;
	/**
	 * Clears `out` and appends mesh instances whose bounds overlap `bounds`.
	 *
	 * @param bounds - World-space AABB used for overlap testing.
	 * @param out - Caller-owned output array cleared before results are appended.
	 * @param options - Optional visibility and result-count constraints.
	 * @returns The same `out` array.
	 * @sideEffects May rebuild or refit dirty internal acceleration structures.
	 */
	queryBoundsInto(
		bounds: SpatialBounds3D,
		out: MeshInstance[],
		options?: SpatialQueryOptions
	): MeshInstance[];
	/**
	 * Clears `out` and appends mesh instances whose bounds overlap `frustum`.
	 *
	 * @param frustum - Camera frustum used for culling.
	 * @param out - Caller-owned output array cleared before results are appended.
	 * @param options - Optional visibility and result-count constraints.
	 * @returns The same `out` array.
	 * @sideEffects May rebuild or refit dirty internal acceleration structures.
	 */
	queryFrustumInto(
		frustum: Frustum,
		out: MeshInstance[],
		options?: SpatialQueryOptions
	): MeshInstance[];
	/**
	 * Clears `out` and appends sorted ray/AABB candidates.
	 *
	 * @param origin - World-space ray origin.
	 * @param direction - Non-zero ray direction; implementations normalize it.
	 * @param out - Caller-owned output array cleared before results are appended.
	 * @param options - Optional visibility, distance, and result-count constraints.
	 * @returns The same `out` array, sorted by distance, entity id, then mesh id.
	 * @sideEffects May rebuild or refit dirty internal acceleration structures.
	 */
	queryRayDetailedInto(
		origin: { x: number; y: number; z: number },
		direction: { x: number; y: number; z: number },
		out: SpatialRayHit[],
		options?: SpatialRayQueryOptions
	): SpatialRayHit[];
}

export type SpatialIndexMode = "bvh" | "hybrid";
