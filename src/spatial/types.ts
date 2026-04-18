import type { Frustum } from "../maths/Frustum";
import type { MeshInstance } from "../meshes";

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
	queryBounds(
		bounds: {
			min: { x: number; y: number; z: number };
			max: { x: number; y: number; z: number };
		},
		options?: SpatialQueryOptions
	): MeshInstance[];
	queryFrustum(
		frustum: Frustum,
		options?: SpatialQueryOptions
	): MeshInstance[];
	queryRay(
		origin: { x: number; y: number; z: number },
		direction: { x: number; y: number; z: number },
		options?: SpatialRayQueryOptions
	): MeshInstance[];
	queryRayDetailed(
		origin: { x: number; y: number; z: number },
		direction: { x: number; y: number; z: number },
		options?: SpatialRayQueryOptions
	): SpatialRayHit[];
}

export type SpatialIndexMode = "bvh" | "hybrid";
