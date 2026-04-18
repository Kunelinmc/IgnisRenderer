import type { Node } from "../core/Node";
import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../core/types";
import { Matrix4 } from "../maths/Matrix4";
import { MeshInstance } from "../meshes";
import type {
	CollisionGeometryBounds,
	CollisionGeometryTriangleQuery,
	CollisionGeometryTriangles,
	ICollisionGeometryProvider,
} from "./types";

export class DefaultCollisionGeometryProvider implements ICollisionGeometryProvider {
	private _trianglesCache = new Map<string, CollisionGeometryTriangles>();
	private _geometryKeyByMeshId = new Map<string, string>();

	public getBounds(node: Node): CollisionGeometryBounds | null {
		const worldBox = node.getWorldBoundingBox();
		const center = {
			x: (worldBox.min.x + worldBox.max.x) * 0.5,
			y: (worldBox.min.y + worldBox.max.y) * 0.5,
			z: (worldBox.min.z + worldBox.max.z) * 0.5,
		};
		const dx = worldBox.max.x - worldBox.min.x;
		const dy = worldBox.max.y - worldBox.min.y;
		const dz = worldBox.max.z - worldBox.min.z;
		const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5;
		return {
			box: worldBox,
			sphere: {
				center,
				radius: Number.isFinite(radius) ? radius : 0,
			},
		};
	}

	public getTriangles(
		node: Node,
		options: CollisionGeometryTriangleQuery = {}
	): CollisionGeometryTriangles | null {
		if (!(node instanceof MeshInstance)) return null;
		const space = options.space === "world" ? "world" : "local";
		const geometryKey = computeGeometryKey(node);
		const cacheKey = `${space}:${geometryKey}`;
		const useCache = options.useCache !== false;

		const previousKey = this._geometryKeyByMeshId.get(node.mesh.id);
		if (previousKey && previousKey !== geometryKey) {
			this._trianglesCache.delete(`local:${previousKey}`);
			this._trianglesCache.delete(`world:${previousKey}`);
		}
		this._geometryKeyByMeshId.set(node.mesh.id, geometryKey);

		if (useCache) {
			const cached = this._trianglesCache.get(cacheKey);
			if (cached) return cached;
		}

		const vertices: number[] = [];
		const indices: number[] = [];
		let vertexOffset = 0;

		for (const primitive of node.mesh.primitives) {
			if (
				(primitive.topology ?? DEFAULT_PRIMITIVE_DRAW_TOPOLOGY) !==
				DEFAULT_PRIMITIVE_DRAW_TOPOLOGY
			) {
				continue;
			}

			const geometry = primitive.geometry;
			const positions = geometry.positions;
			const primitiveIndices = geometry.indices;
			if (!positions || !primitiveIndices) continue;

			const vertexCount = Math.floor(positions.length / 3);
			if (space === "local") {
				for (let i = 0; i < positions.length; i++) {
					vertices.push(positions[i]);
				}
			} else {
				for (let i = 0; i < vertexCount; i++) {
					const local = {
						x: positions[i * 3],
						y: positions[i * 3 + 1],
						z: positions[i * 3 + 2],
					};
					const world = Matrix4.transformPoint(node.worldMatrix, local);
					vertices.push(world.x, world.y, world.z);
				}
			}

			for (let i = 0; i < primitiveIndices.length; i++) {
				indices.push(primitiveIndices[i] + vertexOffset);
			}

			vertexOffset += vertexCount;
		}

		if (vertices.length === 0 || indices.length === 0) return null;
		const triangles: CollisionGeometryTriangles = {
			vertices: new Float32Array(vertices),
			indices: new Uint32Array(indices),
			space,
			geometryKey,
		};
		if (useCache) {
			this._trianglesCache.set(cacheKey, triangles);
		}
		return triangles;
	}
}

function computeGeometryKey(meshInstance: MeshInstance): string {
	const primitiveKeys: string[] = [];
	for (const primitive of meshInstance.mesh.primitives) {
		if (
			(primitive.topology ?? DEFAULT_PRIMITIVE_DRAW_TOPOLOGY) !==
			DEFAULT_PRIMITIVE_DRAW_TOPOLOGY
		) {
			continue;
		}
		const positionsLength = primitive.geometry.positions?.length ?? 0;
		const indicesLength = primitive.geometry.indices?.length ?? 0;
		primitiveKeys.push(
			`${primitive.id}:${primitive.geometryVersion}:${positionsLength}:${indicesLength}`
		);
	}
	return `${meshInstance.mesh.id}|${primitiveKeys.join(";")}`;
}
