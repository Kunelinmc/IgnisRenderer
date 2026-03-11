import type { Node } from "../core/Node";
import { Matrix4 } from "../maths/Matrix4";
import { MeshInstance } from "../meshes";
import type {
	CollisionGeometryBounds,
	CollisionGeometryTriangles,
	ICollisionGeometryProvider,
} from "./types";

export class DefaultCollisionGeometryProvider implements ICollisionGeometryProvider {
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

	public getTriangles(node: Node): CollisionGeometryTriangles | null {
		if (!(node instanceof MeshInstance)) return null;

		const vertices: number[] = [];
		const indices: number[] = [];
		let vertexOffset = 0;

		for (const primitive of node.mesh.primitives) {
			const geometry = primitive.geometry;
			const positions = geometry.positions;
			const primitiveIndices = geometry.indices;
			if (!positions || !primitiveIndices) continue;

			const vertexCount = Math.floor(positions.length / 3);
			for (let i = 0; i < vertexCount; i++) {
				const local = {
					x: positions[i * 3],
					y: positions[i * 3 + 1],
					z: positions[i * 3 + 2],
				};
				const world = Matrix4.transformPoint(node.worldMatrix, local);
				vertices.push(world.x, world.y, world.z);
			}

			for (let i = 0; i < primitiveIndices.length; i++) {
				indices.push(primitiveIndices[i] + vertexOffset);
			}

			vertexOffset += vertexCount;
		}

		if (vertices.length === 0 || indices.length === 0) return null;
		return {
			vertices: new Float32Array(vertices),
			indices: new Uint32Array(indices),
		};
	}
}
