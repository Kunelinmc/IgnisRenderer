import { EventEmitter } from "../core/EventEmitter";
import { Vector3 } from "../maths/Vector3";
import { Matrix4 } from "../maths/Matrix4";
import type { IVector3, Point } from "../maths/types";
import type {
	IModel,
	IVertex,
	IFace,
	ProjectedFace,
	BoundingSphere,
	BoundingBox,
} from "../core/types";

export interface ModelVertex extends IVertex {
	[key: string]: unknown;
}

export interface ModelFace extends IFace {
	vertices: ModelVertex[];
	projected?: Point[];
	[key: string]: unknown;
}

/**
 * A simple model that can be used to store and render 3D models.
 */
export class SimpleModel extends EventEmitter implements IModel {
	public faces: ModelFace[];
	public transform: {
		rotation: Vector3;
		position: Vector3;
		scale: Vector3;
	};
	public boundingSphere: BoundingSphere;
	public boundingBox: BoundingBox;

	constructor(faces: ModelFace[] = []) {
		super();
		this.faces = faces;

		this.transform = {
			rotation: new Vector3(0, 0, 0),
			position: new Vector3(0, 0, 0),
			scale: new Vector3(1, 1, 1),
		};

		this.boundingSphere = this._calculateBoundingSphere();
		this.boundingBox = this._calculateBoundingBox();

		if (this.faces.length > 0 && !this._hasVertexNormals()) {
			this._computeVertexNormals();
		}
	}

	private _hasVertexNormals(): boolean {
		if (this.faces.length === 0) return false;
		let totalVerts = 0;
		let normalCount = 0;
		for (const face of this.faces) {
			for (const v of face.vertices) {
				totalVerts++;
				if (v.normal) normalCount++;
			}
		}
		return normalCount > totalVerts * 0.8;
	}

	private _computeVertexNormals(): void {
		const vertexNormals = new Map<string, IVector3>();

		for (const face of this.faces) {
			const v0 = face.vertices[0];
			const v1 = face.vertices[1];
			const v2 = face.vertices[2];

			if (!face.normal) {
				const ux = v1.x - v0.x,
					uy = v1.y - v0.y,
					uz = v1.z - v0.z;
				const vx = v2.x - v0.x,
					vy = v2.y - v0.y,
					vz = v2.z - v0.z;
				const nx = uy * vz - uz * vy;
				const ny = uz * vx - ux * vz;
				const nz = ux * vy - uy * vx;
				const len = Math.hypot(nx, ny, nz) || 1e-6;
				face.normal = { x: nx / len, y: ny / len, z: nz / len };
			}

			for (const v of face.vertices) {
				const key = `${v.x.toFixed(5)},${v.y.toFixed(5)},${v.z.toFixed(5)}`;
				if (!vertexNormals.has(key)) {
					vertexNormals.set(key, { x: 0, y: 0, z: 0 });
				}
				const sum = vertexNormals.get(key)!;
				sum.x += face.normal.x;
				sum.y += face.normal.y;
				sum.z += face.normal.z;
			}
		}

		const normalizedNormals = new Map<string, IVector3>();
		for (const [key, sum] of vertexNormals) {
			const len = Math.hypot(sum.x, sum.y, sum.z) || 1e-6;
			normalizedNormals.set(key, {
				x: sum.x / len,
				y: sum.y / len,
				z: sum.z / len,
			});
		}

		for (const face of this.faces) {
			for (const v of face.vertices) {
				const key = `${v.x.toFixed(5)},${v.y.toFixed(5)},${v.z.toFixed(5)}`;
				const normal = normalizedNormals.get(key);
				if (normal) {
					v.normal = { ...normal };
				}
			}
		}
	}

	private _calculateBoundingSphere(): BoundingSphere {
		if (this.faces.length === 0)
			return { center: { x: 0, y: 0, z: 0 }, radius: 0 };

		let minX = Infinity,
			maxX = -Infinity;
		let minY = Infinity,
			maxY = -Infinity;
		let minZ = Infinity,
			maxZ = -Infinity;

		for (const face of this.faces) {
			for (const v of face.vertices) {
				if (v.x < minX) minX = v.x;
				if (v.x > maxX) maxX = v.x;
				if (v.y < minY) minY = v.y;
				if (v.y > maxY) maxY = v.y;
				if (v.z < minZ) minZ = v.z;
				if (v.z > maxZ) maxZ = v.z;
			}
		}

		const center = {
			x: (minX + maxX) / 2,
			y: (minY + maxY) / 2,
			z: (minZ + maxZ) / 2,
		};

		let maxDistSq = 0;
		for (const face of this.faces) {
			for (const v of face.vertices) {
				const dx = v.x - center.x;
				const dy = v.y - center.y;
				const dz = v.z - center.z;
				const d2 = dx * dx + dy * dy + dz * dz;
				if (d2 > maxDistSq) maxDistSq = d2;
			}
		}

		return { center, radius: Math.sqrt(maxDistSq) };
	}

	private _calculateBoundingBox(): BoundingBox {
		if (this.faces.length === 0)
			return { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 } };

		let minX = Infinity,
			minY = Infinity,
			minZ = Infinity;
		let maxX = -Infinity,
			maxY = -Infinity,
			maxZ = -Infinity;

		for (const face of this.faces) {
			for (const v of face.vertices) {
				if (v.x < minX) minX = v.x;
				if (v.y < minY) minY = v.y;
				if (v.z < minZ) minZ = v.z;
				if (v.x > maxX) maxX = v.x;
				if (v.y > maxY) maxY = v.y;
				if (v.z > maxZ) maxZ = v.z;
			}
		}

		return {
			min: { x: minX, y: minY, z: minZ },
			max: { x: maxX, y: maxY, z: maxZ },
		};
	}

	public getWorldBoundingBox(): BoundingBox {
		const box = this.boundingBox;
		const transform = this.transform;

		const scaleMat = Matrix4.fromScale([
			transform.scale.x,
			transform.scale.y,
			transform.scale.z,
		]);
		const rotMat = Matrix4.rotationFromEuler(
			transform.rotation.x,
			transform.rotation.y,
			transform.rotation.z
		);
		const transMat = Matrix4.fromTranslation([
			transform.position.x,
			transform.position.y,
			transform.position.z,
		]);
		const modelMatrix: Matrix4 = Matrix4.multiply(
			transMat,
			Matrix4.multiply(rotMat, scaleMat)
		);

		const corners: IVector3[] = [
			{ x: box.min.x, y: box.min.y, z: box.min.z },
			{ x: box.max.x, y: box.min.y, z: box.min.z },
			{ x: box.min.x, y: box.max.y, z: box.min.z },
			{ x: box.max.x, y: box.max.y, z: box.min.z },
			{ x: box.min.x, y: box.min.y, z: box.max.z },
			{ x: box.max.x, y: box.min.y, z: box.max.z },
			{ x: box.min.x, y: box.max.y, z: box.max.z },
			{ x: box.max.x, y: box.max.y, z: box.max.z },
		];

		let minX = Infinity,
			minY = Infinity,
			minZ = Infinity;
		let maxX = -Infinity,
			maxY = -Infinity,
			maxZ = -Infinity;

		for (const corner of corners) {
			const worldPoint = Matrix4.transformPoint(modelMatrix, corner);
			if (worldPoint.x! < minX) minX = worldPoint.x!;
			if (worldPoint.y! < minY) minY = worldPoint.y!;
			if (worldPoint.z! < minZ) minZ = worldPoint.z!;
			if (worldPoint.x! > maxX) maxX = worldPoint.x!;
			if (worldPoint.y! > maxY) maxY = worldPoint.y!;
			if (worldPoint.z! > maxZ) maxZ = worldPoint.z!;
		}

		return {
			min: { x: minX, y: minY, z: minZ },
			max: { x: maxX, y: maxY, z: maxZ },
		};
	}
}
