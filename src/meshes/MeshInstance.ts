import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import { Node, type NodeParams } from "../core/Node";
import { MeshAsset } from "./MeshAsset";
import type { BoundingBox, BoundingSphere } from "../core/types";
import type { Skeleton } from "../animation/Skeleton";

export interface MeshInstanceParams extends NodeParams {
	mesh: MeshAsset;
	skeleton?: Skeleton | null;
	morphWeights?: ArrayLike<number>[];
}

export class MeshInstance extends Node {
	public mesh: MeshAsset;
	public skeleton: Skeleton | null;
	public morphWeights: Float32Array[];
	private _localBoundsCorner: IVector3 = { x: 0, y: 0, z: 0 };
	private _worldBoundsCorner: IVector3 = { x: 0, y: 0, z: 0 };

	constructor(params: MeshInstanceParams) {
		super({
			...params,
			idPrefix: "meshInstance",
		});
		this.mesh = params.mesh;
		this.skeleton = params.skeleton ?? null;
		this.morphWeights =
			params.morphWeights?.map((weights) => new Float32Array(weights)) ??
			this.mesh.defaultMorphWeights.map((weights) => new Float32Array(weights));
	}

	public getWorldBoundingSphere(out?: BoundingSphere): BoundingSphere {
		const worldCenter = Matrix4.transformPoint(
			this.worldMatrix,
			this.mesh.boundingSphere.center
		);
		const worldScale = getMaxScaleFromMatrix(this.worldMatrix) || 1;
		const target = out ?? {
			center: { x: 0, y: 0, z: 0 },
			radius: 0,
		};
		target.center.x = worldCenter.x;
		target.center.y = worldCenter.y;
		target.center.z = worldCenter.z;
		target.radius = this.mesh.boundingSphere.radius * worldScale;
		return target;
	}

	protected override getOwnWorldBoundingBox(out?: BoundingBox): BoundingBox {
		const box = this.mesh.boundingBox;
		const localCorner = this._localBoundsCorner;
		const worldCorner = this._worldBoundsCorner;

		let minX = Infinity;
		let minY = Infinity;
		let minZ = Infinity;
		let maxX = -Infinity;
		let maxY = -Infinity;
		let maxZ = -Infinity;

		for (let cornerIndex = 0; cornerIndex < 8; cornerIndex++) {
			localCorner.x = (cornerIndex & 1) === 0 ? box.min.x : box.max.x;
			localCorner.y = (cornerIndex & 2) === 0 ? box.min.y : box.max.y;
			localCorner.z = (cornerIndex & 4) === 0 ? box.min.z : box.max.z;
			Matrix4.transformPoint(this.worldMatrix, localCorner, worldCorner);
			if (worldCorner.x < minX) minX = worldCorner.x;
			if (worldCorner.y < minY) minY = worldCorner.y;
			if (worldCorner.z < minZ) minZ = worldCorner.z;
			if (worldCorner.x > maxX) maxX = worldCorner.x;
			if (worldCorner.y > maxY) maxY = worldCorner.y;
			if (worldCorner.z > maxZ) maxZ = worldCorner.z;
		}

		const target =
			out ??
			{
				min: { x: 0, y: 0, z: 0 },
				max: { x: 0, y: 0, z: 0 },
			};
		target.min.x = minX;
		target.min.y = minY;
		target.min.z = minZ;
		target.max.x = maxX;
		target.max.y = maxY;
		target.max.z = maxZ;
		return target;
	}

	protected override _createCloneInstance(): this {
		return new MeshInstance({
			mesh: this.mesh,
			skeleton: this.skeleton,
			morphWeights: this.morphWeights.map(
				(weights) => new Float32Array(weights)
			),
		}) as this;
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.mesh = this.mesh;
		target.skeleton = this.skeleton;
		target.morphWeights = this.morphWeights.map(
			(weights) => new Float32Array(weights)
		);
	}
}

function getMaxScaleFromMatrix(matrix: Matrix4): number {
	const elements = matrix.elements;
	const x = Math.hypot(elements[0][0], elements[1][0], elements[2][0]);
	const y = Math.hypot(elements[0][1], elements[1][1], elements[2][1]);
	const z = Math.hypot(elements[0][2], elements[1][2], elements[2][2]);
	return Math.max(x, y, z);
}
