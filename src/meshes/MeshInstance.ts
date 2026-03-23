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

	protected override getOwnWorldBoundingBox(): BoundingBox {
		const box = this.mesh.boundingBox;
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

		let minX = Infinity;
		let minY = Infinity;
		let minZ = Infinity;
		let maxX = -Infinity;
		let maxY = -Infinity;
		let maxZ = -Infinity;

		for (const corner of corners) {
			const worldPoint = Matrix4.transformPoint(this.worldMatrix, corner);
			if (worldPoint.x < minX) minX = worldPoint.x;
			if (worldPoint.y < minY) minY = worldPoint.y;
			if (worldPoint.z < minZ) minZ = worldPoint.z;
			if (worldPoint.x > maxX) maxX = worldPoint.x;
			if (worldPoint.y > maxY) maxY = worldPoint.y;
			if (worldPoint.z > maxZ) maxZ = worldPoint.z;
		}

		return {
			min: { x: minX, y: minY, z: minZ },
			max: { x: maxX, y: maxY, z: maxZ },
		};
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
