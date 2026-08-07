import { Matrix4 } from "../maths/Matrix4";
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
	private _mesh: MeshAsset;
	public skeleton: Skeleton | null;
	public morphWeights: Float32Array[];
	private readonly _worldBoundingBox: BoundingBox = {
		min: { x: 0, y: 0, z: 0 },
		max: { x: 0, y: 0, z: 0 },
	};
	private readonly _worldBoundingSphere: BoundingSphere = {
		center: { x: 0, y: 0, z: 0 },
		radius: 0,
	};
	private readonly _worldBoundsMatrixSnapshot = new Float64Array(16);
	private _worldBoundsMesh: MeshAsset | null = null;
	private _worldBoundsMeshVersion = -1;
	private _worldBoundsVersion = 0;
	private _worldBoundsInitialized = false;

	constructor(params: MeshInstanceParams) {
		super({
			...params,
			idPrefix: "meshInstance",
		});
		this._mesh = params.mesh;
		this.skeleton = params.skeleton ?? null;
		this.morphWeights =
			params.morphWeights?.map((weights) => new Float32Array(weights)) ??
			this.mesh.defaultMorphWeights.map((weights) => new Float32Array(weights));
	}

	public get mesh(): MeshAsset {
		return this._mesh;
	}

	public set mesh(mesh: MeshAsset) {
		if (this._mesh === mesh) return;
		this._mesh = mesh;
		this._worldBoundsInitialized = false;
		this.scene?.invalidate("transform");
	}

	/**
	 * Monotonic own-world-bounds revision used by `Scene` and spatial indexes.
	 *
	 * @internal Owned by scene/spatial synchronization; consumers should query
	 * bounds instead of depending on this token.
	 */
	public get worldBoundsVersion(): number {
		this._ensureWorldBoundsFresh();
		return this._worldBoundsVersion;
	}

	public getWorldBoundingSphere(out?: BoundingSphere): BoundingSphere {
		this._ensureWorldBoundsFresh();
		const target = out ?? {
			center: { x: 0, y: 0, z: 0 },
			radius: 0,
		};
		target.center.x = this._worldBoundingSphere.center.x;
		target.center.y = this._worldBoundingSphere.center.y;
		target.center.z = this._worldBoundingSphere.center.z;
		target.radius = this._worldBoundingSphere.radius;
		return target;
	}

	/**
	 * Copies the world AABB of this instance's own mesh geometry.
	 *
	 * @param out - Optional destination reused by the caller.
	 * @returns `out`, or a new bounding box when `out` is omitted.
	 * @internal Owned by spatial indexes. Consumers that need subtree aggregate
	 * bounds should use `Node.getWorldBoundingBox()`.
	 */
	public override getOwnWorldBoundingBox(out?: BoundingBox): BoundingBox {
		this._ensureWorldBoundsFresh();
		const target =
			out ??
			{
				min: { x: 0, y: 0, z: 0 },
				max: { x: 0, y: 0, z: 0 },
			};
		target.min.x = this._worldBoundingBox.min.x;
		target.min.y = this._worldBoundingBox.min.y;
		target.min.z = this._worldBoundingBox.min.z;
		target.max.x = this._worldBoundingBox.max.x;
		target.max.y = this._worldBoundingBox.max.y;
		target.max.z = this._worldBoundingBox.max.z;
		return target;
	}

	private _ensureWorldBoundsFresh(): void {
		const meshVersion = this._mesh.boundsVersion;
		const matrixChanged = this._captureWorldMatrixIfChanged();
		if (
			this._worldBoundsInitialized &&
			!matrixChanged &&
			this._worldBoundsMesh === this._mesh &&
			this._worldBoundsMeshVersion === meshVersion
		) {
			return;
		}

		const box = this._mesh.boundingBox;
		const sphere = this._mesh.boundingSphere;
		const elements = this.worldMatrix.elements;
		const centerX = (box.min.x + box.max.x) * 0.5;
		const centerY = (box.min.y + box.max.y) * 0.5;
		const centerZ = (box.min.z + box.max.z) * 0.5;
		const extentX = (box.max.x - box.min.x) * 0.5;
		const extentY = (box.max.y - box.min.y) * 0.5;
		const extentZ = (box.max.z - box.min.z) * 0.5;
		const worldCenterX =
			elements[0][0] * centerX +
			elements[0][1] * centerY +
			elements[0][2] * centerZ +
			elements[0][3];
		const worldCenterY =
			elements[1][0] * centerX +
			elements[1][1] * centerY +
			elements[1][2] * centerZ +
			elements[1][3];
		const worldCenterZ =
			elements[2][0] * centerX +
			elements[2][1] * centerY +
			elements[2][2] * centerZ +
			elements[2][3];
		const worldExtentX =
			Math.abs(elements[0][0]) * extentX +
			Math.abs(elements[0][1]) * extentY +
			Math.abs(elements[0][2]) * extentZ;
		const worldExtentY =
			Math.abs(elements[1][0]) * extentX +
			Math.abs(elements[1][1]) * extentY +
			Math.abs(elements[1][2]) * extentZ;
		const worldExtentZ =
			Math.abs(elements[2][0]) * extentX +
			Math.abs(elements[2][1]) * extentY +
			Math.abs(elements[2][2]) * extentZ;

		this._worldBoundingBox.min.x = worldCenterX - worldExtentX;
		this._worldBoundingBox.min.y = worldCenterY - worldExtentY;
		this._worldBoundingBox.min.z = worldCenterZ - worldExtentZ;
		this._worldBoundingBox.max.x = worldCenterX + worldExtentX;
		this._worldBoundingBox.max.y = worldCenterY + worldExtentY;
		this._worldBoundingBox.max.z = worldCenterZ + worldExtentZ;

		const sphereCenter = sphere.center;
		this._worldBoundingSphere.center.x =
			elements[0][0] * sphereCenter.x +
			elements[0][1] * sphereCenter.y +
			elements[0][2] * sphereCenter.z +
			elements[0][3];
		this._worldBoundingSphere.center.y =
			elements[1][0] * sphereCenter.x +
			elements[1][1] * sphereCenter.y +
			elements[1][2] * sphereCenter.z +
			elements[1][3];
		this._worldBoundingSphere.center.z =
			elements[2][0] * sphereCenter.x +
			elements[2][1] * sphereCenter.y +
			elements[2][2] * sphereCenter.z +
			elements[2][3];
		this._worldBoundingSphere.radius =
			sphere.radius * (getMaxScaleFromMatrix(this.worldMatrix) || 1);

		this._worldBoundsMesh = this._mesh;
		this._worldBoundsMeshVersion = meshVersion;
		this._worldBoundsInitialized = true;
		this._worldBoundsVersion++;
	}

	private _captureWorldMatrixIfChanged(): boolean {
		const elements = this.worldMatrix.elements;
		let changed = !this._worldBoundsInitialized;
		let cursor = 0;
		for (let row = 0; row < 4; row++) {
			for (let column = 0; column < 4; column++) {
				const value = elements[row][column];
				if (this._worldBoundsMatrixSnapshot[cursor] !== value) changed = true;
				this._worldBoundsMatrixSnapshot[cursor++] = value;
			}
		}
		return changed;
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
