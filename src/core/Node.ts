import { Matrix4 } from "../maths/Matrix4";
import { Quaternion } from "../maths/Quaternion";
import type { IVector3 } from "../maths/types";
import { Vector3 } from "../maths/Vector3";
import { IdGenerator } from "../foundation/IdGenerator";
import type { BoundingBox } from "./types";
import type { Scene } from "./Scene";

interface QuaternionLike {
	x: number;
	y: number;
	z: number;
	w: number;
}

export interface NodeParams {
	idPrefix?: string;
	name?: string;
	visible?: boolean;
	position?: IVector3;
	quaternion?: Quaternion | QuaternionLike;
	scale?: IVector3;
}

export class Node {
	public readonly id: string;
	public name: string;
	public visible: boolean;
	public parent: Node | null;
	public children: Node[];
	public position: Vector3;
	public quaternion: Quaternion;
	public scale: Vector3;
	public localMatrix: Matrix4;
	public worldMatrix: Matrix4;
	private _scene: Scene | null;
	private _entityId: number | null;

	constructor(params: NodeParams = {}) {
		this.id = IdGenerator.nextId(params.idPrefix ?? "node");
		this.name = params.name ?? this.id;
		this.visible = params.visible ?? true;
		this.parent = null;
		this.children = [];
		this.position = new Vector3();
		this.position.copy(params.position ?? { x: 0, y: 0, z: 0 });
		this.quaternion = createQuaternion(params.quaternion);
		this.scale = new Vector3();
		this.scale.copy(params.scale ?? { x: 1, y: 1, z: 1 });
		this.localMatrix = Matrix4.identity();
		this.worldMatrix = Matrix4.identity();
		this._scene = null;
		this._entityId = null;
		this.updateLocalMatrix();
		copyMatrix(this.worldMatrix, this.localMatrix);
	}

	public get entityId(): number | null {
		return this._entityId;
	}

	public get scene(): Scene | null {
		return this._scene;
	}

	/**
	 * @internal Scene-owned ECS binding. External code must not call this.
	 */
	public _setSceneInternal(scene: Scene | null): void {
		this._scene = scene;
	}

	/**
	 * @internal Scene-owned ECS binding. External code must not call this.
	 */
	public _setEntityIdInternal(entityId: number | null): void {
		this._entityId = entityId;
	}

	public addChild(child: Node): Node {
		if (child === this) {
			throw new Error("Node cannot be parent of itself");
		}

		if (this._isAncestorOf(child)) {
			throw new Error(`Cannot create cycle while adding child "${child.id}"`);
		}

		if (child.parent) {
			if (
				this._scene &&
				child._scene === this._scene &&
				child.parent._scene === this._scene
			) {
				this._scene.markNodeReparenting(child, true);
			}
			child.parent.removeChild(child);
			if (this._scene && child._scene === this._scene && this._scene) {
				this._scene.markNodeReparenting(child, false);
			}
		}
		child.parent = this;
		this.children.push(child);
		if (this._scene) {
			this._scene.onNodeAttachedFromAPI(this, child);
		}
		return child;
	}

	public removeChild(child: Node): boolean {
		const index = this.children.indexOf(child);
		if (index === -1) return false;
		this.children.splice(index, 1);
		child.parent = null;
		if (this._scene) {
			this._scene.onNodeDetachedFromAPI(this, child);
		}
		return true;
	}

	public traverse(visitor: (node: Node) => void): void {
		visitor(this);
		for (const child of this.children) {
			child.traverse(visitor);
		}
	}

	public setRotationFromEuler(x: number, y: number, z: number): this {
		this.quaternion = Quaternion.fromEuler(x, y, z).normalize();
		this.updateLocalMatrix();
		return this;
	}

	public rotateByQuaternion(q: Quaternion): this {
		this.quaternion = Quaternion.multiply(q, this.quaternion).normalize();
		this.updateLocalMatrix();
		return this;
	}

	public updateLocalMatrix(): void {
		const qx = this.quaternion.x;
		const qy = this.quaternion.y;
		const qz = this.quaternion.z;
		const qw = this.quaternion.w;
		const sx = this.scale.x;
		const sy = this.scale.y;
		const sz = this.scale.z;
		const px = this.position.x;
		const py = this.position.y;
		const pz = this.position.z;
		const elements = this.localMatrix.elements;

		const x2 = qx + qx;
		const y2 = qy + qy;
		const z2 = qz + qz;
		const xx = qx * x2;
		const xy = qx * y2;
		const xz = qx * z2;
		const yy = qy * y2;
		const yz = qy * z2;
		const zz = qz * z2;
		const wx = qw * x2;
		const wy = qw * y2;
		const wz = qw * z2;

		elements[0][0] = (1 - (yy + zz)) * sx;
		elements[0][1] = (xy - wz) * sy;
		elements[0][2] = (xz + wy) * sz;
		elements[0][3] = px;

		elements[1][0] = (xy + wz) * sx;
		elements[1][1] = (1 - (xx + zz)) * sy;
		elements[1][2] = (yz - wx) * sz;
		elements[1][3] = py;

		elements[2][0] = (xz - wy) * sx;
		elements[2][1] = (yz + wx) * sy;
		elements[2][2] = (1 - (xx + yy)) * sz;
		elements[2][3] = pz;

		elements[3][0] = 0;
		elements[3][1] = 0;
		elements[3][2] = 0;
		elements[3][3] = 1;
	}

	public updateWorldMatrix(parentWorldMatrix?: Matrix4): void {
		this.updateLocalMatrix();

		if (parentWorldMatrix) {
			Matrix4.multiply(parentWorldMatrix, this.localMatrix, this.worldMatrix);
		} else {
			copyMatrix(this.worldMatrix, this.localMatrix);
		}

		for (const child of this.children) {
			child.updateWorldMatrix(this.worldMatrix);
		}
	}

	public getWorldPosition(out?: IVector3): IVector3 {
		const transformed = Matrix4.transformPoint(this.worldMatrix, {
			x: 0,
			y: 0,
			z: 0,
		});
		const target = out ?? { x: 0, y: 0, z: 0 };
		target.x = transformed.x;
		target.y = transformed.y;
		target.z = transformed.z;
		return target;
	}

	public getWorldBoundingBox(): BoundingBox {
		let minX = Infinity;
		let minY = Infinity;
		let minZ = Infinity;
		let maxX = -Infinity;
		let maxY = -Infinity;
		let maxZ = -Infinity;

		this.traverse((node) => {
			const box = node.getOwnWorldBoundingBox();
			if (box) {
				if (box.min.x < minX) minX = box.min.x;
				if (box.min.y < minY) minY = box.min.y;
				if (box.min.z < minZ) minZ = box.min.z;
				if (box.max.x > maxX) maxX = box.max.x;
				if (box.max.y > maxY) maxY = box.max.y;
				if (box.max.z > maxZ) maxZ = box.max.z;
			}
		});

		if (minX === Infinity) {
			const pos = this.getWorldPosition();
			return {
				min: { x: pos.x, y: pos.y, z: pos.z },
				max: { x: pos.x, y: pos.y, z: pos.z },
			};
		}

		return {
			min: { x: minX, y: minY, z: minZ },
			max: { x: maxX, y: maxY, z: maxZ },
		};
	}

	public getWorldDirection(localDirection: IVector3, out?: IVector3): IVector3 {
		const transformed = Matrix4.transformDirection(this.worldMatrix, localDirection);
		const length = Math.hypot(transformed.x, transformed.y, transformed.z) || 1;
		const target = out ?? { x: 0, y: 0, z: 0 };
		target.x = transformed.x / length;
		target.y = transformed.y / length;
		target.z = transformed.z / length;
		return target;
	}

	public clone(recursive: boolean = true): this {
		const cloned = this._createCloneInstance();
		this._copyClonePropertiesTo(cloned);

		if (recursive) {
			for (const child of this.children) {
				cloned.addChild(child.clone(true));
			}
		}

		return cloned;
	}

	protected _createCloneInstance(): this {
		const Constructor = this.constructor as unknown as { new (): Node };
		try {
			return new Constructor() as this;
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			throw new Error(
				`Node.clone failed for "${this.constructor.name}". ` +
					`Override _createCloneInstance in this class. Cause: ${message}`,
			);
		}
	}

	protected _copyClonePropertiesTo(target: this): void {
		target.name = this.name;
		target.visible = this.visible;
		target.position.copy(this.position);
		target.quaternion = createQuaternion(this.quaternion);
		target.scale.copy(this.scale);
		copyMatrix(target.localMatrix, this.localMatrix);
		copyMatrix(target.worldMatrix, this.worldMatrix);
	}

	protected getOwnWorldBoundingBox(): BoundingBox | null {
		return null;
	}

	private _isAncestorOf(candidate: Node): boolean {
		let current = this.parent;
		while (current) {
			if (current === candidate) return true;
			current = current.parent;
		}
		return false;
	}
}

function copyMatrix(target: Matrix4, source: Matrix4): void {
	const targetElements = target.elements;
	const sourceElements = source.elements;
	for (let row = 0; row < 4; row++) {
		targetElements[row][0] = sourceElements[row][0];
		targetElements[row][1] = sourceElements[row][1];
		targetElements[row][2] = sourceElements[row][2];
		targetElements[row][3] = sourceElements[row][3];
	}
}

function createQuaternion(
	value: Quaternion | QuaternionLike | undefined
): Quaternion {
	if (!value) {
		return new Quaternion();
	}
	return new Quaternion(value.x, value.y, value.z, value.w).normalize();
}
