import { Matrix4 } from "../maths/Matrix4";
import { Quaternion } from "../maths/Quaternion";
import type { IVector3 } from "../maths/types";
import { Vector3 } from "../maths/Vector3";
import { IdGenerator } from "../utils/IdGenerator";

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
		this.updateLocalMatrix();
		copyMatrix(this.worldMatrix, this.localMatrix);
	}

	public addChild(child: Node): Node {
		if (child === this) {
			throw new Error("Node cannot be parent of itself");
		}

		if (this._isAncestorOf(child)) {
			throw new Error(`Cannot create cycle while adding child "${child.id}"`);
		}

		if (child.parent) {
			child.parent.removeChild(child);
		}
		child.parent = this;
		this.children.push(child);
		return child;
	}

	public removeChild(child: Node): boolean {
		const index = this.children.indexOf(child);
		if (index === -1) return false;
		this.children.splice(index, 1);
		child.parent = null;
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
		const rotation = Matrix4.fromQuaternion([
			this.quaternion.x,
			this.quaternion.y,
			this.quaternion.z,
			this.quaternion.w,
		]).elements;

		this.localMatrix = new Matrix4([
			[
				rotation[0][0] * this.scale.x,
				rotation[0][1] * this.scale.y,
				rotation[0][2] * this.scale.z,
				this.position.x,
			],
			[
				rotation[1][0] * this.scale.x,
				rotation[1][1] * this.scale.y,
				rotation[1][2] * this.scale.z,
				this.position.y,
			],
			[
				rotation[2][0] * this.scale.x,
				rotation[2][1] * this.scale.y,
				rotation[2][2] * this.scale.z,
				this.position.z,
			],
			[0, 0, 0, 1],
		]);
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

	public getWorldDirection(localDirection: IVector3, out?: IVector3): IVector3 {
		const transformed = Matrix4.transformDirection(
			this.worldMatrix,
			localDirection
		);
		const length = Math.hypot(transformed.x, transformed.y, transformed.z) || 1;
		const target = out ?? { x: 0, y: 0, z: 0 };
		target.x = transformed.x / length;
		target.y = transformed.y / length;
		target.z = transformed.z / length;
		return target;
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

	if (value instanceof Quaternion) {
		return new Quaternion(value.x, value.y, value.z, value.w).normalize();
	}

	return new Quaternion(value.x, value.y, value.z, value.w).normalize();
}
