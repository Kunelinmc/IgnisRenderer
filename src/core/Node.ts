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
	renderLayers?: number;
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
	private _renderLayers: number;

	constructor(params: NodeParams = {}) {
		this.id = IdGenerator.nextId(params.idPrefix ?? "node");
		this.name = params.name ?? this.id;
		this.visible = params.visible ?? true;
		this._renderLayers = normalizeRenderLayerMask(params.renderLayers ?? 1);
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
		this.localMatrix.copyTo(this.worldMatrix);
	}

	public get entityId(): number | null {
		return this._entityId;
	}

	public get scene(): Scene | null {
		return this._scene;
	}

	/**
	 * Bitmask selecting which renderer layers this node belongs to.
	 *
	 * @returns A 32-bit unsigned render-layer mask. The default is layer bit 0.
	 * @sideEffects None.
	 */
	public get renderLayers(): number {
		return this._renderLayers;
	}

	/**
	 * Sets the renderer layer bitmask used by features such as decal receivers.
	 *
	 * @param value - Layer mask to store; non-finite values resolve to bit 0.
	 * @sideEffects Marks the owning scene dirty so prepared render lists rebuild.
	 */
	public set renderLayers(value: number) {
		const next = normalizeRenderLayerMask(value);
		if (this._renderLayers === next) {
			return;
		}
		this._renderLayers = next;
		this._scene?.invalidate("unknown");
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
		Matrix4.compose(this.position, this.quaternion, this.scale, this.localMatrix);
	}

	public updateWorldMatrix(parentWorldMatrix?: Matrix4): void {
		this.updateLocalMatrix();

		if (parentWorldMatrix) {
			Matrix4.multiply(parentWorldMatrix, this.localMatrix, this.worldMatrix);
		} else {
			this.localMatrix.copyTo(this.worldMatrix);
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

	public getWorldBoundingBox(out?: BoundingBox): BoundingBox {
		let minX = Infinity;
		let minY = Infinity;
		let minZ = Infinity;
		let maxX = -Infinity;
		let maxY = -Infinity;
		let maxZ = -Infinity;
		const ownBounds = createBoundingBox();

		this.traverse((node) => {
			const box = node.getOwnWorldBoundingBox(ownBounds);
			if (box) {
				if (box.min.x < minX) minX = box.min.x;
				if (box.min.y < minY) minY = box.min.y;
				if (box.min.z < minZ) minZ = box.min.z;
				if (box.max.x > maxX) maxX = box.max.x;
				if (box.max.y > maxY) maxY = box.max.y;
				if (box.max.z > maxZ) maxZ = box.max.z;
			}
		});

		const target = out ?? createBoundingBox();
		if (minX === Infinity) {
			const elements = this.worldMatrix.elements;
			const x = elements[0][3];
			const y = elements[1][3];
			const z = elements[2][3];
			target.min.x = x;
			target.min.y = y;
			target.min.z = z;
			target.max.x = x;
			target.max.y = y;
			target.max.z = z;
			return target;
		}

		target.min.x = minX;
		target.min.y = minY;
		target.min.z = minZ;
		target.max.x = maxX;
		target.max.y = maxY;
		target.max.z = maxZ;
		return target;
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
		target.renderLayers = this.renderLayers;
		target.position.copy(this.position);
		target.quaternion = createQuaternion(this.quaternion);
		target.scale.copy(this.scale);
		this.localMatrix.copyTo(target.localMatrix);
		this.worldMatrix.copyTo(target.worldMatrix);
	}

	protected getOwnWorldBoundingBox(_out?: BoundingBox): BoundingBox | null {
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

/**
 * Normalizes a renderer layer mask into an unsigned 32-bit value.
 *
 * @param value - Candidate mask value.
 * @param fallback - Mask used when `value` is not finite.
 * @returns A non-zero unsigned mask; zero is preserved to allow opt-out nodes.
 * @sideEffects None.
 */
export function normalizeRenderLayerMask(value: number, fallback = 1): number {
	if (!Number.isFinite(value)) {
		return fallback >>> 0;
	}
	return Math.max(0, Math.floor(value)) >>> 0;
}

function createQuaternion(
	value: Quaternion | QuaternionLike | undefined
): Quaternion {
	if (!value) {
		return new Quaternion();
	}
	return new Quaternion(value.x, value.y, value.z, value.w).normalize();
}

function createBoundingBox(): BoundingBox {
	return {
		min: { x: 0, y: 0, z: 0 },
		max: { x: 0, y: 0, z: 0 },
	};
}
