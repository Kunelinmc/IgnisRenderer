import type { Camera } from "../cameras/Camera";
import type { Node } from "../core/Node";
import type { Scene } from "../core/Scene";
import { Quaternion } from "../maths/Quaternion";
import type { IVector3 } from "../maths/types";
import { Vector3 } from "../maths/Vector3";
import type { InteractionPointerState } from "./Interactable";
import { screenToWorldRay } from "./screenToWorldRay";
import type {
	GizmoMode,
	InteractionGizmoState,
	GizmoPivot,
	GizmoSpace,
	InteractionTransformEvent,
} from "./types";

type AxisKey = "x" | "y" | "z";

interface GizmoTransformSnapshot {
	position: IVector3;
	quaternion: Quaternion;
	scale: IVector3;
}

interface ActiveGizmoState {
	mode: GizmoMode;
	space: GizmoSpace;
	pivot: GizmoPivot;
	axisLock: AxisKey | null;
	planeLock: AxisKey | null;
	pointerStart: { x: number; y: number };
	rayStartOrigin: IVector3;
	rayStartDirection: IVector3;
	pivotWorld: IVector3;
	snapshot: GizmoTransformSnapshot;
}

export class TransformGizmoController {
	private _scene: Scene | null = null;
	private _camera: Camera | null = null;
	private _activeGizmo: ActiveGizmoState | null = null;
	private _gizmoSpace: GizmoSpace = "world";
	private _gizmoPivot: GizmoPivot = "object-origin";

	public attach(scene: Scene, camera: Camera): void {
		this._scene = scene;
		this._camera = camera;
	}

	public detach(): void {
		this._scene = null;
		this._camera = null;
		this._activeGizmo = null;
	}

	public isActive(): boolean {
		return this._activeGizmo !== null;
	}

	public getState(): InteractionGizmoState | null {
		if (!this._activeGizmo) return null;
		return {
			mode: this._activeGizmo.mode,
			space: this._activeGizmo.space,
			pivot: this._activeGizmo.pivot,
		};
	}

	public setSpace(space: GizmoSpace): void {
		this._gizmoSpace = space;
	}

	public getSpace(): GizmoSpace {
		return this._gizmoSpace;
	}

	public setPivot(pivot: GizmoPivot): void {
		this._gizmoPivot = pivot;
	}

	public getPivot(): GizmoPivot {
		return this._gizmoPivot;
	}

	public begin(
		mode: GizmoMode,
		node: Node,
		pointer: InteractionPointerState
	): void {
		if (!this._camera) return;
		const ray = screenToWorldRay(this._camera, {
			screenX: pointer.screenX,
			screenY: pointer.screenY,
			viewportWidth: pointer.viewportWidth,
			viewportHeight: pointer.viewportHeight,
		});
		const pivotWorld =
			this._gizmoPivot === "bounds-center" ?
				getNodeBoundsCenter(node)
			: node.getWorldPosition({ x: 0, y: 0, z: 0 });
		this._activeGizmo = {
			mode,
			space: this._gizmoSpace,
			pivot: this._gizmoPivot,
			axisLock: null,
			planeLock: null,
			pointerStart: { x: pointer.screenX, y: pointer.screenY },
			rayStartOrigin: ray.origin,
			rayStartDirection: ray.direction,
			pivotWorld,
			snapshot: {
				position: {
					x: node.position.x,
					y: node.position.y,
					z: node.position.z,
				},
				quaternion: new Quaternion(
					node.quaternion.x,
					node.quaternion.y,
					node.quaternion.z,
					node.quaternion.w
				),
				scale: {
					x: node.scale.x,
					y: node.scale.y,
					z: node.scale.z,
				},
			},
		};
	}

	public setAxisConstraint(axis: AxisKey, plane: boolean): void {
		if (!this._activeGizmo) return;
		if (plane) {
			this._activeGizmo.planeLock = axis;
			this._activeGizmo.axisLock = null;
		} else {
			this._activeGizmo.axisLock = axis;
			this._activeGizmo.planeLock = null;
		}
	}

	public updateTransform(
		node: Node,
		pointer: InteractionPointerState
	): boolean {
		if (!this._activeGizmo || !this._scene) {
			return false;
		}

		const gizmo = this._activeGizmo;
		const dx = pointer.screenX - gizmo.pointerStart.x;
		const dy = pointer.screenY - gizmo.pointerStart.y;

		if (gizmo.mode === "translate") {
			this._applyTranslateGizmo(node, gizmo, pointer);
		} else if (gizmo.mode === "rotate") {
			const axis = this._resolveGizmoAxis(node, gizmo);
			const angle = (dx + dy) * 0.01;
			const delta = Quaternion.fromAxisAngle(axis, angle).normalize();
			const result = Quaternion.multiply(
				delta,
				gizmo.snapshot.quaternion
			).normalize();
			node.quaternion.copy(result);
		} else {
			const amount = Math.max(0.01, 1 + (dx + dy) * 0.005);
			node.scale.set(
				gizmo.snapshot.scale.x,
				gizmo.snapshot.scale.y,
				gizmo.snapshot.scale.z
			);
			if (gizmo.axisLock) {
				applyAxisScale(node, gizmo.axisLock, amount);
			} else if (gizmo.planeLock) {
				applyPlaneScale(node, gizmo.planeLock, amount);
			} else {
				node.scale.set(
					gizmo.snapshot.scale.x * amount,
					gizmo.snapshot.scale.y * amount,
					gizmo.snapshot.scale.z * amount
				);
			}
		}

		node.updateLocalMatrix();
		this._scene.invalidate("transform");
		return true;
	}

	public commit(node: Node | null): InteractionTransformEvent | null {
		if (!this._activeGizmo || !node || !this._scene) {
			this._activeGizmo = null;
			return null;
		}
		const mode = this._activeGizmo.mode;
		this._activeGizmo = null;
		return { node, mode };
	}

	public cancel(node: Node | null): InteractionTransformEvent | null {
		if (!this._activeGizmo || !node || !this._scene) {
			this._activeGizmo = null;
			return null;
		}

		node.position.set(
			this._activeGizmo.snapshot.position.x,
			this._activeGizmo.snapshot.position.y,
			this._activeGizmo.snapshot.position.z
		);
		node.quaternion.copy(this._activeGizmo.snapshot.quaternion);
		node.scale.set(
			this._activeGizmo.snapshot.scale.x,
			this._activeGizmo.snapshot.scale.y,
			this._activeGizmo.snapshot.scale.z
		);
		node.updateLocalMatrix();
		this._scene.invalidate("transform");

		const mode = this._activeGizmo.mode;
		this._activeGizmo = null;
		return { node, mode };
	}

	private _applyTranslateGizmo(
		node: Node,
		gizmo: ActiveGizmoState,
		pointer: InteractionPointerState
	): void {
		if (!this._camera) return;
		const ray = screenToWorldRay(this._camera, {
			screenX: pointer.screenX,
			screenY: pointer.screenY,
			viewportWidth: pointer.viewportWidth,
			viewportHeight: pointer.viewportHeight,
		});
		const planeNormal = this._resolveTranslatePlaneNormal(node, gizmo);
		const startPoint = intersectRayPlane(
			gizmo.rayStartOrigin,
			gizmo.rayStartDirection,
			gizmo.pivotWorld,
			planeNormal
		);
		const currentPoint = intersectRayPlane(
			ray.origin,
			ray.direction,
			gizmo.pivotWorld,
			planeNormal
		);
		if (!startPoint || !currentPoint) return;

		let delta = Vector3.sub(currentPoint, startPoint);
		if (gizmo.axisLock) {
			const axis = this._resolveGizmoAxis(node, gizmo);
			const projected = Vector3.dot(delta, axis);
			delta = Vector3.scale(axis, projected);
		}
		node.position.set(
			gizmo.snapshot.position.x + delta.x,
			gizmo.snapshot.position.y + delta.y,
			gizmo.snapshot.position.z + delta.z
		);
	}

	private _resolveTranslatePlaneNormal(
		node: Node,
		gizmo: ActiveGizmoState
	): IVector3 {
		if (!this._camera) {
			return { x: 0, y: 0, z: 1 };
		}
		if (gizmo.planeLock) {
			return this._resolveAxisVector(node, gizmo.planeLock, gizmo.space);
		}
		if (gizmo.axisLock) {
			const axis = this._resolveAxisVector(node, gizmo.axisLock, gizmo.space);
			const cameraForward = this._camera.getWorldDirection(
				{ x: 0, y: 0, z: -1 },
				{ x: 0, y: 0, z: -1 }
			);
			const side = Vector3.cross(axis, cameraForward).normalize();
			return Vector3.cross(axis, side).normalize();
		}
		return this._camera.getWorldDirection(
			{ x: 0, y: 0, z: -1 },
			{ x: 0, y: 0, z: -1 }
		);
	}

	private _resolveGizmoAxis(node: Node, gizmo: ActiveGizmoState): IVector3 {
		if (!this._camera) {
			return { x: 1, y: 0, z: 0 };
		}
		if (gizmo.axisLock) {
			return this._resolveAxisVector(node, gizmo.axisLock, gizmo.space);
		}
		if (gizmo.mode === "rotate") {
			return this._camera.getWorldDirection(
				{ x: 0, y: 0, z: -1 },
				{ x: 0, y: 0, z: -1 }
			);
		}
		return { x: 1, y: 0, z: 0 };
	}

	private _resolveAxisVector(
		node: Node,
		axis: AxisKey,
		space: GizmoSpace
	): IVector3 {
		const localAxis =
			axis === "x" ?
				{ x: 1, y: 0, z: 0 }
			: axis === "y" ?
				{ x: 0, y: 1, z: 0 }
			: { x: 0, y: 0, z: 1 };
		if (space === "world") {
			return localAxis;
		}
		return node.getWorldDirection(localAxis, { x: 0, y: 0, z: 0 });
	}
}

function getNodeBoundsCenter(node: Node): IVector3 {
	const bounds = node.getWorldBoundingBox();
	return {
		x: (bounds.min.x + bounds.max.x) * 0.5,
		y: (bounds.min.y + bounds.max.y) * 0.5,
		z: (bounds.min.z + bounds.max.z) * 0.5,
	};
}

function applyAxisScale(node: Node, axis: AxisKey, value: number): void {
	if (axis === "x") {
		node.scale.x *= value;
		return;
	}
	if (axis === "y") {
		node.scale.y *= value;
		return;
	}
	node.scale.z *= value;
}

function applyPlaneScale(node: Node, axis: AxisKey, value: number): void {
	if (axis !== "x") {
		node.scale.x *= value;
	}
	if (axis !== "y") {
		node.scale.y *= value;
	}
	if (axis !== "z") {
		node.scale.z *= value;
	}
}

function intersectRayPlane(
	rayOrigin: IVector3,
	rayDirection: IVector3,
	planePoint: IVector3,
	planeNormal: IVector3
): IVector3 | null {
	const denom = Vector3.dot(rayDirection, planeNormal);
	if (Math.abs(denom) < 1e-8) {
		return null;
	}
	const t =
		Vector3.dot(Vector3.sub(planePoint, rayOrigin), planeNormal) / denom;
	if (!Number.isFinite(t)) {
		return null;
	}
	return {
		x: rayOrigin.x + rayDirection.x * t,
		y: rayOrigin.y + rayDirection.y * t,
		z: rayOrigin.z + rayDirection.z * t,
	};
}
