import { Camera } from "../cameras/Camera";
import { EventEmitter } from "../core/EventEmitter";
import type { Node } from "../core/Node";
import type { Scene } from "../core/Scene";
import { MeshInstance } from "../meshes";
import type {
	InteractionGizmoState,
	InteractionOutlineStyle,
	InteractionTransientState,
} from "../pipeline/types";
import { INTERACTION_TRANSIENT_STATE_KEY } from "../pipeline/types";
import type { PhysicsQueryHit } from "../physics/types";
import type { PhysicsSystem } from "../physics/PhysicsSystem";
import { Quaternion } from "../maths/Quaternion";
import type { IVector3 } from "../maths/types";
import { Vector3 } from "../maths/Vector3";
import type {
	FrameTransientContributor,
	FrameTransientContributorContext,
	Renderer,
} from "../renderers/Renderer";
import { screenToWorldRay } from "../interaction/screenToWorldRay";

const DEFAULT_OUTLINE_STYLE: InteractionOutlineStyle = {
	color: { r: 255, g: 196, b: 64, a: 1 },
	thickness: 2,
	opacity: 0.9,
	xray: true,
	shape: "circle",
};

const DEFAULT_MAX_RAY_DISTANCE = 10000;

export type GizmoMode = "translate" | "rotate" | "scale";
export type GizmoSpace = "world" | "local";
export type GizmoPivot = "object-origin" | "bounds-center";

type AxisKey = "x" | "y" | "z";

export interface InteractionManagerOptions {
	maxRayDistance?: number;
	outline?: Partial<InteractionOutlineStyle>;
}

export interface InteractionPointerEventLike {
	type: "move" | "down" | "up" | "leave" | "cancel" | "key";
	screenX?: number;
	screenY?: number;
	button?: number;
	key?: string;
	shiftKey?: boolean;
	ctrlKey?: boolean;
	metaKey?: boolean;
	altKey?: boolean;
	viewportWidth?: number;
	viewportHeight?: number;
}

interface HitResult {
	node: Node;
	entityId: number;
	distance: number;
	source: "physics" | "bvh";
}

interface DragRectState {
	startX: number;
	startY: number;
	endX: number;
	endY: number;
	active: boolean;
}

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

export interface InteractionEvents {
	hoverChanged: [{ entityId: number | null; node: Node | null }];
	selectionChanged: [{ entityId: number | null; node: Node | null }];
	transformCommitted: [{ entityId: number; node: Node; mode: GizmoMode }];
	transformCancelled: [{ entityId: number; node: Node; mode: GizmoMode }];
	[key: string]: any[];
}

export class InteractionManager extends EventEmitter<InteractionEvents> {
	private _renderer: Renderer | null = null;
	private _scene: Scene | null = null;
	private _camera: Camera | null = null;
	private _physics: PhysicsSystem | null = null;
	private _maxRayDistance: number;
	private _outlineStyle: InteractionOutlineStyle;
	private _activeEntityId: number | null = null;
	private _hoveredEntityId: number | null = null;
	private _dragRect: DragRectState | null = null;
	private _activeGizmo: ActiveGizmoState | null = null;
	private _gizmoSpace: GizmoSpace = "world";
	private _gizmoPivot: GizmoPivot = "object-origin";
	private _lastPointer = { x: 0, y: 0, width: 1, height: 1 };
	private _transientContributor: FrameTransientContributor;

	constructor(options: InteractionManagerOptions = {}) {
		super();
		this._maxRayDistance = Number.isFinite(options.maxRayDistance)
			? Math.max(1, Number(options.maxRayDistance))
			: DEFAULT_MAX_RAY_DISTANCE;
		this._outlineStyle = {
			...DEFAULT_OUTLINE_STYLE,
			...(options.outline ?? {}),
		};
		this._transientContributor = (context) => {
			this._writeTransientState(context);
		};
	}

	public attach(
		renderer: Renderer,
		scene: Scene = renderer.scene,
		camera: Camera = renderer.camera,
		physicsSystem: PhysicsSystem | null = null
	): this {
		if (
			this._renderer === renderer &&
			this._scene === scene &&
			this._camera === camera &&
			this._physics === physicsSystem
		) {
			return this;
		}
		this.detach();
		this._renderer = renderer;
		this._scene = scene;
		this._camera = camera;
		this._physics = physicsSystem;
		renderer.registerFrameTransientContributor(this._transientContributor);
		return this;
	}

	public detach(): void {
		if (this._renderer) {
			this._renderer.unregisterFrameTransientContributor(
				this._transientContributor
			);
		}
		this._renderer = null;
		this._scene = null;
		this._camera = null;
		this._physics = null;
		this._dragRect = null;
		this._activeGizmo = null;
	}

	public getSelection(): number | null {
		return this._activeEntityId;
	}

	public setOutlineStyle(style: Partial<InteractionOutlineStyle>): void {
		this._outlineStyle = {
			...this._outlineStyle,
			...style,
		};
		this._renderer?.requestRender("interaction");
	}

	public setGizmoSpace(space: GizmoSpace): void {
		this._gizmoSpace = space;
	}

	public setGizmoPivot(pivot: GizmoPivot): void {
		this._gizmoPivot = pivot;
	}

	public updatePointer(event: InteractionPointerEventLike): void {
		if (!this._scene || !this._camera) return;

		if (event.type === "key") {
			this._handleKeyEvent(event);
			return;
		}

		if (typeof event.screenX === "number") {
			this._lastPointer.x = event.screenX;
		}
		if (typeof event.screenY === "number") {
			this._lastPointer.y = event.screenY;
		}
		if (typeof event.viewportWidth === "number") {
			this._lastPointer.width = Math.max(1, event.viewportWidth);
		}
		if (typeof event.viewportHeight === "number") {
			this._lastPointer.height = Math.max(1, event.viewportHeight);
		}

		switch (event.type) {
			case "move":
				this._handlePointerMove();
				break;
			case "down":
				this._handlePointerDown(event.button ?? 0);
				break;
			case "up":
				this._handlePointerUp(event.button ?? 0);
				break;
			case "leave":
			case "cancel":
				this._setHover(null);
				this._dragRect = null;
				break;
		}
	}

	private _handlePointerMove(): void {
		if (this._activeGizmo) {
			this._updateGizmoTransform();
			this._renderer?.requestRender("transform");
			return;
		}
		if (this._dragRect?.active) {
			this._dragRect.endX = this._lastPointer.x;
			this._dragRect.endY = this._lastPointer.y;
			this._renderer?.requestRender("interaction");
			return;
		}
		const hit = this._performHitTest(this._lastPointer.x, this._lastPointer.y);
		this._setHover(hit?.entityId ?? null);
	}

	private _handlePointerDown(button: number): void {
		if (button === 2) {
			if (this._activeGizmo) {
				this._cancelGizmo();
			}
			return;
		}

		if (this._activeGizmo && button === 0) {
			this._commitGizmo();
			return;
		}

		const hit = this._performHitTest(this._lastPointer.x, this._lastPointer.y);
		if (hit) {
			this._setSelection(hit.entityId);
			return;
		}

		this._setSelection(null);
		this._dragRect = {
			startX: this._lastPointer.x,
			startY: this._lastPointer.y,
			endX: this._lastPointer.x,
			endY: this._lastPointer.y,
			active: true,
		};
		this._renderer?.requestRender("interaction");
	}

	private _handlePointerUp(button: number): void {
		if (button === 2) {
			if (this._activeGizmo) {
				this._cancelGizmo();
			}
			return;
		}
		if (button !== 0) return;

		if (this._dragRect?.active) {
			const selected = this._resolveSelectionFromDragRect(this._dragRect);
			this._dragRect = null;
			this._setSelection(selected);
			return;
		}
	}

	private _handleKeyEvent(event: InteractionPointerEventLike): void {
		const key = (event.key ?? "").toLowerCase();
		if (!key) return;

		if (key === "escape") {
			if (this._activeGizmo) {
				this._cancelGizmo();
			}
			return;
		}

		if (key === "enter") {
			if (this._activeGizmo) {
				this._commitGizmo();
			}
			return;
		}

		if (key === "q") {
			this._gizmoSpace = this._gizmoSpace === "world" ? "local" : "world";
			this._renderer?.requestRender("interaction");
			return;
		}

		if (key === ".") {
			this._gizmoPivot =
				this._gizmoPivot === "object-origin" ? "bounds-center" : "object-origin";
			this._renderer?.requestRender("interaction");
			return;
		}

		if (key === "x" || key === "y" || key === "z") {
			if (this._activeGizmo) {
				if (event.shiftKey) {
					this._activeGizmo.planeLock = key as AxisKey;
					this._activeGizmo.axisLock = null;
				} else {
					this._activeGizmo.axisLock = key as AxisKey;
					this._activeGizmo.planeLock = null;
				}
			}
			return;
		}

		if (key === "g" || key === "r" || key === "s") {
			if (this._activeEntityId === null || !this._scene) return;
			const node = this._scene.ecs.getNodeByEntity(this._activeEntityId);
			if (!node) return;
			const mode: GizmoMode =
				key === "g" ? "translate" : key === "r" ? "rotate" : "scale";
			this._beginGizmo(mode, node);
		}
	}

	private _beginGizmo(mode: GizmoMode, node: Node): void {
		const ray = screenToWorldRay(this._camera!, {
			screenX: this._lastPointer.x,
			screenY: this._lastPointer.y,
			viewportWidth: this._lastPointer.width,
			viewportHeight: this._lastPointer.height,
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
			pointerStart: { x: this._lastPointer.x, y: this._lastPointer.y },
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
		this._renderer?.requestRender("interaction");
	}

	private _commitGizmo(): void {
		if (!this._activeGizmo || this._activeEntityId === null || !this._scene) {
			this._activeGizmo = null;
			return;
		}
		const node = this._scene.ecs.getNodeByEntity(this._activeEntityId);
		if (node) {
		this.emit("transformCommitted", {
			entityId: this._activeEntityId,
			node,
			mode: this._activeGizmo.mode,
		});
		}
		this._activeGizmo = null;
		this._renderer?.requestRender("transform");
	}

	private _cancelGizmo(): void {
		if (!this._activeGizmo || this._activeEntityId === null || !this._scene) {
			this._activeGizmo = null;
			return;
		}
		const node = this._scene.ecs.getNodeByEntity(this._activeEntityId);
		if (!node) {
			this._activeGizmo = null;
			return;
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

		this.emit("transformCancelled", {
			entityId: this._activeEntityId,
			node,
			mode: this._activeGizmo.mode,
		});
		this._activeGizmo = null;
		this._renderer?.requestRender("transform");
	}

	private _updateGizmoTransform(): void {
		if (!this._activeGizmo || !this._scene || this._activeEntityId === null) {
			return;
		}
		const node = this._scene.ecs.getNodeByEntity(this._activeEntityId);
		if (!node) return;

		const gizmo = this._activeGizmo;
		const dx = this._lastPointer.x - gizmo.pointerStart.x;
		const dy = this._lastPointer.y - gizmo.pointerStart.y;

		if (gizmo.mode === "translate") {
			this._applyTranslateGizmo(node, gizmo);
		} else if (gizmo.mode === "rotate") {
			const axis = this._resolveGizmoAxis(node, gizmo);
			const angle = (dx + dy) * 0.01;
			const delta = Quaternion.fromAxisAngle(axis, angle).normalize();
			const result = Quaternion.multiply(delta, gizmo.snapshot.quaternion).normalize();
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
	}

	private _applyTranslateGizmo(node: Node, gizmo: ActiveGizmoState): void {
		const ray = screenToWorldRay(this._camera!, {
			screenX: this._lastPointer.x,
			screenY: this._lastPointer.y,
			viewportWidth: this._lastPointer.width,
			viewportHeight: this._lastPointer.height,
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
		if (gizmo.planeLock) {
			return this._resolveAxisVector(node, gizmo.planeLock, gizmo.space);
		}
		if (gizmo.axisLock) {
			const axis = this._resolveAxisVector(node, gizmo.axisLock, gizmo.space);
			const cameraForward = this._camera!.getWorldDirection(
				{ x: 0, y: 0, z: -1 },
				{ x: 0, y: 0, z: -1 }
			);
			const side = Vector3.cross(axis, cameraForward).normalize();
			return Vector3.cross(axis, side).normalize();
		}
		return this._camera!.getWorldDirection(
			{ x: 0, y: 0, z: -1 },
			{ x: 0, y: 0, z: -1 }
		);
	}

	private _resolveGizmoAxis(node: Node, gizmo: ActiveGizmoState): IVector3 {
		if (gizmo.axisLock) {
			return this._resolveAxisVector(node, gizmo.axisLock, gizmo.space);
		}
		if (gizmo.mode === "rotate") {
			return this._camera!.getWorldDirection(
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

	private _resolveSelectionFromDragRect(rect: DragRectState): number | null {
		if (!this._scene || !this._camera) return null;
		this._scene.updateWorldMatrices();
		this._camera.updateMatrices();
		const minX = Math.min(rect.startX, rect.endX);
		const minY = Math.min(rect.startY, rect.endY);
		const maxX = Math.max(rect.startX, rect.endX);
		const maxY = Math.max(rect.startY, rect.endY);
		if (maxX - minX < 2 || maxY - minY < 2) {
			return null;
		}

		let bestEntityId: number | null = null;
		let bestDepth = Infinity;
		for (const meshInstance of this._scene.getMeshInstances()) {
			if (meshInstance.visible === false) continue;
			const entityId = meshInstance.entityId;
			if (typeof entityId !== "number") continue;
			const sphere = meshInstance.getWorldBoundingSphere();
			const clip = screenProject(
				this._camera.viewProjectionMatrix,
				sphere.center,
				this._lastPointer.width,
				this._lastPointer.height
			);
			if (!clip || clip.w <= 0) continue;
			if (
				clip.screenX < minX ||
				clip.screenX > maxX ||
				clip.screenY < minY ||
				clip.screenY > maxY
			) {
				continue;
			}
			if (
				clip.depth < bestDepth ||
				(clip.depth === bestDepth &&
					(bestEntityId === null || entityId < bestEntityId))
			) {
				bestDepth = clip.depth;
				bestEntityId = entityId;
			}
		}
		return bestEntityId;
	}

	private _performHitTest(screenX: number, screenY: number): HitResult | null {
		if (!this._scene || !this._camera) return null;
		this._scene.updateWorldMatrices();
		this._camera.updateMatrices();
		const ray = screenToWorldRay(this._camera, {
			screenX,
			screenY,
			viewportWidth: this._lastPointer.width,
			viewportHeight: this._lastPointer.height,
		});

		const physicsHit = this._pickPhysicsHit(ray.origin, ray.direction);
		if (physicsHit) {
			return physicsHit;
		}
		return this._pickBVHFallbackHit(ray.origin, ray.direction);
	}

	private _pickPhysicsHit(origin: IVector3, direction: IVector3): HitResult | null {
		if (!this._physics) return null;
		const hits = this._physics.raycastAll({
			origin,
			direction,
			maxDistance: this._maxRayDistance,
			filter: {
				includeTriggers: false,
			},
		});
		if (hits.length === 0) return null;

		let best: HitResult | null = null;
		for (const hit of hits) {
			const node = this._physics.resolveHitNode(hit);
			if (!node) continue;
			const entityId =
				typeof node.entityId === "number" ?
					node.entityId
				: this._physics.resolveHitEntityId(hit);
			if (typeof entityId !== "number") continue;
			if (
				!best ||
				hit.distance < best.distance ||
				(hit.distance === best.distance && entityId < best.entityId)
			) {
				best = {
					node,
					entityId,
					distance: hit.distance,
					source: "physics",
				};
			}
		}
		return best;
	}

	private _pickBVHFallbackHit(
		origin: IVector3,
		direction: IVector3
	): HitResult | null {
		if (!this._scene) return null;
		const meshInstances = this._scene
			.getMeshInstances()
			.filter((meshInstance) => meshInstance.visible !== false);
		if (meshInstances.length === 0) return null;

		const spatial = this._scene.rebuildSpatialIndex(meshInstances);
		const candidates = spatial.queryRayDetailed(origin, direction, {
			includeInvisible: false,
			maxDistance: this._maxRayDistance,
			maxResults: 64,
		});
		if (candidates.length === 0) return null;

		let best: HitResult | null = null;
		for (const candidate of candidates) {
			const meshInstance = candidate.meshInstance;
			const entityId = meshInstance.entityId;
			if (typeof entityId !== "number") continue;
			let distance = candidate.distance;

			if (isAnimationDrivenMesh(meshInstance)) {
				const sphereHit = intersectRaySphere(
					origin,
					direction,
					meshInstance.getWorldBoundingSphere().center,
					meshInstance.getWorldBoundingSphere().radius,
					this._maxRayDistance
				);
				if (sphereHit === null) continue;
				distance = sphereHit;
			} else {
				const triangleHit = intersectRayMeshTriangles(
					origin,
					direction,
					meshInstance,
					this._maxRayDistance
				);
				if (triangleHit === null) continue;
				distance = triangleHit;
			}

			if (
				!best ||
				distance < best.distance ||
				(distance === best.distance && entityId < best.entityId)
			) {
				best = {
					node: meshInstance,
					entityId,
					distance,
					source: "bvh",
				};
			}
		}
		return best;
	}

	private _setHover(entityId: number | null): void {
		if (this._hoveredEntityId === entityId) {
			return;
		}
		this._hoveredEntityId = entityId;
		const node =
			entityId !== null && this._scene ?
				this._scene.ecs.getNodeByEntity(entityId)
			: null;
		this.emit("hoverChanged", {
			entityId,
			node,
		});
		this._renderer?.requestRender("interaction");
	}

	private _setSelection(entityId: number | null): void {
		if (this._activeEntityId === entityId) {
			return;
		}
		this._activeEntityId = entityId;
		const node =
			entityId !== null && this._scene ?
				this._scene.ecs.getNodeByEntity(entityId)
			: null;
		this.emit("selectionChanged", {
			entityId,
			node,
		});
		this._renderer?.requestRender("interaction");
	}

	private _writeTransientState(context: FrameTransientContributorContext): void {
		const interactionState: InteractionTransientState = {
			selectedEntityIds:
				typeof this._activeEntityId === "number" ? [this._activeEntityId] : [],
			hoveredEntityId: this._hoveredEntityId,
			outline: {
				color: { ...this._outlineStyle.color },
				thickness: this._outlineStyle.thickness,
				opacity: this._outlineStyle.opacity,
				xray: this._outlineStyle.xray,
				shape: this._outlineStyle.shape,
			},
			gizmo:
				this._activeGizmo ?
					{
						mode: this._activeGizmo.mode,
						space: this._activeGizmo.space,
						pivot: this._activeGizmo.pivot,
					}
				: null,
			dragRect:
				this._dragRect ?
					{
						startX: this._dragRect.startX,
						startY: this._dragRect.startY,
						endX: this._dragRect.endX,
						endY: this._dragRect.endY,
						active: this._dragRect.active,
					}
				: null,
		};
		context.transient.set(INTERACTION_TRANSIENT_STATE_KEY, interactionState);
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

function screenProject(
	viewProjection: { elements: number[][] },
	position: IVector3,
	width: number,
	height: number
): { screenX: number; screenY: number; depth: number; w: number } | null {
	const me = viewProjection.elements;
	const x = position.x;
	const y = position.y;
	const z = position.z;
	const clipX = me[0][0] * x + me[0][1] * y + me[0][2] * z + me[0][3];
	const clipY = me[1][0] * x + me[1][1] * y + me[1][2] * z + me[1][3];
	const clipZ = me[2][0] * x + me[2][1] * y + me[2][2] * z + me[2][3];
	const clipW = me[3][0] * x + me[3][1] * y + me[3][2] * z + me[3][3];
	if (Math.abs(clipW) < 1e-8) return null;
	const invW = 1 / clipW;
	const ndcX = clipX * invW;
	const ndcY = clipY * invW;
	const ndcZ = clipZ * invW;
	return {
		screenX: (ndcX * 0.5 + 0.5) * width,
		screenY: (0.5 - ndcY * 0.5) * height,
		depth: ndcZ,
		w: clipW,
	};
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

function isAnimationDrivenMesh(meshInstance: MeshInstance): boolean {
	if (meshInstance.skeleton) return true;
	for (const primitive of meshInstance.mesh.primitives) {
		if ((primitive.geometry.morphTargets?.length ?? 0) > 0) {
			return true;
		}
	}
	return false;
}

function intersectRaySphere(
	origin: IVector3,
	direction: IVector3,
	center: IVector3,
	radius: number,
	maxDistance: number
): number | null {
	const radiusClamped = Math.max(0.001, radius);
	const oc = Vector3.sub(origin, center);
	const b = Vector3.dot(oc, direction);
	const c = Vector3.dot(oc, oc) - radiusClamped * radiusClamped;
	if (c > 0 && b > 0) return null;
	const discriminant = b * b - c;
	if (discriminant < 0) return null;
	let distance = -b - Math.sqrt(discriminant);
	if (distance < 0) distance = 0;
	if (distance > maxDistance) return null;
	return distance;
}

function intersectRayMeshTriangles(
	origin: IVector3,
	direction: IVector3,
	meshInstance: MeshInstance,
	maxDistance: number
): number | null {
	let bestDistance = Infinity;
	for (const primitive of meshInstance.mesh.primitives) {
		if (primitive.visible === false) continue;
		const geometry = primitive.geometry;
		const positions = geometry.positions;
		const indices = geometry.indices;
		if (!positions || !indices || positions.length < 9 || indices.length < 3) {
			continue;
		}

		for (let triangleIndex = 0; triangleIndex + 2 < indices.length; triangleIndex += 3) {
			const i0 = indices[triangleIndex] * 3;
			const i1 = indices[triangleIndex + 1] * 3;
			const i2 = indices[triangleIndex + 2] * 3;
			const v0 = transformPosition(meshInstance, positions, i0);
			const v1 = transformPosition(meshInstance, positions, i1);
			const v2 = transformPosition(meshInstance, positions, i2);
			const distance = intersectRayTriangle(origin, direction, v0, v1, v2);
			if (distance === null) continue;
			if (distance > maxDistance) continue;
			if (distance < bestDistance) {
				bestDistance = distance;
			}
		}
	}
	return Number.isFinite(bestDistance) ? bestDistance : null;
}

function transformPosition(
	meshInstance: MeshInstance,
	positions: Float32Array,
	index: number
): IVector3 {
	const world = meshInstance.worldMatrix.elements;
	const x = positions[index];
	const y = positions[index + 1];
	const z = positions[index + 2];
	return {
		x: world[0][0] * x + world[0][1] * y + world[0][2] * z + world[0][3],
		y: world[1][0] * x + world[1][1] * y + world[1][2] * z + world[1][3],
		z: world[2][0] * x + world[2][1] * y + world[2][2] * z + world[2][3],
	};
}

function intersectRayTriangle(
	origin: IVector3,
	direction: IVector3,
	v0: IVector3,
	v1: IVector3,
	v2: IVector3
): number | null {
	const epsilon = 1e-8;
	const edge1 = Vector3.sub(v1, v0);
	const edge2 = Vector3.sub(v2, v0);
	const h = Vector3.cross(direction, edge2);
	const a = Vector3.dot(edge1, h);
	if (Math.abs(a) < epsilon) return null;

	const f = 1 / a;
	const s = Vector3.sub(origin, v0);
	const u = f * Vector3.dot(s, h);
	if (u < 0 || u > 1) return null;

	const q = Vector3.cross(s, edge1);
	const v = f * Vector3.dot(direction, q);
	if (v < 0 || u + v > 1) return null;

	const t = f * Vector3.dot(edge2, q);
	if (t < 0) return null;
	return t;
}
