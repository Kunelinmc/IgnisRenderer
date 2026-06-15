import type { Camera } from "../cameras/Camera";
import { EventEmitter } from "../core/EventEmitter";
import type { Scene } from "../core/Scene";
import type { PhysicsSystem } from "../physics/PhysicsSystem";
import {
	InteractableRegistry,
	type InteractionPointerState,
} from "./Interactable";
import { InteractionPicker } from "./InteractionPicker";
import { InteractionSelectionState } from "./InteractionSelectionState";
import { TransformGizmoController } from "./TransformGizmoController";
import {
	DEFAULT_MAX_RAY_DISTANCE,
	type GizmoMode,
	type GizmoPivot,
	type GizmoSpace,
	type InteractionClickEvent,
	type InteractionControllerOptions,
	type InteractionDragRectState,
	type InteractionEntityEvent,
	type InteractionEvents,
	type InteractionGizmoState,
	type InteractionPointerEventLike,
	type InteractionSelectionMode,
	type InteractionState,
	type InteractionTransformEvent,
} from "./types";

export class InteractionController extends EventEmitter<InteractionEvents> {
	/**
	 * Registry used by this controller to map scene nodes to interaction
	 * behavior for picking, callbacks, and selection filtering.
	 */
	public readonly interactables: InteractableRegistry;
	private _scene: Scene | null = null;
	private _camera: Camera | null = null;
	private _physics: PhysicsSystem | null = null;
	private _selectionMode: InteractionSelectionMode;
	private _lastPointer: InteractionPointerState = {
		screenX: 0,
		screenY: 0,
		viewportWidth: 1,
		viewportHeight: 1,
	};
	private _dragRect: InteractionDragRectState | null = null;
	private _picker: InteractionPicker;
	private _selection: InteractionSelectionState;
	private _gizmo: TransformGizmoController;

	/**
	 * Creates a controller for registry-backed scene interaction.
	 *
	 * @param options - Interaction registry, limits, outline style, and
	 * selection mode.
	 * @returns A detached controller. Call `attach()` before sending input.
	 * @sideEffects Creates a default interactable registry when one is not
	 * supplied.
	 */
	public constructor(options: InteractionControllerOptions = {}) {
		super();
		this.interactables = options.interactables ?? new InteractableRegistry();
		const maxRayDistance = Number.isFinite(options.maxRayDistance)
			? Math.max(1, Number(options.maxRayDistance))
			: DEFAULT_MAX_RAY_DISTANCE;
		this._selectionMode = options.selectionMode ?? "single";
		this._picker = new InteractionPicker(maxRayDistance, this.interactables);
		this._selection = new InteractionSelectionState(
			this._selectionMode,
			(event, ...args) => this.emit(event, ...args),
			this.interactables
		);
		this._gizmo = new TransformGizmoController();
	}

	/**
	 * Attaches the controller to scene picking data.
	 *
	 * @param scene - Scene whose nodes are resolved against `interactables`.
	 * @param camera - Camera used to convert pointer coordinates into rays.
	 * @param physicsSystem - Optional physics system used before BVH picking.
	 * @returns This controller for chaining.
	 * @sideEffects Resets interaction state when the attachment changes.
	 */
	public attach(
		scene: Scene,
		camera: Camera,
		physicsSystem: PhysicsSystem | null = null
	): this {
		if (
			this._scene === scene &&
			this._camera === camera &&
			this._physics === physicsSystem
		) {
			return this;
		}
		this.detach();
		this._scene = scene;
		this._camera = camera;
		this._physics = physicsSystem;
		this._picker.attach(scene, camera, physicsSystem);
		this._selection.setScene(scene);
		this._gizmo.attach(scene, camera);
		return this;
	}

	/**
	 * Detaches from the current scene and clears interaction state.
	 *
	 * @returns Nothing.
	 * @sideEffects Clears picking, selection, drag, and gizmo state.
	 */
	public detach(): void {
		this._scene = null;
		this._camera = null;
		this._physics = null;
		this._dragRect = null;
		this._picker.detach();
		this._selection.setScene(null);
		this._gizmo.detach();
	}

	/**
	 * Releases renderer attachment held by the controller.
	 *
	 * @returns Nothing.
	 * @sideEffects Equivalent to `detach()`.
	 */
	public dispose(): void {
		this.detach();
	}

	/**
	 * Reads the primary selected entity.
	 *
	 * @returns The first selected entity id, or `null` when nothing is selected.
	 * @sideEffects None.
	 */
	public getSelection(): number | null {
		return this._selection.getSelection();
	}

	/**
	 * Reads all selected entities.
	 *
	 * @returns A copy of the selected entity id list.
	 * @sideEffects None.
	 */
	public getSelectedEntities(): number[] {
		return this._selection.getSelectedEntities();
	}

	/**
	 * Reads the current interaction state without submitting rendering work.
	 *
	 * @returns A detached snapshot of selection, hover, gizmo, and drag state.
	 * @sideEffects None.
	 */
	public getState(): InteractionState {
		return {
			selectedEntityIds: this._selection.getSelectedEntities(),
			hoveredEntityId: this._selection.getHoveredEntity(),
			gizmo: this._gizmo.getState(),
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
	}

	/**
	 * Updates whether selection is single-entity or multi-entity.
	 *
	 * @param selectionMode - `single` keeps one entity; `multiple` allows many.
	 * @returns Nothing.
	 * @sideEffects May deselect entities and emit interaction callbacks.
	 */
	public setSelectionMode(selectionMode: InteractionSelectionMode): void {
		if (this._selectionMode === selectionMode) return;
		this._selectionMode = selectionMode;
		this._selection.setSelectionMode(selectionMode);
	}

	/**
	 * Sets the transform gizmo coordinate space used by future gizmo sessions.
	 *
	 * @param space - World-space or local-space gizmo axes.
	 * @returns Nothing.
	 * @sideEffects Updates future gizmo sessions.
	 */
	public setGizmoSpace(space: GizmoSpace): void {
		this._gizmo.setSpace(space);
	}

	/**
	 * Sets the transform gizmo pivot used by future gizmo sessions.
	 *
	 * @param pivot - Object-origin or bounds-center pivot mode.
	 * @returns Nothing.
	 * @sideEffects Updates future gizmo sessions.
	 */
	public setGizmoPivot(pivot: GizmoPivot): void {
		this._gizmo.setPivot(pivot);
	}

	/**
	 * Feeds a normalized pointer or keyboard event into the controller.
	 *
	 * @param event - Pointer/key event data in viewport pixel coordinates.
	 * @returns The interaction state after processing the event.
	 * @sideEffects May update registry-backed interaction state, invoke callbacks,
	 * emit events, and modify a node during transform-gizmo interaction. It does
	 * not request or submit rendering work.
	 */
	public updatePointer(event: InteractionPointerEventLike): InteractionState {
		if (!this._scene || !this._camera) return this.getState();

		this._updatePointerSnapshot(event);
		if (event.type === "key") {
			this._handleKeyEvent(event);
			return this.getState();
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
				this._selection.setHover(null, this._snapshotPointer());
				this._dragRect = null;
				if (event.type === "cancel" && this._gizmo.isActive()) {
					const cancelled = this._gizmo.cancel(this._selection.getSelection());
					if (cancelled) {
						this.emit("transformCancelled", cancelled);
					}
				}
				break;
		}
		return this.getState();
	}

	private _handlePointerMove(): void {
		const pointer = this._snapshotPointer();
		if (this._gizmo.isActive()) {
			const entityId = this._selection.getSelection();
			if (
				entityId !== null &&
				this._picker.isEntitySelectable(entityId) &&
				this._gizmo.updateTransform(entityId, pointer)
			) {
				return;
			}
			const cancelled = this._gizmo.cancel(entityId);
			if (cancelled) {
				this.emit("transformCancelled", cancelled);
			}
			return;
		}
		if (this._dragRect?.active) {
			this._dragRect.endX = pointer.screenX;
			this._dragRect.endY = pointer.screenY;
			return;
		}
		const hit = this._picker.pick(
			pointer.screenX,
			pointer.screenY,
			{ width: pointer.viewportWidth, height: pointer.viewportHeight },
			"hover"
		);
		this._selection.setHover(hit?.entityId ?? null, pointer);
	}

	private _handlePointerDown(button: number): void {
		const pointer = this._snapshotPointer();
		if (button === 2) {
			if (this._gizmo.isActive()) {
				const cancelled = this._gizmo.cancel(this._selection.getSelection());
				if (cancelled) {
					this.emit("transformCancelled", cancelled);
				}
			}
			return;
		}

		if (this._gizmo.isActive() && button === 0) {
			const committed = this._gizmo.commit(this._selection.getSelection());
			if (committed) {
				this.emit("transformCommitted", committed);
			}
			return;
		}

		const hit = this._picker.pick(
			pointer.screenX,
			pointer.screenY,
			{ width: pointer.viewportWidth, height: pointer.viewportHeight },
			"select"
		);
		if (hit) {
			if (this._selectionMode === "multiple" && pointer.shiftKey) {
				this._selection.toggleSelection(hit.entityId, pointer);
			} else {
				this._selection.replaceSelection([hit.entityId], pointer);
			}
			this._selection.emitClick(hit.entityId, pointer);
			return;
		}

		this._dragRect = {
			startX: pointer.screenX,
			startY: pointer.screenY,
			endX: pointer.screenX,
			endY: pointer.screenY,
			active: true,
		};
	}

	private _handlePointerUp(button: number): void {
		if (button !== 0) return;
		if (!this._dragRect?.active) return;
		const pointer = this._snapshotPointer();
		const selected = this._picker.pickDragRect(this._dragRect, {
			width: pointer.viewportWidth,
			height: pointer.viewportHeight,
		});
		this._dragRect = null;
		if (this._selectionMode === "multiple" && pointer.shiftKey) {
			this._selection.addSelection(selected, pointer);
		} else {
			this._selection.replaceSelection(selected, pointer);
		}
	}

	private _handleKeyEvent(event: InteractionPointerEventLike): void {
		const key = (event.key ?? "").toLowerCase();
		if (!key) return;

		if (key === "escape") {
			if (this._gizmo.isActive()) {
				const cancelled = this._gizmo.cancel(this._selection.getSelection());
				if (cancelled) {
					this.emit("transformCancelled", cancelled);
				}
			}
			return;
		}

		if (key === "enter") {
			if (this._gizmo.isActive()) {
				const committed = this._gizmo.commit(this._selection.getSelection());
				if (committed) {
					this.emit("transformCommitted", committed);
				}
			}
			return;
		}

		if (key === "q") {
			this._gizmo.setSpace(
				this._gizmo.getSpace() === "world" ? "local" : "world"
			);
			return;
		}

		if (key === ".") {
			this._gizmo.setPivot(
				this._gizmo.getPivot() === "object-origin" ?
					"bounds-center"
				: "object-origin"
			);
			return;
		}

		if (key === "x" || key === "y" || key === "z") {
			this._gizmo.setAxisConstraint(key, !!event.shiftKey);
			return;
		}

		if (key === "g" || key === "r" || key === "s") {
			const entityId = this._selection.getSelection();
			if (entityId === null || !this._scene) return;
			if (!this._picker.isEntitySelectable(entityId)) return;
			const node = this._scene.ecs.getNodeByEntity(entityId);
			if (!node) return;
			const mode: GizmoMode =
				key === "g" ? "translate" : key === "r" ? "rotate" : "scale";
			this._gizmo.begin(mode, node, this._snapshotPointer());
		}
	}

	private _updatePointerSnapshot(event: InteractionPointerEventLike): void {
		if (typeof event.screenX === "number") {
			this._lastPointer.screenX = event.screenX;
		}
		if (typeof event.screenY === "number") {
			this._lastPointer.screenY = event.screenY;
		}
		if (typeof event.viewportWidth === "number") {
			this._lastPointer.viewportWidth = Math.max(1, event.viewportWidth);
		}
		if (typeof event.viewportHeight === "number") {
			this._lastPointer.viewportHeight = Math.max(1, event.viewportHeight);
		}
		if (typeof event.button === "number") {
			this._lastPointer.button = event.button;
		}
		this._lastPointer.shiftKey = event.shiftKey === true;
		this._lastPointer.ctrlKey = event.ctrlKey === true;
		this._lastPointer.metaKey = event.metaKey === true;
		this._lastPointer.altKey = event.altKey === true;
	}

	private _snapshotPointer(): InteractionPointerState {
		return {
			screenX: this._lastPointer.screenX,
			screenY: this._lastPointer.screenY,
			viewportWidth: this._lastPointer.viewportWidth,
			viewportHeight: this._lastPointer.viewportHeight,
			button: this._lastPointer.button,
			shiftKey: this._lastPointer.shiftKey,
			ctrlKey: this._lastPointer.ctrlKey,
			metaKey: this._lastPointer.metaKey,
			altKey: this._lastPointer.altKey,
		};
	}
}

export type {
	GizmoMode,
	GizmoPivot,
	GizmoSpace,
	InteractionControllerOptions,
	InteractionClickEvent,
	InteractionDragRectState,
	InteractionEntityEvent,
	InteractionEvents,
	InteractionGizmoState,
	InteractionPointerEventLike,
	InteractionSelectionMode,
	InteractionState,
	InteractionTransformEvent,
};
