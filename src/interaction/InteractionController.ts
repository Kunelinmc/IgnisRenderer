import type { Camera } from "../cameras/Camera";
import { EventEmitter } from "../core/EventEmitter";
import type { Scene } from "../core/Scene";
import type { InteractionPointerState } from "../ecs";
import type {
	InteractionOutlineStyle,
	InteractionTransientState,
} from "../pipeline/types";
import { INTERACTION_TRANSIENT_STATE_KEY } from "../pipeline/types";
import type { PhysicsSystem } from "../physics/PhysicsSystem";
import type {
	FrameTransientContributor,
	FrameTransientContributorContext,
	Renderer,
} from "../renderers/Renderer";
import { InteractionPicker } from "./InteractionPicker";
import { InteractionSelectionState } from "./InteractionSelectionState";
import { TransformGizmoController } from "./TransformGizmoController";
import {
	DEFAULT_INTERACTION_OUTLINE_STYLE,
	DEFAULT_MAX_RAY_DISTANCE,
	type GizmoMode,
	type GizmoPivot,
	type GizmoSpace,
	type InteractionClickEvent,
	type InteractionControllerOptions,
	type InteractionDragRectState,
	type InteractionEntityEvent,
	type InteractionEvents,
	type InteractionPointerEventLike,
	type InteractionSelectionMode,
	type InteractionTransformEvent,
} from "./types";

export class InteractionController extends EventEmitter<InteractionEvents> {
	private _renderer: Renderer | null = null;
	private _scene: Scene | null = null;
	private _camera: Camera | null = null;
	private _physics: PhysicsSystem | null = null;
	private _outlineStyle: InteractionOutlineStyle;
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
	private _transientContributor: FrameTransientContributor;

	/**
	 * Creates a controller for ECS-backed scene interaction.
	 *
	 * @param options - Interaction limits, outline style, and selection mode.
	 * @returns A detached controller. Call `attach()` before sending input.
	 * @sideEffects None.
	 */
	public constructor(options: InteractionControllerOptions = {}) {
		super();
		const maxRayDistance = Number.isFinite(options.maxRayDistance)
			? Math.max(1, Number(options.maxRayDistance))
			: DEFAULT_MAX_RAY_DISTANCE;
		this._outlineStyle = {
			...DEFAULT_INTERACTION_OUTLINE_STYLE,
			...(options.outline ?? {}),
		};
		this._selectionMode = options.selectionMode ?? "single";
		this._picker = new InteractionPicker(maxRayDistance);
		this._selection = new InteractionSelectionState(
			this._selectionMode,
			(event, ...args) => this.emit(event, ...args)
		);
		this._gizmo = new TransformGizmoController();
		this._transientContributor = (context) => {
			this._writeTransientState(context);
		};
	}

	/**
	 * Attaches the controller to renderer frame state and scene picking data.
	 *
	 * @param renderer - Renderer that receives transient interaction state.
	 * @param scene - Scene whose ECS entities are queried for `Interactable`.
	 * @param camera - Camera used to convert pointer coordinates into rays.
	 * @param physicsSystem - Optional physics system used before BVH picking.
	 * @returns This controller for chaining.
	 * @sideEffects Registers a frame transient contributor on `renderer`.
	 */
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
		this._picker.attach(scene, camera, physicsSystem);
		this._selection.setScene(scene);
		this._gizmo.attach(scene, camera);
		renderer.registerFrameTransientContributor(this._transientContributor);
		return this;
	}

	/**
	 * Detaches from the current renderer and clears transient interaction state.
	 *
	 * @returns Nothing.
	 * @sideEffects Unregisters the frame transient contributor when attached.
	 */
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
	 * Updates whether selection is single-entity or multi-entity.
	 *
	 * @param selectionMode - `single` keeps one entity; `multiple` allows many.
	 * @returns Nothing.
	 * @sideEffects May deselect entities and requests an interaction render.
	 */
	public setSelectionMode(selectionMode: InteractionSelectionMode): void {
		if (this._selectionMode === selectionMode) return;
		this._selectionMode = selectionMode;
		this._selection.setSelectionMode(selectionMode);
		this._requestRender("interaction");
	}

	/**
	 * Updates the global interaction outline style.
	 *
	 * @param style - Partial style merged into the current outline style.
	 * @returns Nothing.
	 * @sideEffects Requests an interaction render when attached.
	 */
	public setOutlineStyle(style: Partial<InteractionOutlineStyle>): void {
		this._outlineStyle = {
			...this._outlineStyle,
			...style,
		};
		this._requestRender("interaction");
	}

	/**
	 * Sets the transform gizmo coordinate space used by future gizmo sessions.
	 *
	 * @param space - World-space or local-space gizmo axes.
	 * @returns Nothing.
	 * @sideEffects Requests an interaction render when attached.
	 */
	public setGizmoSpace(space: GizmoSpace): void {
		this._gizmo.setSpace(space);
		this._requestRender("interaction");
	}

	/**
	 * Sets the transform gizmo pivot used by future gizmo sessions.
	 *
	 * @param pivot - Object-origin or bounds-center pivot mode.
	 * @returns Nothing.
	 * @sideEffects Requests an interaction render when attached.
	 */
	public setGizmoPivot(pivot: GizmoPivot): void {
		this._gizmo.setPivot(pivot);
		this._requestRender("interaction");
	}

	/**
	 * Feeds a normalized pointer or keyboard event into the controller.
	 *
	 * @param event - Pointer/key event data in viewport pixel coordinates.
	 * @returns Nothing.
	 * @sideEffects May update ECS-backed interaction state, callbacks, and render dirtiness.
	 */
	public updatePointer(event: InteractionPointerEventLike): void {
		if (!this._scene || !this._camera) return;

		this._updatePointerSnapshot(event);
		if (event.type === "key") {
			this._handleKeyEvent(event);
			return;
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
				if (this._selection.setHover(null, this._snapshotPointer())) {
					this._requestRender("interaction");
				}
				this._dragRect = null;
				if (event.type === "cancel" && this._gizmo.isActive()) {
					const cancelled = this._gizmo.cancel(this._selection.getSelection());
					if (cancelled) {
						this.emit("transformCancelled", cancelled);
					}
					this._requestRender("transform");
				}
				break;
		}
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
				this._requestRender("transform");
				return;
			}
			const cancelled = this._gizmo.cancel(entityId);
			if (cancelled) {
				this.emit("transformCancelled", cancelled);
			}
			this._requestRender("transform");
			return;
		}
		if (this._dragRect?.active) {
			this._dragRect.endX = pointer.screenX;
			this._dragRect.endY = pointer.screenY;
			this._requestRender("interaction");
			return;
		}
		const hit = this._picker.pick(
			pointer.screenX,
			pointer.screenY,
			{ width: pointer.viewportWidth, height: pointer.viewportHeight },
			"hover"
		);
		if (this._selection.setHover(hit?.entityId ?? null, pointer)) {
			this._requestRender("interaction");
		}
	}

	private _handlePointerDown(button: number): void {
		const pointer = this._snapshotPointer();
		if (button === 2) {
			if (this._gizmo.isActive()) {
				const cancelled = this._gizmo.cancel(this._selection.getSelection());
				if (cancelled) {
					this.emit("transformCancelled", cancelled);
				}
				this._requestRender("transform");
			}
			return;
		}

		if (this._gizmo.isActive() && button === 0) {
			const committed = this._gizmo.commit(this._selection.getSelection());
			if (committed) {
				this.emit("transformCommitted", committed);
			}
			this._requestRender("transform");
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
			this._requestRender("interaction");
			return;
		}

		this._dragRect = {
			startX: pointer.screenX,
			startY: pointer.screenY,
			endX: pointer.screenX,
			endY: pointer.screenY,
			active: true,
		};
		this._requestRender("interaction");
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
		this._requestRender("interaction");
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
				this._requestRender("transform");
			}
			return;
		}

		if (key === "enter") {
			if (this._gizmo.isActive()) {
				const committed = this._gizmo.commit(this._selection.getSelection());
				if (committed) {
					this.emit("transformCommitted", committed);
				}
				this._requestRender("transform");
			}
			return;
		}

		if (key === "q") {
			this._gizmo.setSpace(
				this._gizmo.getSpace() === "world" ? "local" : "world"
			);
			this._requestRender("interaction");
			return;
		}

		if (key === ".") {
			this._gizmo.setPivot(
				this._gizmo.getPivot() === "object-origin" ?
					"bounds-center"
				: "object-origin"
			);
			this._requestRender("interaction");
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
			this._requestRender("interaction");
		}
	}

	private _writeTransientState(context: FrameTransientContributorContext): void {
		const interactionState: InteractionTransientState = {
			selectedEntityIds: this._selection.getSelectedEntities(),
			hoveredEntityId: this._selection.getHoveredEntity(),
			outline: {
				color: { ...this._outlineStyle.color },
				thickness: this._outlineStyle.thickness,
				opacity: this._outlineStyle.opacity,
				xray: this._outlineStyle.xray,
				shape: this._outlineStyle.shape,
			},
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
		context.transient.set(INTERACTION_TRANSIENT_STATE_KEY, interactionState);
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

	private _requestRender(reason: "interaction" | "transform"): void {
		this._renderer?.requestRender(reason);
	}
}

export type {
	GizmoMode,
	GizmoPivot,
	GizmoSpace,
	InteractionControllerOptions,
	InteractionClickEvent,
	InteractionEntityEvent,
	InteractionEvents,
	InteractionPointerEventLike,
	InteractionSelectionMode,
	InteractionTransformEvent,
};
