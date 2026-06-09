import type { Node } from "../core/Node";
import type { Scene } from "../core/Scene";
import type {
	InteractionCallback,
	InteractionCallbackContext,
	InteractionPointerState,
	InteractableComponent,
} from "../ecs";
import type {
	InteractionClickEvent,
	InteractionEntityEvent,
	InteractionEvents,
	InteractionSelectionMode,
} from "./types";

type InteractionEmitter = <K extends keyof InteractionEvents>(
	event: K,
	...args: InteractionEvents[K]
) => boolean;

export class InteractionSelectionState {
	private _scene: Scene | null = null;
	private _hoveredEntityId: number | null = null;
	private _selectedEntityIds: number[] = [];
	private _selectionMode: InteractionSelectionMode;
	private _emit: InteractionEmitter;

	public constructor(
		selectionMode: InteractionSelectionMode,
		emit: InteractionEmitter
	) {
		this._selectionMode = selectionMode;
		this._emit = emit;
	}

	public setScene(scene: Scene | null): void {
		this._scene = scene;
		this._hoveredEntityId = null;
		this._selectedEntityIds = [];
	}

	public setSelectionMode(selectionMode: InteractionSelectionMode): void {
		if (this._selectionMode === selectionMode) return;
		this._selectionMode = selectionMode;
		if (selectionMode === "single" && this._selectedEntityIds.length > 1) {
			this._applySelection(this._selectedEntityIds.slice(0, 1), null);
		}
	}

	public getSelection(): number | null {
		return this._selectedEntityIds[0] ?? null;
	}

	public getSelectedEntities(): number[] {
		return this._selectedEntityIds.slice();
	}

	public getHoveredEntity(): number | null {
		return this._hoveredEntityId;
	}

	public setHover(
		entityId: number | null,
		pointer: InteractionPointerState | null
	): boolean {
		if (this._hoveredEntityId === entityId) {
			return false;
		}
		const previous = this._hoveredEntityId;
		this._hoveredEntityId = entityId;
		if (previous !== null) {
			this._invokeCallback(previous, "hover-leave", pointer);
		}
		if (entityId !== null) {
			this._invokeCallback(entityId, "hover-enter", pointer);
		}
		this._emit("hoverChanged", this._createEntityEvent(entityId));
		return true;
	}

	public replaceSelection(
		entityIds: number[],
		pointer: InteractionPointerState | null
	): boolean {
		const next = this._normalizeSelection(entityIds);
		return this._applySelection(next, pointer);
	}

	public addSelection(
		entityIds: number[],
		pointer: InteractionPointerState | null
	): boolean {
		if (this._selectionMode === "single") {
			return this.replaceSelection(entityIds.slice(0, 1), pointer);
		}
		const next = this._normalizeSelection([
			...this._selectedEntityIds,
			...entityIds,
		]);
		return this._applySelection(next, pointer);
	}

	public toggleSelection(
		entityId: number,
		pointer: InteractionPointerState | null
	): boolean {
		if (this._selectionMode === "single") {
			return this.replaceSelection(
				this._selectedEntityIds[0] === entityId ? [] : [entityId],
				pointer
			);
		}
		const selected = new Set(this._selectedEntityIds);
		if (selected.has(entityId)) {
			selected.delete(entityId);
		} else {
			selected.add(entityId);
		}
		return this._applySelection(Array.from(selected), pointer);
	}

	public emitClick(
		entityId: number,
		pointer: InteractionPointerState | null
	): void {
		this._invokeCallback(entityId, "click", pointer);
		const event: InteractionClickEvent = {
			...this._createEntityEvent(entityId),
			pointer,
		};
		this._emit("click", event);
	}

	private _applySelection(
		next: number[],
		pointer: InteractionPointerState | null
	): boolean {
		const previous = this._selectedEntityIds;
		if (arraysEqual(previous, next)) {
			return false;
		}
		this._selectedEntityIds = next;
		const nextSet = new Set(next);
		const previousSet = new Set(previous);
		for (const entityId of previous) {
			if (!nextSet.has(entityId)) {
				this._invokeCallback(entityId, "deselect", pointer);
			}
		}
		for (const entityId of next) {
			if (!previousSet.has(entityId)) {
				this._invokeCallback(entityId, "select", pointer);
			}
		}
		this._emitSelectionChanged();
		return true;
	}

	private _emitSelectionChanged(): void {
		this._emit(
			"selectionChanged",
			this._createEntityEvent(this._selectedEntityIds)
		);
	}

	private _normalizeSelection(entityIds: number[]): number[] {
		const result: number[] = [];
		const seen = new Set<number>();
		for (const entityId of entityIds) {
			if (!this._isSelectable(entityId) || seen.has(entityId)) continue;
			seen.add(entityId);
			result.push(entityId);
			if (this._selectionMode === "single") {
				break;
			}
		}
		return result;
	}

	private _isSelectable(entityId: number): boolean {
		const component = this._getInteractable(entityId);
		return !!component && component.selectable !== false;
	}

	private _getInteractable(entityId: number): InteractableComponent | null {
		if (!this._scene || !this._scene.ecs.hasEntity(entityId)) return null;
		const component = this._scene.ecs.getComponent(entityId, "Interactable");
		if (!component || component.enabled === false) return null;
		return component;
	}

	private _invokeCallback(
		entityId: number,
		phase: InteractionCallbackContext["phase"],
		pointer: InteractionPointerState | null
	): void {
		const component = this._getInteractable(entityId);
		if (!component) return;
		const callback = resolveCallback(component, phase);
		if (!callback) return;
		const node = this._scene?.ecs.getNodeByEntity(entityId);
		if (!node) return;
		callback({
			entityId,
			node,
			phase,
			selectedEntityIds: this.getSelectedEntities(),
			pointer,
		});
	}

	private _createEntityEvent(
		entityIdOrIds: number | number[] | null
	): InteractionEntityEvent {
		const entityIds =
			Array.isArray(entityIdOrIds) ? entityIdOrIds
			: entityIdOrIds === null ? []
			: [entityIdOrIds];
		const nodes: Node[] = [];
		for (const entityId of entityIds) {
			const node = this._scene?.ecs.getNodeByEntity(entityId);
			if (node) {
				nodes.push(node);
			}
		}
		return {
			entityId: entityIds[0] ?? null,
			entityIds: entityIds.slice(),
			node: nodes[0] ?? null,
			nodes,
		};
	}
}

function resolveCallback(
	component: InteractableComponent,
	phase: InteractionCallbackContext["phase"]
): InteractionCallback | undefined {
	switch (phase) {
		case "hover-enter":
			return component.onHoverEnter;
		case "hover-leave":
			return component.onHoverLeave;
		case "select":
			return component.onSelect;
		case "deselect":
			return component.onDeselect;
		case "click":
			return component.onClick;
	}
}

function arraysEqual(left: number[], right: number[]): boolean {
	if (left.length !== right.length) return false;
	for (let index = 0; index < left.length; index++) {
		if (left[index] !== right[index]) return false;
	}
	return true;
}
