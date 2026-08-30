import type { Node } from "../core/Node";
import type {
	Interactable,
	InteractableRegistry,
	InteractionCallback,
	InteractionCallbackContext,
	InteractionPointerState,
} from "./Interactable";
import type {
	InteractionClickEvent,
	InteractionEvents,
	InteractionNodeEvent,
	InteractionSelectionMode,
} from "./types";

type InteractionEmitter = <K extends keyof InteractionEvents>(
	event: K,
	...args: InteractionEvents[K]
) => boolean;

export class InteractionSelectionState {
	private _hoveredNode: Node | null = null;
	private _selectedNodes: Node[] = [];
	private _selectionMode: InteractionSelectionMode;
	private _emit: InteractionEmitter;
	private _interactables: InteractableRegistry;

	public constructor(
		selectionMode: InteractionSelectionMode,
		emit: InteractionEmitter,
		interactables: InteractableRegistry,
	) {
		this._selectionMode = selectionMode;
		this._emit = emit;
		this._interactables = interactables;
	}

	public reset(): void {
		this._hoveredNode = null;
		this._selectedNodes = [];
	}

	public setSelectionMode(selectionMode: InteractionSelectionMode): void {
		if (this._selectionMode === selectionMode) return;
		this._selectionMode = selectionMode;
		if (selectionMode === "single" && this._selectedNodes.length > 1) {
			this._applySelection(this._selectedNodes.slice(0, 1), null);
		}
	}

	public getSelection(): Node | null {
		return this._selectedNodes[0] ?? null;
	}

	public getSelectedNodes(): Node[] {
		return this._selectedNodes.slice();
	}

	public getHoveredNode(): Node | null {
		return this._hoveredNode;
	}

	public setHover(node: Node | null, pointer: InteractionPointerState | null): boolean {
		if (this._hoveredNode === node) return false;
		const previous = this._hoveredNode;
		this._hoveredNode = node;
		if (previous) this._invokeCallback(previous, "hover-leave", pointer);
		if (node) this._invokeCallback(node, "hover-enter", pointer);
		this._emit("hoverChanged", this._createNodeEvent(node));
		return true;
	}

	public replaceSelection(nodes: Node[], pointer: InteractionPointerState | null): boolean {
		return this._applySelection(this._normalizeSelection(nodes), pointer);
	}

	public addSelection(nodes: Node[], pointer: InteractionPointerState | null): boolean {
		if (this._selectionMode === "single") {
			return this.replaceSelection(nodes.slice(0, 1), pointer);
		}
		return this._applySelection(
			this._normalizeSelection([...this._selectedNodes, ...nodes]),
			pointer,
		);
	}

	public toggleSelection(node: Node, pointer: InteractionPointerState | null): boolean {
		if (this._selectionMode === "single") {
			return this.replaceSelection(
				this._selectedNodes[0] === node ? [] : [node],
				pointer,
			);
		}
		const selected = new Set(this._selectedNodes);
		if (selected.has(node)) selected.delete(node);
		else selected.add(node);
		return this._applySelection(Array.from(selected), pointer);
	}

	public emitClick(node: Node, pointer: InteractionPointerState | null): void {
		this._invokeCallback(node, "click", pointer);
		const event: InteractionClickEvent = {
			...this._createNodeEvent(node),
			pointer,
		};
		this._emit("click", event);
	}

	private _applySelection(next: Node[], pointer: InteractionPointerState | null): boolean {
		const previous = this._selectedNodes;
		if (arraysEqual(previous, next)) return false;
		this._selectedNodes = next;
		const nextSet = new Set(next);
		const previousSet = new Set(previous);
		for (const node of previous) {
			if (!nextSet.has(node)) this._invokeCallback(node, "deselect", pointer);
		}
		for (const node of next) {
			if (!previousSet.has(node)) this._invokeCallback(node, "select", pointer);
		}
		this._emit("selectionChanged", this._createNodeEvent(next));
		return true;
	}

	private _normalizeSelection(nodes: Node[]): Node[] {
		const result: Node[] = [];
		const seen = new Set<string>();
		for (const node of nodes) {
			if (!this._isSelectable(node) || seen.has(node.id)) continue;
			seen.add(node.id);
			result.push(node);
			if (this._selectionMode === "single") break;
		}
		return result;
	}

	private _isSelectable(node: Node): boolean {
		const component = this._getInteractable(node);
		return !!component && component.selectable !== false;
	}

	private _getInteractable(node: Node): Interactable | null {
		const interactable = this._interactables.get(node);
		if (!interactable || interactable.enabled === false) return null;
		return interactable;
	}

	private _invokeCallback(
		node: Node,
		phase: InteractionCallbackContext["phase"],
		pointer: InteractionPointerState | null,
	): void {
		const component = this._getInteractable(node);
		const callback = component ? resolveCallback(component, phase) : undefined;
		if (!callback) return;
		callback({ node, phase, selectedNodes: this.getSelectedNodes(), pointer });
	}

	private _createNodeEvent(nodeOrNodes: Node | Node[] | null): InteractionNodeEvent {
		const nodes = Array.isArray(nodeOrNodes)
			? nodeOrNodes.slice()
			: nodeOrNodes ? [nodeOrNodes] : [];
		return { node: nodes[0] ?? null, nodes };
	}
}

function resolveCallback(
	component: Interactable,
	phase: InteractionCallbackContext["phase"],
): InteractionCallback | undefined {
	switch (phase) {
		case "hover-enter": return component.onHoverEnter;
		case "hover-leave": return component.onHoverLeave;
		case "select": return component.onSelect;
		case "deselect": return component.onDeselect;
		case "click": return component.onClick;
	}
}

function arraysEqual(left: Node[], right: Node[]): boolean {
	if (left.length !== right.length) return false;
	for (let index = 0; index < left.length; index++) {
		if (left[index] !== right[index]) return false;
	}
	return true;
}
