import type { Node } from "../core/Node";

export interface InteractionPointerState {
	screenX: number;
	screenY: number;
	viewportWidth: number;
	viewportHeight: number;
	button?: number;
	shiftKey?: boolean;
	ctrlKey?: boolean;
	metaKey?: boolean;
	altKey?: boolean;
}

export type InteractionEventPhase =
	| "hover-enter"
	| "hover-leave"
	| "select"
	| "deselect"
	| "click";

export interface InteractionCallbackContext {
	entityId: number;
	node: Node;
	phase: InteractionEventPhase;
	selectedEntityIds: number[];
	pointer: InteractionPointerState | null;
}

export type InteractionCallback = (
	context: InteractionCallbackContext
) => void;

/**
 * Describes how a scene node participates in interaction picking.
 */
export interface Interactable {
	/** Enables or disables all interaction for the registered node. */
	enabled?: boolean;
	/** Allows hover state and hover callbacks when not set to `false`. */
	hoverable?: boolean;
	/** Allows click, drag, and gizmo selection when not set to `false`. */
	selectable?: boolean;
	/** Higher values win when multiple interactable nodes overlap. */
	priority?: number;
	/** Callback invoked when the pointer starts hovering the node. */
	onHoverEnter?: InteractionCallback;
	/** Callback invoked when the pointer stops hovering the node. */
	onHoverLeave?: InteractionCallback;
	/** Callback invoked when the node becomes selected. */
	onSelect?: InteractionCallback;
	/** Callback invoked when the node is removed from selection. */
	onDeselect?: InteractionCallback;
	/** Callback invoked when the node receives a click selection event. */
	onClick?: InteractionCallback;
}

/** @deprecated Use `Interactable` for new interaction definitions. */
export type InteractableComponent = Interactable;

/**
 * Stores node-owned interaction definitions outside ECS and `Node` instances.
 */
export class InteractableRegistry {
	private _interactables = new WeakMap<Node, Interactable>();

	/**
	 * Associates interaction behavior with a scene node.
	 *
	 * @param node - Node that should respond to interaction picking.
	 * @param interactable - Interaction flags, priority, and callbacks.
	 * @returns This registry for call chaining.
	 * @sideEffects Replaces any existing interactable definition for `node`.
	 */
	public set(node: Node, interactable: Interactable): this {
		this._interactables.set(node, interactable);
		return this;
	}

	/**
	 * Reads the interaction behavior associated with a scene node.
	 *
	 * @param node - Node whose interactable definition should be resolved.
	 * @returns The registered interactable definition, or `null` when absent.
	 * @sideEffects None.
	 */
	public get(node: Node): Interactable | null {
		return this._interactables.get(node) ?? null;
	}

	/**
	 * Checks whether a node has an interaction definition.
	 *
	 * @param node - Node to test.
	 * @returns `true` when an interactable definition is registered.
	 * @sideEffects None.
	 */
	public has(node: Node): boolean {
		return this._interactables.has(node);
	}

	/**
	 * Removes a node interaction definition.
	 *
	 * @param node - Node whose interactable definition should be removed.
	 * @returns `true` when a definition existed and was removed.
	 * @sideEffects The node stops participating in interaction picking.
	 */
	public delete(node: Node): boolean {
		return this._interactables.delete(node);
	}

	/**
	 * Removes all interaction definitions from this registry.
	 *
	 * @returns Nothing.
	 * @sideEffects Existing registered nodes stop participating in interaction picking.
	 */
	public clear(): void {
		this._interactables = new WeakMap<Node, Interactable>();
	}
}
