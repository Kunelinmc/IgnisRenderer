import type { Node } from "../core/Node";
import type {
	InteractableRegistry,
	InteractionPointerState,
} from "./Interactable";

export type GizmoMode = "translate" | "rotate" | "scale";
export type GizmoSpace = "world" | "local";
export type GizmoPivot = "object-origin" | "bounds-center";
export type InteractionSelectionMode = "single" | "multiple";

export interface InteractionControllerOptions {
	interactables?: InteractableRegistry;
	maxRayDistance?: number;
	selectionMode?: InteractionSelectionMode;
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

export interface InteractionNodeEvent {
	node: Node | null;
	nodes: Node[];
}

export interface InteractionClickEvent extends InteractionNodeEvent {
	pointer: InteractionPointerState | null;
}

export interface InteractionTransformEvent {
	node: Node;
	mode: GizmoMode;
}

export interface InteractionEvents {
	hoverChanged: [InteractionNodeEvent];
	selectionChanged: [InteractionNodeEvent];
	click: [InteractionClickEvent];
	transformCommitted: [InteractionTransformEvent];
	transformCancelled: [InteractionTransformEvent];
	[key: string]: any[];
}

export interface InteractionHitResult {
	node: Node;
	distance: number;
	priority: number;
	source: "physics" | "bvh";
}

export interface InteractionViewport {
	width: number;
	height: number;
}

export interface InteractionDragRectState {
	startX: number;
	startY: number;
	endX: number;
	endY: number;
	active: boolean;
}

export interface InteractionGizmoState {
	mode: GizmoMode;
	space: GizmoSpace;
	pivot: GizmoPivot;
}

export interface InteractionState {
	selectedNodes: Node[];
	hoveredNode: Node | null;
	gizmo: InteractionGizmoState | null;
	dragRect: InteractionDragRectState | null;
}

export const DEFAULT_MAX_RAY_DISTANCE = 10000;
