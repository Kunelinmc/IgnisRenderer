import type { Node } from "../core/Node";
import type { InteractionOutlineStyle } from "../pipeline/types";
import type { InteractionPointerState } from "../ecs";

export type GizmoMode = "translate" | "rotate" | "scale";
export type GizmoSpace = "world" | "local";
export type GizmoPivot = "object-origin" | "bounds-center";
export type InteractionSelectionMode = "single" | "multiple";

export interface InteractionControllerOptions {
	maxRayDistance?: number;
	outline?: Partial<InteractionOutlineStyle>;
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

export interface InteractionEntityEvent {
	entityId: number | null;
	entityIds: number[];
	node: Node | null;
	nodes: Node[];
}

export interface InteractionClickEvent extends InteractionEntityEvent {
	pointer: InteractionPointerState | null;
}

export interface InteractionTransformEvent {
	entityId: number;
	node: Node;
	mode: GizmoMode;
}

export interface InteractionEvents {
	hoverChanged: [InteractionEntityEvent];
	selectionChanged: [InteractionEntityEvent];
	click: [InteractionClickEvent];
	transformCommitted: [InteractionTransformEvent];
	transformCancelled: [InteractionTransformEvent];
	[key: string]: any[];
}

export interface InteractionHitResult {
	node: Node;
	entityId: number;
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

export const DEFAULT_INTERACTION_OUTLINE_STYLE: InteractionOutlineStyle = {
	color: { r: 255, g: 196, b: 64, a: 1 },
	thickness: 2,
	opacity: 0.9,
	xray: true,
	shape: "circle",
};

export const DEFAULT_MAX_RAY_DISTANCE = 10000;
