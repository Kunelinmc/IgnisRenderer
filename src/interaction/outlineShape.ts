import type { InteractionOutlineShape } from "../pipeline/types";

export const INTERACTION_OUTLINE_SHAPE_CODE_CIRCLE = 0;
export const INTERACTION_OUTLINE_SHAPE_CODE_SQUARE = 1;
export const INTERACTION_OUTLINE_SHAPE_CODE_DIAMOND = 2;
export const INTERACTION_OUTLINE_SHAPE_CODE_OCTAGON = 3;

export function resolveInteractionOutlineShape(
	shape: unknown
): InteractionOutlineShape {
	switch (shape) {
		case "square":
		case "diamond":
		case "octagon":
		case "circle":
			return shape;
		default:
			return "circle";
	}
}

export function getInteractionOutlineShapeCode(shape: unknown): number {
	switch (resolveInteractionOutlineShape(shape)) {
		case "square":
			return INTERACTION_OUTLINE_SHAPE_CODE_SQUARE;
		case "diamond":
			return INTERACTION_OUTLINE_SHAPE_CODE_DIAMOND;
		case "octagon":
			return INTERACTION_OUTLINE_SHAPE_CODE_OCTAGON;
		default:
			return INTERACTION_OUTLINE_SHAPE_CODE_CIRCLE;
	}
}

export function computeInteractionOutlineShapeDistance(
	dx: number,
	dy: number,
	shape: InteractionOutlineShape
): number {
	const absX = Math.abs(dx);
	const absY = Math.abs(dy);
	switch (shape) {
		case "square":
			return Math.max(absX, absY) * Math.SQRT2;
		case "diamond":
			return absX + absY;
		case "octagon":
			return Math.max(Math.max(absX, absY), (absX + absY) * Math.SQRT1_2);
		default:
			return Math.hypot(dx, dy);
	}
}
