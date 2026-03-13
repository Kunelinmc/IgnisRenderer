import type { IVector3 } from "../maths/types";

export const DEFAULT_GRAVITY: IVector3 = Object.freeze({
	x: 0,
	y: -9.8,
	z: 0,
});

export const TRANSFORM_EPSILON = 1e-6;
export const DEFAULT_BROADPHASE_BODY_RADIUS = 0.5;
export const BROADPHASE_CELL_SIZE = 4;
export const BROADPHASE_MAX_DIRTY_CELLS = 512;
