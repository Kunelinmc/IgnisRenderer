import type { RGB } from "../foundation/Color";
import type { Tuple } from "../foundation/types";

export interface IVector2 {
	x: number;
	y: number;
}

export interface IVector3 {
	x: number;
	y: number;
	z: number;
}

export interface IVector4 {
	x: number;
	y: number;
	z: number;
	w: number;
}

export interface Point extends IVector3 {
	w?: number;
	depth?: number;
}

export type Matrix4Arr = Tuple<Tuple<number, 4>, 4>;
export type Matrix3Arr = Tuple<Tuple<number, 3>, 3>;

/**
 * Spherical Harmonics coefficients for 3nd order (L=3)
 * Contains exactly 16 RGB coefficients.
 */
export type SHCoefficients = Tuple<RGB, 16>;
