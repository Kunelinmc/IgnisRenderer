import type { RGB } from "../foundation/Color";

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

export type Matrix4Arr = number[][];
export type Matrix3Arr = number[][];

/**
 * Spherical Harmonics coefficients for 3nd order (L=3)
 * Contains exactly 16 RGB coefficients.
 */
export type SHCoefficients = RGB[];
