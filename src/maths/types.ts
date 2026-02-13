import type { RGB } from "../utils/Color";

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
 * Spherical Harmonics coefficients for 2nd order (L=2)
 * Contains exactly 9 RGB coefficients.
 */
export type SHCoefficients = RGB[];
