/**
 * Common math utilities
 */

import type { Point } from "./types";

export function d2r(d: number): number {
	return (d * Math.PI) / 180;
}

export function r2d(r: number): number {
	return (r * 180) / Math.PI;
}

export function clamp(val: number, min = 0, max = 1): number {
	return Math.max(min, Math.min(max, val));
}

/**
 * sRGB EOTF (Electro-Optical Transfer Function) — decode sRGB to linear.
 *
 * Piecewise function per IEC 61966-2-1:
 *   x ≤ 0.04045 → x / 12.92
 *   x >  0.04045 → ((x + 0.055) / 1.055) ^ 2.4
 *
 * @param x  sRGB-encoded value in [0, 1]
 * @returns  Linear-light value in [0, 1]
 */
export function sRGBToLinear(x: number): number {
	return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}

/**
 * sRGB OETF (Opto-Electronic Transfer Function) — encode linear to sRGB.
 *
 * Piecewise function per IEC 61966-2-1:
 *   x ≤ 0.0031308 → 12.92 * x
 *   x >  0.0031308 → 1.055 * x^(1/2.4) − 0.055
 *
 * @param x  Linear-light value in [0, 1]
 * @returns  sRGB-encoded value in [0, 1]
 */
export function linearToSRGB(x: number): number {
	return x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1.0 / 2.4) - 0.055;
}

export function interpolatePoint(a: Point, b: Point, t: number): Point {
	t = clamp(t, 0, 1);
	return {
		x: a.x + (b.x - a.x) * t,
		y: a.y + (b.y - a.y) * t,
		z: a.z + (b.z - a.z) * t,
		depth: (a.depth || 0) + ((b.depth || 0) - (a.depth || 0)) * t,
	};
}

export function rotatePoint(
	point: Point,
	rotX: number,
	rotY: number,
	rotZ: number
): Point {
	const { x, y, z } = point;
	const [radX, radY, radZ] = [rotX, rotY, rotZ].map(d2r);

	const [cX, sX] = [Math.cos(radX), Math.sin(radX)];
	const [cY, sY] = [Math.cos(radY), Math.sin(radY)];
	const [cZ, sZ] = [Math.cos(radZ), Math.sin(radZ)];

	// Z rotation
	const x_z = x * cZ - y * sZ;
	const y_z = x * sZ + y * cZ;

	// Y rotation
	const x_y = x_z * cY + z * sY;
	const y_y = y_z;
	const z_y = -x_z * sY + z * cY;

	// X rotation
	return {
		x: x_y,
		y: y_y * cX - z_y * sX,
		z: y_y * sX + z_y * cX,
	};
}
