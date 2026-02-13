/**
 * Spherical Harmonics (SH) utility functions
 * Using 2nd order SH (L=2, 9 coefficients)
 */

import type { IVector3, SHCoefficients } from "./types";
import type { RGB } from "../utils/Color";

export class SH {
	/**
	 * Compute SH basis functions for a given direction vector (normal)
	 * @param {IVector3} n - Direction vector {x, y, z}, must be normalized
	 * @returns {number[]} 9 SH basis values
	 */
	public static evalBasis(n: IVector3): number[] {
		const { x, y, z } = n;
		// Basis constants from Peter-Pike Sloan's paper or similar sources
		// Adjusted for Y-up coordinate system (Y is the polar axis)
		const Y00 = 0.282095;
		const Y1_1 = 0.488603 * x; // m = -1
		const Y10 = 0.488603 * y; // m = 0 (UP)
		const Y11 = 0.488603 * z; // m = 1
		const Y2_2 = 1.092548 * x * z; // m = -2
		const Y2_1 = 1.092548 * x * y; // m = -1
		const Y20 = 0.315392 * (3 * y * y - 1); // m = 0
		const Y21 = 1.092548 * y * z; // m = 1
		const Y22 = 0.546274 * (x * x - z * z); // m = 2

		return [Y00, Y1_1, Y10, Y11, Y2_2, Y2_1, Y20, Y21, Y22];
	}

	/**
	 * Project a directional light source into SH coefficients
	 * @param {IVector3} dir - Direction vector {x, y, z} towards the light
	 * @param {RGB} color - Light color {r, g, b}
	 * @returns {SHCoefficients} 9 SH coefficients, each is {r, g, b}
	 */
	public static projectDirectionalLight(
		dir: IVector3,
		color: RGB
	): SHCoefficients {
		const basis = this.evalBasis(dir);
		return basis.map((b) => ({
			r: color.r * b,
			g: color.g * b,
			b: color.b * b,
		})) as SHCoefficients;
	}

	/**
	 * Reconstruct irradiance from SH coefficients
	 * @param {IVector3} n - Surface normal {x, y, z}
	 * @param {SHCoefficients} coeffs - 9 SH coefficients
	 * @returns {RGB} Irradiance color {r, g, b}
	 */
	public static calculateIrradiance(n: IVector3, coeffs: SHCoefficients): RGB {
		const basis = this.evalBasis(n);

		// Convolution constants for diffuse irradiance
		const c1 = Math.PI; // pi
		const c2 = (2 * Math.PI) / 3; // 2pi/3
		const c3 = Math.PI / 4; // pi/4

		const factors = [c1, c2, c2, c2, c3, c3, c3, c3, c3];

		let r = 0,
			g = 0,
			b = 0;
		for (let i = 0; i < 9; i++) {
			const weight = basis[i] * factors[i];
			r += coeffs[i].r * weight;
			g += coeffs[i].g * weight;
			b += coeffs[i].b * weight;
		}

		return {
			r: Math.max(0, r),
			g: Math.max(0, g),
			b: Math.max(0, b),
		};
	}

	/**
	 * Add two sets of SH coefficients
	 */
	public static addCoeffs(
		a: SHCoefficients,
		b: SHCoefficients
	): SHCoefficients {
		const result: RGB[] = [];
		for (let i = 0; i < 9; i++) {
			result.push({
				r: a[i].r + b[i].r,
				g: a[i].g + b[i].g,
				b: a[i].b + b[i].b,
			});
		}
		return result as SHCoefficients;
	}

	/**
	 * Create empty (zero) SH coefficients
	 */
	public static empty(): SHCoefficients {
		return Array.from({ length: 9 }, () => ({
			r: 0,
			g: 0,
			b: 0,
		})) as SHCoefficients;
	}

	/**
	 * Serialize SH coefficients to a flat array for storage or transmission
	 * and deserialize back to RGB[] format
	 */
	public static serialize(coeffs: SHCoefficients): number[] {
		const flat: number[] = [];
		for (let i = 0; i < 9; i++) {
			flat.push(coeffs[i].r, coeffs[i].g, coeffs[i].b);
		}
		return flat;
	}

	/**
	 * Deserialize a flat array back to SH coefficients format
	 */
	public static deserialize(flat: number[]): SHCoefficients {
		const coeffs: RGB[] = [];
		for (let i = 0; i < 9; i++) {
			coeffs.push({
				r: flat[i * 3],
				g: flat[i * 3 + 1],
				b: flat[i * 3 + 2],
			});
		}
		return coeffs as SHCoefficients;
	}
}
