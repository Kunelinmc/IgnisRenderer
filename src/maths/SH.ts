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
	 * @returns {number[]} 16 SH basis values (L=0..3)
	 */
	public static evalBasis(n: IVector3): number[] {
		const { x, y, z } = n;
		// Basis constants adjusted for Y-up coordinate system
		// L=0
		const Y00 = 0.282095;

		// L=1
		const Y1_1 = 0.488603 * x; // m = -1
		const Y10 = 0.488603 * y; // m = 0 (UP)
		const Y11 = 0.488603 * z; // m = 1

		// L=2
		const Y2_2 = 1.092548 * x * z; // m = -2
		const Y2_1 = 1.092548 * x * y; // m = -1
		const Y20 = 0.315392 * (3 * y * y - 1); // m = 0
		const Y21 = 1.092548 * y * z; // m = 1
		const Y22 = 0.546274 * (x * x - z * z); // m = 2

		// L=3
		const Y3_3 = 0.590835 * x * (x * x - 3 * z * z); // m = -3
		const Y3_2 = 2.893641 * x * y * z; // m = -2
		const Y3_1 = 0.457619 * x * (5 * y * y - 1); // m = -1
		const Y30 = 0.373176 * y * (5 * y * y - 3); // m = 0
		const Y31 = 0.457619 * z * (5 * y * y - 1); // m = 1
		const Y32 = 1.446821 * y * (x * x - z * z); // m = 2
		const Y33 = 0.590835 * z * (3 * x * x - z * z); // m = 3

		return [
			Y00,
			Y1_1,
			Y10,
			Y11,
			Y2_2,
			Y2_1,
			Y20,
			Y21,
			Y22,
			Y3_3,
			Y3_2,
			Y3_1,
			Y30,
			Y31,
			Y32,
			Y33,
		];
	}

	/**
	 * Project a directional light source into SH coefficients
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
	 * @param {SHCoefficients} coeffs - 16 SH coefficients
	 * @returns {RGB} Irradiance color {r, g, b}
	 */
	public static calculateIrradiance(n: IVector3, coeffs: SHCoefficients): RGB {
		const basis = this.evalBasis(n);

		// Convolution constants for diffuse irradiance
		const c1 = Math.PI; // Band 0
		const c2 = (2 * Math.PI) / 3; // Band 1
		const c3 = Math.PI / 4; // Band 2
		const c4 = 0; // Band 3 (Odd bands > 1 have zero Lambertian response)

		const factors = [
			c1,
			c2,
			c2,
			c2,
			c3,
			c3,
			c3,
			c3,
			c3,
			c4,
			c4,
			c4,
			c4,
			c4,
			c4,
			c4,
		];

		let r = 0,
			g = 0,
			b = 0;
		const numCoeffs = Math.min(factors.length, coeffs.length);
		for (let i = 0; i < numCoeffs; i++) {
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
		const len = Math.max(a.length, b.length);
		const result: RGB[] = [];
		for (let i = 0; i < len; i++) {
			const rgbA = a[i] || { r: 0, g: 0, b: 0 };
			const rgbB = b[i] || { r: 0, g: 0, b: 0 };
			result.push({
				r: rgbA.r + rgbB.r,
				g: rgbA.g + rgbB.g,
				b: rgbA.b + rgbB.b,
			});
		}
		return result as SHCoefficients;
	}

	/**
	 * Create empty (zero) SH coefficients
	 * @param orderSH The order of SH (e.g. 3 for L=3, 16 coefficients)
	 */
	public static empty(orderSH = 3): SHCoefficients {
		const count = (orderSH + 1) * (orderSH + 1);
		return Array.from({ length: count }, () => ({
			r: 0,
			g: 0,
			b: 0,
		})) as SHCoefficients;
	}

	/**
	 * Serialize SH coefficients to a flat array
	 */
	public static serialize(coeffs: SHCoefficients): number[] {
		const flat: number[] = [];
		for (let i = 0; i < coeffs.length; i++) {
			flat.push(coeffs[i].r, coeffs[i].g, coeffs[i].b);
		}
		return flat;
	}

	/**
	 * Deserialize a flat array back to SH coefficients format
	 */
	public static deserialize(flat: number[]): SHCoefficients {
		const coeffs: RGB[] = [];
		const count = Math.floor(flat.length / 3);
		for (let i = 0; i < count; i++) {
			coeffs.push({
				r: flat[i * 3],
				g: flat[i * 3 + 1],
				b: flat[i * 3 + 2],
			});
		}
		return coeffs as SHCoefficients;
	}
}
