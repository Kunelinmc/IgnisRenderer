/**
 * Spherical Harmonics (SH) utility functions
 * Using 3rd order SH (L=3, 16 coefficients)
 */

import type { IVector3, SHCoefficients } from "./types";
import type { RGB } from "../foundation/Color";

type SHBasisBuffer = number[] | Float32Array;

const SH_COEFFICIENT_COUNT = 16;
const SH_SERIALIZED_COMPONENT_COUNT = SH_COEFFICIENT_COUNT * 3;

export class SH {
	/**
	 * Compute SH basis functions for a given direction vector.
	 *
	 * @param n Direction vector, which must be normalized by the caller.
	 * @param out Optional 16-element destination buffer. When provided, this
	 * method writes all basis values into `out` and returns the same buffer.
	 * @returns The 16 SH basis values for L=0..3.
	 * @sideEffects Writes indices 0..15 of `out` when a destination is provided.
	 */
	public static evalBasis(n: IVector3): number[];
	public static evalBasis(n: IVector3, out: Float32Array): Float32Array;
	public static evalBasis(n: IVector3, out: number[]): number[];
	public static evalBasis(n: IVector3, out?: SHBasisBuffer): SHBasisBuffer {
		const { x, y, z } = n;
		const basis = out ?? new Array<number>(16);
		// Basis constants adjusted for Y-up coordinate system
		// L=0
		basis[0] = 0.282095;

		// L=1
		basis[1] = 0.488603 * x; // m = -1
		basis[2] = 0.488603 * y; // m = 0 (UP)
		basis[3] = 0.488603 * z; // m = 1

		// L=2
		const yy = y * y;
		basis[4] = 1.092548 * x * z; // m = -2
		basis[5] = 1.092548 * x * y; // m = -1
		basis[6] = 0.315392 * (3 * yy - 1); // m = 0
		basis[7] = 1.092548 * y * z; // m = 1
		basis[8] = 0.546274 * (x * x - z * z); // m = 2

		// L=3
		basis[9] = 0.590835 * x * (x * x - 3 * z * z); // m = -3
		basis[10] = 2.893641 * x * y * z; // m = -2
		basis[11] = 0.457619 * x * (5 * yy - 1); // m = -1
		basis[12] = 0.373176 * y * (5 * yy - 3); // m = 0
		basis[13] = 0.457619 * z * (5 * yy - 1); // m = 1
		basis[14] = 1.446821 * y * (x * x - z * z); // m = 2
		basis[15] = 0.590835 * z * (3 * x * x - z * z); // m = 3
		return basis;
	}

	/**
	 * Project a directional light source into SH coefficients
	 */
	public static projectDirectionalLight(
		dir: IVector3,
		color: RGB
	): SHCoefficients {
		const basis = this.evalBasis(dir);
		const coefficients = this.empty();
		for (let i = 0; i < SH_COEFFICIENT_COUNT; i++) {
			const value = basis[i];
			coefficients[i].r = color.r * value;
			coefficients[i].g = color.g * value;
			coefficients[i].b = color.b * value;
		}
		return coefficients;
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
		const result = this.empty();
		for (let i = 0; i < SH_COEFFICIENT_COUNT; i++) {
			result[i].r = a[i].r + b[i].r;
			result[i].g = a[i].g + b[i].g;
			result[i].b = a[i].b + b[i].b;
		}
		return result;
	}

	/** Creates the engine's empty L=3 spherical harmonics coefficients. */
	public static empty(): SHCoefficients {
		return Array.from({ length: SH_COEFFICIENT_COUNT }, () => ({
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
		if (flat.length !== SH_SERIALIZED_COMPONENT_COUNT) {
			throw new RangeError(
				`Serialized SH data must contain exactly ${SH_SERIALIZED_COMPONENT_COUNT} values.`
			);
		}

		const coefficients = this.empty();
		for (let i = 0; i < SH_COEFFICIENT_COUNT; i++) {
			coefficients[i] = {
				r: flat[i * 3],
				g: flat[i * 3 + 1],
				b: flat[i * 3 + 2],
			};
		}
		return coefficients;
	}
}
