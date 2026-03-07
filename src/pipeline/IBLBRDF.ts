import { Texture } from "../core/Texture";
import { Vector3 } from "../maths/Vector3";
import { hammersley, importanceSampleGGX_VNDF } from "../maths/Sampling";
import { lerp } from "../maths/Common";

/**
 * Split-Sum Approximation for IBL Specular.
 */
export class IBLBRDF {
	private static _lut: Texture | null = null;

	public static getLUT(): Texture {
		if (!this._lut) {
			this._lut = this._generateBRDFLUT(64, 64);
		}
		return this._lut;
	}

	private static _generateBRDFLUT(width: number, height: number): Texture {
		const data = new Float32Array(width * height * 4);

		for (let j = 0; j < height; j++) {
			// Square distribution for better resolution in smooth areas
			const roughness = Math.pow((j + 0.5) / height, 2);
			const sampleCount = Math.floor(lerp(128, 32, roughness));

			for (let i = 0; i < width; i++) {
				const NdotV = (i + 0.5) / width;
				const res = this._integrateBRDF(NdotV, roughness, sampleCount);
				const idx = (j * width + i) * 4;
				data[idx] = res.x;
				data[idx + 1] = res.y;
				data[idx + 2] = 0;
				data[idx + 3] = 1;
			}
		}

		return new Texture(data, width, height, "HDR");
	}

	private static _integrateBRDF(
		NdotV: number,
		roughness: number,
		sampleCount: number
	): { x: number; y: number } {
		const V = {
			x: Math.sqrt(Math.max(0.0, 1.0 - NdotV * NdotV)),
			y: 0.0,
			z: NdotV,
		};
		const N = { x: 0.0, y: 0.0, z: 1.0 };
		let A = 0;
		let B = 0;

		for (let i = 0; i < sampleCount; i++) {
			const xi = hammersley(i, sampleCount);
			// Use VNDF sampling for better convergence
			const H = importanceSampleGGX_VNDF(xi, V, N, roughness);

			const dotVH = Vector3.dot(V, H);
			const L = {
				x: 2.0 * dotVH * H.x - V.x,
				y: 2.0 * dotVH * H.y - V.y,
				z: 2.0 * dotVH * H.z - V.z,
			};

			const NdotL = Math.max(L.z, 0.0);
			const VdotH = Math.max(dotVH, 0.0);

			if (NdotL > 0) {
				const G = this._geometrySmith(NdotV, NdotL, roughness);
				const G1V = this._geometrySchlickG1(NdotV, roughness);

				// For VNDF, the weight is G / G1V
				const weight = G / G1V;
				const Fc = Math.pow(1.0 - VdotH, 5.0);

				A += (1.0 - Fc) * weight;
				B += Fc * weight;
			}
		}

		return { x: A / sampleCount, y: B / sampleCount };
	}

	private static _geometrySchlickG1(NdotV: number, roughness: number): number {
		const k = (roughness * roughness) / 2.0;
		return NdotV / (NdotV * (1.0 - k) + k);
	}

	private static _geometrySmith(
		NdotV: number,
		NdotL: number,
		roughness: number
	): number {
		const k = (roughness * roughness) / 2.0; // IBL remapping
		const G1V = NdotV / (NdotV * (1.0 - k) + k);
		const G1L = NdotL / (NdotL * (1.0 - k) + k);
		return G1V * G1L;
	}
}
