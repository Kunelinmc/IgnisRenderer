import { Texture } from "../core/Texture";
import { Vector3 } from "../maths/Vector3";
import type { IVector3 } from "../maths/types";

/**
 * Split-Sum Approximation for IBL Specular.
 */
export class IBLBRDF {
	private static _lut: Texture | null = null;

	public static getLUT(): Texture {
		if (!this._lut) {
			this._lut = this._generateBRDFLUT(128, 128);
		}
		return this._lut;
	}

	private static _generateBRDFLUT(width: number, height: number): Texture {
		const data = new Float32Array(width * height * 4);
		const SAMPLE_COUNT = 512;

		for (let j = 0; j < height; j++) {
			const roughness = (j + 0.5) / height;
			for (let i = 0; i < width; i++) {
				const NdotV = (i + 0.5) / width;
				const res = this._integrateBRDF(NdotV, roughness, SAMPLE_COUNT);
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
			x: Math.sqrt(1.0 - NdotV * NdotV),
			y: 0.0,
			z: NdotV,
		};
		const N = { x: 0.0, y: 0.0, z: 1.0 };
		let A = 0;
		let B = 0;

		for (let i = 0; i < sampleCount; i++) {
			const xi = this._hammersley(i, sampleCount);
			const H = this._importanceSampleGGX(xi, N, roughness);
			const L = this._reflect(V, H);

			const NdotL = Math.max(L.z, 0.0);
			const NdotH = Math.max(H.z, 0.0);
			const VdotH = Math.max(Vector3.dot(V, H), 0.0);

			if (NdotL > 0) {
				const G = this._geometrySmith(NdotV, NdotL, roughness);
				const G_Vis = (G * VdotH) / (NdotH * NdotV);
				const Fc = Math.pow(1.0 - VdotH, 5.0);

				A += (1.0 - Fc) * G_Vis;
				B += Fc * G_Vis;
			}
		}

		return { x: A / sampleCount, y: B / sampleCount };
	}

	private static _hammersley(i: number, n: number): { x: number; y: number } {
		let bits = i;
		bits = (bits << 16) | (bits >>> 16);
		bits = ((bits & 0x55555555) << 1) | ((bits & 0xaaaaaaaa) >>> 1);
		bits = ((bits & 0x33333333) << 2) | ((bits & 0xcccccccc) >>> 2);
		bits = ((bits & 0x0f0f0f0f) << 4) | ((bits & 0xf0f0f0f0) >>> 4);
		bits = ((bits & 0x00ff00ff) << 8) | ((bits & 0xff00ff00) >>> 8);

		// Use manual bits extraction because bits * fractional power might overflow
		const fraction = (bits >>> 0) * 2.3283064365386963e-10;
		return { x: i / n, y: fraction };
	}

	private static _importanceSampleGGX(
		xi: { x: number; y: number },
		N: IVector3,
		roughness: number
	): IVector3 {
		const a = roughness * roughness;
		const a2 = a * a;
		const phi = 2.0 * Math.PI * xi.x;
		const cosTheta = Math.sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
		const sinTheta = Math.sqrt(1.0 - cosTheta * cosTheta);

		const H = {
			x: Math.cos(phi) * sinTheta,
			y: Math.sin(phi) * sinTheta,
			z: cosTheta,
		};

		// Local to world space (N is {0,0,1} in our integrate case since it's just a LUT)
		if (N.x === 0 && N.y === 0 && N.z === 1) return H;

		const up =
			Math.abs(N.z) < 0.999 ? { x: 0, y: 0, z: 1 } : { x: 1, y: 0, z: 0 };
		const tangent = Vector3.normalize(Vector3.cross(up, N));
		const bitangent = Vector3.cross(N, tangent);

		const worldH = {
			x: tangent.x * H.x + bitangent.x * H.y + N.x * H.z,
			y: tangent.y * H.x + bitangent.y * H.y + N.y * H.z,
			z: tangent.z * H.x + bitangent.z * H.y + N.z * H.z,
		};

		return Vector3.normalize(worldH);
	}

	private static _reflect(V: IVector3, N: IVector3): IVector3 {
		const NdotV = Vector3.dot(N, V);
		return {
			x: 2.0 * NdotV * N.x - V.x,
			y: 2.0 * NdotV * N.y - V.y,
			z: 2.0 * NdotV * N.z - V.z,
		};
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
