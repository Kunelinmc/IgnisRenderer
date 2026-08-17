import { Texture } from "../../core/Texture";
import { lerp } from "../../maths/Common";
import { hammersley, importanceSampleGGX_VNDF } from "../../maths/Sampling";
import { Vector3 } from "../../maths/Vector3";

/** Split-sum approximation lookup data for IBL specular lighting. */
export class IBLBRDF {
	private static _lut: Texture | null = null;

	public static getLUT(): Texture {
		if (!this._lut) this._lut = this._generateBRDFLUT(64, 64);
		return this._lut;
	}

	private static _generateBRDFLUT(width: number, height: number): Texture {
		const data = new Float32Array(width * height * 4);
		for (let y = 0; y < height; y++) {
			const roughness = Math.pow((y + 0.5) / height, 2);
			const sampleCount = Math.floor(lerp(128, 32, roughness));
			for (let x = 0; x < width; x++) {
				const nDotV = (x + 0.5) / width;
				const result = this._integrateBRDF(
					nDotV,
					roughness,
					sampleCount,
				);
				const index = (y * width + x) * 4;
				data[index] = result.x;
				data[index + 1] = result.y;
				data[index + 2] = 0;
				data[index + 3] = 1;
			}
		}
		return new Texture({ data, width, height, colorSpace: "HDR" });
	}

	private static _integrateBRDF(
		nDotV: number,
		roughness: number,
		sampleCount: number,
	): { x: number; y: number } {
		const view = {
			x: Math.sqrt(Math.max(0, 1 - nDotV * nDotV)),
			y: 0,
			z: nDotV,
		};
		const normal = { x: 0, y: 0, z: 1 };
		let scale = 0;
		let bias = 0;
		for (let index = 0; index < sampleCount; index++) {
			const sample = hammersley(index, sampleCount);
			const half = importanceSampleGGX_VNDF(
				sample,
				view,
				normal,
				roughness,
			);
			const viewDotHalf = Vector3.dot(view, half);
			const lightZ = 2 * viewDotHalf * half.z - view.z;
			const nDotL = Math.max(lightZ, 0);
			if (nDotL <= 0) continue;
			const visibility = this._geometrySmith(nDotV, nDotL, roughness);
			const viewVisibility = this._geometrySchlickG1(nDotV, roughness);
			const weight = visibility / viewVisibility;
			const fresnel = Math.pow(1 - Math.max(viewDotHalf, 0), 5);
			scale += (1 - fresnel) * weight;
			bias += fresnel * weight;
		}
		return { x: scale / sampleCount, y: bias / sampleCount };
	}

	private static _geometrySchlickG1(
		nDotV: number,
		roughness: number,
	): number {
		const k = roughness * roughness / 2;
		return nDotV / (nDotV * (1 - k) + k);
	}

	private static _geometrySmith(
		nDotV: number,
		nDotL: number,
		roughness: number,
	): number {
		const k = roughness * roughness / 2;
		const view = nDotV / (nDotV * (1 - k) + k);
		const light = nDotL / (nDotL * (1 - k) + k);
		return view * light;
	}
}
