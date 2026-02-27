import { SH } from "../maths/SH";
import { linearToSRGB, sRGBToLinear } from "../maths/Common";
import {
	Light,
	LightType,
	type LightContribution,
	type SurfacePoint,
} from "./Light";
import type { SHCoefficients } from "../maths/types";
import type { Texture } from "../core/Texture";

/**
 * LightProbe provides irregular or environment lighting via Spherical Harmonics
 */
export class LightProbe extends Light<LightType.LightProbe> {
	public sh: SHCoefficients;
	private static readonly DC_IRRADIANCE_SCALE = Math.PI * 0.282095;

	constructor(sh: SHCoefficients | null = null, intensity = 1.0) {
		super(LightType.LightProbe, { intensity });
		// Deep copy to prevent external mutation of passed array
		this.sh = sh ? JSON.parse(JSON.stringify(sh)) : SH.empty();
	}

	/**
	 * Compute the light's contribution. For a LightProbe, this is usually
	 * used as an ambient-like contribution based on the surface normal.
	 * NOTE: Standard computeContribution takes a point, but SH irradiance
	 * fundamentally depends on the normal.
	 */
	public computeContribution(surface: SurfacePoint): LightContribution | null {
		let irrR = 0,
			irrG = 0,
			irrB = 0;

		if (surface.normal) {
			const irr = SH.calculateIrradiance(surface.normal, this.sh);
			irrR = Math.max(0, irr.r);
			irrG = Math.max(0, irr.g);
			irrB = Math.max(0, irr.b);
		} else {
			const dc = this.sh[0];
			irrR = Math.max(0, dc.r * LightProbe.DC_IRRADIANCE_SCALE);
			irrG = Math.max(0, dc.g * LightProbe.DC_IRRADIANCE_SCALE);
			irrB = Math.max(0, dc.b * LightProbe.DC_IRRADIANCE_SCALE);
		}

		if (irrR <= 0 && irrG <= 0 && irrB <= 0) return null;

		const toSrgb255 = (linear255: number): number => {
			const linear01 = Math.max(0, linear255 / 255);
			return linearToSRGB(Math.min(1, linear01)) * 255;
		};

		return {
			type: "irradiance",
			color: {
				r: toSrgb255(irrR),
				g: toSrgb255(irrG),
				b: toSrgb255(irrB),
			},
			intensity: this.intensity,
		};
	}

	/**
	 * Create a light probe from an environment map
	 * @param envMap Equirectangular environment texture
	 * @returns A new LightProbe instance
	 */
	public static fromEnvironmentMap(envMap: Texture): LightProbe {
		return LightProbe._projectEquirectToSH(envMap);
	}

	/**
	 * Project an environment map to SH coefficients using numerical integration
	 * @param envMap Equirectangular environment texture
	 * @returns A new LightProbe instance with computed coefficients
	 */
	private static _projectEquirectToSH(envMap: Texture): LightProbe {
		if (!envMap || !envMap.data) {
			return new LightProbe();
		}

		const { width, height, data } = envMap;
		const sh = SH.empty();

		// Integration step sizes
		const dTheta = Math.PI / height;
		const dPhi = (2 * Math.PI) / width;

		let totalWeight = 0;

		// Perform numerical integration over the sphere
		// L_lm = ∫ L(s) * Y_lm(s) * dΩ
		// dΩ = sin(θ) * dθ * dφ
		for (let j = 0; j < height; j++) {
			const theta = (j + 0.5) * dTheta;
			const sinTheta = Math.sin(theta);
			const cosTheta = Math.cos(theta);
			const weight = sinTheta * dTheta * dPhi;

			for (let i = 0; i < width; i++) {
				const phi = (i + 0.5) * dPhi;

				// Convert spherical coordinates to cartesian direction
				// Latitude θ maps to Y axis (UP), Longitude φ maps to XZ plane
				const x = sinTheta * Math.sin(phi);
				const y = cosTheta;
				const z = sinTheta * Math.cos(phi);

				const basis = SH.evalBasis({ x, y, z });

				const idx = (j * width + i) * 4;

				// Convert texture values to linear and keep engine-wide 0..255 light units.
				// HDR data is already linear; sRGB (Uint8) needs sRGB EOTF decode.
				const isLinear =
					envMap.colorSpace === "HDR" || envMap.colorSpace === "Linear";
				const r = isLinear
					? data[idx] * 255
					: sRGBToLinear(data[idx] / 255) * 255;
				const g = isLinear
					? data[idx + 1] * 255
					: sRGBToLinear(data[idx + 1] / 255) * 255;
				const b = isLinear
					? data[idx + 2] * 255
					: sRGBToLinear(data[idx + 2] / 255) * 255;

				for (let k = 0; k < 9; k++) {
					const bK = basis[k] * weight;
					sh[k].r += r * bK;
					sh[k].g += g * bK;
					sh[k].b += b * bK;
				}

				totalWeight += weight;
			}
		}

		// Normalize by 1/PI so that the reconstructed irradiance matches the
		// texture color values for constant radiance, consistent with AmbientLight.
		// Also normalize by totalWeight to compensate for discrete sum approximation.
		const normFactor = (4 * Math.PI) / totalWeight;

		for (let k = 0; k < 9; k++) {
			sh[k].r *= normFactor;
			sh[k].g *= normFactor;
			sh[k].b *= normFactor;
		}

		return new LightProbe(sh);
	}

	/**
	 * Clone this light probe
	 * @returns A new LightProbe instance with the same coefficients and intensity
	 */
	public clone(): LightProbe {
		return new LightProbe(this.sh, this.intensity);
	}

	/**
	 * Copy coefficients and intensity from another probe or a raw SH array
	 * @param source Source LightProbe or raw SH coefficients array
	 * @returns This instance for chaining
	 */
	public copy(source: LightProbe | SHCoefficients): LightProbe {
		const sourceSH = source instanceof LightProbe ? source.sh : source;
		const sourceIntensity =
			source instanceof LightProbe ? source.intensity : this.intensity;

		for (let i = 0; i < 9; i++) {
			this.sh[i].r = sourceSH[i].r;
			this.sh[i].g = sourceSH[i].g;
			this.sh[i].b = sourceSH[i].b;
		}

		this.intensity = sourceIntensity;
		return this;
	}
}
