import { SH } from "../maths/SH";
import { Texture } from "../core/Texture";
import { Vector3 } from "../maths/Vector3";
import { linearToSRGB, sRGBToLinear } from "../maths/Common";
import {
	Light,
	LightType,
	type LightContribution,
	type SurfacePoint,
} from "./Light";
import type { SHCoefficients, IVector3 } from "../maths/types";

/**
 * LightProbe provides irregular or environment lighting via Spherical Harmonics (diffuse)
 * and pre-filtered maps (specular).
 */
export class LightProbe extends Light<LightType.LightProbe> {
	public sh: SHCoefficients;
	public prefilteredMap: Texture | null = null;
	private static readonly DC_IRRADIANCE_SCALE = Math.PI * 0.282095;

	constructor(
		sh: SHCoefficients | null = null,
		intensity = 1.0,
		prefilteredMap: Texture | null = null
	) {
		super(LightType.LightProbe, { intensity });
		// Deep copy to prevent external mutation of passed array
		this.sh = sh ? JSON.parse(JSON.stringify(sh)) : SH.empty();
		this.prefilteredMap = prefilteredMap;
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

		const MAX_WIDTH = 128;
		const MAX_HEIGHT = 64;

		const sampleWidth = Math.min(width, MAX_WIDTH);
		const sampleHeight = Math.min(height, MAX_HEIGHT);

		const stepX = width / sampleWidth;
		const stepY = height / sampleHeight;

		const dTheta = Math.PI / sampleHeight;
		const dPhi = (2 * Math.PI) / sampleWidth;

		let totalWeight = 0;

		for (let sj = 0; sj < sampleHeight; sj++) {
			const theta = (sj + 0.5) * dTheta;
			const sinTheta = Math.sin(theta);
			const cosTheta = Math.cos(theta);
			const weight = sinTheta * dTheta * dPhi;

			const j = Math.floor((sj + 0.5) * stepY);

			for (let si = 0; si < sampleWidth; si++) {
				const phi = (si + 0.5) * dPhi;

				const x = sinTheta * Math.sin(phi);
				const y = cosTheta;
				const z = sinTheta * Math.cos(phi);

				const basis = SH.evalBasis({ x, y, z });

				const i = Math.floor((si + 0.5) * stepX);
				const idx = (j * width + i) * 4;

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

				for (let k = 0; k < sh.length; k++) {
					const bK = basis[k] * weight;
					sh[k].r += r * bK;
					sh[k].g += g * bK;
					sh[k].b += b * bK;
				}

				totalWeight += weight;
			}
		}

		const normFactor = (4 * Math.PI) / totalWeight;

		for (let k = 0; k < sh.length; k++) {
			sh[k].r *= normFactor;
			sh[k].g *= normFactor;
			sh[k].b *= normFactor;
		}

		return new LightProbe(sh);
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
		const prb = LightProbe._projectEquirectToSH(envMap);
		prb.prefilteredMap = LightProbe._prefilterEnvMap(envMap);
		return prb;
	}

	/**
	 * Pre-filter an environment map for different roughness levels.
	 */
	private static _prefilterEnvMap(envMap: Texture): Texture {
		const baseWidth = Math.min(envMap.width, 128);
		const baseHeight = Math.min(envMap.height, 64);
		const maxMipLevels = 5;

		const prefiltered = new Texture(null, baseWidth, baseHeight, "HDR");
		prefiltered.mipmaps = [];

		const SAMPLE_COUNT = 256;

		for (let level = 0; level < maxMipLevels; level++) {
			const roughness = level / (maxMipLevels - 1);
			const w = Math.max(1, baseWidth >> level);
			const h = Math.max(1, baseHeight >> level);
			const data = new Float32Array(w * h * 4);

			for (let j = 0; j < h; j++) {
				const theta = ((j + 0.5) / h) * Math.PI;
				for (let i = 0; i < w; i++) {
					const phi = ((i + 0.5) / w) * 2 * Math.PI;

					const N = {
						x: Math.sin(theta) * Math.sin(phi),
						y: Math.cos(theta),
						z: Math.sin(theta) * Math.cos(phi),
					};

					const radiance = this._prefilterSpecular(
						envMap,
						N,
						roughness,
						SAMPLE_COUNT
					);
					const idx = (j * w + i) * 4;
					data[idx] = radiance.r;
					data[idx + 1] = radiance.g;
					data[idx + 2] = radiance.b;
					data[idx + 3] = 1.0;
				}
			}
			prefiltered.mipmaps.push(data);
		}

		prefiltered.data = prefiltered.mipmaps[0];
		return prefiltered;
	}

	private static _prefilterSpecular(
		envMap: Texture,
		N: IVector3,
		roughness: number,
		sampleCount: number
	): { r: number; g: number; b: number } {
		let totalWeight = 0;
		let prefilteredColor = { r: 0, g: 0, b: 0 };

		for (let i = 0; i < sampleCount; i++) {
			const xi = this._hammersley(i, sampleCount);
			const H = this._importanceSampleGGX(xi, N, roughness);
			const V = N; // Approximation: V = N
			const NdotH = Math.max(Vector3.dot(N, H), 0);
			const L = Vector3.normalize({
				x: 2.0 * NdotH * H.x - V.x,
				y: 2.0 * NdotH * H.y - V.y,
				z: 2.0 * NdotH * H.z - V.z,
			});

			const NdotL = Math.max(Vector3.dot(N, L), 0);
			if (NdotL > 0) {
				const phi = Math.atan2(L.x, L.z);
				const theta = Math.acos(Math.max(-1, Math.min(1, L.y)));
				const u = (phi + Math.PI) / (2 * Math.PI);
				const v = theta / Math.PI;

				const sample = envMap.sample(u, v);
				const isLinear =
					envMap.colorSpace === "HDR" || envMap.colorSpace === "Linear";

				const r = isLinear ? sample.r / 255 : sRGBToLinear(sample.r / 255);
				const g = isLinear ? sample.g / 255 : sRGBToLinear(sample.g / 255);
				const b = isLinear ? sample.b / 255 : sRGBToLinear(sample.b / 255);

				prefilteredColor.r += r * NdotL;
				prefilteredColor.g += g * NdotL;
				prefilteredColor.b += b * NdotL;
				totalWeight += NdotL;
			}
		}

		if (totalWeight > 0) {
			prefilteredColor.r /= totalWeight;
			prefilteredColor.g /= totalWeight;
			prefilteredColor.b /= totalWeight;
		}

		return prefilteredColor;
	}

	private static _hammersley(i: number, n: number): { x: number; y: number } {
		let bits = i;
		bits = (bits << 16) | (bits >>> 16);
		bits = ((bits & 0x55555555) << 1) | ((bits & 0xaaaaaaaa) >>> 1);
		bits = ((bits & 0x33333333) << 2) | ((bits & 0xcccccccc) >>> 2);
		bits = ((bits & 0x0f0f0f0f) << 4) | ((bits & 0xf0f0f0f0) >>> 4);
		bits = ((bits & 0x00ff00ff) << 8) | ((bits & 0xff00ff00) >>> 8);
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

		const up =
			Math.abs(N.y) < 0.999 ? { x: 0, y: 1, z: 0 } : { x: 0, y: 0, z: 1 };
		const tangent = Vector3.normalize(Vector3.cross(up, N));
		const bitangent = Vector3.cross(N, tangent);

		const worldH = {
			x: tangent.x * H.x + bitangent.x * H.y + N.x * H.z,
			y: tangent.y * H.x + bitangent.y * H.y + N.y * H.z,
			z: tangent.z * H.x + bitangent.z * H.y + N.z * H.z,
		};

		return Vector3.normalize(worldH);
	}

	/**
	 * Clone this light probe
	 * @returns A new LightProbe instance with the same coefficients and intensity
	 */
	public clone(): LightProbe {
		const cloned = new LightProbe(this.sh, this.intensity);
		cloned.prefilteredMap = this.prefilteredMap;
		return cloned;
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

		for (let i = 0; i < this.sh.length; i++) {
			this.sh[i].r = sourceSH[i].r;
			this.sh[i].g = sourceSH[i].g;
			this.sh[i].b = sourceSH[i].b;
		}

		this.intensity = sourceIntensity;

		if (source instanceof LightProbe) {
			this.prefilteredMap = source.prefilteredMap;
		}

		return this;
	}
}
