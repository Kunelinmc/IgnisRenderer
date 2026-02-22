import { Vector3 } from "../maths/Vector3";
import { SH } from "../maths/SH";
import { isShadowCastingLight } from "../lights";
import { LightingConstants } from "../core/Constants";
import { clamp, sRGBToLinear } from "../maths/Common";
import type { IVector3 } from "../maths/types";
import type { RGB } from "../utils/Color";
import type {
	ILightingStrategy,
	PBRSurfaceProperties,
	ShaderContext,
} from "./types";

/**
 * Cook-Torrance PBR lighting strategy.
 *
 * Color pipeline:
 *   sRGB material inputs (albedo, f0, emissive) [0-255]
 *     → sRGB EOTF decode: sRGBToLinear(x/255) → linear [0-1]
 *     → all BRDF math in linear space
 *     → ACES tone map: scene-linear → display-linear [0-1]
 *     → scale to [0-255] and return
 *     → PostProcessor.applyGamma applies the sRGB OETF for final sRGB output
 *
 * Internal conventions:
 * - All colors processed in this method are in **linear space [0-1]** unless noted.
 * - RGB inputs from materials/lights are assumed sRGB-encoded [0-255] and decoded
 *   via sRGBToLinear() — the exact IEC 61966-2-1 piecewise transfer function.
 * - SH coefficients are pre-converted to linear space in Renderer.updateSH(), so SH irradiance
 *   output only needs normalization (/255), NOT additional sRGB decoding.
 * - The output RGB [0-255] is in display-linear space (post tone map, pre sRGB encode).
 */
export class PBRStrategy implements ILightingStrategy<PBRSurfaceProperties> {
	public calculate(
		world: IVector3,
		normal: IVector3,
		viewDir: IVector3,
		surface: PBRSurfaceProperties,
		context: ShaderContext
	): RGB {
		// Inputs N and V are already normalized in LitShader
		const N = normal;
		const V = viewDir;
		const shAmbient = context.shAmbientCoeffs;
		const hasSHAmbient =
			!!shAmbient &&
			(shAmbient[0].r !== 0 || shAmbient[0].g !== 0 || shAmbient[0].b !== 0);
		// Clamp to small positive value to avoid division-by-zero in Cook-Torrance denominator.
		// This same NdotV is shared with _GeometrySmith to keep G and denominator consistent.
		const NdotV = Math.max(Vector3.dot(N, V), LightingConstants.PBR_MIN_NDOTV);
		const useSHAmbient = context.enableSH && hasSHAmbient;

		let totalR = 0,
			totalG = 0,
			totalB = 0;
		let ambientLightR = 0,
			ambientLightG = 0,
			ambientLightB = 0;

		// 1. Linear Workflow: Decode sRGB material inputs → linear [0-1]
		// Material colors (albedo, f0, emissive) arrive as sRGB-encoded [0-255].
		// sRGBToLinear() applies the exact IEC 61966-2-1 piecewise EOTF.
		const alb = {
			r: sRGBToLinear(Math.max(0, surface.albedo.r / 255)),
			g: sRGBToLinear(Math.max(0, surface.albedo.g / 255)),
			b: sRGBToLinear(Math.max(0, surface.albedo.b / 255)),
		};
		const metal = clamp(surface.metalness, 0.0, 1.0);
		const rough = clamp(surface.roughness, 0.04, 1.0);
		const occlusion = clamp(surface.occlusion, 0.0, 1.0);
		const clearcoat = clamp(surface.clearcoat, 0.0, 1.0);
		const clearcoatRoughness = clamp(surface.clearcoatRoughness, 0.04, 1.0);

		// Dielectric F0 from reflectance (0.5 -> 0.04)
		const reflectance = clamp(surface.reflectance, 0.0, 1.0);
		const f0Val = 0.16 * reflectance * reflectance;
		const f0_norm = { r: f0Val, g: f0Val, b: f0Val };

		// Metalness workflow: metals have albedo as F0, non-metals use f0 computed from reflectance
		const realF0 = {
			r: (1 - metal) * f0_norm.r + metal * alb.r,
			g: (1 - metal) * f0_norm.g + metal * alb.g,
			b: (1 - metal) * f0_norm.b + metal * alb.b,
		};

		// Emissive: sRGB [0-255] → linear, then scaled by intensity.
		// Emissive is additive and bypasses ambient occlusion.
		const emissiveScale = surface.emissiveIntensity ?? 1.0;
		const emissive = {
			r: sRGBToLinear(Math.max(0, surface.emissive.r / 255)) * emissiveScale,
			g: sRGBToLinear(Math.max(0, surface.emissive.g / 255)) * emissiveScale,
			b: sRGBToLinear(Math.max(0, surface.emissive.b / 255)) * emissiveScale,
		};

		for (const light of context.lights) {
			const contrib = light.computeContribution(world);
			if (!contrib) continue;
			const lightIntensity = contrib.intensity ?? 1.0;

			if (contrib.type === "ambient") {
				if (!useSHAmbient) {
					// Ambient light color: sRGB [0-255] → linear [0-1].
					// Accumulated in linear space; applied in the ambient term below.
					// Skipped when SH ambient is active (SH replaces flat ambient).
					ambientLightR += sRGBToLinear(contrib.color.r / 255) * lightIntensity;
					ambientLightG += sRGBToLinear(contrib.color.g / 255) * lightIntensity;
					ambientLightB += sRGBToLinear(contrib.color.b / 255) * lightIntensity;
				}
				continue;
			}

			const L = Vector3.normalize(contrib.direction);
			const NdotL = Math.max(Vector3.dot(N, L), 0);
			if (NdotL <= 0) continue; // Back-facing to this light; NdotL > 0 guaranteed below

			const H = Vector3.normalize(Vector3.add(L, V));
			// Direct light radiance: sRGB [0-255] → linear [0-1], scaled by intensity
			const radiance = {
				r: sRGBToLinear(contrib.color.r / 255) * lightIntensity,
				g: sRGBToLinear(contrib.color.g / 255) * lightIntensity,
				b: sRGBToLinear(contrib.color.b / 255) * lightIntensity,
			};

			let shadow = { r: 1, g: 1, b: 1 };
			if (context.enableShadows && isShadowCastingLight(light)) {
				const shadowMap = context.renderer.shadowMaps.get(light);
				if (shadowMap) {
					shadow = shadowMap.getShadowFactor(world, N);
				}
			}

			// Cook-Torrance math
			const NDF = this._DistributionGGX(N, H, rough);
			const G = this._GeometrySmith(NdotV, NdotL, rough);
			const F = this._FresnelSchlick(Math.max(Vector3.dot(H, V), 0), realF0);

			const nominator = {
				r: NDF * G * F.r,
				g: NDF * G * F.g,
				b: NDF * G * F.b,
			};
			const denominator =
				4 * NdotV * NdotL + LightingConstants.PBR_DENOM_EPSILON;

			const specular = {
				r: nominator.r / denominator,
				g: nominator.g / denominator,
				b: nominator.b / denominator,
			};

			const kS = F;
			const kD = {
				r: (1 - kS.r) * (1 - metal),
				g: (1 - kS.g) * (1 - metal),
				b: (1 - kS.b) * (1 - metal),
			};

			// Clearcoat calculations
			let ccSpecular = { r: 0, g: 0, b: 0 };
			let ccFresnel = { r: 0, g: 0, b: 0 };
			if (clearcoat > 0) {
				const HdotV = Math.max(Vector3.dot(H, V), 0);
				const NDF_cc = this._DistributionGGX(N, H, clearcoatRoughness);
				const G_cc = this._GeometrySmithClearcoat(
					NdotV,
					NdotL,
					clearcoatRoughness
				);
				const F_cc = this._FresnelSchlickScalar(HdotV, 0.04);
				ccFresnel = { r: F_cc, g: F_cc, b: F_cc };

				const ccDenom = 4 * NdotV * NdotL + LightingConstants.PBR_DENOM_EPSILON;
				const ccValue = (NDF_cc * G_cc * F_cc) / ccDenom;
				ccSpecular = { r: ccValue, g: ccValue, b: ccValue };
			}

			// Attenuate base layer by (1 - F_cc * clearcoat)
			const baseAttenuation = {
				r: 1.0 - ccFresnel.r * clearcoat,
				g: 1.0 - ccFresnel.g * clearcoat,
				b: 1.0 - ccFresnel.b * clearcoat,
			};

			totalR +=
				(((kD.r * alb.r) / Math.PI + specular.r) * baseAttenuation.r +
					ccSpecular.r * clearcoat) *
				radiance.r *
				NdotL *
				shadow.r;
			totalG +=
				(((kD.g * alb.g) / Math.PI + specular.g) * baseAttenuation.g +
					ccSpecular.g * clearcoat) *
				radiance.g *
				NdotL *
				shadow.g;
			totalB +=
				(((kD.b * alb.b) / Math.PI + specular.b) * baseAttenuation.b +
					ccSpecular.b * clearcoat) *
				radiance.b *
				NdotL *
				shadow.b;
		}

		// Improved Ambient/IBL
		let ambR = 0,
			ambG = 0,
			ambB = 0;

		if (useSHAmbient && shAmbient) {
			// SH coefficients were pre-converted to linear space in Renderer.updateSH(),
			// so calculateIrradiance returns linear values scaled to [0-255].
			// Only normalization (/255) is needed here — no gamma decode.
			const irr = SH.calculateIrradiance(N, shAmbient);
			const irrLinear = {
				r: irr.r / 255,
				g: irr.g / 255,
				b: irr.b / 255,
			};

			const F_amb = this._FresnelSchlick(NdotV, realF0);
			const kD_amb = {
				r: (1.0 - F_amb.r) * (1.0 - metal),
				g: (1.0 - F_amb.g) * (1.0 - metal),
				b: (1.0 - F_amb.b) * (1.0 - metal),
			};

			// Clearcoat Ambient
			let ccAmbFresnel = 0;
			let ccAmbSpec = 0;
			if (clearcoat > 0) {
				ccAmbFresnel = this._FresnelSchlickScalar(NdotV, 0.04);
				const ccSpecFactor = Math.max(
					LightingConstants.PBR_SPEC_FALLBACK,
					(1.0 - clearcoatRoughness) * 0.5
				);
				ccAmbSpec = ccAmbFresnel * ccSpecFactor;
			}

			const baseAttenuationAmb = 1.0 - ccAmbFresnel * clearcoat;

			ambR = irrLinear.r * alb.r * kD_amb.r * baseAttenuationAmb;
			ambG = irrLinear.g * alb.g * kD_amb.g * baseAttenuationAmb;
			ambB = irrLinear.b * alb.b * kD_amb.b * baseAttenuationAmb;

			// Simplified Specular IBL fallback
			const specFactor = Math.max(
				LightingConstants.PBR_SPEC_FALLBACK,
				(1.0 - rough) * 0.5
			);
			ambR +=
				irrLinear.r * F_amb.r * specFactor * baseAttenuationAmb +
				irrLinear.r * ccAmbSpec * clearcoat;
			ambG +=
				irrLinear.g * F_amb.g * specFactor * baseAttenuationAmb +
				irrLinear.g * ccAmbSpec * clearcoat;
			ambB +=
				irrLinear.b * F_amb.b * specFactor * baseAttenuationAmb +
				irrLinear.b * ccAmbSpec * clearcoat;
		} else {
			const ambientCol = {
				r: ambientLightR,
				g: ambientLightG,
				b: ambientLightB,
			};
			if (ambientLightR + ambientLightG + ambientLightB === 0) {
				// Fallback ambient when no ambient lights exist.
				// 0.05 is a sRGB reference value; decode to linear for consistency.
				const fallback = sRGBToLinear(0.05);
				ambientCol.r = fallback;
				ambientCol.g = fallback;
				ambientCol.b = fallback;
			}

			// Fresnel-based kD for ambient diffuse (consistent with SH branch)
			const F_amb = this._FresnelSchlick(NdotV, realF0);
			const kD_amb = {
				r: (1.0 - F_amb.r) * (1.0 - metal),
				g: (1.0 - F_amb.g) * (1.0 - metal),
				b: (1.0 - F_amb.b) * (1.0 - metal),
			};

			// Clearcoat for simple ambient
			const ccAmbFresnel =
				clearcoat > 0 ? this._FresnelSchlickScalar(NdotV, 0.04) : 0;
			const baseAttenuationAmb = 1.0 - ccAmbFresnel * clearcoat;

			ambR = ambientCol.r * alb.r * kD_amb.r * baseAttenuationAmb;
			ambG = ambientCol.g * alb.g * kD_amb.g * baseAttenuationAmb;
			ambB = ambientCol.b * alb.b * kD_amb.b * baseAttenuationAmb;

			const specFactor = LightingConstants.PBR_SPEC_FALLBACK;
			const ccSpecFactor = Math.max(
				LightingConstants.PBR_SPEC_FALLBACK,
				(1.0 - clearcoatRoughness) * 0.5
			);

			ambR +=
				ambientCol.r * realF0.r * specFactor * baseAttenuationAmb +
				ambientCol.r * ccAmbFresnel * ccSpecFactor * clearcoat;
			ambG +=
				ambientCol.g * realF0.g * specFactor * baseAttenuationAmb +
				ambientCol.g * ccAmbFresnel * ccSpecFactor * clearcoat;
			ambB +=
				ambientCol.b * realF0.b * specFactor * baseAttenuationAmb +
				ambientCol.b * ccAmbFresnel * ccSpecFactor * clearcoat;
		}

		ambR *= occlusion;
		ambG *= occlusion;
		ambB *= occlusion;

		// Final combined scene-linear color (unbounded HDR range)
		let finalR = totalR + ambR + emissive.r;
		let finalG = totalG + ambG + emissive.g;
		let finalB = totalB + ambB + emissive.b;

		// 2. Tone Mapping (Narkowicz ACES approximation)
		// Maps scene-linear HDR → display-linear [0-1]
		finalR = this._acesFilm(finalR);
		finalG = this._acesFilm(finalG);
		finalB = this._acesFilm(finalB);

		// 3. Scale to [0-255] and return.
		// Output is display-linear (post tone map, pre sRGB encode).
		// PostProcessor.applyGamma() will apply the sRGB OETF for final sRGB output.
		return {
			r: clamp(finalR * 255, 0, 255),
			g: clamp(finalG * 255, 0, 255),
			b: clamp(finalB * 255, 0, 255),
		};
	}

	private _DistributionGGX(N: IVector3, H: IVector3, roughness: number) {
		const a = roughness * roughness;
		const a2 = a * a;
		const NdotH = Math.max(Vector3.dot(N, H), 0);
		const NdotH2 = NdotH * NdotH;
		const nom = a2;
		let denom = NdotH2 * (a2 - 1.0) + 1.0;
		denom = Math.PI * denom * denom;
		return nom / Math.max(denom, LightingConstants.GGX_EPSILON);
	}

	/**
	 * Schlick-GGX geometry function (Smith's method).
	 * Uses the direct lighting remapping: k = (roughness + 1)² / 8.
	 * NdotV and NdotL are pre-clamped by the caller (NdotV ≥ PBR_MIN_NDOTV, NdotL > 0)
	 * to stay consistent with the Cook-Torrance denominator.
	 */
	private _GeometrySmith(NdotV: number, NdotL: number, roughness: number) {
		const r = roughness + 1.0;
		const k = (r * r) / 8.0;
		const G1V = NdotV / (NdotV * (1.0 - k) + k);
		const G1L = NdotL / (NdotL * (1.0 - k) + k);
		return G1V * G1L;
	}

	/** Schlick's approximation for Fresnel reflectance. F0 is in linear space. */
	private _FresnelSchlick(cosTheta: number, F0: RGB) {
		const f = Math.pow(Math.max(1.0 - cosTheta, 0), 5.0);
		return {
			r: F0.r + (1.0 - F0.r) * f,
			g: F0.g + (1.0 - F0.g) * f,
			b: F0.b + (1.0 - F0.b) * f,
		};
	}

	/** Schlick Fresnel for a single scalar value (useful for Clearcoat F0=0.04). */
	private _FresnelSchlickScalar(cosTheta: number, F0: number) {
		const f = Math.pow(Math.max(1.0 - cosTheta, 0), 5.0);
		return F0 + (1.0 - F0) * f;
	}

	/**
	 * Geometry function for clearcoat using Schlick-GGX (Smith's method).
	 * Uses the isotropic remapping k = α² / 2 (where α = roughness²),
	 * per the Filament/glTF clearcoat convention. This differs from the base
	 * layer's direct lighting remapping k = (roughness + 1)² / 8, and better
	 * models the smooth, isotropic nature of the clearcoat layer.
	 */
	private _GeometrySmithClearcoat(
		NdotV: number,
		NdotL: number,
		roughness: number
	) {
		const a = roughness * roughness;
		const k = a / 2.0;
		const G1V = NdotV / (NdotV * (1.0 - k) + k);
		const G1L = NdotL / (NdotL * (1.0 - k) + k);
		return G1V * G1L;
	}

	/** Narkowicz ACES fitted curve. Input: scene-linear HDR. Output: display-linear [0-1]. */
	private _acesFilm(x: number): number {
		const a = 2.51;
		const b = 0.03;
		const c = 2.43;
		const d = 0.59;
		const e = 0.14;
		return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1.0);
	}
}
