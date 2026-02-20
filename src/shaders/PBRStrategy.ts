import { Vector3 } from "../maths/Vector3";
import { SH } from "../maths/SH";
import { isShadowCastingLight } from "../lights";
import { LightingConstants } from "../core/Constants";
import { clamp } from "../maths/Common";
import type { IVector3 } from "../maths/types";
import type { RGB } from "../utils/Color";
import type {
	ILightingStrategy,
	PBRSurfaceProperties,
	ShaderContext,
} from "./types";

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
		const NdotV = Math.max(Vector3.dot(N, V), LightingConstants.PBR_MIN_NDOTV);
		const useSHAmbient = context.enableSH && !!context.shAmbientCoeffs;

		let totalR = 0,
			totalG = 0,
			totalB = 0;
		let ambientLightR = 0,
			ambientLightG = 0,
			ambientLightB = 0;

		const gamma = context.gamma;

		// 1. Linear Workflow: Convert inputs to Linear Space
		const alb = {
			r: Math.pow(Math.max(0, surface.albedo.r / 255), gamma),
			g: Math.pow(Math.max(0, surface.albedo.g / 255), gamma),
			b: Math.pow(Math.max(0, surface.albedo.b / 255), gamma),
		};
		const metal = clamp(surface.metalness, 0.0, 1.0);
		const rough = clamp(surface.roughness, 0.04, 1.0);
		const occlusion = clamp(surface.occlusion, 0.0, 1.0);

		// Common PBR practice: non-metals have a base F0 of 0.04
		const F0_NON_METAL = 0.04;
		const f0_norm = {
			r: Math.pow(Math.max(0, surface.f0.r / 255), gamma),
			g: Math.pow(Math.max(0, surface.f0.g / 255), gamma),
			b: Math.pow(Math.max(0, surface.f0.b / 255), gamma),
		};

		// Metalness workflow: metals have albedo as F0, non-metals use a small constant or f0 param
		const realF0 = {
			r: (1 - metal) * Math.max(F0_NON_METAL, f0_norm.r) + metal * alb.r,
			g: (1 - metal) * Math.max(F0_NON_METAL, f0_norm.g) + metal * alb.g,
			b: (1 - metal) * Math.max(F0_NON_METAL, f0_norm.b) + metal * alb.b,
		};

		const emissiveScale = surface.emissiveIntensity ?? 1.0;
		const emissive = {
			r: Math.pow(Math.max(0, surface.emissive.r / 255), gamma) * emissiveScale,
			g: Math.pow(Math.max(0, surface.emissive.g / 255), gamma) * emissiveScale,
			b: Math.pow(Math.max(0, surface.emissive.b / 255), gamma) * emissiveScale,
		};

		for (const light of context.lights) {
			const contrib = light.computeContribution(world);
			if (!contrib) continue;
			const lightIntensity = contrib.intensity ?? 1.0;

			if (contrib.type === "ambient") {
				if (!useSHAmbient) {
					// Convert ambient light to linear
					ambientLightR += Math.pow(contrib.color.r / 255, gamma) * lightIntensity;
					ambientLightG += Math.pow(contrib.color.g / 255, gamma) * lightIntensity;
					ambientLightB += Math.pow(contrib.color.b / 255, gamma) * lightIntensity;
				}
				continue;
			}

			const L = Vector3.normalize(contrib.direction);
			const NdotL = Math.max(Vector3.dot(N, L), 0);
			if (NdotL <= 0) continue;

			const H = Vector3.normalize(Vector3.add(L, V));
			const radiance = {
				r: Math.pow(contrib.color.r / 255, gamma) * lightIntensity,
				g: Math.pow(contrib.color.g / 255, gamma) * lightIntensity,
				b: Math.pow(contrib.color.b / 255, gamma) * lightIntensity,
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
			const G = this._GeometrySmith(N, V, L, rough);
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

			totalR +=
				((kD.r * alb.r) / Math.PI + specular.r) * radiance.r * NdotL * shadow.r;
			totalG +=
				((kD.g * alb.g) / Math.PI + specular.g) * radiance.g * NdotL * shadow.g;
			totalB +=
				((kD.b * alb.b) / Math.PI + specular.b) * radiance.b * NdotL * shadow.b;
		}

		// Improved Ambient/IBL
		let ambR = 0,
			ambG = 0,
			ambB = 0;

		if (useSHAmbient && context.shAmbientCoeffs) {
			const irr = SH.calculateIrradiance(N, context.shAmbientCoeffs);
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
			ambR = irrLinear.r * alb.r * kD_amb.r;
			ambG = irrLinear.g * alb.g * kD_amb.g;
			ambB = irrLinear.b * alb.b * kD_amb.b;

			// Simplified Specular IBL fallback
			const specFactor = Math.max(
				LightingConstants.PBR_SPEC_FALLBACK,
				(1.0 - rough) * 0.5
			);
			ambR += irrLinear.r * F_amb.r * specFactor;
			ambG += irrLinear.g * F_amb.g * specFactor;
			ambB += irrLinear.b * F_amb.b * specFactor;
		} else {
			const ambientCol = {
				r: ambientLightR,
				g: ambientLightG,
				b: ambientLightB,
			};
			if (ambientLightR + ambientLightG + ambientLightB === 0) {
				// Constant low ambient if no lights
				const fallback = Math.pow(0.05, gamma);
				ambientCol.r = fallback;
				ambientCol.g = fallback;
				ambientCol.b = fallback;
			}

			ambR = ambientCol.r * alb.r * (1 - metal);
			ambG = ambientCol.g * alb.g * (1 - metal);
			ambB = ambientCol.b * alb.b * (1 - metal);

			const specFactor = LightingConstants.PBR_SPEC_FALLBACK;
			ambR += ambientCol.r * realF0.r * specFactor;
			ambG += ambientCol.g * realF0.g * specFactor;
			ambB += ambientCol.b * realF0.b * specFactor;
		}

		ambR *= occlusion;
		ambG *= occlusion;
		ambB *= occlusion;

		// Final Combined Linear Color
		let finalR = totalR + ambR + emissive.r;
		let finalG = totalG + ambG + emissive.g;
		let finalB = totalB + ambB + emissive.b;

		// 2. Tone Mapping (ACES Approximation)
		finalR = this._acesFilm(finalR);
		finalG = this._acesFilm(finalG);
		finalB = this._acesFilm(finalB);

		// 3. Shader output stays in linear space; gamma encode happens in post-process.
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

	private _GeometrySmith(
		N: IVector3,
		V: IVector3,
		L: IVector3,
		roughness: number
	) {
		const r = roughness + 1.0;
		const k = (r * r) / 8.0;
		const G1V =
			Math.max(Vector3.dot(N, V), 0) /
			(Math.max(Vector3.dot(N, V), 0) * (1.0 - k) + k);
		const G1L =
			Math.max(Vector3.dot(N, L), 0) /
			(Math.max(Vector3.dot(N, L), 0) * (1.0 - k) + k);
		return G1V * G1L;
	}

	private _FresnelSchlick(cosTheta: number, F0: RGB) {
		const f = Math.pow(Math.max(1.0 - cosTheta, 0), 5.0);
		return {
			r: F0.r + (1.0 - F0.r) * f,
			g: F0.g + (1.0 - F0.g) * f,
			b: F0.b + (1.0 - F0.b) * f,
		};
	}

	private _acesFilm(x: number): number {
		const a = 2.51;
		const b = 0.03;
		const c = 2.43;
		const d = 0.59;
		const e = 0.14;
		return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1.0);
	}
}
