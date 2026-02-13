import { Vector3 } from "../maths/Vector3";
import { SH } from "../maths/SH";
import { isShadowCastingLight } from "../lights";
import { LightingConstants } from "../core/Constants";
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
		const N = Vector3.normalize(normal);
		const V = Vector3.normalize(viewDir);
		const NdotV = Math.max(Vector3.dot(N, V), LightingConstants.PBR_MIN_NDOTV);
		const useSHAmbient = context.enableSH && !!context.shAmbientCoeffs;
		const shadowLight =
			context.enableShadows ? context.lights.find(isShadowCastingLight) : null;

		let totalR = 0,
			totalG = 0,
			totalB = 0;
		let ambientLightR = 0,
			ambientLightG = 0,
			ambientLightB = 0;

		const alb = {
			r: surface.albedo.r / 255,
			g: surface.albedo.g / 255,
			b: surface.albedo.b / 255,
		};
		const metal = surface.metalness;
		const rough = surface.roughness;
		const f0_norm = {
			r: surface.f0.r / 255,
			g: surface.f0.g / 255,
			b: surface.f0.b / 255,
		};

		// Metalness workflow: metals have albedo as F0
		const realF0 = {
			r: (1 - metal) * f0_norm.r + metal * alb.r,
			g: (1 - metal) * f0_norm.g + metal * alb.g,
			b: (1 - metal) * f0_norm.b + metal * alb.b,
		};

		for (const light of context.lights) {
			const contrib = light.computeContribution(world);
			if (!contrib) continue;

			if (contrib.type === "ambient") {
				if (!useSHAmbient) {
					ambientLightR += contrib.color.r / 255;
					ambientLightG += contrib.color.g / 255;
					ambientLightB += contrib.color.b / 255;
				}
				continue;
			}

			const L = Vector3.normalize(contrib.direction);
			const NdotL = Math.max(Vector3.dot(N, L), 0);
			if (NdotL <= 0) continue;

			const H = Vector3.normalize(Vector3.add(L, V));
			const radiance = {
				r: contrib.color.r / 255,
				g: contrib.color.g / 255,
				b: contrib.color.b / 255,
			};
			let shadow = 1.0;
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
				((kD.r * alb.r) / Math.PI + specular.r) * radiance.r * NdotL * shadow;
			totalG +=
				((kD.g * alb.g) / Math.PI + specular.g) * radiance.g * NdotL * shadow;
			totalB +=
				((kD.b * alb.b) / Math.PI + specular.b) * radiance.b * NdotL * shadow;
		}

		// Improved Ambient/IBL: Diffuse + Simplified Specular
		let ambR = 0,
			ambG = 0,
			ambB = 0;
		if (useSHAmbient && context.shAmbientCoeffs) {
			const irr = SH.calculateIrradiance(N, context.shAmbientCoeffs);
			const irrNorm = { r: irr.r / 255, g: irr.g / 255, b: irr.b / 255 };

			// 1. Diffuse IBL
			const kD_amb = 1.0 - metal;
			ambR = irrNorm.r * alb.r * kD_amb;
			ambG = irrNorm.g * alb.g * kD_amb;
			ambB = irrNorm.b * alb.b * kD_amb;

			// 2. Simplified Specular IBL fallback
			// In a full implementation, we would use a pre-filtered env map.
			// Here we use a small fraction of irradiance based on roughness.
			const specFactor = Math.max(
				LightingConstants.PBR_SPEC_FALLBACK,
				(1.0 - rough) * 0.5
			);
			const F_amb = this._FresnelSchlick(NdotV, realF0);
			ambR += irrNorm.r * F_amb.r * specFactor;
			ambG += irrNorm.g * F_amb.g * specFactor;
			ambB += irrNorm.b * F_amb.b * specFactor;
		} else {
			// Fallback to simple ambient lights in scene
			const intensity =
				ambientLightR + ambientLightG + ambientLightB > 0 ? 1.0 : 0.05;
			const ambientCol =
				intensity === 0.05 ?
					{ r: 1, g: 1, b: 1 }
				:	{ r: ambientLightR, g: ambientLightG, b: ambientLightB };

			ambR = ambientCol.r * alb.r * (1 - metal);
			ambG = ambientCol.g * alb.g * (1 - metal);
			ambB = ambientCol.b * alb.b * (1 - metal);

			// Small constant specular ambient for fallback
			const specFactor = LightingConstants.PBR_SPEC_FALLBACK;
			ambR += ambientCol.r * realF0.r * specFactor;
			ambG += ambientCol.g * realF0.g * specFactor;
			ambB += ambientCol.b * realF0.b * specFactor;
		}

		return {
			r: Math.min(1, totalR + ambR + surface.emissive.r / 255) * 255,
			g: Math.min(1, totalG + ambG + surface.emissive.g / 255) * 255,
			b: Math.min(1, totalB + ambB + surface.emissive.b / 255) * 255,
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
}
