import { Vector3 } from "../maths/Vector3";
import { SH } from "../maths/SH";
import { isShadowCastingLight } from "../lights";
import { clamp } from "../maths/Common";
import type { IVector3 } from "../maths/types";
import type { RGB } from "../utils/Color";
import type {
	ILightingStrategy,
	PhongSurfaceProperties,
	ShaderContext,
} from "./types";

export class BlinnPhongStrategy implements ILightingStrategy<PhongSurfaceProperties> {
	public calculate(
		world: IVector3,
		normal: IVector3,
		viewDir: IVector3,
		surface: PhongSurfaceProperties,
		context: ShaderContext
	): RGB {
		// N and V are already normalized in LitShader
		const N = normal;
		const V = viewDir;
		const useSHAmbient = context.enableSH && !!context.shAmbientCoeffs;

		let ambR = 0,
			ambG = 0,
			ambB = 0;
		let diffR = 0,
			diffG = 0,
			diffB = 0;
		let specR = 0,
			specG = 0,
			specB = 0;

		const gamma = context.gamma;
		const invGamma = 1.0 / gamma;

		const alb = {
			r: Math.pow(Math.max(0, surface.albedo.r / 255), gamma),
			g: Math.pow(Math.max(0, surface.albedo.g / 255), gamma),
			b: Math.pow(Math.max(0, surface.albedo.b / 255), gamma),
		};

		// Ambient IBL or simple
		if (useSHAmbient && context.shAmbientCoeffs) {
			const irr = SH.calculateIrradiance(N, context.shAmbientCoeffs);
			ambR = irr.r / 255;
			ambG = irr.g / 255;
			ambB = irr.b / 255;
		}

		for (const light of context.lights) {
			const contrib = light.computeContribution(world);
			if (!contrib) continue;
			const lightIntensity = contrib.intensity ?? 1.0;

			if (contrib.type === "ambient") {
				if (useSHAmbient) continue;
				ambR += Math.pow(contrib.color.r / 255, gamma) * lightIntensity;
				ambG += Math.pow(contrib.color.g / 255, gamma) * lightIntensity;
				ambB += Math.pow(contrib.color.b / 255, gamma) * lightIntensity;
				continue;
			}

			const L = Vector3.normalize(contrib.direction);
			const NdotL = Math.max(0, Vector3.dot(N, L));

			let shadow = { r: 1, g: 1, b: 1 };
			if (context.enableShadows && isShadowCastingLight(light)) {
				const shadowMap = context.renderer.shadowMaps.get(light);
				if (shadowMap) {
					shadow = shadowMap.getShadowFactor(world, N);
				}
			}

			const radiance = {
				r: Math.pow(contrib.color.r / 255, gamma) * lightIntensity,
				g: Math.pow(contrib.color.g / 255, gamma) * lightIntensity,
				b: Math.pow(contrib.color.b / 255, gamma) * lightIntensity,
			};

			// Diffuse
			diffR += radiance.r * NdotL * shadow.r;
			diffG += radiance.g * NdotL * shadow.g;
			diffB += radiance.b * NdotL * shadow.b;

			// Specular
			const H = Vector3.normalize(Vector3.add(L, V));
			const NdotH = Math.max(0, Vector3.dot(N, H));
			const specFactor = NdotL > 0 ? Math.pow(NdotH, surface.shininess) : 0;

			specR += radiance.r * specFactor * shadow.r;
			specG += radiance.g * specFactor * shadow.g;
			specB += radiance.b * specFactor * shadow.b;
		}

		const specColor = {
			r: Math.pow(Math.max(0, surface.specular.r / 255), gamma),
			g: Math.pow(Math.max(0, surface.specular.g / 255), gamma),
			b: Math.pow(Math.max(0, surface.specular.b / 255), gamma),
		};

		const finalR = alb.r * (ambR + diffR) + specR * specColor.r;
		const finalG = alb.g * (ambG + diffG) + specG * specColor.g;
		const finalB = alb.b * (ambB + diffB) + specB * specColor.b;

		// No Tone mapping for Blinn-Phong to keep it simple/classic, but back to sRGB
		if (context.enableGamma) {
			return {
				r: clamp(Math.max(0, finalR) * 255, 0, 255),
				g: clamp(Math.max(0, finalG) * 255, 0, 255),
				b: clamp(Math.max(0, finalB) * 255, 0, 255),
			};
		}

		return {
			r: clamp(Math.pow(Math.max(0, finalR), invGamma) * 255, 0, 255),
			g: clamp(Math.pow(Math.max(0, finalG), invGamma) * 255, 0, 255),
			b: clamp(Math.pow(Math.max(0, finalB), invGamma) * 255, 0, 255),
		};
	}
}
