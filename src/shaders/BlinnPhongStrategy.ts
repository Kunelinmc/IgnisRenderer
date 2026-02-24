import { Vector3 } from "../maths/Vector3";
import { SH } from "../maths/SH";
import { isShadowCastingLight } from "../lights";
import { clamp, sRGBToLinear } from "../maths/Common";
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
		const shAmbient = context.shAmbientCoeffs;
		const hasSHAmbient =
			!!shAmbient &&
			(shAmbient[0].r !== 0 || shAmbient[0].g !== 0 || shAmbient[0].b !== 0);
		const useSHAmbient = context.enableSH && hasSHAmbient;

		let ambR = 0,
			ambG = 0,
			ambB = 0;
		let diffR = 0,
			diffG = 0,
			diffB = 0;
		let specR = 0,
			specG = 0,
			specB = 0;

		const alb = {
			r: sRGBToLinear(Math.max(0, surface.albedo.r / 255)),
			g: sRGBToLinear(Math.max(0, surface.albedo.g / 255)),
			b: sRGBToLinear(Math.max(0, surface.albedo.b / 255)),
		};
		const ambColor = {
			r: sRGBToLinear(Math.max(0, surface.ambient.r / 255)),
			g: sRGBToLinear(Math.max(0, surface.ambient.g / 255)),
			b: sRGBToLinear(Math.max(0, surface.ambient.b / 255)),
		};

		// Ambient IBL or simple
		if (useSHAmbient && shAmbient) {
			const irr = SH.calculateIrradiance(N, shAmbient);
			ambR = irr.r / 255;
			ambG = irr.g / 255;
			ambB = irr.b / 255;
		}

		for (const light of context.lights) {
			const contrib = light.computeContribution({ position: world, normal: N });
			if (!contrib) continue;
			const lightIntensity = contrib.intensity ?? 1.0;

			if (contrib.type === "ambient" || contrib.type === "irradiance") {
				if (useSHAmbient) continue;
				ambR += sRGBToLinear(contrib.color.r / 255) * lightIntensity;
				ambG += sRGBToLinear(contrib.color.g / 255) * lightIntensity;
				ambB += sRGBToLinear(contrib.color.b / 255) * lightIntensity;
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
				r: sRGBToLinear(contrib.color.r / 255) * lightIntensity,
				g: sRGBToLinear(contrib.color.g / 255) * lightIntensity,
				b: sRGBToLinear(contrib.color.b / 255) * lightIntensity,
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
			r: sRGBToLinear(Math.max(0, surface.specular.r / 255)),
			g: sRGBToLinear(Math.max(0, surface.specular.g / 255)),
			b: sRGBToLinear(Math.max(0, surface.specular.b / 255)),
		};

		const finalR = ambR * ambColor.r + diffR * alb.r + specR * specColor.r;
		const finalG = ambG * ambColor.g + diffG * alb.g + specG * specColor.g;
		const finalB = ambB * ambColor.b + diffB * alb.b + specB * specColor.b;

		// Shader output stays in linear space; optional gamma encode happens in post-process
		return {
			r: clamp(Math.max(0, finalR) * 255, 0, 255),
			g: clamp(Math.max(0, finalG) * 255, 0, 255),
			b: clamp(Math.max(0, finalB) * 255, 0, 255),
		};
	}
}
