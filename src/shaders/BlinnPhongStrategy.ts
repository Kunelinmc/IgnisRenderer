import { Vector3 } from "../maths/Vector3";
import { SH } from "../maths/SH";
import { isShadowCastingLight } from "../lights";
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
		const N = Vector3.normalize(normal);
		const V = Vector3.normalize(viewDir);
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

		// Ambient IBL or simple
		if (useSHAmbient && context.shAmbientCoeffs) {
			const irr = SH.calculateIrradiance(N, context.shAmbientCoeffs);
			ambR = irr.r;
			ambG = irr.g;
			ambB = irr.b;
		}

		for (const light of context.lights) {
			const contrib = light.computeContribution(world);
			if (!contrib) continue;

			if (contrib.type === "ambient") {
				if (useSHAmbient) continue;
				ambR += contrib.color.r;
				ambG += contrib.color.g;
				ambB += contrib.color.b;
				continue;
			}

			const L = Vector3.normalize(contrib.direction);
			const NdotL = Math.max(0, Vector3.dot(N, L));

			let shadow = 1.0;
			if (context.enableShadows && isShadowCastingLight(light)) {
				const shadowMap = context.renderer.shadowMaps.get(light);
				if (shadowMap) {
					shadow = shadowMap.getShadowFactor(world, N);
				}
			}

			// Diffuse
			diffR += contrib.color.r * NdotL * shadow;
			diffG += contrib.color.g * NdotL * shadow;
			diffB += contrib.color.b * NdotL * shadow;

			// Specular
			const H = Vector3.normalize(Vector3.add(L, V));
			const NdotH = Math.max(0, Vector3.dot(N, H));
			const specFactor = NdotL > 0 ? Math.pow(NdotH, surface.shininess) : 0;

			specR += contrib.color.r * specFactor * shadow;
			specG += contrib.color.g * specFactor * shadow;
			specB += contrib.color.b * specFactor * shadow;
		}

		const specColor = surface.specular;
		return {
			r:
				(surface.albedo.r * (ambR + diffR)) / 255 + (specR * specColor.r) / 255,
			g:
				(surface.albedo.g * (ambG + diffG)) / 255 + (specG * specColor.g) / 255,
			b:
				(surface.albedo.b * (ambB + diffB)) / 255 + (specB * specColor.b) / 255,
		};
	}
}
