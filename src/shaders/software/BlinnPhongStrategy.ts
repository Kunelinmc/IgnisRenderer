import { Vector3 } from "../../maths/Vector3";
import { SH } from "../../maths/SH";
import {
	isShadowCastingLight,
} from "../../lights";
import { sampleActiveIrradianceProbeGrid } from "../../lights/runtime/irradianceProbeGridRuntime";
import {
	createLightContribution,
	evaluateLightContribution,
	type SurfacePoint,
} from "../../lights/runtime/lightEvaluator";
import { clamp, sRGBToLinear } from "../../maths/Common";
import type { IVector3, SHCoefficients } from "../../maths/types";
import type { RGB } from "../../foundation/Color";
import type {
	ILightingStrategy,
	PhongSurfaceProperties,
	ShaderContext,
} from "./types";

export class BlinnPhongStrategy implements ILightingStrategy<PhongSurfaceProperties> {
	private _surfacePoint: SurfacePoint = {
		position: { x: 0, y: 0, z: 0 },
		normal: { x: 0, y: 0, z: 1 },
	};
	private _lightContribution = createLightContribution();
	private _gridSHAmbient: SHCoefficients = SH.empty();

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
		const baseSHAmbient = context.shAmbientCoeffs;
		const gridAmbient =
			context.enableSH ?
				sampleActiveIrradianceProbeGrid(
					context.lights,
					world,
					context.cameraPos,
					this._gridSHAmbient
				)
			:	null;
		const gridCoverage = gridAmbient?.coverage ?? 0;
		const shAmbient = this._resolveSHAmbient(
			baseSHAmbient,
			gridAmbient?.sh ?? null,
			gridCoverage
		);
		const hasSHAmbient = this._hasNonZeroSH(shAmbient);
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
		const surfacePoint = this._surfacePoint;
		surfacePoint.position.x = world.x;
		surfacePoint.position.y = world.y;
		surfacePoint.position.z = world.z;
		surfacePoint.normal!.x = N.x;
		surfacePoint.normal!.y = N.y;
		surfacePoint.normal!.z = N.z;

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
			const contrib = evaluateLightContribution(
				light,
				surfacePoint,
				this._lightContribution
			);
			if (!contrib) continue;
			const lightIntensity = contrib.intensity ?? 1.0;

			if (contrib.type === "ambient" || contrib.type === "irradiance") {
				const fallbackScale = this._resolveAmbientFallbackScale(
					useSHAmbient,
					baseSHAmbient,
					gridCoverage
				);
				if (useSHAmbient && fallbackScale <= 0) continue;
				const intensity = lightIntensity * fallbackScale;
				ambR += sRGBToLinear(contrib.color.r / 255) * intensity;
				ambG += sRGBToLinear(contrib.color.g / 255) * intensity;
				ambB += sRGBToLinear(contrib.color.b / 255) * intensity;
				continue;
			}

			const L = Vector3.normalize(contrib.direction);
			const NdotL = Math.max(0, Vector3.dot(N, L));

			let shadow = { r: 1, g: 1, b: 1 };
			if (
				context.enableShadows &&
				isShadowCastingLight(light) &&
				context.sampleShadow
			) {
				shadow = context.sampleShadow(light, world, N);
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

	private _hasNonZeroSH(coeffs: ShaderContext["shAmbientCoeffs"]): boolean {
		if (!coeffs) return false;

		for (const coeff of coeffs) {
			if (coeff.r !== 0 || coeff.g !== 0 || coeff.b !== 0) return true;
		}

		return false;
	}

	private _resolveSHAmbient(
		base: SHCoefficients | null,
		grid: SHCoefficients | null,
		gridCoverage: number
	): SHCoefficients | null {
		if (!grid || gridCoverage <= 0) {
			return base;
		}
		if (!base || gridCoverage >= 1) {
			return grid;
		}
		for (let i = 0; i < grid.length; i++) {
			grid[i].r = base[i].r * (1 - gridCoverage) + grid[i].r * gridCoverage;
			grid[i].g = base[i].g * (1 - gridCoverage) + grid[i].g * gridCoverage;
			grid[i].b = base[i].b * (1 - gridCoverage) + grid[i].b * gridCoverage;
		}
		return grid;
	}

	private _resolveAmbientFallbackScale(
		useSHAmbient: boolean,
		baseSHAmbient: SHCoefficients | null,
		gridCoverage: number
	): number {
		if (!useSHAmbient) return 1;
		if (baseSHAmbient || gridCoverage <= 0 || gridCoverage >= 1) return 0;
		return 1 - gridCoverage;
	}
}
