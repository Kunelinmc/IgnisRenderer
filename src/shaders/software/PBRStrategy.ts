import { Vector3 } from "../../maths/Vector3";
import { SH } from "../../maths/SH";
import {
	isShadowCastingLight,
} from "../../lights";
import { sampleActiveIrradianceProbeGrid } from "../../lights/runtime/irradianceProbeGridRuntime";
import { sampleReflectionProbesSpecular } from "../../lights/runtime/reflectionProbeRuntime";
import {
	createLightContribution,
	evaluateLightContribution,
	type SurfacePoint,
} from "./LightEvaluator";
import {
	GGX_EPSILON,
	PBR_AMBIENT_FALLBACK_LINEAR,
	PBR_DENOM_EPSILON,
	PBR_MIN_NDOTV,
	PBR_SPEC_FALLBACK,
} from "../../lights/constants";
import { clamp, sRGBToLinear } from "../../maths/Common";
import type { IVector3, SHCoefficients } from "../../maths/types";
import type { RGB, RGBA } from "../../foundation/Color";
import type {
	ILightingStrategy,
	PBRSurfaceProperties,
	ShaderContext,
} from "./types";

const SH_IRRADIANCE_FACTORS: readonly number[] = [
	Math.PI,
	(2 * Math.PI) / 3,
	(2 * Math.PI) / 3,
	(2 * Math.PI) / 3,
	Math.PI / 4,
	Math.PI / 4,
	Math.PI / 4,
	Math.PI / 4,
	Math.PI / 4,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
];

/**
 * Cook-Torrance PBR lighting strategy.
 *
 * All material colors entering this strategy are already in linear space,
 * stored in the engine's usual 0..255 material range by PBREvaluator. This
 * method normalizes them to 0..1 and returns normalized linear radiance.
 */
export class PBRStrategy implements ILightingStrategy<PBRSurfaceProperties> {
	private _surfacePoint: SurfacePoint = {
		position: { x: 0, y: 0, z: 0 },
		normal: { x: 0, y: 0, z: 1 },
	};
	private _lightContribution = createLightContribution();
	private _reflectionDir: IVector3 = { x: 0, y: 0, z: 1 };
	private _refractionDir: IVector3 = { x: 0, y: 0, z: 1 };
	private _clearcoatReflectionDir: IVector3 = { x: 0, y: 0, z: 1 };
	private _lightDir: IVector3 = { x: 0, y: 0, z: 1 };
	private _halfDir: IVector3 = { x: 0, y: 0, z: 1 };
	private _anisotropicBentNormal: IVector3 = { x: 0, y: 0, z: 1 };
	private _shBasis = new Float32Array(16);
	private _gridSHAmbient: SHCoefficients = SH.empty();
	private _shIrradiance: RGB = { r: 0, g: 0, b: 0 };
	private _shRadiance: RGB = { r: 0, g: 0, b: 0 };
	private _iridescenceFresnel: RGB = { r: 0, g: 0, b: 0 };
	private _iridescenceSensitivity: RGB = { r: 0, g: 0, b: 0 };

	public calculate(
		world: IVector3,
		normal: IVector3,
		viewDir: IVector3,
		surface: PBRSurfaceProperties,
		context: ShaderContext
	): RGB {
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
		const NdotVRaw = Vector3.dot(N, V);
		const NdotV = Math.max(NdotVRaw, PBR_MIN_NDOTV);
		const useSHAmbient = context.enableSH && hasSHAmbient;
		const reflectionDir = this._reflectionDir;
		this._reflectViewDirection(N, V, NdotVRaw, reflectionDir);

		let totalR = 0;
		let totalG = 0;
		let totalB = 0;
		let ambientLightR = 0;
		let ambientLightG = 0;
		let ambientLightB = 0;

		const surfacePoint = this._surfacePoint;
		surfacePoint.position.x = world.x;
		surfacePoint.position.y = world.y;
		surfacePoint.position.z = world.z;
		surfacePoint.normal!.x = N.x;
		surfacePoint.normal!.y = N.y;
		surfacePoint.normal!.z = N.z;

		const albR = Math.max(0, surface.albedo.r / 255);
		const albG = Math.max(0, surface.albedo.g / 255);
		const albB = Math.max(0, surface.albedo.b / 255);

		const metal = clamp(surface.metalness, 0.0, 1.0);
		const rough = clamp(surface.roughness, 0.04, 1.0);
		const occlusion = clamp(surface.occlusion, 0.0, 1.0);
		const clearcoat = clamp(surface.clearcoat, 0.0, 1.0);
		const clearcoatRoughness = clamp(surface.clearcoatRoughness, 0.04, 1.0);
		const anisotropy = clamp(surface.anisotropyStrength ?? 0.0, 0.0, 1.0);
		const anisotropyT = surface.anisotropyTangent;
		const anisotropyB = surface.anisotropyBitangent;
		const hasAnisotropy =
			anisotropy > 1e-6 &&
			!!anisotropyT &&
			!!anisotropyB;
		if (hasAnisotropy) {
			this._resolveAnisotropicReflectionDirection(
				N,
				V,
				anisotropyB,
				rough,
				anisotropy,
				reflectionDir
			);
		}

		const sheenInput = surface.sheenColor;
		const sheenColorR = (sheenInput?.r ?? 0) / 255;
		const sheenColorG = (sheenInput?.g ?? 0) / 255;
		const sheenColorB = (sheenInput?.b ?? 0) / 255;
		const sheenRoughness = clamp(surface.sheenRoughness ?? 0.0, 0.04, 1.0);
		const maxSheenColor = Math.max(sheenColorR, sheenColorG, sheenColorB);

		const transmission = clamp(surface.transmission ?? 0.0, 0.0, 1.0);
		const iridescence = clamp(surface.iridescence ?? 0.0, 0.0, 1.0);
		const iridescenceIor = Math.max(surface.iridescenceIor ?? 1.3, 1.0);
		const iridescenceThickness = Math.max(
			surface.iridescenceThickness ?? 400.0,
			0.0
		);
		const thickness = surface.thickness ?? 0.0;
		const attenuationDist = surface.attenuationDistance ?? Infinity;
		const attenuationInput = surface.attenuationColor;
		const attenuationColorR = clamp((attenuationInput?.r ?? 255) / 255, 0.0, 1.0);
		const attenuationColorG = clamp((attenuationInput?.g ?? 255) / 255, 0.0, 1.0);
		const attenuationColorB = clamp((attenuationInput?.b ?? 255) / 255, 0.0, 1.0);

		let volumeAttenuationR = 1.0;
		let volumeAttenuationG = 1.0;
		let volumeAttenuationB = 1.0;
		if (thickness > 0 && attenuationDist < Infinity && attenuationDist > 0) {
			const absorbR = -Math.log(attenuationColorR) / attenuationDist;
			const absorbG = -Math.log(attenuationColorG) / attenuationDist;
			const absorbB = -Math.log(attenuationColorB) / attenuationDist;
			volumeAttenuationR = Math.exp(-absorbR * thickness);
			volumeAttenuationG = Math.exp(-absorbG * thickness);
			volumeAttenuationB = Math.exp(-absorbB * thickness);
		}

		const reflectance = clamp(surface.reflectance, 0.0, 1.0);
		const baseF0Val = 0.16 * reflectance * reflectance;
		const oneMinusMetal = 1.0 - metal;

		const specularFactor = surface.specularFactor ?? 1.0;
		const specColorInput = surface.specularColor;
		const specColorR = (specColorInput?.r ?? 255) / 255;
		const specColorG = (specColorInput?.g ?? 255) / 255;
		const specColorB = (specColorInput?.b ?? 255) / 255;

		const f0NormR = Math.min(baseF0Val * specColorR * specularFactor, 1.0);
		const f0NormG = Math.min(baseF0Val * specColorG * specularFactor, 1.0);
		const f0NormB = Math.min(baseF0Val * specColorB * specularFactor, 1.0);

		const realF0R = oneMinusMetal * f0NormR + metal * albR;
		const realF0G = oneMinusMetal * f0NormG + metal * albG;
		const realF0B = oneMinusMetal * f0NormB + metal * albB;

		let energyCompensationR = 1.0;
		let energyCompensationG = 1.0;
		let energyCompensationB = 1.0;
		let brdfValue: RGBA | null = null;
		if (context.brdfLUT) {
			brdfValue = context.brdfLUT.sample(NdotV, Math.sqrt(rough));
			const E = brdfValue.r + brdfValue.g;
			if (E > 0.0 && E < 1.0) {
				const factor = 1.0 / E - 1.0;
				energyCompensationR = 1.0 + realF0R * factor;
				energyCompensationG = 1.0 + realF0G * factor;
				energyCompensationB = 1.0 + realF0B * factor;
			}
		}

		const emissiveScale = surface.emissiveIntensity ?? 1.0;
		const emissiveR = (surface.emissive.r / 255) * emissiveScale;
		const emissiveG = (surface.emissive.g / 255) * emissiveScale;
		const emissiveB = (surface.emissive.b / 255) * emissiveScale;

		const Nc = surface.clearcoatNormal ?? N;
		const NcdotV = Math.max(Vector3.dot(Nc, V), PBR_MIN_NDOTV);
		const clearcoatTransmissionFresnel =
			clearcoat > 0 ? this._FresnelSchlickScalar(NcdotV, 0.04) : 0;
		const transmissionAttenuation = 1.0 - clearcoatTransmissionFresnel * clearcoat;
		const oneMinusTransmission = 1.0 - transmission;

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
				if (!useSHAmbient || fallbackScale > 0) {
					const intensity = lightIntensity * fallbackScale;
					ambientLightR += sRGBToLinear(contrib.color.r / 255) * intensity;
					ambientLightG += sRGBToLinear(contrib.color.g / 255) * intensity;
					ambientLightB += sRGBToLinear(contrib.color.b / 255) * intensity;
				}
				continue;
			}

			const L = this._lightDir;
			L.x = contrib.direction.x;
			L.y = contrib.direction.y;
			L.z = contrib.direction.z;
			Vector3.normalizeInPlace(L);

			const NdotLRaw = Vector3.dot(N, L);
			const NdotL = Math.max(NdotLRaw, 0);
			const NdotLTransmission = Math.max(-NdotLRaw, 0);
			if (NdotL <= 0 && NdotLTransmission <= 0) continue;

			const radianceR = sRGBToLinear(contrib.color.r / 255) * lightIntensity;
			const radianceG = sRGBToLinear(contrib.color.g / 255) * lightIntensity;
			const radianceB = sRGBToLinear(contrib.color.b / 255) * lightIntensity;

			let shadowR = 1.0;
			let shadowG = 1.0;
			let shadowB = 1.0;
			if (
				context.enableShadows &&
				isShadowCastingLight(light) &&
				context.sampleShadow
			) {
				const shadow = context.sampleShadow(light, world, N);
				shadowR = shadow.r;
				shadowG = shadow.g;
				shadowB = shadow.b;
			}

			const fView = this._resolveIridescenceFresnel(
				NdotV,
				realF0R,
				realF0G,
				realF0B,
				iridescence,
				iridescenceIor,
				iridescenceThickness,
				this._iridescenceFresnel
			);
			const FviewR = fView.r;
			const FviewG = fView.g;
			const FviewB = fView.b;

			const kTR = (1.0 - FviewR) * oneMinusMetal * transmission;
			const kTG = (1.0 - FviewG) * oneMinusMetal * transmission;
			const kTB = (1.0 - FviewB) * oneMinusMetal * transmission;
			const transmittedDiffuseR = (kTR * volumeAttenuationR * albR) / Math.PI;
			const transmittedDiffuseG = (kTG * volumeAttenuationG * albG) / Math.PI;
			const transmittedDiffuseB = (kTB * volumeAttenuationB * albB) / Math.PI;

			let specularR = 0;
			let specularG = 0;
			let specularB = 0;
			let diffuseR = 0;
			let diffuseG = 0;
			let diffuseB = 0;
			let clearcoatAttenuationR = 1.0;
			let clearcoatAttenuationG = 1.0;
			let clearcoatAttenuationB = 1.0;
			let baseLayerAttenuationR = 1.0;
			let baseLayerAttenuationG = 1.0;
			let baseLayerAttenuationB = 1.0;
			let ccSpecularR = 0;
			let ccSpecularG = 0;
			let ccSpecularB = 0;
			let sheenSpecularR = 0;
			let sheenSpecularG = 0;
			let sheenSpecularB = 0;

			if (NdotL > 0) {
				const H = this._halfDir;
				H.x = L.x + V.x;
				H.y = L.y + V.y;
				H.z = L.z + V.z;
				Vector3.normalizeInPlace(H);

				const HdotV = Math.max(Vector3.dot(H, V), 0);
				const fresnel = this._resolveIridescenceFresnel(
					HdotV,
					realF0R,
					realF0G,
					realF0B,
					iridescence,
					iridescenceIor,
					iridescenceThickness,
					this._iridescenceFresnel
				);
				const FR = fresnel.r;
				const FG = fresnel.g;
				const FB = fresnel.b;
				if (hasAnisotropy) {
					const alphaRoughness = rough * rough;
					const at =
						alphaRoughness * (1.0 - anisotropy * anisotropy) +
						anisotropy * anisotropy;
					const ab = alphaRoughness;
					const NdotH = Math.max(Vector3.dot(N, H), 0);
					const TdotV = Vector3.dot(anisotropyT, V);
					const BdotV = Vector3.dot(anisotropyB, V);
					const TdotL = Vector3.dot(anisotropyT, L);
					const BdotL = Vector3.dot(anisotropyB, L);
					const TdotH = Vector3.dot(anisotropyT, H);
					const BdotH = Vector3.dot(anisotropyB, H);
					const anisotropicNDF = this._DistributionAnisotropicGGX(
						NdotH,
						TdotH,
						BdotH,
						at,
						ab
					);
					const anisotropicVisibility = this._VisibilityAnisotropicGGX(
						NdotL,
						NdotV,
						BdotV,
						TdotV,
						TdotL,
						BdotL,
						at,
						ab
					);
					specularR =
						anisotropicNDF *
						anisotropicVisibility *
						FR *
						energyCompensationR;
					specularG =
						anisotropicNDF *
						anisotropicVisibility *
						FG *
						energyCompensationG;
					specularB =
						anisotropicNDF *
						anisotropicVisibility *
						FB *
						energyCompensationB;
				} else {
					const NDF = this._DistributionGGX(N, H, rough);
					const G = this._GeometrySmith(NdotV, NdotL, rough);
					const denominator = 4 * NdotV * NdotL + PBR_DENOM_EPSILON;
					specularR = (NDF * G * FR) / denominator * energyCompensationR;
					specularG = (NDF * G * FG) / denominator * energyCompensationG;
					specularB = (NDF * G * FB) / denominator * energyCompensationB;
				}

				const fresnelMax = Math.max(FR, FG, FB);
				const kDR =
					(1.0 - (iridescence > 0 ? fresnelMax : FR)) *
					oneMinusMetal *
					oneMinusTransmission;
				const kDG =
					(1.0 - (iridescence > 0 ? fresnelMax : FG)) *
					oneMinusMetal *
					oneMinusTransmission;
				const kDB =
					(1.0 - (iridescence > 0 ? fresnelMax : FB)) *
					oneMinusMetal *
					oneMinusTransmission;
				diffuseR = (kDR * albR) / Math.PI;
				diffuseG = (kDG * albG) / Math.PI;
				diffuseB = (kDB * albB) / Math.PI;

				let ccFresnel = 0;
				if (clearcoat > 0) {
					const NcdotL = Math.max(Vector3.dot(Nc, L), 0);
					if (NcdotL > 0) {
						const ndfCc = this._DistributionGGX(Nc, H, clearcoatRoughness);
						const gCc = this._GeometrySmithClearcoat(
							NcdotV,
							NcdotL,
							clearcoatRoughness
						);
						const fCc = this._FresnelSchlickScalar(HdotV, 0.04);
						ccFresnel = fCc;

						const ccDenom = 4 * NcdotV * NcdotL + PBR_DENOM_EPSILON;
						const ccValue = (ndfCc * gCc * fCc) / ccDenom;
						ccSpecularR = ccValue;
						ccSpecularG = ccValue;
						ccSpecularB = ccValue;
					}
				}

				let albedoSheenScalingR = 1.0;
				let albedoSheenScalingG = 1.0;
				let albedoSheenScalingB = 1.0;
				if (maxSheenColor > 0) {
					const NdotH = Math.max(Vector3.dot(N, H), 0);
					const sheenNDF = this._DistributionCharlie(NdotH, sheenRoughness);
					const sheenV = this._VisibilityAshikhmin(NdotL, NdotV);
					sheenSpecularR = sheenColorR * sheenNDF * sheenV;
					sheenSpecularG = sheenColorG * sheenNDF * sheenV;
					sheenSpecularB = sheenColorB * sheenNDF * sheenV;

					const sheenFresnelFactor = this._pow5(Math.max(1.0 - HdotV, 0));
					const sheenFresnelR =
						sheenColorR + (1.0 - sheenColorR) * sheenFresnelFactor;
					const sheenFresnelG =
						sheenColorG + (1.0 - sheenColorG) * sheenFresnelFactor;
					const sheenFresnelB =
						sheenColorB + (1.0 - sheenColorB) * sheenFresnelFactor;
					albedoSheenScalingR = Math.max(0, 1.0 - sheenFresnelR);
					albedoSheenScalingG = Math.max(0, 1.0 - sheenFresnelG);
					albedoSheenScalingB = Math.max(0, 1.0 - sheenFresnelB);
				}

				const ccAttenuation = 1.0 - ccFresnel * clearcoat;
				clearcoatAttenuationR = ccAttenuation;
				clearcoatAttenuationG = ccAttenuation;
				clearcoatAttenuationB = ccAttenuation;
				baseLayerAttenuationR = ccAttenuation * albedoSheenScalingR;
				baseLayerAttenuationG = ccAttenuation * albedoSheenScalingG;
				baseLayerAttenuationB = ccAttenuation * albedoSheenScalingB;
			}

			totalR +=
				(((diffuseR + specularR) * baseLayerAttenuationR +
					ccSpecularR * clearcoat +
					sheenSpecularR * clearcoatAttenuationR) *
					NdotL +
					transmittedDiffuseR * transmissionAttenuation * NdotLTransmission) *
				radianceR *
				shadowR;
			totalG +=
				(((diffuseG + specularG) * baseLayerAttenuationG +
					ccSpecularG * clearcoat +
					sheenSpecularG * clearcoatAttenuationG) *
					NdotL +
					transmittedDiffuseG * transmissionAttenuation * NdotLTransmission) *
				radianceG *
				shadowG;
			totalB +=
				(((diffuseB + specularB) * baseLayerAttenuationB +
					ccSpecularB * clearcoat +
					sheenSpecularB * clearcoatAttenuationB) *
					NdotL +
					transmittedDiffuseB * transmissionAttenuation * NdotLTransmission) *
				radianceB *
				shadowB;
		}

		let ambR = 0;
		let ambG = 0;
		let ambB = 0;
		const ior = surface.ior ?? 1.5;

		if (useSHAmbient && shAmbient) {
			const irr = this._shIrradiance;
			this._calculateSHIrradiance(N, shAmbient, irr);
			const irrLinearR = irr.r / 255;
			const irrLinearG = irr.g / 255;
			const irrLinearB = irr.b / 255;

			const specRadiance = this._shRadiance;
			this._sampleSHRadiance(reflectionDir, shAmbient, specRadiance);
			const specRadianceLinearR = specRadiance.r / 255;
			const specRadianceLinearG = specRadiance.g / 255;
			const specRadianceLinearB = specRadiance.b / 255;

			const fAmbient = this._resolveIridescenceFresnel(
				NdotV,
				realF0R,
				realF0G,
				realF0B,
				iridescence,
				iridescenceIor,
				iridescenceThickness,
				this._iridescenceFresnel
			);
			const FambR = fAmbient.r;
			const FambG = fAmbient.g;
			const FambB = fAmbient.b;

			const hasRefraction =
				transmission > 0 && this._refract(V, N, ior, this._refractionDir);
			const isTIR = transmission > 0 && !hasRefraction;
			const effectiveFambR = isTIR ? 1.0 : FambR;
			const effectiveFambG = isTIR ? 1.0 : FambG;
			const effectiveFambB = isTIR ? 1.0 : FambB;

			const effectiveFambMax = Math.max(
				effectiveFambR,
				effectiveFambG,
				effectiveFambB
			);
			const diffuseFambR = iridescence > 0 ? effectiveFambMax : effectiveFambR;
			const diffuseFambG = iridescence > 0 ? effectiveFambMax : effectiveFambG;
			const diffuseFambB = iridescence > 0 ? effectiveFambMax : effectiveFambB;
			const kDambR =
				(1.0 - diffuseFambR) * oneMinusMetal * oneMinusTransmission;
			const kDambG =
				(1.0 - diffuseFambG) * oneMinusMetal * oneMinusTransmission;
			const kDambB =
				(1.0 - diffuseFambB) * oneMinusMetal * oneMinusTransmission;
			const kTambR = (1.0 - effectiveFambR) * oneMinusMetal * transmission;
			const kTambG = (1.0 - effectiveFambG) * oneMinusMetal * transmission;
			const kTambB = (1.0 - effectiveFambB) * oneMinusMetal * transmission;

			let ccAmbFresnel = 0;
			let ccAmbSpecR = 0;
			let ccAmbSpecG = 0;
			let ccAmbSpecB = 0;
			if (clearcoat > 0) {
				const NcdotVAmb = Math.max(Vector3.dot(Nc, V), PBR_MIN_NDOTV);
				ccAmbFresnel = this._FresnelSchlickScalar(NcdotVAmb, 0.04);

				const ccReflectionDir = this._clearcoatReflectionDir;
				this._reflectViewDirection(Nc, V, Vector3.dot(Nc, V), ccReflectionDir);
				const ccPrefiltered = this._sampleEnvironmentSpecular(
					world,
					ccReflectionDir,
					clearcoatRoughness,
					context
				);
				if (ccPrefiltered && context.brdfLUT) {
					const ccBrdf = context.brdfLUT.sample(
						NcdotVAmb,
						Math.sqrt(clearcoatRoughness)
					);
					const ccMix = ccAmbFresnel * ccBrdf.r + ccBrdf.g;
					ccAmbSpecR = ccPrefiltered.r * ccMix;
					ccAmbSpecG = ccPrefiltered.g * ccMix;
					ccAmbSpecB = ccPrefiltered.b * ccMix;
				} else {
					const ccSpecFactor = Math.max(
						PBR_SPEC_FALLBACK,
						(1.0 - clearcoatRoughness) * 0.5
					);
					const ccSpec = ccAmbFresnel * ccSpecFactor;
					ccAmbSpecR = specRadianceLinearR * ccSpec;
					ccAmbSpecG = specRadianceLinearG * ccSpec;
					ccAmbSpecB = specRadianceLinearB * ccSpec;
				}
			}

			const clearcoatAttenuationAmb = 1.0 - ccAmbFresnel * clearcoat;
			const baseAttenuationAmbR =
				clearcoatAttenuationAmb * (1.0 - sheenColorR * 0.5);
			const baseAttenuationAmbG =
				clearcoatAttenuationAmb * (1.0 - sheenColorG * 0.5);
			const baseAttenuationAmbB =
				clearcoatAttenuationAmb * (1.0 - sheenColorB * 0.5);

			ambR = irrLinearR * albR * kDambR * baseAttenuationAmbR;
			ambG = irrLinearG * albG * kDambG * baseAttenuationAmbG;
			ambB = irrLinearB * albB * kDambB * baseAttenuationAmbB;

			if (transmission > 0 && hasRefraction) {
				let transmRadianceR = 0;
				let transmRadianceG = 0;
				let transmRadianceB = 0;
				const prefilteredRefraction = this._sampleEnvironmentSpecular(
					world,
					this._refractionDir,
					rough,
					context
				);
				if (prefilteredRefraction) {
					transmRadianceR = prefilteredRefraction.r;
					transmRadianceG = prefilteredRefraction.g;
					transmRadianceB = prefilteredRefraction.b;
				} else {
					this._sampleSHRadiance(this._refractionDir, shAmbient, specRadiance);
					transmRadianceR = specRadiance.r;
					transmRadianceG = specRadiance.g;
					transmRadianceB = specRadiance.b;
				}

				ambR +=
					transmRadianceR *
					albR *
					kTambR *
					volumeAttenuationR *
					clearcoatAttenuationAmb;
				ambG +=
					transmRadianceG *
					albG *
					kTambG *
					volumeAttenuationG *
					clearcoatAttenuationAmb;
				ambB +=
					transmRadianceB *
					albB *
					kTambB *
					volumeAttenuationB *
					clearcoatAttenuationAmb;
			}

			const prefiltered = this._sampleEnvironmentSpecular(
				world,
				reflectionDir,
				rough,
				context
			);
			if (prefiltered && context.brdfLUT && brdfValue) {
				const specR =
					prefiltered.r *
					(effectiveFambR * brdfValue.r + brdfValue.g) *
					clearcoatAttenuationAmb *
					energyCompensationR;
				const specG =
					prefiltered.g *
					(effectiveFambG * brdfValue.r + brdfValue.g) *
					clearcoatAttenuationAmb *
					energyCompensationG;
				const specB =
					prefiltered.b *
					(effectiveFambB * brdfValue.r + brdfValue.g) *
					clearcoatAttenuationAmb *
					energyCompensationB;

				ambR += specR + ccAmbSpecR * clearcoat;
				ambG += specG + ccAmbSpecG * clearcoat;
				ambB += specB + ccAmbSpecB * clearcoat;
			} else {
				const specFactor = Math.max(PBR_SPEC_FALLBACK, (1.0 - rough) * 0.5);
				ambR +=
					specRadianceLinearR *
						effectiveFambR *
						specFactor *
						clearcoatAttenuationAmb +
					ccAmbSpecR * clearcoat;
				ambG +=
					specRadianceLinearG *
						effectiveFambG *
						specFactor *
						clearcoatAttenuationAmb +
					ccAmbSpecG * clearcoat;
				ambB +=
					specRadianceLinearB *
						effectiveFambB *
						specFactor *
						clearcoatAttenuationAmb +
					ccAmbSpecB * clearcoat;
			}

			if (maxSheenColor > 0) {
				const sheenAmb = Math.max(
					PBR_SPEC_FALLBACK,
					(1.0 - sheenRoughness) * 0.5
				);
				ambR +=
					specRadianceLinearR *
					sheenColorR *
					sheenAmb *
					clearcoatAttenuationAmb;
				ambG +=
					specRadianceLinearG *
					sheenColorG *
					sheenAmb *
					clearcoatAttenuationAmb;
				ambB +=
					specRadianceLinearB *
					sheenColorB *
					sheenAmb *
					clearcoatAttenuationAmb;
			}
		} else {
			let ambientColR = ambientLightR;
			let ambientColG = ambientLightG;
			let ambientColB = ambientLightB;
			if (ambientLightR + ambientLightG + ambientLightB === 0) {
				const fallback = PBR_AMBIENT_FALLBACK_LINEAR;
				ambientColR = fallback;
				ambientColG = fallback;
				ambientColB = fallback;
			}

			const ambientRadianceR = ambientColR / Math.PI;
			const ambientRadianceG = ambientColG / Math.PI;
			const ambientRadianceB = ambientColB / Math.PI;
			const fAmbient = this._resolveIridescenceFresnel(
				NdotV,
				realF0R,
				realF0G,
				realF0B,
				iridescence,
				iridescenceIor,
				iridescenceThickness,
				this._iridescenceFresnel
			);
			const FambR = fAmbient.r;
			const FambG = fAmbient.g;
			const FambB = fAmbient.b;

			const hasRefraction =
				transmission > 0 && this._refract(V, N, ior, this._refractionDir);
			const isTIR = transmission > 0 && !hasRefraction;
			const effectiveFambR = isTIR ? 1.0 : FambR;
			const effectiveFambG = isTIR ? 1.0 : FambG;
			const effectiveFambB = isTIR ? 1.0 : FambB;

			const effectiveFambMax = Math.max(
				effectiveFambR,
				effectiveFambG,
				effectiveFambB
			);
			const diffuseFambR = iridescence > 0 ? effectiveFambMax : effectiveFambR;
			const diffuseFambG = iridescence > 0 ? effectiveFambMax : effectiveFambG;
			const diffuseFambB = iridescence > 0 ? effectiveFambMax : effectiveFambB;
			const kDambR =
				(1.0 - diffuseFambR) * oneMinusMetal * oneMinusTransmission;
			const kDambG =
				(1.0 - diffuseFambG) * oneMinusMetal * oneMinusTransmission;
			const kDambB =
				(1.0 - diffuseFambB) * oneMinusMetal * oneMinusTransmission;
			const kTambR = (1.0 - effectiveFambR) * oneMinusMetal * transmission;
			const kTambG = (1.0 - effectiveFambG) * oneMinusMetal * transmission;
			const kTambB = (1.0 - effectiveFambB) * oneMinusMetal * transmission;

			const ccAmbFresnel =
				clearcoat > 0 ? this._FresnelSchlickScalar(NdotV, 0.04) : 0;
			const clearcoatAttenuationAmb = 1.0 - ccAmbFresnel * clearcoat;
			const baseAttenuationAmbR =
				clearcoatAttenuationAmb * (1.0 - sheenColorR * 0.5);
			const baseAttenuationAmbG =
				clearcoatAttenuationAmb * (1.0 - sheenColorG * 0.5);
			const baseAttenuationAmbB =
				clearcoatAttenuationAmb * (1.0 - sheenColorB * 0.5);

			ambR = ambientColR * albR * kDambR * baseAttenuationAmbR;
			ambG = ambientColG * albG * kDambG * baseAttenuationAmbG;
			ambB = ambientColB * albB * kDambB * baseAttenuationAmbB;

			if (transmission > 0 && hasRefraction) {
				ambR +=
					ambientColR *
					albR *
					kTambR *
					volumeAttenuationR *
					clearcoatAttenuationAmb;
				ambG +=
					ambientColG *
					albG *
					kTambG *
					volumeAttenuationG *
					clearcoatAttenuationAmb;
				ambB +=
					ambientColB *
					albB *
					kTambB *
					volumeAttenuationB *
					clearcoatAttenuationAmb;
			}

			const specFactor = Math.max(PBR_SPEC_FALLBACK, (1.0 - rough) * 0.5);
			const ccSpecFactor = Math.max(
				PBR_SPEC_FALLBACK,
				(1.0 - clearcoatRoughness) * 0.5
			);

			ambR +=
				ambientRadianceR *
					effectiveFambR *
					specFactor *
					clearcoatAttenuationAmb +
				ambientRadianceR * ccAmbFresnel * ccSpecFactor * clearcoat;
			ambG +=
				ambientRadianceG *
					effectiveFambG *
					specFactor *
					clearcoatAttenuationAmb +
				ambientRadianceG * ccAmbFresnel * ccSpecFactor * clearcoat;
			ambB +=
				ambientRadianceB *
					effectiveFambB *
					specFactor *
					clearcoatAttenuationAmb +
				ambientRadianceB * ccAmbFresnel * ccSpecFactor * clearcoat;

			if (maxSheenColor > 0) {
				const sheenAmb = Math.max(
					PBR_SPEC_FALLBACK,
					(1.0 - sheenRoughness) * 0.5
				);
				ambR +=
					ambientRadianceR *
					sheenColorR *
					sheenAmb *
					clearcoatAttenuationAmb;
				ambG +=
					ambientRadianceG *
					sheenColorG *
					sheenAmb *
					clearcoatAttenuationAmb;
				ambB +=
					ambientRadianceB *
					sheenColorB *
					sheenAmb *
					clearcoatAttenuationAmb;
			}
		}

		ambR *= occlusion;
		ambG *= occlusion;
		ambB *= occlusion;

		const finalR = Math.max(0, totalR + ambR + emissiveR);
		const finalG = Math.max(0, totalG + ambG + emissiveG);
		const finalB = Math.max(0, totalB + ambB + emissiveB);

		return {
			r: finalR,
			g: finalG,
			b: finalB,
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
		return nom / Math.max(denom, GGX_EPSILON);
	}

	/**
	 * Schlick-GGX geometry function (Smith's method).
	 * Uses the direct lighting remapping: k = (roughness + 1)^2 / 8.
	 */
	private _GeometrySmith(NdotV: number, NdotL: number, roughness: number) {
		const r = roughness + 1.0;
		const k = (r * r) / 8.0;
		const G1V = NdotV / (NdotV * (1.0 - k) + k);
		const G1L = NdotL / (NdotL * (1.0 - k) + k);
		return G1V * G1L;
	}

	private _DistributionAnisotropicGGX(
		NdotH: number,
		TdotH: number,
		BdotH: number,
		at: number,
		ab: number
	): number {
		const a2 = Math.max(at * ab, 1e-6);
		const fx = ab * TdotH;
		const fy = at * BdotH;
		const fz = a2 * NdotH;
		const denom = Math.max(fx * fx + fy * fy + fz * fz, GGX_EPSILON);
		const w2 = a2 / denom;
		return (a2 * w2 * w2) / Math.PI;
	}

	private _VisibilityAnisotropicGGX(
		NdotL: number,
		NdotV: number,
		BdotV: number,
		TdotV: number,
		TdotL: number,
		BdotL: number,
		at: number,
		ab: number
	): number {
		const ggxV =
			NdotL *
			Math.hypot(at * TdotV, ab * BdotV, NdotV);
		const ggxL =
			NdotV *
			Math.hypot(at * TdotL, ab * BdotL, NdotL);
		return clamp(0.5 / Math.max(ggxV + ggxL, GGX_EPSILON), 0.0, 1.0);
	}

	private _pow5(value: number): number {
		const value2 = value * value;
		return value2 * value2 * value;
	}

	private _DistributionCharlie(NdotH: number, roughness: number) {
		const invAlpha = 1.0 / Math.max(roughness * roughness, 1e-6);
		const cos2h = NdotH * NdotH;
		const sin2h = Math.max(1.0 - cos2h, 0.0078125);
		return (
			((2.0 + invAlpha) * Math.pow(sin2h, invAlpha * 0.5)) / (2.0 * Math.PI)
		);
	}

	private _VisibilityAshikhmin(NdotL: number, NdotV: number) {
		return 1.0 / (4.0 * (NdotL + NdotV - NdotL * NdotV));
	}

	private _FresnelSchlickScalar(cosTheta: number, F0: number) {
		const f = Math.pow(Math.max(1.0 - cosTheta, 0), 5.0);
		return F0 + (1.0 - F0) * f;
	}

	private _resolveIridescenceFresnel(
		cosTheta: number,
		baseF0R: number,
		baseF0G: number,
		baseF0B: number,
		iridescence: number,
		iridescenceIor: number,
		iridescenceThickness: number,
		out: RGB
	): RGB {
		const baseR = this._FresnelSchlickScalar(cosTheta, baseF0R);
		const baseG = this._FresnelSchlickScalar(cosTheta, baseF0G);
		const baseB = this._FresnelSchlickScalar(cosTheta, baseF0B);
		const strength = clamp(iridescence, 0.0, 1.0);
		if (strength <= 1e-6 || iridescenceThickness <= 0.0) {
			out.r = baseR;
			out.g = baseG;
			out.b = baseB;
			return out;
		}

		this._iridescentFresnel(
			cosTheta,
			baseF0R,
			baseF0G,
			baseF0B,
			iridescenceIor,
			iridescenceThickness,
			out
		);
		out.r = clamp(baseR * (1.0 - strength) + out.r * strength, 0.0, 1.0);
		out.g = clamp(baseG * (1.0 - strength) + out.g * strength, 0.0, 1.0);
		out.b = clamp(baseB * (1.0 - strength) + out.b * strength, 0.0, 1.0);
		return out;
	}

	private _iridescentFresnel(
		cosTheta1: number,
		baseF0R: number,
		baseF0G: number,
		baseF0B: number,
		iridescenceIor: number,
		iridescenceThickness: number,
		out: RGB
	): RGB {
		const outsideIor = 1.0;
		const filmIor = Math.max(iridescenceIor, 1e-6);
		const cos1 = clamp(cosTheta1, 0.0, 1.0);
		const eta = outsideIor / filmIor;
		const sinTheta2Sq = eta * eta * (1.0 - cos1 * cos1);
		if (sinTheta2Sq > 1.0) {
			out.r = 1.0;
			out.g = 1.0;
			out.b = 1.0;
			return out;
		}

		const cosTheta2 = Math.sqrt(Math.max(1.0 - sinTheta2Sq, 0.0));
		const r0 = this._iorToFresnel0(filmIor, outsideIor);
		const r12 = this._FresnelSchlickScalar(cos1, r0);
		const t121 = 1.0 - r12;

		const baseIorR = this._fresnel0ToIor(
			clamp(baseF0R + 0.0001, 0.0, 0.9999)
		);
		const baseIorG = this._fresnel0ToIor(
			clamp(baseF0G + 0.0001, 0.0, 0.9999)
		);
		const baseIorB = this._fresnel0ToIor(
			clamp(baseF0B + 0.0001, 0.0, 0.9999)
		);
		const r23R = this._FresnelSchlickScalar(
			cosTheta2,
			this._iorToFresnel0(baseIorR, filmIor)
		);
		const r23G = this._FresnelSchlickScalar(
			cosTheta2,
			this._iorToFresnel0(baseIorG, filmIor)
		);
		const r23B = this._FresnelSchlickScalar(
			cosTheta2,
			this._iorToFresnel0(baseIorB, filmIor)
		);

		const phi12 = filmIor < outsideIor ? Math.PI : 0.0;
		const phi21 = Math.PI - phi12;
		const phiR = phi21 + (baseIorR < filmIor ? Math.PI : 0.0);
		const phiG = phi21 + (baseIorG < filmIor ? Math.PI : 0.0);
		const phiB = phi21 + (baseIorB < filmIor ? Math.PI : 0.0);

		const opd = 2.0 * filmIor * iridescenceThickness * cosTheta2;
		const r123R = clamp(r12 * r23R, 1e-5, 0.9999);
		const r123G = clamp(r12 * r23G, 1e-5, 0.9999);
		const r123B = clamp(r12 * r23B, 1e-5, 0.9999);
		const sqrtR123R = Math.sqrt(r123R);
		const sqrtR123G = Math.sqrt(r123G);
		const sqrtR123B = Math.sqrt(r123B);
		const t121Sq = t121 * t121;
		const rsR = (t121Sq * r23R) / (1.0 - r123R);
		const rsG = (t121Sq * r23G) / (1.0 - r123G);
		const rsB = (t121Sq * r23B) / (1.0 - r123B);

		let irR = r12 + rsR;
		let irG = r12 + rsG;
		let irB = r12 + rsB;
		let cmR = rsR - t121;
		let cmG = rsG - t121;
		let cmB = rsB - t121;
		for (let order = 1; order <= 2; order++) {
			cmR *= sqrtR123R;
			cmG *= sqrtR123G;
			cmB *= sqrtR123B;
			const sensitivity = this._evalIridescenceSensitivity(
				order * opd,
				order * phiR,
				order * phiG,
				order * phiB,
				this._iridescenceSensitivity
			);
			irR += cmR * 2.0 * sensitivity.r;
			irG += cmG * 2.0 * sensitivity.g;
			irB += cmB * 2.0 * sensitivity.b;
		}

		out.r = Math.max(irR, 0.0);
		out.g = Math.max(irG, 0.0);
		out.b = Math.max(irB, 0.0);
		return out;
	}

	private _evalIridescenceSensitivity(
		opd: number,
		shiftR: number,
		shiftG: number,
		shiftB: number,
		out: RGB
	): RGB {
		const phase = 2.0 * Math.PI * opd * 1.0e-9;
		const phaseSq = phase * phase;
		const sqrtTwoPi = Math.sqrt(2.0 * Math.PI);
		let x =
			5.4856e-13 *
			sqrtTwoPi *
			Math.sqrt(4.3278e9) *
			Math.cos(1.6810e6 * phase + shiftR) *
			Math.exp(-4.3278e9 * phaseSq);
		const y =
			4.4201e-13 *
			sqrtTwoPi *
			Math.sqrt(9.3046e9) *
			Math.cos(1.7953e6 * phase + shiftG) *
			Math.exp(-9.3046e9 * phaseSq);
		const z =
			5.2481e-13 *
			sqrtTwoPi *
			Math.sqrt(6.6121e9) *
			Math.cos(2.2084e6 * phase + shiftB) *
			Math.exp(-6.6121e9 * phaseSq);
		x +=
			9.7470e-14 *
			sqrtTwoPi *
			Math.sqrt(4.5282e9) *
			Math.cos(2.2399e6 * phase + shiftR) *
			Math.exp(-4.5282e9 * phaseSq);

		const scale = 1.0 / 1.0685e-7;
		x *= scale;
		const scaledY = y * scale;
		const scaledZ = z * scale;
		out.r = 3.2404542 * x - 1.5371385 * scaledY - 0.4985314 * scaledZ;
		out.g = -0.9692660 * x + 1.8760108 * scaledY + 0.0415560 * scaledZ;
		out.b = 0.0556434 * x - 0.2040259 * scaledY + 1.0572252 * scaledZ;
		return out;
	}

	private _iorToFresnel0(transmittedIor: number, incidentIor: number): number {
		const denom = transmittedIor + incidentIor;
		if (Math.abs(denom) <= 1e-6) return 1.0;
		const value = (transmittedIor - incidentIor) / denom;
		return value * value;
	}

	private _fresnel0ToIor(f0: number): number {
		const sqrtF0 = Math.sqrt(clamp(f0, 0.0, 0.9999));
		return (1.0 + sqrtF0) / Math.max(1.0 - sqrtF0, 1e-6);
	}

	private _reflectViewDirection(
		N: IVector3,
		V: IVector3,
		NdotV: number,
		out: IVector3
	): IVector3 {
		out.x = 2 * NdotV * N.x - V.x;
		out.y = 2 * NdotV * N.y - V.y;
		out.z = 2 * NdotV * N.z - V.z;
		Vector3.normalizeInPlace(out);
		return out;
	}

	private _resolveAnisotropicReflectionDirection(
		N: IVector3,
		V: IVector3,
		B: IVector3,
		roughness: number,
		anisotropy: number,
		out: IVector3
	): IVector3 {
		const cx = B.y * V.z - B.z * V.y;
		const cy = B.z * V.x - B.x * V.z;
		const cz = B.x * V.y - B.y * V.x;
		const bent = this._anisotropicBentNormal;
		bent.x = cy * B.z - cz * B.y;
		bent.y = cz * B.x - cx * B.z;
		bent.z = cx * B.y - cy * B.x;
		Vector3.normalizeInPlace(bent);

		const oneMinusBent = 1.0 - anisotropy * (1.0 - roughness);
		const blendToNormal = oneMinusBent * oneMinusBent * oneMinusBent * oneMinusBent;
		bent.x = bent.x * (1.0 - blendToNormal) + N.x * blendToNormal;
		bent.y = bent.y * (1.0 - blendToNormal) + N.y * blendToNormal;
		bent.z = bent.z * (1.0 - blendToNormal) + N.z * blendToNormal;
		Vector3.normalizeInPlace(bent);

		const bentDotV = Vector3.dot(bent, V);
		out.x = 2 * bentDotV * bent.x - V.x;
		out.y = 2 * bentDotV * bent.y - V.y;
		out.z = 2 * bentDotV * bent.z - V.z;
		const roughnessSq = roughness * roughness;
		out.x = out.x * (1.0 - roughnessSq) + bent.x * roughnessSq;
		out.y = out.y * (1.0 - roughnessSq) + bent.y * roughnessSq;
		out.z = out.z * (1.0 - roughnessSq) + bent.z * roughnessSq;
		Vector3.normalizeInPlace(out);
		return out;
	}

	private _refract(
		V: IVector3,
		N: IVector3,
		ior: number,
		out: IVector3
	): boolean {
		const cosThetaI = Vector3.dot(V, N); // V points towards camera, so this is cosTheta with N
		let eta = 1.0;
		let nx = N.x;
		let ny = N.y;
		let nz = N.z;
		if (cosThetaI > 0) {
			eta = 1.0 / ior;
		} else {
			eta = ior;
			nx = -N.x;
			ny = -N.y;
			nz = -N.z;
		}

		const absCosThetaI = Math.abs(cosThetaI);
		const sin2ThetaT = eta * eta * (1.0 - absCosThetaI * absCosThetaI);
		if (sin2ThetaT > 1.0) return false; // Total internal reflection

		const cosThetaT = Math.sqrt(1.0 - sin2ThetaT);
		const refractScale = eta * absCosThetaI - cosThetaT;
		out.x = eta * -V.x + refractScale * nx;
		out.y = eta * -V.y + refractScale * ny;
		out.z = eta * -V.z + refractScale * nz;
		Vector3.normalizeInPlace(out);
		return true;
	}

	private _sampleSHRadiance(
		direction: IVector3,
		coeffs: SHCoefficients,
		out: RGB
	): RGB {
		const basis = this._evaluateSHBasis(direction);
		const count = Math.min(basis.length, coeffs.length);
		let r = 0,
			g = 0,
			b = 0;

		for (let i = 0; i < count; i++) {
			const weight = basis[i];
			r += coeffs[i].r * weight;
			g += coeffs[i].g * weight;
			b += coeffs[i].b * weight;
		}

		out.r = Math.max(0, r);
		out.g = Math.max(0, g);
		out.b = Math.max(0, b);
		return out;
	}

	private _calculateSHIrradiance(
		direction: IVector3,
		coeffs: SHCoefficients,
		out: RGB
	): RGB {
		const basis = this._evaluateSHBasis(direction);
		const count = Math.min(coeffs.length, SH_IRRADIANCE_FACTORS.length);
		let r = 0;
		let g = 0;
		let b = 0;

		for (let i = 0; i < count; i++) {
			const weight = basis[i] * SH_IRRADIANCE_FACTORS[i];
			r += coeffs[i].r * weight;
			g += coeffs[i].g * weight;
			b += coeffs[i].b * weight;
		}

		out.r = Math.max(0, r);
		out.g = Math.max(0, g);
		out.b = Math.max(0, b);
		return out;
	}

	private _evaluateSHBasis(direction: IVector3): Float32Array {
		return SH.evalBasis(direction, this._shBasis);
	}

	/**
	 * Geometry function for clearcoat using Schlick-GGX (Smith's method).
	 * Uses the isotropic remapping k = alpha^2 / 2.
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

	private _hasNonZeroSH(coeffs: SHCoefficients | null): boolean {
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

	private _sampleEnvironmentSpecular(
		worldPosition: IVector3,
		direction: IVector3,
		roughness: number,
		context: ShaderContext
	): RGB | null {
		return sampleReflectionProbesSpecular(
			worldPosition,
			direction,
			roughness,
			context.reflectionProbes ?? [],
			context.reflectionProbeFallbackMap ?? null
		);
	}
}
