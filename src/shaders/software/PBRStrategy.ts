import { Vector3 } from "../../maths/Vector3";
import { SH } from "../../maths/SH";
import { Texture } from "../../core/Texture";
import {
	isShadowCastingLight,
} from "../../lights";
import {
	createLightContribution,
	evaluateLightContribution,
	type SurfacePoint,
} from "../../renderers/software/LightEvaluator";
import { LightingConstants } from "../../core/constants";
import { clamp, sRGBToLinear } from "../../maths/Common";
import type { IVector3, SHCoefficients } from "../../maths/types";
import type { RGB } from "../../foundation/Color";
import type {
	ILightingStrategy,
	PBRSurfaceProperties,
	ShaderContext,
} from "./types";

/**
 * Cook-Torrance PBR lighting strategy.
 *
 * All material colors entering this strategy are already in linear space,
 * stored in the engine's usual 0..255 range by PBREvaluator. This method
 * normalizes them to 0..1, evaluates BRDF terms in linear space, and returns
 * linear 0..255 for the renderer's existing gamma pass.
 */
export class PBRStrategy implements ILightingStrategy<PBRSurfaceProperties> {
	private _surfacePoint: SurfacePoint = {
		position: { x: 0, y: 0, z: 0 },
		normal: { x: 0, y: 0, z: 1 },
	};
	private _lightContribution = createLightContribution();

	public calculate(
		world: IVector3,
		normal: IVector3,
		viewDir: IVector3,
		surface: PBRSurfaceProperties,
		context: ShaderContext
	): RGB {
		const N = normal;
		const V = viewDir;
		const shAmbient = context.shAmbientCoeffs;
		const hasSHAmbient = this._hasNonZeroSH(shAmbient);
		const NdotVRaw = Vector3.dot(N, V);
		const NdotV = Math.max(NdotVRaw, LightingConstants.PBR_MIN_NDOTV);
		const useSHAmbient = context.enableSH && hasSHAmbient;
		const reflectionDir = this._reflectViewDirection(N, V, NdotVRaw);

		let totalR = 0,
			totalG = 0,
			totalB = 0;
		let ambientLightR = 0,
			ambientLightG = 0,
			ambientLightB = 0;
		const surfacePoint = this._surfacePoint;
		surfacePoint.position.x = world.x;
		surfacePoint.position.y = world.y;
		surfacePoint.position.z = world.z;
		surfacePoint.normal!.x = N.x;
		surfacePoint.normal!.y = N.y;
		surfacePoint.normal!.z = N.z;

		const alb = {
			r: Math.max(0, surface.albedo.r / 255),
			g: Math.max(0, surface.albedo.g / 255),
			b: Math.max(0, surface.albedo.b / 255),
		};
		const metal = clamp(surface.metalness, 0.0, 1.0);
		const rough = clamp(surface.roughness, 0.04, 1.0);
		const occlusion = clamp(surface.occlusion, 0.0, 1.0);
		const clearcoat = clamp(surface.clearcoat, 0.0, 1.0);
		const clearcoatRoughness = clamp(surface.clearcoatRoughness, 0.04, 1.0);

		const sSheen = surface.sheenColor ?? { r: 0, g: 0, b: 0 };
		const sheenColor = {
			r: sSheen.r / 255,
			g: sSheen.g / 255,
			b: sSheen.b / 255,
		};
		const sheenRoughness = clamp(surface.sheenRoughness ?? 0.0, 0.04, 1.0);
		const maxSheenColor = Math.max(sheenColor.r, sheenColor.g, sheenColor.b);

		const transmission = clamp(surface.transmission ?? 0.0, 0.0, 1.0);
		const thickness = surface.thickness ?? 0.0;
		const attenuationDist = surface.attenuationDistance ?? Infinity;
		const sAtten = surface.attenuationColor ?? { r: 255, g: 255, b: 255 };
		const attenuationColor = {
			r: clamp(sAtten.r / 255, 0.0, 1.0),
			g: clamp(sAtten.g / 255, 0.0, 1.0),
			b: clamp(sAtten.b / 255, 0.0, 1.0),
		};

		let volumeAttenuation = { r: 1.0, g: 1.0, b: 1.0 };
		if (thickness > 0 && attenuationDist < Infinity && attenuationDist > 0) {
			const absorb = {
				r: -Math.log(attenuationColor.r) / attenuationDist,
				g: -Math.log(attenuationColor.g) / attenuationDist,
				b: -Math.log(attenuationColor.b) / attenuationDist,
			};
			volumeAttenuation = {
				r: Math.exp(-absorb.r * thickness),
				g: Math.exp(-absorb.g * thickness),
				b: Math.exp(-absorb.b * thickness),
			};
		}

		const reflectance = clamp(surface.reflectance, 0.0, 1.0);
		const baseF0Val = 0.16 * reflectance * reflectance;

		const specularFactor = surface.specularFactor ?? 1.0;
		const specColorInput = surface.specularColor ?? {
			r: 255,
			g: 255,
			b: 255,
		};
		const specColor = {
			r: specColorInput.r / 255,
			g: specColorInput.g / 255,
			b: specColorInput.b / 255,
		};

		const f0Norm = {
			r: Math.min(baseF0Val * specColor.r * specularFactor, 1.0),
			g: Math.min(baseF0Val * specColor.g * specularFactor, 1.0),
			b: Math.min(baseF0Val * specColor.b * specularFactor, 1.0),
		};

		const realF0 = {
			r: (1 - metal) * f0Norm.r + metal * alb.r,
			g: (1 - metal) * f0Norm.g + metal * alb.g,
			b: (1 - metal) * f0Norm.b + metal * alb.b,
		};

		const emissiveScale = surface.emissiveIntensity ?? 1.0;
		const emissive = {
			r: (surface.emissive.r / 255) * emissiveScale,
			g: (surface.emissive.g / 255) * emissiveScale,
			b: (surface.emissive.b / 255) * emissiveScale,
		};

		for (const light of context.lights) {
			const contrib = evaluateLightContribution(
				light,
				surfacePoint,
				this._lightContribution
			);
			if (!contrib) continue;
			const lightIntensity = contrib.intensity ?? 1.0;

			if (contrib.type === "ambient" || contrib.type === "irradiance") {
				if (!useSHAmbient) {
					ambientLightR += sRGBToLinear(contrib.color.r / 255) * lightIntensity;
					ambientLightG += sRGBToLinear(contrib.color.g / 255) * lightIntensity;
					ambientLightB += sRGBToLinear(contrib.color.b / 255) * lightIntensity;
				}
				continue;
			}

			const L = Vector3.normalize(contrib.direction);
			const NdotLRaw = Vector3.dot(N, L);
			const NdotL = Math.max(NdotLRaw, 0);
			const NdotLTransmission = Math.max(-NdotLRaw, 0);
			if (NdotL <= 0 && NdotLTransmission <= 0) continue;

			const radiance = {
				r: sRGBToLinear(contrib.color.r / 255) * lightIntensity,
				g: sRGBToLinear(contrib.color.g / 255) * lightIntensity,
				b: sRGBToLinear(contrib.color.b / 255) * lightIntensity,
			};

			let shadow = { r: 1, g: 1, b: 1 };
			if (
				context.enableShadows &&
				isShadowCastingLight(light) &&
				context.sampleShadow
			) {
				shadow = context.sampleShadow(light, world, N);
			}

			const Fview = this._FresnelSchlick(NdotV, realF0);
			const kT = {
				r: (1.0 - Fview.r) * (1.0 - metal) * transmission,
				g: (1.0 - Fview.g) * (1.0 - metal) * transmission,
				b: (1.0 - Fview.b) * (1.0 - metal) * transmission,
			};
			const transmittedDiffuse = {
				r: (kT.r * volumeAttenuation.r * alb.r) / Math.PI,
				g: (kT.g * volumeAttenuation.g * alb.g) / Math.PI,
				b: (kT.b * volumeAttenuation.b * alb.b) / Math.PI,
			};

			let specular = { r: 0, g: 0, b: 0 };
			let diffuse = { r: 0, g: 0, b: 0 };
			let clearcoatAttenuation = { r: 1.0, g: 1.0, b: 1.0 };
			let baseLayerAttenuation = { r: 1.0, g: 1.0, b: 1.0 };

			const Nc = surface.clearcoatNormal ?? N;
			const NcdotV = Math.max(
				Vector3.dot(Nc, V),
				LightingConstants.PBR_MIN_NDOTV
			);

			const clearcoatTransmissionFresnel =
				clearcoat > 0 ? this._FresnelSchlickScalar(NcdotV, 0.04) : 0;
			const transmissionAttenuation = {
				r: 1.0 - clearcoatTransmissionFresnel * clearcoat,
				g: 1.0 - clearcoatTransmissionFresnel * clearcoat,
				b: 1.0 - clearcoatTransmissionFresnel * clearcoat,
			};

			let ccSpecular = { r: 0, g: 0, b: 0 };
			let sheenSpecular = { r: 0, g: 0, b: 0 };

			if (NdotL > 0) {
				const H = Vector3.normalize(Vector3.add(L, V));
				const NDF = this._DistributionGGX(N, H, rough);
				const G = this._GeometrySmith(NdotV, NdotL, rough);
				const F = this._FresnelSchlick(Math.max(Vector3.dot(H, V), 0), realF0);
				const denominator =
					4 * NdotV * NdotL + LightingConstants.PBR_DENOM_EPSILON;

				specular = {
					r: (NDF * G * F.r) / denominator,
					g: (NDF * G * F.g) / denominator,
					b: (NDF * G * F.b) / denominator,
				};

				const kD = {
					r: (1 - F.r) * (1 - metal) * (1.0 - transmission),
					g: (1 - F.g) * (1 - metal) * (1.0 - transmission),
					b: (1 - F.b) * (1 - metal) * (1.0 - transmission),
				};
				diffuse = {
					r: (kD.r * alb.r) / Math.PI,
					g: (kD.g * alb.g) / Math.PI,
					b: (kD.b * alb.b) / Math.PI,
				};

				let ccFresnel = { r: 0, g: 0, b: 0 };
				if (clearcoat > 0) {
					const NcdotL = Math.max(Vector3.dot(Nc, L), 0);
					if (NcdotL > 0) {
						const Hcc = Vector3.normalize(Vector3.add(L, V));
						const HccdotV = Math.max(Vector3.dot(Hcc, V), 0);
						const ndfCc = this._DistributionGGX(Nc, Hcc, clearcoatRoughness);
						const gCc = this._GeometrySmithClearcoat(
							NcdotV,
							NcdotL,
							clearcoatRoughness
						);
						const fCc = this._FresnelSchlickScalar(HccdotV, 0.04);
						ccFresnel = { r: fCc, g: fCc, b: fCc };

						const ccDenom =
							4 * NcdotV * NcdotL + LightingConstants.PBR_DENOM_EPSILON;
						const ccValue = (ndfCc * gCc * fCc) / ccDenom;
						ccSpecular = { r: ccValue, g: ccValue, b: ccValue };
					}
				}

				let albedoSheenScaling = { r: 1.0, g: 1.0, b: 1.0 };
				if (maxSheenColor > 0) {
					const NdotH = Math.max(Vector3.dot(N, H), 0);
					const sheenNDF = this._DistributionCharlie(NdotH, sheenRoughness);
					const sheenV = this._VisibilityAshikhmin(NdotL, NdotV);
					sheenSpecular = {
						r: sheenColor.r * sheenNDF * sheenV,
						g: sheenColor.g * sheenNDF * sheenV,
						b: sheenColor.b * sheenNDF * sheenV,
					};

					const HdotV = Math.max(Vector3.dot(H, V), 0);
					const sheenFresnel = this._FresnelSchlick(HdotV, sheenColor);
					albedoSheenScaling = {
						r: Math.max(0, 1.0 - sheenFresnel.r),
						g: Math.max(0, 1.0 - sheenFresnel.g),
						b: Math.max(0, 1.0 - sheenFresnel.b),
					};
				}

				clearcoatAttenuation = {
					r: 1.0 - ccFresnel.r * clearcoat,
					g: 1.0 - ccFresnel.g * clearcoat,
					b: 1.0 - ccFresnel.b * clearcoat,
				};
				baseLayerAttenuation = {
					r: clearcoatAttenuation.r * albedoSheenScaling.r,
					g: clearcoatAttenuation.g * albedoSheenScaling.g,
					b: clearcoatAttenuation.b * albedoSheenScaling.b,
				};
			}

			totalR +=
				(((diffuse.r + specular.r) * baseLayerAttenuation.r +
					ccSpecular.r * clearcoat +
					sheenSpecular.r * clearcoatAttenuation.r) *
					NdotL +
					transmittedDiffuse.r *
						transmissionAttenuation.r *
						NdotLTransmission) *
				radiance.r *
				shadow.r;
			totalG +=
				(((diffuse.g + specular.g) * baseLayerAttenuation.g +
					ccSpecular.g * clearcoat +
					sheenSpecular.g * clearcoatAttenuation.g) *
					NdotL +
					transmittedDiffuse.g *
						transmissionAttenuation.g *
						NdotLTransmission) *
				radiance.g *
				shadow.g;
			totalB +=
				(((diffuse.b + specular.b) * baseLayerAttenuation.b +
					ccSpecular.b * clearcoat +
					sheenSpecular.b * clearcoatAttenuation.b) *
					NdotL +
					transmittedDiffuse.b *
						transmissionAttenuation.b *
						NdotLTransmission) *
				radiance.b *
				shadow.b;
		}

		let ambR = 0,
			ambG = 0,
			ambB = 0;

		if (useSHAmbient && shAmbient) {
			const irr = SH.calculateIrradiance(N, shAmbient);
			const irrLinear = {
				r: irr.r / 255,
				g: irr.g / 255,
				b: irr.b / 255,
			};
			const specRadiance = this._sampleSHRadiance(reflectionDir, shAmbient);
			const specRadianceLinear = {
				r: specRadiance.r / 255,
				g: specRadiance.g / 255,
				b: specRadiance.b / 255,
			};

			const Famb = this._FresnelSchlick(NdotV, realF0);
			const refractionDir =
				transmission > 0 ? this._refract(V, N, surface.ior) : null;

			// Handle Total Internal Reflection (TIR)
			const isTIR = transmission > 0 && refractionDir === null;
			const effectiveFamb = isTIR ? { r: 1, g: 1, b: 1 } : Famb;

			const kDamb = {
				r: (1.0 - effectiveFamb.r) * (1.0 - metal) * (1.0 - transmission),
				g: (1.0 - effectiveFamb.g) * (1.0 - metal) * (1.0 - transmission),
				b: (1.0 - effectiveFamb.b) * (1.0 - metal) * (1.0 - transmission),
			};
			const kTamb = {
				r: (1.0 - effectiveFamb.r) * (1.0 - metal) * transmission,
				g: (1.0 - effectiveFamb.g) * (1.0 - metal) * transmission,
				b: (1.0 - effectiveFamb.b) * (1.0 - metal) * transmission,
			};

			let ccAmbFresnel = 0;
			let ccAmbSpecR = 0;
			let ccAmbSpecG = 0;
			let ccAmbSpecB = 0;

			if (clearcoat > 0) {
				const Nc = surface.clearcoatNormal ?? N;
				const NcdotV = Math.max(
					Vector3.dot(Nc, V),
					LightingConstants.PBR_MIN_NDOTV
				);
				ccAmbFresnel = this._FresnelSchlickScalar(NcdotV, 0.04);
				if (context.envSpecularMap && context.brdfLUT) {
					const ccReflectionDir = this._reflectViewDirection(
						Nc,
						V,
						Vector3.dot(Nc, V)
					);
					const ccPrefiltered = this._samplePrefiltered(
						ccReflectionDir,
						clearcoatRoughness,
						context.envSpecularMap
					);
					const ccBrdf = context.brdfLUT.sample(
						NcdotV,
						Math.sqrt(clearcoatRoughness)
					);
					ccAmbSpecR = ccPrefiltered.r * (ccAmbFresnel * ccBrdf.r + ccBrdf.g);
					ccAmbSpecG = ccPrefiltered.g * (ccAmbFresnel * ccBrdf.r + ccBrdf.g);
					ccAmbSpecB = ccPrefiltered.b * (ccAmbFresnel * ccBrdf.r + ccBrdf.g);
				} else {
					const ccSpecFactor = Math.max(
						LightingConstants.PBR_SPEC_FALLBACK,
						(1.0 - clearcoatRoughness) * 0.5
					);
					const ccSpec = ccAmbFresnel * ccSpecFactor;
					ccAmbSpecR = specRadianceLinear.r * ccSpec;
					ccAmbSpecG = specRadianceLinear.g * ccSpec;
					ccAmbSpecB = specRadianceLinear.b * ccSpec;
				}
			}

			const clearcoatAttenuationAmb = 1.0 - ccAmbFresnel * clearcoat;
			const baseAttenuationAmb = {
				r: clearcoatAttenuationAmb * (1.0 - sheenColor.r * 0.5),
				g: clearcoatAttenuationAmb * (1.0 - sheenColor.g * 0.5),
				b: clearcoatAttenuationAmb * (1.0 - sheenColor.b * 0.5),
			};

			ambR = irrLinear.r * alb.r * kDamb.r * baseAttenuationAmb.r;
			ambG = irrLinear.g * alb.g * kDamb.g * baseAttenuationAmb.g;
			ambB = irrLinear.b * alb.b * kDamb.b * baseAttenuationAmb.b;

			if (transmission > 0 && refractionDir) {
				const transmRadiance = context.envSpecularMap
					? this._samplePrefiltered(
							refractionDir,
							rough,
							context.envSpecularMap
						)
					: this._sampleSHRadiance(refractionDir, shAmbient!);

				ambR +=
					transmRadiance.r *
					alb.r *
					kTamb.r *
					volumeAttenuation.r *
					clearcoatAttenuationAmb;
				ambG +=
					transmRadiance.g *
					alb.g *
					kTamb.g *
					volumeAttenuation.g *
					clearcoatAttenuationAmb;
				ambB +=
					transmRadiance.b *
					alb.b *
					kTamb.b *
					volumeAttenuation.b *
					clearcoatAttenuationAmb;
			}

			if (context.envSpecularMap && context.brdfLUT) {
				const prefiltered = this._samplePrefiltered(
					reflectionDir,
					rough,
					context.envSpecularMap
				);
				// LUT stores scale at R (red), bias at G (green)
				const brdf = context.brdfLUT.sample(NdotV, Math.sqrt(rough));
				const specR =
					prefiltered.r *
					(effectiveFamb.r * brdf.r + brdf.g) *
					clearcoatAttenuationAmb;
				const specG =
					prefiltered.g *
					(effectiveFamb.g * brdf.r + brdf.g) *
					clearcoatAttenuationAmb;
				const specB =
					prefiltered.b *
					(effectiveFamb.b * brdf.r + brdf.g) *
					clearcoatAttenuationAmb;

				ambR += specR + ccAmbSpecR * clearcoat;
				ambG += specG + ccAmbSpecG * clearcoat;
				ambB += specB + ccAmbSpecB * clearcoat;
			} else {
				const specFactor = Math.max(
					LightingConstants.PBR_SPEC_FALLBACK,
					(1.0 - rough) * 0.5
				);
				ambR +=
					specRadianceLinear.r *
						effectiveFamb.r *
						specFactor *
						clearcoatAttenuationAmb +
					ccAmbSpecR * clearcoat;
				ambG +=
					specRadianceLinear.g *
						effectiveFamb.g *
						specFactor *
						clearcoatAttenuationAmb +
					ccAmbSpecG * clearcoat;
				ambB +=
					specRadianceLinear.b *
						effectiveFamb.b *
						specFactor *
						clearcoatAttenuationAmb +
					ccAmbSpecB * clearcoat;
			}

			if (maxSheenColor > 0) {
				const sheenAmb = Math.max(
					LightingConstants.PBR_SPEC_FALLBACK,
					(1.0 - sheenRoughness) * 0.5
				);
				ambR +=
					specRadianceLinear.r *
					sheenColor.r *
					sheenAmb *
					clearcoatAttenuationAmb;
				ambG +=
					specRadianceLinear.g *
					sheenColor.g *
					sheenAmb *
					clearcoatAttenuationAmb;
				ambB +=
					specRadianceLinear.b *
					sheenColor.b *
					sheenAmb *
					clearcoatAttenuationAmb;
			}
		} else {
			const ambientCol = {
				r: ambientLightR,
				g: ambientLightG,
				b: ambientLightB,
			};
			if (ambientLightR + ambientLightG + ambientLightB === 0) {
				const fallback = LightingConstants.PBR_AMBIENT_FALLBACK_LINEAR;
				ambientCol.r = fallback;
				ambientCol.g = fallback;
				ambientCol.b = fallback;
			}

			const ambientRadiance = {
				r: ambientCol.r / Math.PI,
				g: ambientCol.g / Math.PI,
				b: ambientCol.b / Math.PI,
			};
			const Famb = this._FresnelSchlick(NdotV, realF0);
			const refractionDir =
				transmission > 0 ? this._refract(V, N, surface.ior) : null;
			const isTIR = transmission > 0 && refractionDir === null;
			const effectiveFamb = isTIR ? { r: 1, g: 1, b: 1 } : Famb;

			const kDamb = {
				r: (1.0 - effectiveFamb.r) * (1.0 - metal) * (1.0 - transmission),
				g: (1.0 - effectiveFamb.g) * (1.0 - metal) * (1.0 - transmission),
				b: (1.0 - effectiveFamb.b) * (1.0 - metal) * (1.0 - transmission),
			};
			const kTamb = {
				r: (1.0 - effectiveFamb.r) * (1.0 - metal) * transmission,
				g: (1.0 - effectiveFamb.g) * (1.0 - metal) * transmission,
				b: (1.0 - effectiveFamb.b) * (1.0 - metal) * transmission,
			};

			const ccAmbFresnel =
				clearcoat > 0 ? this._FresnelSchlickScalar(NdotV, 0.04) : 0;
			const clearcoatAttenuationAmb = 1.0 - ccAmbFresnel * clearcoat;
			const baseAttenuationAmb = {
				r: clearcoatAttenuationAmb * (1.0 - sheenColor.r * 0.5),
				g: clearcoatAttenuationAmb * (1.0 - sheenColor.g * 0.5),
				b: clearcoatAttenuationAmb * (1.0 - sheenColor.b * 0.5),
			};

			ambR = ambientCol.r * alb.r * kDamb.r * baseAttenuationAmb.r;
			ambG = ambientCol.g * alb.g * kDamb.g * baseAttenuationAmb.g;
			ambB = ambientCol.b * alb.b * kDamb.b * baseAttenuationAmb.b;

			if (transmission > 0 && refractionDir) {
				// Fallback: use SH/Ambient color for refraction if no map, but sample SH if possible?
				// Here we just use the ambient color tilted by albedo.
				// However, if we have a skybox, maybe we can sample it?
				// But fallback usually means we don't have high-res maps.
				ambR +=
					ambientCol.r *
					alb.r *
					kTamb.r *
					volumeAttenuation.r *
					clearcoatAttenuationAmb;
				ambG +=
					ambientCol.g *
					alb.g *
					kTamb.g *
					volumeAttenuation.g *
					clearcoatAttenuationAmb;
				ambB +=
					ambientCol.b *
					alb.b *
					kTamb.b *
					volumeAttenuation.b *
					clearcoatAttenuationAmb;
			}

			const specFactor = Math.max(
				LightingConstants.PBR_SPEC_FALLBACK,
				(1.0 - rough) * 0.5
			);
			const ccSpecFactor = Math.max(
				LightingConstants.PBR_SPEC_FALLBACK,
				(1.0 - clearcoatRoughness) * 0.5
			);

			ambR +=
				ambientRadiance.r *
					effectiveFamb.r *
					specFactor *
					clearcoatAttenuationAmb +
				ambientRadiance.r * ccAmbFresnel * ccSpecFactor * clearcoat;
			ambG +=
				ambientRadiance.g *
					effectiveFamb.g *
					specFactor *
					clearcoatAttenuationAmb +
				ambientRadiance.g * ccAmbFresnel * ccSpecFactor * clearcoat;
			ambB +=
				ambientRadiance.b *
					effectiveFamb.b *
					specFactor *
					clearcoatAttenuationAmb +
				ambientRadiance.b * ccAmbFresnel * ccSpecFactor * clearcoat;

			if (maxSheenColor > 0) {
				const sheenAmb = Math.max(
					LightingConstants.PBR_SPEC_FALLBACK,
					(1.0 - sheenRoughness) * 0.5
				);
				ambR +=
					ambientRadiance.r * sheenColor.r * sheenAmb * clearcoatAttenuationAmb;
				ambG +=
					ambientRadiance.g * sheenColor.g * sheenAmb * clearcoatAttenuationAmb;
				ambB +=
					ambientRadiance.b * sheenColor.b * sheenAmb * clearcoatAttenuationAmb;
			}
		}

		ambR *= occlusion;
		ambG *= occlusion;
		ambB *= occlusion;

		const finalR = Math.max(0, totalR + ambR + emissive.r);
		const finalG = Math.max(0, totalG + ambG + emissive.g);
		const finalB = Math.max(0, totalB + ambB + emissive.b);

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
	 * Uses the direct lighting remapping: k = (roughness + 1)^2 / 8.
	 */
	private _GeometrySmith(NdotV: number, NdotL: number, roughness: number) {
		const r = roughness + 1.0;
		const k = (r * r) / 8.0;
		const G1V = NdotV / (NdotV * (1.0 - k) + k);
		const G1L = NdotL / (NdotL * (1.0 - k) + k);
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

	private _reflectViewDirection(
		N: IVector3,
		V: IVector3,
		NdotV: number
	): IVector3 {
		const reflected = {
			x: 2 * NdotV * N.x - V.x,
			y: 2 * NdotV * N.y - V.y,
			z: 2 * NdotV * N.z - V.z,
		};
		Vector3.normalizeInPlace(reflected);
		return reflected;
	}

	private _refract(V: IVector3, N: IVector3, ior: number): IVector3 | null {
		const cosThetaI = Vector3.dot(V, N); // V points towards camera, so this is cosTheta with N
		let eta, n;
		if (cosThetaI > 0) {
			// Outside
			eta = 1.0 / ior;
			n = N;
		} else {
			// Inside
			eta = ior;
			n = { x: -N.x, y: -N.y, z: -N.z };
		}

		const absCosThetaI = Math.abs(cosThetaI);
		const sin2ThetaT = eta * eta * (1.0 - absCosThetaI * absCosThetaI);
		if (sin2ThetaT > 1.0) return null; // Total internal reflection

		const cosThetaT = Math.sqrt(1.0 - sin2ThetaT);
		const refraction = {
			x: eta * -V.x + (eta * absCosThetaI - cosThetaT) * n.x,
			y: eta * -V.y + (eta * absCosThetaI - cosThetaT) * n.y,
			z: eta * -V.z + (eta * absCosThetaI - cosThetaT) * n.z,
		};
		Vector3.normalizeInPlace(refraction);
		return refraction;
	}

	private _sampleSHRadiance(direction: IVector3, coeffs: SHCoefficients): RGB {
		const basis = SH.evalBasis(direction);
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

		return {
			r: Math.max(0, r),
			g: Math.max(0, g),
			b: Math.max(0, b),
		};
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

	private _samplePrefiltered(
		R: IVector3,
		roughness: number,
		envMap: Texture
	): RGB {
		const phi = Math.atan2(R.x, R.z);
		const theta = Math.acos(Math.max(-1, Math.min(1, R.y)));
		const u = (phi + Math.PI) / (2 * Math.PI);
		const v = theta / Math.PI;

		const mipCount = envMap.mipmaps.length;
		const level = roughness * (mipCount - 1);

		const sample = envMap.sampleLevel(u, v, level);
		return {
			r: sample.r / 255,
			g: sample.g / 255,
			b: sample.b / 255,
		};
	}
}
