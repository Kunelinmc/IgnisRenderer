#if WEBGL_SCENE_ENVIRONMENT_SPECULAR
vec2 sampleBRDFLUT(float nDotV, float roughness) {
	vec2 halfTexel = vec2(0.5) / vec2(textureSize(uBrdfLUT, 0));
	vec2 uv = vec2(clamp(nDotV, 0.0, 1.0), sqrt(clamp(roughness, 0.0, 1.0)));
	return textureLod(uBrdfLUT, clamp(uv, halfTexel, vec2(1.0) - halfTexel), 0.0).rg;
}
#endif

// Cosine-weighted isotropic thin-surface GGX transmission, matching WebGPU.
float evaluatePBRTransmissionLobe(
	vec3 normal,
	vec3 viewDirection,
	vec3 lightDirection,
	float roughness,
	float ior,
	out float vDotH
) {
	vDotH = 0.0;
	float nDotV = clamp(dot(normal, viewDirection), 0.0, 1.0);
	float nDotL = clamp(-dot(normal, lightDirection), 0.0, 1.0);
	if (nDotV <= 0.0 || nDotL <= 0.0) return 0.0;

	// Mirroring the incoming light puts the transmission lobe along the view ray.
	vec3 mirroredLight = reflect(lightDirection, normal);
	vec3 halfVector = safeNormalize(viewDirection + mirroredLight, normal);
	vDotH = clamp(dot(viewDirection, halfVector), 0.0, 1.0);
	if (vDotH <= 0.0 || dot(halfVector, lightDirection) >= 0.0) return 0.0;
	float nDotH = clamp(dot(normal, halfVector), 0.0, 1.0);
	float alpha = max(roughness * roughness * clamp(2.0 * ior - 2.0, 0.0, 1.0), 0.0016);
	float alphaSq = alpha * alpha;
	float nDotHSq = nDotH * nDotH;
	// Preserve the narrow GGX peak instead of clamping its squared denominator.
	float distributionRatio = alpha / ((1.0 - nDotHSq) + alphaSq * nDotHSq);
	float distribution = distributionRatio * distributionRatio / PI;
	float smithL = nDotL + sqrt(alphaSq + (1.0 - alphaSq) * nDotL * nDotL);
	float smithV = nDotV + sqrt(alphaSq + (1.0 - alphaSq) * nDotV * nDotV);
	return distribution * nDotL / (smithL * smithV);
}

vec3 resolvePBRShadowNormal(vec3 normal, vec3 lightDirection, float transmission) {
	// Transmission needs back-side occlusion and a bias toward the incident light.
	return transmission > 0.0 ? faceforward(normal, -lightDirection, normal) : normal;
}

vec3 evalPBRLight(
	vec3 albedo,
	vec3 pbrNormal,
	vec3 viewDir,
	vec3 lightDir,
	vec3 radiance,
	float roughness,
	float metalness,
	float transmission,
	float ior,
	vec3 f0,
	float nDotV,
	vec3 energyCompensation,
	vec3 volumeAttenuation,
	float clearcoat,
	float clearcoatRoughness,
	vec3 clearcoatNormal,
	vec3 sheenColor,
	float sheenRoughness,
	float anisotropyStrength,
	vec3 anisotropyTangent,
	vec3 anisotropyBitangent,
	float iridescence,
	float iridescenceThickness,
	float iridescenceIor
) {
	float nDotLRaw = dot(pbrNormal, lightDir);
	float nDotL = max(nDotLRaw, 0.0);
	float nDotLTransmission = max(-nDotLRaw, 0.0);
	if (nDotL <= 0.0 && nDotLTransmission <= 0.0) {
		return vec3(0.0);
	}
	vec3 transmitted = vec3(0.0);
	if (transmission > 0.0 && metalness < 1.0) {
		float transmissionVdotH;
		float lobe = evaluatePBRTransmissionLobe(
			pbrNormal, viewDir, lightDir, roughness, ior, transmissionVdotH
		);
		vec3 transmissionFresnel = resolveIridescenceFresnel(
			transmissionVdotH, f0, iridescence, iridescenceThickness, iridescenceIor
		);
		float clearcoatViewFresnel = clearcoat > 0.0 ?
			fresnelSchlickScalar(max(dot(clearcoatNormal, viewDir), PBR_MIN_NDOTV), 0.04)
			: 0.0;
		transmitted = (vec3(1.0) - transmissionFresnel) *
			((1.0 - metalness) * transmission * lobe) * volumeAttenuation * albedo *
			(1.0 - clearcoatViewFresnel * clearcoat) * radiance;
	}
	if (nDotL <= 0.0) return transmitted;

	vec3 halfVector = safeNormalize(viewDir + lightDir, viewDir);
	vec3 fresnel = resolveIridescenceFresnel(
		max(dot(halfVector, viewDir), 0.0),
		f0,
		iridescence,
		iridescenceThickness,
		iridescenceIor
	);
	vec3 specular;
#if WEBGL_MATERIAL_ANISOTROPY || WEBGL_MATERIAL_MODEL_FULL
	if (anisotropyStrength > EPSILON) {
		specular = resolveAnisotropicSpecular(
			fresnel,
			roughness,
			anisotropyStrength,
			nDotL,
			nDotV,
			max(dot(pbrNormal, halfVector), 0.0),
			dot(anisotropyTangent, viewDir),
			dot(anisotropyBitangent, viewDir),
			dot(anisotropyTangent, lightDir),
			dot(anisotropyBitangent, lightDir),
			dot(anisotropyTangent, halfVector),
			dot(anisotropyBitangent, halfVector)
		);
	} else {
		float ndf = distributionGGX(pbrNormal, halfVector, roughness);
		float geometry = geometrySmith(nDotV, nDotL, roughness);
		float denominator = max(4.0 * nDotV * nDotL, 0.0001);
		specular = (ndf * geometry * fresnel) / denominator;
	}
#else
	float ndf = distributionGGX(pbrNormal, halfVector, roughness);
	float geometry = geometrySmith(nDotV, nDotL, roughness);
	float denominator = max(4.0 * nDotV * nDotL, 0.0001);
	specular = (ndf * geometry * fresnel) / denominator;
#endif

	specular *= energyCompensation;
	vec3 kd =
		diffuseFresnelWeight(fresnel, iridescence) *
		(1.0 - metalness) *
		(1.0 - transmission);
	vec3 diffuse = (kd * albedo) / PI;
	float clearcoatFresnel = 0.0;
	vec3 clearcoatSpecular = vec3(0.0);
	if (clearcoat > 0.0) {
		float ncDotL = max(dot(clearcoatNormal, lightDir), 0.0);
		float ncDotV = max(dot(clearcoatNormal, viewDir), PBR_MIN_NDOTV);
		vec3 clearcoatHalf = safeNormalize(viewDir + lightDir, viewDir);
		clearcoatFresnel = fresnelSchlickScalar(
			max(dot(clearcoatHalf, viewDir), 0.0), 0.04
		);
		float clearcoatNdf = distributionGGX(
			clearcoatNormal, clearcoatHalf, clearcoatRoughness
		);
		float clearcoatGeometry = geometrySmithClearcoat(
			ncDotV, ncDotL, clearcoatRoughness
		);
		clearcoatSpecular = vec3(
			clearcoatNdf * clearcoatGeometry * clearcoatFresnel /
			max(4.0 * ncDotV * ncDotL, 0.0001)
		) * clearcoat;
	}
	vec3 sheenSpecular = vec3(0.0);
	vec3 sheenAttenuation = vec3(1.0);
	if (max(max(sheenColor.r, sheenColor.g), sheenColor.b) > EPSILON) {
		float nDotH = max(dot(pbrNormal, halfVector), 0.0);
		sheenSpecular = sheenColor *
			distributionCharlie(nDotH, max(sheenRoughness, 0.04)) *
			visibilityAshikhmin(nDotL, nDotV);
		sheenAttenuation = max(
			vec3(0.0), vec3(1.0) - fresnelSchlick(
				max(dot(halfVector, viewDir), 0.0), sheenColor
			)
		);
	}
	vec3 baseAttenuation =
		(vec3(1.0) - vec3(clearcoatFresnel * clearcoat)) * sheenAttenuation;
	vec3 reflected =
		((diffuse + specular) * baseAttenuation +
		clearcoatSpecular + sheenSpecular) * radiance * nDotL;
	return reflected + transmitted;
}

vec3 shadePBR(
	vec3 albedo,
	vec3 pbrNormal,
	vec3 shadowNormal,
	vec3 viewDir,
	float roughness,
	float metalness,
	float reflectance,
	float specularFactor,
	vec3 specularColor,
	float transmission,
	float resolvedThickness,
	float clearcoat,
	float clearcoatRoughness,
	vec3 clearcoatNormal,
	vec3 sheenColor,
	float sheenRoughness,
	float anisotropyStrength,
	vec3 anisotropyTangent,
	vec3 anisotropyBitangent,
	float iridescence,
	float iridescenceThickness,
	float iridescenceIor,
	float occlusion
) {
#if WEBGL_MATERIAL_TRANSMISSION || WEBGL_MATERIAL_MODEL_FULL
	float ior = max(uTransmissionVolume.x, 1.0);
	float thickness = max(resolvedThickness, 0.0);
	float attenuationDistance = uTransmissionVolume.z;
	vec3 attenuationColor = clamp(uAttenuationColor.rgb, vec3(0.0001), vec3(1.0));
#else
	float ior = 1.5;
	float thickness = 0.0;
	float attenuationDistance = -1.0;
	vec3 attenuationColor = vec3(1.0);
#endif
	vec3 volumeAttenuation = vec3(1.0);
	if (thickness > 0.0 && attenuationDistance > 0.0) {
		vec3 absorb = -log(attenuationColor) / attenuationDistance;
		volumeAttenuation = exp(-absorb * thickness);
	}
	vec3 dielectricF0 = vec3(0.16 * reflectance * reflectance) *
		specularColor * specularFactor;
	vec3 f0 = mix(dielectricF0, albedo, metalness);
	float nDotV = max(dot(pbrNormal, viewDir), PBR_MIN_NDOTV);
	vec3 energyCompensation = vec3(1.0);
#if WEBGL_SCENE_ENVIRONMENT_SPECULAR
	vec2 compensationBRDF = sampleBRDFLUT(nDotV, roughness);
	float singleScatterEnergy = compensationBRDF.x + compensationBRDF.y;
	if (singleScatterEnergy > 0.0 && singleScatterEnergy < 1.0) {
		energyCompensation += clamp(f0, vec3(0.0), vec3(1.0)) *
			(1.0 / singleScatterEnergy - 1.0);
	}
#endif
	vec3 reflectionDir;
#if WEBGL_MATERIAL_ANISOTROPY || WEBGL_MATERIAL_MODEL_FULL
	vec3 reflectionGeometricNormal = resolvePBRGeometricNormal(vWorldPos, viewDir, shadowNormal);
	if (anisotropyStrength > EPSILON) {
		reflectionDir = resolvePBRReflectionDirection(resolveAnisotropicReflectionDirection(
			pbrNormal,
			viewDir,
			anisotropyBitangent,
			roughness,
			anisotropyStrength
		), reflectionGeometricNormal);
	} else {
		reflectionDir = reflect(-viewDir, pbrNormal);
	}
#else
	reflectionDir = reflect(-viewDir, pbrNormal);
#endif
	ivec2 localProbeIndices = ivec2(-1);
	vec2 localProbeWeights = vec2(0.0);

	bool shAmbientEnabled = false;
	vec3 ambientBase = uAmbientColor;
	vec3 specularAmbientBase = vec3(0.0);
#if WEBGL_SCENE_SH
	if (uEnableSH == 1) {
		shAmbientEnabled = true;
		selectTopTwoLocalLightProbes(vWorldPos, localProbeIndices, localProbeWeights);
		ambientBase =
			sampleDiffuseProbeIrradiance(vWorldPos, pbrNormal) / 255.0;

		vec3 globalSpecularAmbientBase = sampleSHRadiance(reflectionDir);
		vec4 localSpecularAmbientBase = sampleBlendedLocalLightProbeRadiance(
			localProbeIndices,
			localProbeWeights,
			reflectionDir
		);
		specularAmbientBase = mix(
			globalSpecularAmbientBase,
			localSpecularAmbientBase.rgb,
			localSpecularAmbientBase.a
		) / 255.0;
	} else
#endif
	if (ambientBase.x + ambientBase.y + ambientBase.z <= 0.0) {
		ambientBase = vec3(0.0);
		specularAmbientBase = vec3(0.0);
	}
	vec3 ambientFresnel = resolveIridescenceFresnel(
		nDotV,
		f0,
		iridescence,
		iridescenceThickness,
		iridescenceIor
	);
	vec3 ambientDiffuseWeight =
		vec3((1.0 - metalness) * (1.0 - transmission));
	vec3 specularIBLWeight = ambientFresnel * max(PBR_SPEC_FALLBACK, (1.0 - roughness) * 0.5);
#if WEBGL_SCENE_ENVIRONMENT_SPECULAR
	if (uHasEnvSpecularMap == 1) {
		vec3 splitSumFresnel = iridescence > EPSILON ? ambientFresnel : f0;
		specularIBLWeight = resolveIBLSpecularEnergy(compensationBRDF, splitSumFresnel, energyCompensation);
	}
#endif
	if (shAmbientEnabled) {
		ambientDiffuseWeight =
			(vec3(1.0) - specularIBLWeight) *
			(1.0 - metalness) *
			(1.0 - transmission);
	}
	vec3 ambientDiffuse = (ambientBase / PI) * albedo * ambientDiffuseWeight;
	vec3 ambientSpecular;
#if WEBGL_SCENE_ENVIRONMENT_SPECULAR
	if (uHasEnvSpecularMap == 1) {
		vec3 prefiltered = sampleEnvironmentSpecular(
			vWorldPos,
			reflectionDir,
			roughness
		);
		ambientSpecular = prefiltered * specularIBLWeight;
	} else
#endif
	{
		float specularAmbientFactor = max(PBR_SPEC_FALLBACK, (1.0 - roughness) * 0.5);
		ambientSpecular = specularAmbientBase * ambientFresnel * specularAmbientFactor;
	}
	float ambientClearcoatFresnel = clearcoat > 0.0 ?
		fresnelSchlickScalar(
			max(dot(clearcoatNormal, viewDir), PBR_MIN_NDOTV), 0.04
		) : 0.0;
	vec3 ambientSheenFresnel = max(max(sheenColor.r, sheenColor.g), sheenColor.b) > 0.0 ?
		fresnelSchlick(nDotV, sheenColor) : vec3(0.0);
	float diffuseClearcoatFresnel = shAmbientEnabled ?
		ambientClearcoatFresnel : (clearcoat > 0.0 ? 0.04 : 0.0);
	vec3 ambientDiffuseAttenuation =
		(vec3(1.0) - vec3(diffuseClearcoatFresnel * clearcoat)) *
		(shAmbientEnabled ?
			max(vec3(0.0), vec3(1.0) - ambientSheenFresnel) :
			max(vec3(0.0), vec3(1.0) - sheenColor * 0.5));
	vec3 ambientSpecularAttenuation =
		(vec3(1.0) - vec3(ambientClearcoatFresnel * clearcoat)) *
		max(vec3(0.0), vec3(1.0) - ambientSheenFresnel);
	vec3 ambientClearcoat = specularAmbientBase *
		ambientClearcoatFresnel * clearcoat *
		max(PBR_SPEC_FALLBACK, (1.0 - clearcoatRoughness) * 0.5);
	vec3 ambientSheen = specularAmbientBase * sheenColor *
		(1.0 - 0.5 * sheenRoughness);
	vec3 ambientTransmission = vec3(0.0);
	if (transmission > EPSILON) {
		float cosThetaI = dot(viewDir, pbrNormal);
		bool outside = cosThetaI > 0.0;
		float eta = outside ? 1.0 / max(ior, 1.0) : ior;
		vec3 refractNormal = outside ? pbrNormal : -pbrNormal;
		vec3 transmissionDir = refract(-viewDir, refractNormal, eta);
		if (length(transmissionDir) > EPSILON) {
			vec3 transmissionRadiance = specularAmbientBase;
#if WEBGL_SCENE_ENVIRONMENT_SPECULAR
			if (uHasEnvSpecularMap == 1) {
				transmissionRadiance = sampleEnvironmentSpecular(
					vWorldPos,
					transmissionDir,
					roughness
				);
			}
#if WEBGL_SCENE_SH
			else if (uEnableSH == 1) {
				vec3 globalTransmissionRadiance = sampleSHRadiance(transmissionDir);
				vec4 localTransmissionRadiance = sampleBlendedLocalLightProbeRadiance(
					localProbeIndices,
					localProbeWeights,
					transmissionDir
				);
				transmissionRadiance = mix(
					globalTransmissionRadiance,
					localTransmissionRadiance.rgb,
					localTransmissionRadiance.a
				) / 255.0;
			}
#endif
#elif WEBGL_SCENE_SH
			if (uEnableSH == 1) {
				vec3 globalTransmissionRadiance = sampleSHRadiance(transmissionDir);
				vec4 localTransmissionRadiance = sampleBlendedLocalLightProbeRadiance(
					localProbeIndices,
					localProbeWeights,
					transmissionDir
				);
				transmissionRadiance = mix(
					globalTransmissionRadiance,
					localTransmissionRadiance.rgb,
					localTransmissionRadiance.a
				) / 255.0;
			}
#endif
			vec3 ambientTransmissionWeight =
				(vec3(1.0) - ambientFresnel) *
				(1.0 - metalness) *
				transmission;
			ambientTransmission =
				transmissionRadiance *
				albedo *
				ambientTransmissionWeight *
				volumeAttenuation;
		}
	}

	vec3 directLight = vec3(0.0);

	for (int i = 0; i < MAX_DIRECTIONAL_LIGHTS; i++) {
		if (i >= uDirLightCount) break;
		vec3 lightDir = safeNormalize(
			uDirLightDirection[i].xyz,
			vec3(0.0, 1.0, 0.0)
		);
		vec3 shadow = sampleDirectionalShadowVisibility(
			i,
			vWorldPos,
			resolvePBRShadowNormal(shadowNormal, lightDir, transmission),
			lightDir
		);
		directLight += evalPBRLight(
			albedo,
			pbrNormal,
			viewDir,
			lightDir,
			uDirLightColor[i].xyz,
			roughness,
			metalness,
			transmission,
			ior,
			f0,
			nDotV,
			energyCompensation,
			volumeAttenuation,
			clearcoat,
			clearcoatRoughness,
			clearcoatNormal,
			sheenColor,
			sheenRoughness,
			anisotropyStrength,
			anisotropyTangent,
			anisotropyBitangent,
			iridescence,
			iridescenceThickness,
			iridescenceIor
		) * shadow;
	}

#if WEBGL_SCENE_CLUSTERED_LIGHTING
	if (uEnableClusteredLighting == 1) {
		int clusterOffset = 0;
		int clusterCount = 0;
		int clusterMaxPer = 0;
		if (resolveClusterSpan(clusterOffset, clusterCount, clusterMaxPer)) {
			int clusterLimit = min(
				min(clusterCount, clusterMaxPer),
				MAX_CLUSTER_LIGHTS_PER_FRAGMENT
			);
			for (int i = 0; i < MAX_CLUSTER_LIGHTS_PER_FRAGMENT; i++) {
				if (i >= clusterLimit) break;
				int lightIndex = fetchClusterListLightIndex(clusterOffset + i);
				if (lightIndex < 0) {
					continue;
				}
				vec4 lightA = fetchClusterLightRow(lightIndex, 0);
				vec4 lightB = fetchClusterLightRow(lightIndex, 1);
				vec4 lightC = fetchClusterLightRow(lightIndex, 2);
				vec4 lightD = fetchClusterLightRow(lightIndex, 3);
				int lightType = int(floor(lightD.x + 0.5));

				vec3 toLight = lightA.xyz - vWorldPos;
				float distanceSq = dot(toLight, toLight);
				float distanceValue = sqrt(max(distanceSq, EPSILON));
				float lightRange = max(lightA.w, 0.001);
				if (distanceValue > lightRange) {
					continue;
				}
				vec3 lightDir = toLight / distanceValue;

				if (lightType == 0) {
					vec3 radiance = lightC.xyz * pointAttenuation(distanceSq, lightRange);
					directLight += evalPBRLight(
						albedo,
						pbrNormal,
						viewDir,
						lightDir,
						radiance,
						roughness,
						metalness,
						transmission,
						ior,
						f0,
						nDotV,
						energyCompensation,
						volumeAttenuation,
						clearcoat,
						clearcoatRoughness,
						clearcoatNormal,
						sheenColor,
						sheenRoughness,
						anisotropyStrength,
						anisotropyTangent,
						anisotropyBitangent,
						iridescence,
						iridescenceThickness,
						iridescenceIor
					);
				} else if (lightType == 1) {
					vec3 lightToPoint = -lightDir;
					vec3 coneDirection = safeNormalize(lightB.xyz, vec3(0.0, -1.0, 0.0));
					float coneFactor = spotAttenuation(
						dot(lightToPoint, coneDirection),
						lightB.w,
						lightC.w
					);
					if (coneFactor <= 0.0) {
						continue;
					}
					vec3 radiance = lightC.xyz *
						pointAttenuation(distanceSq, lightRange) *
						coneFactor;
					vec3 shadow = vec3(1.0);
					if (lightD.y > 0.5) {
						int shadowIndex = int(floor(lightD.z + 0.5));
						if (shadowIndex >= 0 && shadowIndex < MAX_SPOT_LIGHTS) {
							shadow = sampleSpotShadowVisibility(
								shadowIndex,
								vWorldPos,
								resolvePBRShadowNormal(shadowNormal, lightDir, transmission),
								lightDir
							);
						}
					}
					directLight += evalPBRLight(
						albedo,
						pbrNormal,
						viewDir,
						lightDir,
						radiance,
						roughness,
						metalness,
						transmission,
						ior,
						f0,
						nDotV,
						energyCompensation,
						volumeAttenuation,
						clearcoat,
						clearcoatRoughness,
						clearcoatNormal,
						sheenColor,
						sheenRoughness,
						anisotropyStrength,
						anisotropyTangent,
						anisotropyBitangent,
						iridescence,
						iridescenceThickness,
						iridescenceIor
					) * shadow;
				}
			}
		}
	} else
#endif
	{
		for (int i = 0; i < MAX_POINT_LIGHTS; i++) {
			if (i >= uPointLightCount) break;
			vec3 toLight = uPointLightPositionRange[i].xyz - vWorldPos;
			float distanceSq = dot(toLight, toLight);
			float distanceValue = sqrt(max(distanceSq, EPSILON));
			float lightRange = max(uPointLightPositionRange[i].w, 0.001);
			if (distanceValue > lightRange) {
				continue;
			}
			vec3 lightDir = toLight / distanceValue;
			vec3 radiance = uPointLightColor[i].xyz *
				pointAttenuation(distanceSq, lightRange);
			directLight += evalPBRLight(
				albedo,
				pbrNormal,
				viewDir,
				lightDir,
				radiance,
				roughness,
				metalness,
				transmission,
				ior,
				f0,
				nDotV,
				energyCompensation,
				volumeAttenuation,
				clearcoat,
				clearcoatRoughness,
				clearcoatNormal,
				sheenColor,
				sheenRoughness,
				anisotropyStrength,
				anisotropyTangent,
				anisotropyBitangent,
				iridescence,
				iridescenceThickness,
				iridescenceIor
			);
		}

		for (int i = 0; i < MAX_SPOT_LIGHTS; i++) {
			if (i >= uSpotLightCount) break;
			vec3 toLight = uSpotLightPositionRange[i].xyz - vWorldPos;
			float distanceSq = dot(toLight, toLight);
			float distanceValue = sqrt(max(distanceSq, EPSILON));
			float lightRange = max(uSpotLightPositionRange[i].w, 0.001);
			if (distanceValue > lightRange) {
				continue;
			}
			vec3 lightDir = toLight / distanceValue;
			vec3 lightToPoint = -lightDir;
			vec3 coneDirection = safeNormalize(
				uSpotLightDirectionOuter[i].xyz,
				vec3(0.0, -1.0, 0.0)
			);
			float coneFactor = spotAttenuation(
				dot(lightToPoint, coneDirection),
				uSpotLightDirectionOuter[i].w,
				uSpotLightColorInner[i].w
			);
			if (coneFactor <= 0.0) {
				continue;
			}
			vec3 radiance = uSpotLightColorInner[i].xyz *
				pointAttenuation(distanceSq, lightRange) *
				coneFactor;
			vec3 shadow = sampleSpotShadowVisibility(
				i,
				vWorldPos,
				resolvePBRShadowNormal(shadowNormal, lightDir, transmission),
				lightDir
			);
			directLight += evalPBRLight(
				albedo,
				pbrNormal,
				viewDir,
				lightDir,
				radiance,
				roughness,
				metalness,
				transmission,
				ior,
				f0,
				nDotV,
				energyCompensation,
				volumeAttenuation,
				clearcoat,
				clearcoatRoughness,
				clearcoatNormal,
				sheenColor,
				sheenRoughness,
				anisotropyStrength,
				anisotropyTangent,
				anisotropyBitangent,
				iridescence,
				iridescenceThickness,
				iridescenceIor
			) * shadow;
		}
	}

	vec3 ambient = (
		ambientDiffuse * ambientDiffuseAttenuation +
		ambientSpecular * ambientSpecularAttenuation +
		ambientClearcoat + ambientSheen + ambientTransmission
	) *
		clamp(occlusion, 0.0, 1.0);
	return ambient + directLight;
}
