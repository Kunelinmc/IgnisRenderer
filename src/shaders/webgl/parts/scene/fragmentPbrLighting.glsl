vec3 evalPBRLight(
	vec3 albedo,
	vec3 pbrNormal,
	vec3 viewDir,
	vec3 lightDir,
	vec3 radiance,
	float roughness,
	float metalness,
	float transmission,
	vec3 f0,
	float nDotV,
	float anisotropyStrength,
	vec3 anisotropyTangent,
	vec3 anisotropyBitangent,
	float iridescence,
	float iridescenceThickness,
	float iridescenceIor
) {
	float nDotL = max(dot(pbrNormal, lightDir), 0.0);
	if (nDotL <= 0.0) {
		return vec3(0.0);
	}

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

	vec3 kd =
		diffuseFresnelWeight(fresnel, iridescence) *
		(1.0 - metalness) *
		(1.0 - transmission);
	vec3 diffuse = (kd * albedo) / PI;
	return (diffuse + specular) * radiance * nDotL;
}

vec3 shadePBR(
	vec3 albedo,
	vec3 pbrNormal,
	vec3 shadowNormal,
	vec3 viewDir,
	float roughness,
	float metalness,
	float reflectance,
	float transmission,
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
	float thickness = max(uTransmissionVolume.y, 0.0);
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
	float dielectricF0 = 0.16 * reflectance * reflectance;
	vec3 f0 = mix(vec3(dielectricF0), albedo, metalness);
	float nDotV = max(dot(pbrNormal, viewDir), PBR_MIN_NDOTV);
	vec3 reflectionDir;
#if WEBGL_MATERIAL_ANISOTROPY || WEBGL_MATERIAL_MODEL_FULL
	if (anisotropyStrength > EPSILON) {
		reflectionDir = resolveAnisotropicReflectionDirection(
			pbrNormal,
			viewDir,
			anisotropyBitangent,
			roughness,
			anisotropyStrength
		);
	} else {
		reflectionDir = reflect(-viewDir, pbrNormal);
	}
#else
	reflectionDir = reflect(-viewDir, pbrNormal);
#endif
	ivec2 localProbeIndices = ivec2(-1);
	vec2 localProbeWeights = vec2(0.0);

	vec3 ambientBase = uAmbientColor;
	vec3 specularAmbientBase = ambientBase;
#if WEBGL_SCENE_SH
	if (uEnableSH == 1) {
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
		ambientBase = vec3(PBR_AMBIENT_FALLBACK_LINEAR);
		specularAmbientBase = ambientBase;
	}
	vec3 ambientFresnel = resolveIridescenceFresnel(
		nDotV,
		f0,
		iridescence,
		iridescenceThickness,
		iridescenceIor
	);
	vec3 ambientDiffuse = ambientBase *
		albedo *
		diffuseFresnelWeight(ambientFresnel, iridescence) *
		(1.0 - metalness) *
		(1.0 - transmission);
	vec3 ambientSpecular;
#if WEBGL_SCENE_ENVIRONMENT_SPECULAR
	if (uHasEnvSpecularMap == 1) {
		vec3 prefiltered = sampleEnvironmentSpecular(
			vWorldPos,
			reflectionDir,
			roughness
		);
		vec2 brdf = texture(
			uBrdfLUT,
			vec2(clamp(nDotV, 0.0, 1.0), sqrt(roughness))
		).rg;
		ambientSpecular = prefiltered * (ambientFresnel * brdf.x + vec3(brdf.y));
	} else
#endif
	{
		float specularAmbientFactor = max(PBR_SPEC_FALLBACK, (1.0 - roughness) * 0.5);
		ambientSpecular = specularAmbientBase * ambientFresnel * specularAmbientFactor;
	}
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
			shadowNormal,
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
			f0,
			nDotV,
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
						f0,
						nDotV,
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
								shadowNormal,
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
						f0,
						nDotV,
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
				f0,
				nDotV,
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
				shadowNormal,
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
				f0,
				nDotV,
				anisotropyStrength,
				anisotropyTangent,
				anisotropyBitangent,
				iridescence,
				iridescenceThickness,
				iridescenceIor
			) * shadow;
		}
	}

	vec3 ambient = (ambientDiffuse + ambientSpecular + ambientTransmission) *
		clamp(occlusion, 0.0, 1.0);
	return ambient + directLight;
}
