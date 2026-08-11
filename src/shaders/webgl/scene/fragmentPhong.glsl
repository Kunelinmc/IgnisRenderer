vec3 evaluateNormalizedPhongLight(
	vec3 albedo,
	vec3 f0,
	vec3 n,
	vec3 v,
	vec3 l,
	vec3 radiance,
	float shininess
) {
	float nDotL = max(dot(n, l), 0.0);
	if (nDotL <= 0.0) {
		return vec3(0.0);
	}
	vec3 h = safeNormalize(l + v, v);
	float vDotH = max(dot(v, h), 0.0);
	vec3 fresnel = f0 + (vec3(1.0) - f0) * pow(1.0 - vDotH, 5.0);
	vec3 diffuseBRDF = (vec3(1.0) - fresnel) * albedo / PI;
	float normalizedLobe =
		((shininess + 8.0) / (8.0 * PI)) *
		pow(max(dot(n, h), 0.0), shininess);
	vec3 specularBRDF = fresnel * normalizedLobe;
	return (diffuseBRDF + specularBRDF) * radiance * nDotL;
}

vec3 shadePhong(vec3 albedo, vec3 n, vec3 shadowNormal, vec3 v) {
	vec3 ambientBase = uAmbientColor;
#if WEBGL_SCENE_SH
	if (uEnableSH == 1) {
		ambientBase = sampleDiffuseProbeIrradiance(vWorldPos, n) / 255.0;
	}
#endif
	vec3 lit = ambientBase * uPhongAmbient.rgb / PI;
	vec3 f0 = clamp(uSpecular.rgb, vec3(0.0), vec3(1.0));
	float shininess = max(0.0, uPhong.x);

	for (int i = 0; i < MAX_DIRECTIONAL_LIGHTS; i++) {
		if (i >= uDirLightCount) break;
		vec3 l = safeNormalize(uDirLightDirection[i].xyz, vec3(0.0, 1.0, 0.0));
		vec3 shadow = sampleDirectionalShadowVisibility(
			i,
			vWorldPos,
			shadowNormal,
			l
		);
		lit += evaluateNormalizedPhongLight(
			albedo, f0, n, v, l, uDirLightColor[i].xyz, shininess
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
				vec3 l = safeNormalize(toLight, vec3(0.0, 1.0, 0.0));
				float attenuation = pointAttenuation(distanceSq, lightRange);
				if (lightType == 0) {
					lit += evaluateNormalizedPhongLight(
						albedo, f0, n, v, l, lightC.xyz * attenuation, shininess
					);
				} else if (lightType == 1) {
					vec3 lightToPoint = -l;
					vec3 coneDirection = safeNormalize(lightB.xyz, vec3(0.0, -1.0, 0.0));
					float coneFactor = spotAttenuation(
						dot(lightToPoint, coneDirection),
						lightB.w,
						lightC.w
					);
					if (coneFactor <= 0.0) {
						continue;
					}
					vec3 shadow = vec3(1.0);
					if (lightD.y > 0.5) {
						int shadowIndex = int(floor(lightD.z + 0.5));
						if (shadowIndex >= 0 && shadowIndex < MAX_SPOT_LIGHTS) {
							shadow = sampleSpotShadowVisibility(
								shadowIndex,
								vWorldPos,
								shadowNormal,
								l
							);
						}
					}
					lit += evaluateNormalizedPhongLight(
						albedo, f0, n, v, l,
						lightC.xyz * attenuation * coneFactor, shininess
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
			vec3 l = safeNormalize(toLight, vec3(0.0, 1.0, 0.0));
			float attenuation = pointAttenuation(distanceSq, lightRange);
			lit += evaluateNormalizedPhongLight(
				albedo, f0, n, v, l,
				uPointLightColor[i].xyz * attenuation, shininess
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
			vec3 l = safeNormalize(toLight, vec3(0.0, 1.0, 0.0));
			float attenuation = pointAttenuation(distanceSq, lightRange);
			vec3 lightToPoint = -l;
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
			vec3 shadow = sampleSpotShadowVisibility(
				i,
				vWorldPos,
				shadowNormal,
				l
			);
			lit += evaluateNormalizedPhongLight(
				albedo, f0, n, v, l,
				uSpotLightColorInner[i].xyz * attenuation * coneFactor,
				shininess
			) * shadow;
		}
	}

	return lit;
}
