const float SHADOW_GOLDEN_ANGLE = 2.39996323;
const int MAX_PCSS_FILTER_SAMPLES = 64;
const int MAX_PCSS_SEARCH_SAMPLES = 64;

float hashShadowRotation(vec3 position) {
	return
		fract(
			sin(dot(position, vec3(12.9898, 78.233, 37.719))) * 43758.5453123
		) *
		(2.0 * PI);
}

vec2 vogelDiskSample(int sampleIndex, int sampleCount, float theta) {
	float indexF = float(sampleIndex);
	float countF = max(float(sampleCount), 1.0);
	float radius = sqrt((indexF + 0.5) / countF);
	float angle = indexF * SHADOW_GOLDEN_ANGLE + theta;
	return vec2(cos(angle), sin(angle)) * radius;
}

float sampleParticleShadowVolumeTransmittance(
	int shadowType,
	int index,
	int cascadeIndex,
	vec3 worldPosition
) {
	if (shadowType != 0 || index != 0 || cascadeIndex < 0 || cascadeIndex >= 4) {
		return 1.0;
	}

	vec4 sliceParams = uParticleShadowVolumeSliceParams[cascadeIndex];
	if (sliceParams.x < 0.5 || uParticleShadowVolumeAtlasSize.x <= 0.0) {
		return 1.0;
	}

	float gridWidth = max(uParticleShadowVolumeGridSize.x, 1.0);
	float gridHeight = max(uParticleShadowVolumeGridSize.y, 1.0);
	int gridDepth = int(clamp(floor(uParticleShadowVolumeGridSize.z + 0.5), 1.0, 64.0));
	float tileColumns = max(uParticleShadowVolumeGridSize.w, 1.0);
	bool isCSM =
		uDirShadowParamsC[index].y > 0.5 &&
		uDirShadowParamsC[index].z > 1.5;
	mat4 volumeViewProjection = isCSM ?
		uDirShadowCascadeViewProjection[index * 4 + cascadeIndex] :
		uDirShadowViewProjection[index];
	vec4 clip = volumeViewProjection * vec4(worldPosition, 1.0);
	if (clip.w <= EPSILON) {
		return 1.0;
	}

	vec3 ndc = clip.xyz / clip.w;
	if (
		ndc.x < -1.0 || ndc.x > 1.0 ||
		ndc.y < -1.0 || ndc.y > 1.0 ||
		ndc.z < -1.0 || ndc.z > 1.0
	) {
		return 1.0;
	}

	float vx = clamp(round((ndc.x * 0.5 + 0.5) * (gridWidth - 1.0)), 0.0, gridWidth - 1.0);
	float vy = clamp(round((0.5 - ndc.y * 0.5) * (gridHeight - 1.0)), 0.0, gridHeight - 1.0);
	int zMax = int(clamp(
		round((ndc.z * 0.5 + 0.5) * float(gridDepth - 1)),
		0.0,
		float(gridDepth - 1)
	));
	float opticalDepth = 0.0;
	for (int z = 0; z < 64; z++) {
		if (z > zMax) {
			break;
		}
		float tileIndex = sliceParams.y + float(z);
		float tileX = mod(tileIndex, tileColumns);
		float tileY = floor(tileIndex / tileColumns);
		vec2 atlasUv = (
			vec2(tileX * gridWidth + vx, tileY * gridHeight + vy) + vec2(0.5)
		) / uParticleShadowVolumeAtlasSize;
		opticalDepth += texture(uParticleShadowVolumeAtlas, atlasUv).r;
	}

	return exp(-max(opticalDepth / max(float(gridDepth), 1.0), 0.0));
}

vec3 sampleShadowTransmittance(vec2 atlasUv) {
#ifdef WEBGL_SHADOW_TRANSMITTANCE
	if (uShadowTransmittanceAtlasAvailable == 0) {
		return vec3(1.0);
	}
	return texture(uShadowTransmittanceAtlas, atlasUv).rgb;
#else
	return vec3(1.0);
#endif
}

vec3 sampleShadowVisibility(
	int shadowType,
	int index,
	int cascadeIndex,
	mat4 shadowViewProjection,
	vec4 paramsA,
	vec4 paramsB,
	vec4 paramsC,
	vec4 paramsD,
	vec3 worldPosition,
	vec3 normal,
	vec3 lightDirection,
	float localTileX,
	float localTileY,
	float localTileSpan
) {
	if (uEnableShadows == 0 || paramsA.x < 0.5) {
		return vec3(1.0);
	}
	if (dot(normal, lightDirection) <= 0.0) {
		return vec3(1.0);
	}

	float requestedShadowSize = max(paramsB.z, 1.0);
	float atlasTileSize = max(paramsB.w, requestedShadowSize);
	float resolvedLocalTileSpan = max(localTileSpan, 1.0);
	float subTileSize = max(floor(atlasTileSize / resolvedLocalTileSpan), 1.0);
	float shadowSize = max(1.0, min(requestedShadowSize, subTileSize));
	float slopeBias = max(paramsC.x, 0.0);
	float maxNormalBias = max(paramsA.z, 0.0);
	float minNormalBias = max(paramsA.w, 0.0);
	float cosTheta = max(dot(normal, lightDirection), 0.0);
	float bias = max(paramsA.y + slopeBias * (1.0 - cosTheta), 0.0);
	float normalBias = mix(minNormalBias, maxNormalBias, 1.0 - cosTheta);
	vec3 shadowWorldPosition = worldPosition + normal * normalBias;
	vec4 shadowClip = shadowViewProjection * vec4(shadowWorldPosition, 1.0);
	if (shadowClip.w <= EPSILON) {
		return vec3(1.0);
	}

	vec3 shadowNdc = shadowClip.xyz / shadowClip.w;
	vec2 shadowUv = vec2(shadowNdc.x * 0.5 + 0.5, shadowNdc.y * 0.5 + 0.5);
	float currentDepth = shadowNdc.z * 0.5 + 0.5;
	if (
		shadowUv.x < 0.0 ||
		shadowUv.x > 1.0 ||
		shadowUv.y < 0.0 ||
		shadowUv.y > 1.0 ||
		currentDepth < 0.0 ||
		currentDepth > 1.0
	) {
		return vec3(1.0);
	}

	float pcfRadius = max(paramsB.x, 1.0);
	bool pcssEnabled = paramsD.x > 0.5 && paramsD.y > 0.0;
	float pcssRadius = max(paramsD.y, 0.0);
	int pcssFilterSamples = int(
		clamp(floor(paramsD.z + 0.5), 1.0, float(MAX_PCSS_FILTER_SAMPLES))
	);
	int pcssSearchSamples = int(
		clamp(floor(paramsD.w + 0.5), 1.0, float(MAX_PCSS_SEARCH_SAMPLES))
	);
	vec2 texelPosition = shadowUv * vec2(shadowSize - 1.0);
	vec2 atlasExtent = vec2(textureSize(uShadowAtlas, 0));
	if (atlasExtent.x < 1.0 || atlasExtent.y < 1.0) {
		return vec3(1.0);
	}
	float atlasColumns = max(floor(atlasExtent.x / max(atlasTileSize, 1.0)), 1.0);
	float tileIndex = shadowType == 1 ?
		float(MAX_DIRECTIONAL_LIGHTS + index) :
		float(index);
	float tileX = mod(tileIndex, atlasColumns);
	float tileY = floor(tileIndex / atlasColumns);
	vec2 tileOffset =
		vec2(tileX * atlasTileSize, tileY * atlasTileSize) +
		vec2(localTileX * subTileSize, localTileY * subTileSize);
	vec3 visible = vec3(0.0);
	float sampleCount = 0.0;
	if (pcssEnabled) {
		float theta = hashShadowRotation(worldPosition);
		float blockerDepthSum = 0.0;
		float blockerCount = 0.0;
		for (int i = 0; i < MAX_PCSS_SEARCH_SAMPLES; i++) {
			if (i >= pcssSearchSamples) {
				break;
			}
			vec2 samplePosition =
				texelPosition + vogelDiskSample(i, pcssSearchSamples, theta) * pcssRadius;
			if (
				samplePosition.x < 0.0 ||
				samplePosition.x > shadowSize - 1.0 ||
				samplePosition.y < 0.0 ||
				samplePosition.y > shadowSize - 1.0
			) {
				continue;
			}
			vec2 sampleCoord = round(samplePosition);
			vec2 atlasCoord = tileOffset + sampleCoord + vec2(0.5);
			vec2 atlasUv = atlasCoord / atlasExtent;
			float sampleDepth = texture(uShadowAtlas, atlasUv).r;
			if (currentDepth - bias > sampleDepth) {
				blockerDepthSum += sampleDepth;
				blockerCount += 1.0;
			}
		}

		if (blockerCount <= 0.0) {
			return vec3(1.0);
		}

		float avgBlockerDepth = blockerDepthSum / blockerCount;
		float penumbraRatio = 0.0;
		if (currentDepth > avgBlockerDepth) {
			penumbraRatio = clamp(
				(currentDepth - avgBlockerDepth) / max(avgBlockerDepth, 1e-4),
				0.0,
				1.0
			);
		}
		float filterRadius = pcssRadius * penumbraRatio;
		float effectiveRadius = filterRadius < 0.1 ? pcfRadius : filterRadius;

		for (int i = 0; i < MAX_PCSS_FILTER_SAMPLES; i++) {
			if (i >= pcssFilterSamples) {
				break;
			}
			vec2 samplePosition =
				texelPosition + vogelDiskSample(i, pcssFilterSamples, theta) * effectiveRadius;
			if (
				samplePosition.x < 0.0 ||
				samplePosition.x > shadowSize - 1.0 ||
				samplePosition.y < 0.0 ||
				samplePosition.y > shadowSize - 1.0
			) {
				continue;
			}
			vec2 sampleCoord = round(samplePosition);
			vec2 atlasCoord = tileOffset + sampleCoord + vec2(0.5);
			vec2 atlasUv = atlasCoord / atlasExtent;
			float sampleDepth = texture(uShadowAtlas, atlasUv).r;
			if (currentDepth - bias <= sampleDepth) {
				visible += sampleShadowTransmittance(atlasUv);
			}
			sampleCount += 1.0;
		}
	} else {
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				vec2 samplePosition =
					texelPosition + vec2(float(x), float(y)) * pcfRadius;
				if (
					samplePosition.x < 0.0 ||
					samplePosition.x > shadowSize - 1.0 ||
					samplePosition.y < 0.0 ||
					samplePosition.y > shadowSize - 1.0
				) {
					continue;
				}
				vec2 sampleCoord = round(samplePosition);
				vec2 atlasCoord = tileOffset + sampleCoord + vec2(0.5);
				vec2 atlasUv = atlasCoord / atlasExtent;
				float sampleDepth = texture(uShadowAtlas, atlasUv).r;
				if (currentDepth - bias <= sampleDepth) {
					visible += sampleShadowTransmittance(atlasUv);
				}
				sampleCount += 1.0;
			}
		}
	}

	if (sampleCount <= 0.0) {
		return vec3(1.0);
	}

	vec3 filteredVisibility = visible / sampleCount;
	float strength = clamp(paramsB.y, 0.0, 1.0);
	return (vec3(1.0 - strength) + strength * filteredVisibility) *
		sampleParticleShadowVolumeTransmittance(
			shadowType,
			index,
			cascadeIndex,
			worldPosition
		);
}

int resolveDirectionalCascadeIndex(
	int index,
	float linearDepth,
	int cascadeCount
) {
	int selected = cascadeCount - 1;
	for (int i = 0; i < 4; i++) {
		if (i >= cascadeCount) {
			break;
		}
		int packedIndex = index * 4 + i;
		float splitFar = uDirShadowCascadeSplits[packedIndex].y;
		if (linearDepth <= splitFar) {
			selected = i;
			break;
		}
	}
	return selected;
}

vec3 sampleDirectionalShadowVisibility(
	int index,
	vec3 worldPosition,
	vec3 normal,
	vec3 lightDirection
) {
	vec4 paramsA = uDirShadowParamsA[index];
	vec4 paramsB = uDirShadowParamsB[index];
	vec4 paramsC = uDirShadowParamsC[index];
	vec4 paramsD = uDirShadowParamsD[index];
	bool isCSM = paramsC.y > 0.5 && paramsC.z > 1.5;
	int cascadeCount = isCSM ?
		int(clamp(floor(paramsC.z + 0.5), 1.0, 4.0)) :
		1;
	int cascadeIndex = isCSM ?
		resolveDirectionalCascadeIndex(index, vViewDepth, cascadeCount) :
		0;
	int cascadePackedIndex = index * 4 + cascadeIndex;
	vec4 cascadeSplit = uDirShadowCascadeSplits[cascadePackedIndex];
	float localTileX = isCSM ?
		clamp(floor(cascadeSplit.z + 0.5), 0.0, 1.0) :
		0.0;
	float localTileY = isCSM ?
		clamp(floor(cascadeSplit.w + 0.5), 0.0, 1.0) :
		0.0;
	float localTileSpan = isCSM ? 2.0 : 1.0;
	mat4 shadowViewProjection = isCSM ?
		uDirShadowCascadeViewProjection[cascadePackedIndex] :
		uDirShadowViewProjection[index];

	vec3 baseVisibility = sampleShadowVisibility(
		0,
		index,
		cascadeIndex,
		shadowViewProjection,
		paramsA,
		paramsB,
		paramsC,
		paramsD,
		worldPosition,
		normal,
		lightDirection,
		localTileX,
		localTileY,
		localTileSpan
	);
	if (!isCSM) {
		return baseVisibility;
	}

	float blendRatio = clamp(paramsC.w, 0.0, 1.0);
	bool hasBlend = blendRatio > EPSILON && (cascadeIndex + 1) < cascadeCount;
	if (!hasBlend) {
		return baseVisibility;
	}

	float cascadeRange = max(cascadeSplit.y - cascadeSplit.x, 0.0001);
	float blendStart = cascadeSplit.y - cascadeRange * blendRatio;
	if (vViewDepth <= blendStart) {
		return baseVisibility;
	}

	int nextCascadeIndex = cascadeIndex + 1;
	int nextPackedIndex = index * 4 + nextCascadeIndex;
	vec4 nextCascadeSplit = uDirShadowCascadeSplits[nextPackedIndex];
	float nextLocalTileX = clamp(floor(nextCascadeSplit.z + 0.5), 0.0, 1.0);
	float nextLocalTileY = clamp(floor(nextCascadeSplit.w + 0.5), 0.0, 1.0);
	vec3 nextVisibility = sampleShadowVisibility(
		0,
		index,
		nextCascadeIndex,
		uDirShadowCascadeViewProjection[nextPackedIndex],
		paramsA,
		paramsB,
		paramsC,
		paramsD,
		worldPosition,
		normal,
		lightDirection,
		nextLocalTileX,
		nextLocalTileY,
		2.0
	);
	float blendFactor = clamp(
		(vViewDepth - blendStart) / max(cascadeSplit.y - blendStart, 0.0001),
		0.0,
		1.0
	);
	return mix(baseVisibility, nextVisibility, blendFactor);
}

vec3 sampleSpotShadowVisibility(
	int index,
	vec3 worldPosition,
	vec3 normal,
	vec3 lightDirection
) {
	return sampleShadowVisibility(
		1,
		index,
		0,
		uSpotShadowViewProjection[index],
		uSpotShadowParamsA[index],
		uSpotShadowParamsB[index],
		uSpotShadowParamsC[index],
		uSpotShadowParamsD[index],
		worldPosition,
		normal,
		lightDirection,
		0.0,
		0.0,
		1.0
	);
}
