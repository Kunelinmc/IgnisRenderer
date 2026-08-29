const int SHADOW_FILTER_PCSS = 1;
const float SHADOW_PCF_RADIUS_TEXELS = 1.5;
const float SHADOW_PCSS_SEARCH_RADIUS_TEXELS = 5.0;
const float SHADOW_PCSS_MAX_PENUMBRA_TEXELS = 5.0;
const float SHADOW_PCSS_CONTACT_THRESHOLD_TEXELS = 0.75;
const int MAX_SHADOW_FILTER_SAMPLES = 7;
const int MAX_SHADOW_SEARCH_SAMPLES = 12;

ivec3 resolveShadowSampleCounts(int quality) {
	if (quality <= 0) return ivec3(1, 4, 3);
	if (quality >= 2) return ivec3(5, 12, 7);
	return ivec3(3, 8, 5);
}

float hashShadowRotation(int lightIndex, int cascadeIndex) {
	uint hash = 0x9e3779b9u;
	hash ^= uint(lightIndex) * 0xcb1ab31fu;
	hash ^= uint(cascadeIndex) * 0x165667b1u;
	hash ^= hash >> 16u;
	hash *= 0x7feb352du;
	hash ^= hash >> 15u;
	return float(hash) * (2.0 * PI / 4294967296.0);
}

vec2 shadowFilterDiskSampleBase(int sampleIndex, int sampleCount) {
	if (sampleCount <= 1) return vec2(0.0);
	if (sampleCount <= 3) {
		if (sampleIndex == 0) return vec2(0.0, 0.53);
		if (sampleIndex == 1) return vec2(-0.458993464, -0.265);
		return vec2(0.458993464, -0.265);
	}
	if (sampleCount <= 5) {
		if (sampleIndex == 0) return vec2(0.0);
		if (sampleIndex == 1) return vec2(0.66, 0.0);
		if (sampleIndex == 2) return vec2(-0.66, 0.0);
		if (sampleIndex == 3) return vec2(0.0, 0.66);
		return vec2(0.0, -0.66);
	}
	if (sampleIndex == 0) return vec2(0.0);
	if (sampleIndex == 1) return vec2(0.68, 0.0);
	if (sampleIndex == 2) return vec2(0.34, 0.588897275);
	if (sampleIndex == 3) return vec2(-0.34, 0.588897275);
	if (sampleIndex == 4) return vec2(-0.68, 0.0);
	if (sampleIndex == 5) return vec2(-0.34, -0.588897275);
	return vec2(0.34, -0.588897275);
}

vec2 shadowSearchDiskSampleBase(int sampleIndex) {
	if (sampleIndex == 0) return vec2(-0.191063595, 0.710747050);
	if (sampleIndex == 1) return vec2(0.191063595, -0.710747050);
	if (sampleIndex == 2) return vec2(0.790556673, 0.288710029);
	if (sampleIndex == 3) return vec2(-0.790556673, -0.288710029);
	if (sampleIndex == 4) return vec2(-0.822442486, 0.339492303);
	if (sampleIndex == 5) return vec2(0.822442486, -0.339492303);
	if (sampleIndex == 6) return vec2(-0.364378997, -0.701589586);
	if (sampleIndex == 7) return vec2(0.364378997, 0.701589586);
	if (sampleIndex == 8) return vec2(0.396471625, -0.847236833);
	if (sampleIndex == 9) return vec2(-0.396471625, 0.847236833);
	if (sampleIndex == 10) return vec2(0.571225035, -0.363366609);
	return vec2(-0.571225035, 0.363366609);
}

vec2 rotateShadowDiskSample(vec2 sampleOffset, vec2 rotation) {
	return vec2(
		sampleOffset.x * rotation.x - sampleOffset.y * rotation.y,
		sampleOffset.x * rotation.y + sampleOffset.y * rotation.x
	);
}

vec2 shadowFilterDiskSample(int sampleIndex, int sampleCount, vec2 rotation) {
	return rotateShadowDiskSample(
		shadowFilterDiskSampleBase(sampleIndex, sampleCount),
		rotation
	);
}

vec2 shadowSearchDiskSample(int sampleIndex, vec2 rotation) {
	return rotateShadowDiskSample(shadowSearchDiskSampleBase(sampleIndex), rotation);
}

float linearizeShadowDepth(float depth, vec4 projectionParams) {
	float ndcDepth = depth * 2.0 - 1.0;
	float denominator = ndcDepth * projectionParams.z - projectionParams.x;
	if (abs(denominator) <= 1e-8) return 3.402823466e+38;
	return abs((projectionParams.y - ndcDepth * projectionParams.w) / denominator);
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
	vec4 depthProjectionParams,
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
	float rawBias = max(paramsA.y + slopeBias * (1.0 - cosTheta), 0.0);
	bool compensateCascadeDepthRange =
		shadowType == 0 && paramsC.y > 0.5 && paramsC.z > 1.5;
	vec3 shadowClipDepthRow = vec3(
		shadowViewProjection[0][2],
		shadowViewProjection[1][2],
		shadowViewProjection[2][2]
	);
	float cascadeDepthBiasScale = min(1.0, length(shadowClipDepthRow) * 0.5);
	float bias = rawBias *
		(compensateCascadeDepthRange ? cascadeDepthBiasScale : 1.0);
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
	int quality = int(clamp(floor(paramsD.y + 0.5), 0.0, 2.0));
	ivec3 sampleCounts = resolveShadowSampleCounts(quality);
	float theta = hashShadowRotation(
		int(tileIndex),
		cascadeIndex
	);
	vec2 rotation = vec2(cos(theta), sin(theta));
	int filterSampleCount = sampleCounts.x;
	float filterRadius = SHADOW_PCF_RADIUS_TEXELS;
	bool pcssEnabled = int(floor(paramsD.x + 0.5)) == SHADOW_FILTER_PCSS;
	if (pcssEnabled) {
		float blockerDistanceSum = 0.0;
		float blockerCount = 0.0;
		for (int i = 0; i < MAX_SHADOW_SEARCH_SAMPLES; i++) {
			if (i >= sampleCounts.y) break;
			vec2 samplePosition = clamp(
				texelPosition + shadowSearchDiskSample(i, rotation) *
					SHADOW_PCSS_SEARCH_RADIUS_TEXELS,
				vec2(0.0),
				vec2(shadowSize - 1.0)
			);
			ivec2 sampleCoord = ivec2(round(samplePosition));
			float sampleDepth = texelFetch(
				uShadowAtlas,
				ivec2(tileOffset) + sampleCoord,
				0
			).r;
			if (currentDepth - bias > sampleDepth) {
				float blockerDistance = linearizeShadowDepth(
					sampleDepth,
					depthProjectionParams
				);
				if (blockerDistance < 3.402823466e+38) {
					blockerDistanceSum += blockerDistance;
					blockerCount += 1.0;
				}
			}
		}
		if (blockerCount <= 0.0) return vec3(1.0);
		float receiverDistance = linearizeShadowDepth(
			currentDepth,
			depthProjectionParams
		);
		float blockerDistance = blockerDistanceSum / blockerCount;
		float penumbraRatio = clamp(
			(receiverDistance - blockerDistance) / max(blockerDistance, 1e-6),
			0.0,
			1.0
		);
		float pcssRadius = penumbraRatio * SHADOW_PCSS_MAX_PENUMBRA_TEXELS;
		if (pcssRadius >= SHADOW_PCSS_CONTACT_THRESHOLD_TEXELS) {
			filterSampleCount = sampleCounts.z;
			filterRadius = pcssRadius;
		}
	}

	vec3 visible = vec3(0.0);
	float sampleCount = 0.0;
	for (int i = 0; i < MAX_SHADOW_FILTER_SAMPLES; i++) {
		if (i >= filterSampleCount) break;
		vec2 samplePosition = clamp(
			texelPosition + shadowFilterDiskSample(
				i,
				filterSampleCount,
				rotation
			) * filterRadius,
			vec2(0.0),
			vec2(shadowSize - 1.0)
		);
		ivec2 baseCoord = ivec2(floor(samplePosition));
		ivec2 nextCoord = min(baseCoord + ivec2(1), ivec2(int(shadowSize) - 1));
		vec2 fraction = fract(samplePosition);
		ivec2 atlasBase = ivec2(tileOffset);
		float d00 = texelFetch(uShadowAtlas, atlasBase + baseCoord, 0).r;
		float d10 = texelFetch(
			uShadowAtlas,
			atlasBase + ivec2(nextCoord.x, baseCoord.y),
			0
		).r;
		float d01 = texelFetch(
			uShadowAtlas,
			atlasBase + ivec2(baseCoord.x, nextCoord.y),
			0
		).r;
		float d11 = texelFetch(uShadowAtlas, atlasBase + nextCoord, 0).r;
		float referenceDepth = currentDepth - bias;
		float top = mix(referenceDepth <= d00 ? 1.0 : 0.0,
			referenceDepth <= d10 ? 1.0 : 0.0, fraction.x);
		float bottom = mix(referenceDepth <= d01 ? 1.0 : 0.0,
			referenceDepth <= d11 ? 1.0 : 0.0, fraction.x);
		float comparisonVisibility = mix(top, bottom, fraction.y);
		ivec2 transmittanceCoord = atlasBase + ivec2(round(samplePosition));
		vec2 transmittanceUv = (vec2(transmittanceCoord) + vec2(0.5)) / atlasExtent;
		visible += comparisonVisibility * sampleShadowTransmittance(transmittanceUv);
		sampleCount += 1.0;
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
		uDirShadowDepthProjectionParams[cascadePackedIndex],
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
		uDirShadowDepthProjectionParams[nextPackedIndex],
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
		uSpotShadowDepthProjectionParams[index * 4],
		worldPosition,
		normal,
		lightDirection,
		0.0,
		0.0,
		1.0
	);
}
