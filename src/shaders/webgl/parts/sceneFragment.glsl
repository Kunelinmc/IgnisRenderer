#version 300 es
precision highp float;
__WEBGL_SHADOW_TRANSMITTANCE_DEFINE__
#import <ignis/webgl/constants>
#import <ignis/color/srgb>
#import <ignis/postprocess/fog>

const int MAX_DIRECTIONAL_LIGHTS = __WEBGL_MAX_DIRECTIONAL_LIGHTS__;
const int MAX_POINT_LIGHTS = __WEBGL_MAX_POINT_LIGHTS__;
const int MAX_SPOT_LIGHTS = __WEBGL_MAX_SPOT_LIGHTS__;
const int MAX_LOCAL_LIGHT_PROBES = 8;
const int MAX_REFLECTION_PROBES = 8;
const int MAX_CLUSTER_LIGHTS_PER_FRAGMENT = 128;
const int SH_COEFFICIENT_COUNT = 16;

const float PI = 3.14159265359;
const float EPSILON = 0.000001;
const float PBR_MIN_NDOTV = 0.001;
const float PBR_SPEC_FALLBACK = 0.02;
const float PBR_AMBIENT_FALLBACK_LINEAR = 0.05;
const float TRANSMISSION_ALPHA_FLOOR = 0.12;

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUv;
in vec2 vUv1;
in vec2 vUv2;
in vec2 vUv3;
in vec4 vCurrentClip;
in vec4 vPrevClip;
in float vViewDepth;

uniform vec3 uCameraPosition;
uniform vec3 uAmbientColor;
uniform int uEnableLighting;
uniform int uEnableSH;
uniform vec3 uSHAmbientCoeffs[SH_COEFFICIENT_COUNT];
uniform int uShadingModel;
uniform int uDoubleSided;
uniform vec4 uBaseColor;
uniform vec4 uEmissive;
uniform vec4 uPBR;
uniform vec4 uTransmissionVolume;
uniform vec4 uIridescence;
uniform vec4 uAttenuationColor;
uniform vec4 uAnisotropy;
uniform vec4 uPhong;
uniform vec4 uAlpha;
uniform sampler2D uBaseMap;
uniform int uHasBaseMap;
uniform int uBaseMapIsLinear;
uniform int uBaseMapUV;
uniform vec4 uBaseMapTransformA;
uniform vec2 uBaseMapTransformB;
uniform sampler2D uMetallicRoughnessMap;
uniform int uHasMetallicRoughnessMap;
uniform int uMetallicRoughnessMapUV;
uniform vec4 uMetallicRoughnessMapTransformA;
uniform vec2 uMetallicRoughnessMapTransformB;
uniform sampler2D uNormalMap;
uniform int uHasNormalMap;
uniform int uNormalMapUV;
uniform vec4 uNormalMapTransformA;
uniform vec2 uNormalMapTransformB;
uniform float uNormalScale;
uniform sampler2D uEmissiveMap;
uniform int uHasEmissiveMap;
uniform int uEmissiveMapIsLinear;
uniform int uEmissiveMapUV;
uniform vec4 uEmissiveMapTransformA;
uniform vec2 uEmissiveMapTransformB;
uniform sampler2D uOcclusionMap;
uniform int uHasOcclusionMap;
uniform int uOcclusionMapUV;
uniform vec4 uOcclusionMapTransformA;
uniform vec2 uOcclusionMapTransformB;
uniform float uOcclusionStrength;
uniform sampler2D uIridescenceMap;
uniform int uHasIridescenceMap;
uniform int uIridescenceMapUV;
uniform vec4 uIridescenceMapTransformA;
uniform vec2 uIridescenceMapTransformB;
uniform sampler2D uIridescenceThicknessMap;
uniform int uHasIridescenceThicknessMap;
uniform int uIridescenceThicknessMapUV;
uniform vec4 uIridescenceThicknessMapTransformA;
uniform vec2 uIridescenceThicknessMapTransformB;
uniform int uHasAnisotropyMap;
uniform int uAnisotropyMapUV;
uniform vec4 uAnisotropyMapTransformA;
uniform vec2 uAnisotropyMapTransformB;
uniform sampler2D uEnvSpecularMap;
uniform int uHasEnvSpecularMap;
uniform int uEnvSpecularMapIsLinear;
uniform float uEnvSpecularMaxMipLevel;
uniform sampler2D uEnvSpecularFallbackMap;
uniform int uHasEnvSpecularFallbackMap;
uniform int uEnvSpecularFallbackMapIsLinear;
uniform float uEnvSpecularFallbackMaxMipLevel;
uniform sampler2D uBrdfLUT;
uniform int uLocalLightProbeCount;
uniform vec4 uLocalLightProbeWorldToProbeRow0[MAX_LOCAL_LIGHT_PROBES];
uniform vec4 uLocalLightProbeWorldToProbeRow1[MAX_LOCAL_LIGHT_PROBES];
uniform vec4 uLocalLightProbeWorldToProbeRow2[MAX_LOCAL_LIGHT_PROBES];
uniform vec4 uLocalLightProbeDataA[MAX_LOCAL_LIGHT_PROBES];
uniform vec4 uLocalLightProbeDataB[MAX_LOCAL_LIGHT_PROBES];
uniform sampler2D uLocalLightProbeCoeffs;
uniform vec2 uLocalLightProbeCoeffsSize;
__WEBGL_IRRADIANCE_PROBE_GRID_UNIFORMS__
uniform int uReflectionProbeCount;
uniform vec4 uReflectionProbeWorldToProbeRow0[MAX_REFLECTION_PROBES];
uniform vec4 uReflectionProbeWorldToProbeRow1[MAX_REFLECTION_PROBES];
uniform vec4 uReflectionProbeWorldToProbeRow2[MAX_REFLECTION_PROBES];
uniform vec4 uReflectionProbeProbeToWorldRow0[MAX_REFLECTION_PROBES];
uniform vec4 uReflectionProbeProbeToWorldRow1[MAX_REFLECTION_PROBES];
uniform vec4 uReflectionProbeProbeToWorldRow2[MAX_REFLECTION_PROBES];
uniform vec4 uReflectionProbeDataA[MAX_REFLECTION_PROBES];
uniform vec4 uReflectionProbeDataB[MAX_REFLECTION_PROBES];
uniform vec4 uReflectionProbeDataC[MAX_REFLECTION_PROBES];

uniform int uDirLightCount;
uniform vec4 uDirLightDirection[MAX_DIRECTIONAL_LIGHTS];
uniform vec4 uDirLightColor[MAX_DIRECTIONAL_LIGHTS];

uniform int uPointLightCount;
uniform vec4 uPointLightPositionRange[MAX_POINT_LIGHTS];
uniform vec4 uPointLightColor[MAX_POINT_LIGHTS];

uniform int uSpotLightCount;
uniform vec4 uSpotLightPositionRange[MAX_SPOT_LIGHTS];
uniform vec4 uSpotLightDirectionOuter[MAX_SPOT_LIGHTS];
uniform vec4 uSpotLightColorInner[MAX_SPOT_LIGHTS];
uniform int uEnableShadows;
uniform sampler2D uShadowAtlas;
__WEBGL_SHADOW_TRANSMITTANCE_UNIFORMS__
uniform sampler2D uParticleShadowVolumeAtlas;
uniform vec2 uParticleShadowVolumeAtlasSize;
uniform vec4 uParticleShadowVolumeGridSize;
uniform vec4 uParticleShadowVolumeSliceParams[4];
uniform mat4 uDirShadowViewProjection[MAX_DIRECTIONAL_LIGHTS];
uniform mat4 uDirShadowCascadeViewProjection[MAX_DIRECTIONAL_LIGHTS * 4];
uniform vec4 uDirShadowCascadeSplits[MAX_DIRECTIONAL_LIGHTS * 4];
uniform vec4 uDirShadowParamsA[MAX_DIRECTIONAL_LIGHTS];
uniform vec4 uDirShadowParamsB[MAX_DIRECTIONAL_LIGHTS];
uniform vec4 uDirShadowParamsC[MAX_DIRECTIONAL_LIGHTS];
uniform vec4 uDirShadowParamsD[MAX_DIRECTIONAL_LIGHTS];
uniform mat4 uSpotShadowViewProjection[MAX_SPOT_LIGHTS];
uniform vec4 uSpotShadowParamsA[MAX_SPOT_LIGHTS];
uniform vec4 uSpotShadowParamsB[MAX_SPOT_LIGHTS];
uniform vec4 uSpotShadowParamsC[MAX_SPOT_LIGHTS];
uniform vec4 uSpotShadowParamsD[MAX_SPOT_LIGHTS];
uniform int uEnableClusteredLighting;
uniform vec4 uClusterParams0;
uniform vec4 uClusterParams1;
uniform sampler2D uClusterHeaderTexture;
uniform sampler2D uClusterIndexTexture;
uniform sampler2D uClusterLightTexture;
uniform vec2 uClusterHeaderTexSize;
uniform vec2 uClusterIndexTexSize;
uniform vec2 uClusterLightTexSize;
uniform vec4 uFogParams0;
uniform vec4 uFogParams1;
uniform int uOITPassMode;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragMotion;
layout(location = 2) out vec4 fragNormal;

vec3 safeNormalize(vec3 value, vec3 fallback) {
	float len = length(value);
	return len > EPSILON ? value / len : fallback;
}

float resolveOITWeight(float alpha, float linearDepth) {
	float clampedAlpha = clamp(alpha, 0.0, 1.0);
	float normalizedDepth = clamp(linearDepth / 400.0, 0.0, 1.0);
	float depthWeight = clamp(1.0 - normalizedDepth, 0.05, 1.0);
	float alphaWeight = max(clampedAlpha * 8.0 + 0.01, 0.01);
	float weight = alphaWeight * alphaWeight * alphaWeight * depthWeight;
	return clamp(weight, 1e-2, 3e3);
}

vec2 resolveUV(int uvSet) {
	if (uvSet == 1) return vUv1;
	if (uvSet == 2) return vUv2;
	if (uvSet >= 3) return vUv3;
	return vUv;
}

vec2 applyUVTransform(vec2 uv, vec4 transformA, vec2 transformB) {
	vec2 scaledUv = vec2(uv.x * transformA.x, uv.y * transformA.y);
	vec2 rotatedUv = vec2(
		scaledUv.x * transformB.x - scaledUv.y * transformB.y,
		scaledUv.x * transformB.y + scaledUv.y * transformB.x
	);
	return rotatedUv + transformA.zw;
}

vec2 resolveMappedUV(int uvSet, vec4 transformA, vec2 transformB) {
	return applyUVTransform(resolveUV(uvSet), transformA, transformB);
}

vec3 applyNormalMap(vec3 baseNormal, vec2 uv, vec3 normalSample, float scale) {
	vec3 n = safeNormalize(baseNormal, vec3(0.0, 0.0, 1.0));
	vec3 dp1 = dFdx(vWorldPos);
	vec3 dp2 = dFdy(vWorldPos);
	vec2 duv1 = dFdx(uv);
	vec2 duv2 = dFdy(uv);
	vec3 dp2perp = cross(dp2, n);
	vec3 dp1perp = cross(n, dp1);
	vec3 t = dp2perp * duv1.x + dp1perp * duv2.x;
	vec3 b = dp2perp * duv1.y + dp1perp * duv2.y;
	float maxLenSq = max(dot(t, t), dot(b, b));
	if (maxLenSq <= EPSILON) {
		return n;
	}
	float invMax = inversesqrt(maxLenSq);
	t *= invMax;
	b *= invMax;
	vec3 tangentNormal = vec3(
		(normalSample.x * 2.0 - 1.0) * scale,
		(normalSample.y * 2.0 - 1.0) * scale,
		normalSample.z * 2.0 - 1.0
	);
	return safeNormalize(
		t * tangentNormal.x + b * tangentNormal.y + n * tangentNormal.z,
		n
	);
}

vec3 fallbackTangentFromNormal(vec3 n) {
	vec3 axis = abs(n.y) > 0.999 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
	return safeNormalize(cross(axis, n), vec3(1.0, 0.0, 0.0));
}

void resolveDerivativeTangentFrame(vec3 n, vec2 uv, out vec3 t, out vec3 b) {
	vec3 dp1 = dFdx(vWorldPos);
	vec3 dp2 = dFdy(vWorldPos);
	vec2 duv1 = dFdx(uv);
	vec2 duv2 = dFdy(uv);
	vec3 dp2perp = cross(dp2, n);
	vec3 dp1perp = cross(n, dp1);
	t = dp2perp * duv1.x + dp1perp * duv2.x;
	b = dp2perp * duv1.y + dp1perp * duv2.y;
	float maxLenSq = max(dot(t, t), dot(b, b));
	if (maxLenSq <= EPSILON) {
		t = fallbackTangentFromNormal(n);
		b = safeNormalize(cross(n, t), fallbackTangentFromNormal(n));
		return;
	}
	float invMax = inversesqrt(maxLenSq);
	t *= invMax;
	b *= invMax;
}

vec2 rotateAnisotropyDirection(vec2 direction) {
	return vec2(
		direction.x * uAnisotropy.y - direction.y * uAnisotropy.z,
		direction.x * uAnisotropy.z + direction.y * uAnisotropy.y
	);
}

float distributionAnisotropicGGX(float nDotH, float tDotH, float bDotH, float at, float ab) {
	float a2 = max(at * ab, 1e-6);
	vec3 f = vec3(ab * tDotH, at * bDotH, a2 * nDotH);
	float w2 = a2 / max(dot(f, f), 0.0001);
	return a2 * w2 * w2 / PI;
}

float visibilityAnisotropicGGX(
	float nDotL,
	float nDotV,
	float bDotV,
	float tDotV,
	float tDotL,
	float bDotL,
	float at,
	float ab
) {
	float ggxV = nDotL * length(vec3(at * tDotV, ab * bDotV, nDotV));
	float ggxL = nDotV * length(vec3(at * tDotL, ab * bDotL, nDotL));
	return clamp(0.5 / max(ggxV + ggxL, 0.0001), 0.0, 1.0);
}

vec3 resolveAnisotropicSpecular(
	vec3 fresnel,
	float roughness,
	float anisotropy,
	float nDotL,
	float nDotV,
	float nDotH,
	float tDotV,
	float bDotV,
	float tDotL,
	float bDotL,
	float tDotH,
	float bDotH
) {
	float alphaRoughness = roughness * roughness;
	float at = mix(alphaRoughness, 1.0, anisotropy * anisotropy);
	float ab = alphaRoughness;
	float d = distributionAnisotropicGGX(nDotH, tDotH, bDotH, at, ab);
	float v = visibilityAnisotropicGGX(
		nDotL,
		nDotV,
		bDotV,
		tDotV,
		tDotL,
		bDotL,
		at,
		ab
	);
	return fresnel * d * v;
}

vec3 resolveAnisotropicReflectionDirection(
	vec3 n,
	vec3 v,
	vec3 anisotropicB,
	float roughness,
	float anisotropy
) {
	vec3 bentNormal = cross(anisotropicB, v);
	bentNormal = safeNormalize(cross(bentNormal, anisotropicB), n);
	float a = 1.0 - anisotropy * (1.0 - roughness);
	float blendToNormal = a * a * a * a;
	bentNormal = safeNormalize(mix(bentNormal, n, blendToNormal), n);
	vec3 reflectionDir = safeNormalize(reflect(-v, bentNormal), bentNormal);
	reflectionDir = safeNormalize(
		mix(reflectionDir, bentNormal, roughness * roughness),
		reflectionDir
	);
	return reflectionDir;
}

ivec2 linearIndexToTexel(int linearIndex, vec2 textureSizeValue) {
	int width = max(int(floor(textureSizeValue.x + 0.5)), 1);
	int y = linearIndex / width;
	int x = linearIndex - y * width;
	return ivec2(x, y);
}

vec3 sampleSHAmbientCoeff(int index) {
	return uSHAmbientCoeffs[index];
}

void evalSHBasis(vec3 direction, out float basis[16]) {
	float x = direction.x;
	float y = direction.y;
	float z = direction.z;

	basis[0] = 0.282095;
	basis[1] = 0.488603 * x;
	basis[2] = 0.488603 * y;
	basis[3] = 0.488603 * z;
	basis[4] = 1.092548 * x * z;
	basis[5] = 1.092548 * x * y;
	basis[6] = 0.315392 * (3.0 * y * y - 1.0);
	basis[7] = 1.092548 * y * z;
	basis[8] = 0.546274 * (x * x - z * z);
	basis[9] = 0.590835 * x * (x * x - 3.0 * z * z);
	basis[10] = 2.893641 * x * y * z;
	basis[11] = 0.457619 * x * (5.0 * y * y - 1.0);
	basis[12] = 0.373176 * y * (5.0 * y * y - 3.0);
	basis[13] = 0.457619 * z * (5.0 * y * y - 1.0);
	basis[14] = 1.446821 * y * (x * x - z * z);
	basis[15] = 0.590835 * z * (3.0 * x * x - z * z);
}

vec3 calculateIrradianceFromSH(vec3 normal) {
	float basis[16];
	evalSHBasis(normal, basis);
	float c1 = PI;
	float c2 = (2.0 * PI) / 3.0;
	float c3 = PI / 4.0;
	vec3 result = vec3(0.0);
	for (int i = 0; i < SH_COEFFICIENT_COUNT; i++) {
		float factor = 0.0;
		if (i == 0) {
			factor = c1;
		} else if (i >= 1 && i < 4) {
			factor = c2;
		} else if (i >= 4 && i < 9) {
			factor = c3;
		}
		result += sampleSHAmbientCoeff(i) * basis[i] * factor;
	}
	return max(result, vec3(0.0));
}

vec3 sampleSHRadiance(vec3 direction) {
	float basis[16];
	evalSHBasis(direction, basis);
	vec3 result = vec3(0.0);
	for (int i = 0; i < SH_COEFFICIENT_COUNT; i++) {
		result += sampleSHAmbientCoeff(i) * basis[i];
	}
	return max(result, vec3(0.0));
}

int getLocalLightProbeCount() {
	return clamp(uLocalLightProbeCount, 0, MAX_LOCAL_LIGHT_PROBES);
}

vec3 worldToLocalLightProbePoint(int probeIndex, vec3 worldPosition) {
	vec4 row0 = uLocalLightProbeWorldToProbeRow0[probeIndex];
	vec4 row1 = uLocalLightProbeWorldToProbeRow1[probeIndex];
	vec4 row2 = uLocalLightProbeWorldToProbeRow2[probeIndex];
	return vec3(
		dot(row0.xyz, worldPosition) + row0.w,
		dot(row1.xyz, worldPosition) + row1.w,
		dot(row2.xyz, worldPosition) + row2.w
	);
}

float computeLocalLightProbeMetric(int probeIndex, vec3 worldPosition) {
	vec3 localPosition = worldToLocalLightProbePoint(probeIndex, worldPosition);
	vec4 dataA = uLocalLightProbeDataA[probeIndex];
	float shape = uLocalLightProbeDataB[probeIndex].z;
	if (shape > 0.5) {
		return max(
			max(abs(localPosition.x) * dataA.x, abs(localPosition.y) * dataA.y),
			abs(localPosition.z) * dataA.z
		);
	}
	return length(localPosition) * dataA.w;
}

float computeLocalLightProbeWeight(int probeIndex, float metric) {
	float blendDistance = max(uLocalLightProbeDataB[probeIndex].x, 1e-5);
	float x = clamp((metric - 1.0) / blendDistance, 0.0, 1.0);
	return 1.0 - smoothstep(0.0, 1.0, x);
}

int getLocalLightProbePriority(int probeIndex) {
	return int(uLocalLightProbeDataB[probeIndex].y);
}

bool isBetterLocalLightProbeCandidate(
	float candidateWeight,
	int candidateIndex,
	float currentWeight,
	int currentIndex
) {
	if (candidateWeight > currentWeight + 1e-6) {
		return true;
	}
	if (abs(candidateWeight - currentWeight) <= 1e-6 && candidateIndex < currentIndex) {
		return true;
	}
	return false;
}

vec3 sampleLocalLightProbeCoeff(int probeIndex, int coeffIndex) {
	ivec2 texel = linearIndexToTexel(
		probeIndex * SH_COEFFICIENT_COUNT + coeffIndex,
		uLocalLightProbeCoeffsSize
	);
	return texelFetch(uLocalLightProbeCoeffs, texel, 0).rgb;
}

vec3 calculateIrradianceFromLocalLightProbe(int probeIndex, vec3 normal) {
	float basis[16];
	evalSHBasis(normal, basis);
	float c1 = PI;
	float c2 = (2.0 * PI) / 3.0;
	float c3 = PI / 4.0;
	vec3 result = vec3(0.0);
	for (int i = 0; i < SH_COEFFICIENT_COUNT; i++) {
		float factor = 0.0;
		if (i == 0) {
			factor = c1;
		} else if (i >= 1 && i < 4) {
			factor = c2;
		} else if (i >= 4 && i < 9) {
			factor = c3;
		}
		result += sampleLocalLightProbeCoeff(probeIndex, i) * basis[i] * factor;
	}
	return max(result, vec3(0.0));
}

vec3 sampleLocalLightProbeRadiance(int probeIndex, vec3 direction) {
	float basis[16];
	evalSHBasis(direction, basis);
	vec3 result = vec3(0.0);
	for (int i = 0; i < SH_COEFFICIENT_COUNT; i++) {
		result += sampleLocalLightProbeCoeff(probeIndex, i) * basis[i];
	}
	return max(result, vec3(0.0));
}

void selectTopTwoLocalLightProbes(
	vec3 worldPosition,
	out ivec2 indices,
	out vec2 rawWeights
) {
	indices = ivec2(-1);
	rawWeights = vec2(0.0);
	int probeCount = getLocalLightProbeCount();
	int bestPriority = -2147483647;

	for (int i = 0; i < MAX_LOCAL_LIGHT_PROBES; i++) {
		if (i >= probeCount) {
			break;
		}

		float metric = computeLocalLightProbeMetric(i, worldPosition);
		float weight = computeLocalLightProbeWeight(i, metric);
		if (weight <= 1e-6) {
			continue;
		}

		int priority = getLocalLightProbePriority(i);
		if (priority > bestPriority) {
			bestPriority = priority;
			indices = ivec2(i, -1);
			rawWeights = vec2(weight, 0.0);
			continue;
		}
		if (priority < bestPriority) {
			continue;
		}

		if (
			indices.x < 0 ||
			isBetterLocalLightProbeCandidate(weight, i, rawWeights.x, indices.x)
		) {
			indices.y = indices.x;
			rawWeights.y = rawWeights.x;
			indices.x = i;
			rawWeights.x = weight;
			continue;
		}

		if (
			indices.y < 0 ||
			isBetterLocalLightProbeCandidate(weight, i, rawWeights.y, indices.y)
		) {
			indices.y = i;
			rawWeights.y = weight;
		}
	}
}

vec4 sampleBlendedLocalLightProbeIrradiance(
	ivec2 indices,
	vec2 rawWeights,
	vec3 normal
) {
	float rawSum = rawWeights.x + max(rawWeights.y, 0.0);
	float coverage = clamp(rawSum, 0.0, 1.0);
	if (indices.x < 0 || coverage <= 1e-6) {
		return vec4(0.0);
	}

	float invWeight = 1.0 / max(rawSum, 1e-6);
	vec3 result =
		calculateIrradianceFromLocalLightProbe(indices.x, normal) *
		(rawWeights.x * invWeight);
	if (indices.y >= 0 && rawWeights.y > 1e-6) {
		result +=
			calculateIrradianceFromLocalLightProbe(indices.y, normal) *
			(rawWeights.y * invWeight);
	}
	return vec4(result, coverage);
}

vec4 sampleBlendedLocalLightProbeRadiance(
	ivec2 indices,
	vec2 rawWeights,
	vec3 direction
) {
	float rawSum = rawWeights.x + max(rawWeights.y, 0.0);
	float coverage = clamp(rawSum, 0.0, 1.0);
	if (indices.x < 0 || coverage <= 1e-6) {
		return vec4(0.0);
	}

	float invWeight = 1.0 / max(rawSum, 1e-6);
	vec3 result =
		sampleLocalLightProbeRadiance(indices.x, direction) *
		(rawWeights.x * invWeight);
	if (indices.y >= 0 && rawWeights.y > 1e-6) {
		result +=
			sampleLocalLightProbeRadiance(indices.y, direction) *
			(rawWeights.y * invWeight);
	}
	return vec4(result, coverage);
}

__WEBGL_IRRADIANCE_PROBE_GRID_FUNCTIONS__

vec4 fetchClusterHeader(int clusterIndex) {
	ivec2 texel = linearIndexToTexel(clusterIndex, uClusterHeaderTexSize);
	return texelFetch(uClusterHeaderTexture, texel, 0);
}

int fetchClusterListLightIndex(int listIndex) {
	int texelIndex = listIndex / 4;
	int component = listIndex - texelIndex * 4;
	ivec2 texel = linearIndexToTexel(texelIndex, uClusterIndexTexSize);
	vec4 packed = texelFetch(uClusterIndexTexture, texel, 0);
	if (component == 0) return int(floor(packed.x + 0.5));
	if (component == 1) return int(floor(packed.y + 0.5));
	if (component == 2) return int(floor(packed.z + 0.5));
	return int(floor(packed.w + 0.5));
}

vec4 fetchClusterLightRow(int lightIndex, int row) {
	int texelIndex = lightIndex * 4 + row;
	ivec2 texel = linearIndexToTexel(texelIndex, uClusterLightTexSize);
	return texelFetch(uClusterLightTexture, texel, 0);
}

bool resolveClusterSpan(out int offset, out int count, out int maxLightsPerCluster) {
	offset = 0;
	count = 0;
	maxLightsPerCluster = 0;
	if (uEnableClusteredLighting == 0) {
		return false;
	}
	int tilesX = max(int(floor(uClusterParams0.z + 0.5)), 1);
	int tilesY = max(int(floor(uClusterParams0.w + 0.5)), 1);
	int zSlices = max(int(floor(uClusterParams1.x + 0.5)), 1);
	maxLightsPerCluster = max(int(floor(uClusterParams1.y + 0.5)), 1);
	float logScale = uClusterParams1.z;
	float logBias = uClusterParams1.w;

	float width = max(uClusterParams0.x, 1.0);
	float height = max(uClusterParams0.y, 1.0);
	float xNorm = clamp(gl_FragCoord.x / width, 0.0, 0.999999);
	float yNorm = clamp(gl_FragCoord.y / height, 0.0, 0.999999);
	int tileX = clamp(int(floor(xNorm * float(tilesX))), 0, tilesX - 1);
	int tileY = clamp(int(floor(yNorm * float(tilesY))), 0, tilesY - 1);
	float viewDepth = max(vViewDepth, 1e-4);
	int slice = clamp(
		int(floor(log(viewDepth) * logScale + logBias)),
		0,
		zSlices - 1
	);
	int clusterIndex = tileX + tileY * tilesX + slice * tilesX * tilesY;
	vec4 header = fetchClusterHeader(clusterIndex);
	offset = max(0, int(floor(header.x + 0.5)));
	count = max(0, int(floor(header.y + 0.5)));
	return count > 0;
}

vec2 directionToEquirectUV(vec3 direction) {
	float phi = atan(direction.x, direction.z);
	float theta = acos(clamp(direction.y, -1.0, 1.0));
	return vec2((phi + PI) / (2.0 * PI), theta / PI);
}

vec3 decodeEnvSpecularSample(vec3 sampled) {
	return uEnvSpecularMapIsLinear == 1 ? sampled : srgbToLinear(sampled);
}

vec3 samplePrefilteredEnvSpecularLayer(
	vec3 direction,
	float roughness,
	float layer,
	float layerCount
) {
	vec2 uv = directionToEquirectUV(safeNormalize(direction, vec3(0.0, 1.0, 0.0)));
	if (layerCount > 1.0) {
		uv.x = (uv.x + layer) / layerCount;
	}
	float mipLevel = clamp(
		roughness * max(uEnvSpecularMaxMipLevel, 0.0),
		0.0,
		max(uEnvSpecularMaxMipLevel, 0.0)
	);
	vec3 sampled = textureLod(uEnvSpecularMap, uv, mipLevel).rgb;
	return decodeEnvSpecularSample(sampled);
}

vec3 sampleFallbackEnvSpecular(vec3 direction, float roughness) {
	if (uHasEnvSpecularFallbackMap == 0) {
		return vec3(0.0);
	}

	vec2 uv = directionToEquirectUV(safeNormalize(direction, vec3(0.0, 1.0, 0.0)));
	float mipLevel = clamp(
		roughness * max(uEnvSpecularFallbackMaxMipLevel, 0.0),
		0.0,
		max(uEnvSpecularFallbackMaxMipLevel, 0.0)
	);
	vec3 sampled = textureLod(uEnvSpecularFallbackMap, uv, mipLevel).rgb;
	return uEnvSpecularFallbackMapIsLinear == 1 ?
		sampled
	:	srgbToLinear(sampled);
}

vec3 worldToProbePoint(int probeIndex, vec3 worldPosition) {
	vec4 row0 = uReflectionProbeWorldToProbeRow0[probeIndex];
	vec4 row1 = uReflectionProbeWorldToProbeRow1[probeIndex];
	vec4 row2 = uReflectionProbeWorldToProbeRow2[probeIndex];
	return vec3(
		dot(row0.xyz, worldPosition) + row0.w,
		dot(row1.xyz, worldPosition) + row1.w,
		dot(row2.xyz, worldPosition) + row2.w
	);
}

vec3 worldToProbeDirection(int probeIndex, vec3 worldDirection) {
	vec4 row0 = uReflectionProbeWorldToProbeRow0[probeIndex];
	vec4 row1 = uReflectionProbeWorldToProbeRow1[probeIndex];
	vec4 row2 = uReflectionProbeWorldToProbeRow2[probeIndex];
	return vec3(
		dot(row0.xyz, worldDirection),
		dot(row1.xyz, worldDirection),
		dot(row2.xyz, worldDirection)
	);
}

vec3 probeToWorldPoint(int probeIndex, vec3 probePosition) {
	vec4 row0 = uReflectionProbeProbeToWorldRow0[probeIndex];
	vec4 row1 = uReflectionProbeProbeToWorldRow1[probeIndex];
	vec4 row2 = uReflectionProbeProbeToWorldRow2[probeIndex];
	return vec3(
		dot(row0.xyz, probePosition) + row0.w,
		dot(row1.xyz, probePosition) + row1.w,
		dot(row2.xyz, probePosition) + row2.w
	);
}

float computeReflectionProbeMetric(int probeIndex, vec3 worldPosition) {
	vec3 localPosition = worldToProbePoint(probeIndex, worldPosition);
	vec4 dataA = uReflectionProbeDataA[probeIndex];
	float shape = uReflectionProbeDataB[probeIndex].w;
	if (shape > 0.5) {
		return max(
			max(abs(localPosition.x) * dataA.x, abs(localPosition.y) * dataA.y),
			abs(localPosition.z) * dataA.z
		);
	}
	return length(localPosition) * dataA.w;
}

float computeReflectionProbeWeight(int probeIndex, float metric) {
	float blendDistance = max(uReflectionProbeDataC[probeIndex].y, 1e-5);
	float x = clamp((metric - 1.0) / blendDistance, 0.0, 1.0);
	float weight = 1.0 - smoothstep(0.0, 1.0, x);
	float blendExponent = max(uReflectionProbeDataC[probeIndex].z, 0.01);
	if (abs(blendExponent - 1.0) > 1e-5) {
		weight = pow(max(weight, 0.0), blendExponent);
	}
	return weight;
}

float computeReflectionProbeDepthOcclusion(int probeIndex, float metric) {
	float blendDistance = max(uReflectionProbeDataC[probeIndex].y, 1e-5);
	float normalizedDepth = clamp((1.0 - metric) / blendDistance, 0.0, 1.0);
	return smoothstep(0.0, 1.0, normalizedDepth);
}

bool isBetterReflectionProbeCandidate(
	float candidateWeight,
	int candidateIndex,
	float currentWeight,
	int currentIndex
) {
	if (candidateWeight > currentWeight + 1e-6) {
		return true;
	}
	if (abs(candidateWeight - currentWeight) <= 1e-6 && candidateIndex < currentIndex) {
		return true;
	}
	return false;
}

void selectTopTwoReflectionProbes(
	vec3 worldPosition,
	int probeCount,
	out ivec2 indices,
	out vec2 weights
) {
	indices = ivec2(-1);
	weights = vec2(0.0);

	for (int i = 0; i < MAX_REFLECTION_PROBES; i++) {
		if (i >= probeCount) {
			break;
		}
		float metric = computeReflectionProbeMetric(i, worldPosition);
		float weight = computeReflectionProbeWeight(i, metric);
		if (weight <= 1e-6) {
			continue;
		}

		if (
			indices.x < 0 ||
			isBetterReflectionProbeCandidate(weight, i, weights.x, indices.x)
		) {
			indices.y = indices.x;
			weights.y = weights.x;
			indices.x = i;
			weights.x = weight;
			continue;
		}

		if (
			indices.y < 0 ||
			isBetterReflectionProbeCandidate(weight, i, weights.y, indices.y)
		) {
			indices.y = i;
			weights.y = weight;
		}
	}

	if (indices.x < 0) {
		weights = vec2(0.0);
		return;
	}

	float sumWeight = weights.x + max(weights.y, 0.0);
	if (sumWeight <= 1e-6) {
		indices.y = -1;
		weights = vec2(1.0, 0.0);
		return;
	}

	if (indices.y < 0) {
		weights = vec2(1.0, 0.0);
		return;
	}

	weights /= sumWeight;
}

bool intersectReflectionProbeBox(
	int probeIndex,
	vec3 localOrigin,
	vec3 localDirection,
	out vec3 localHit
) {
	vec4 dataA = uReflectionProbeDataA[probeIndex];
	vec3 halfExtents = vec3(
		1.0 / max(dataA.x, 1e-5),
		1.0 / max(dataA.y, 1e-5),
		1.0 / max(dataA.z, 1e-5)
	);
	float tMin = -1e20;
	float tMax = 1e20;

	if (abs(localDirection.x) <= EPSILON) {
		if (localOrigin.x < -halfExtents.x || localOrigin.x > halfExtents.x) {
			return false;
		}
	} else {
		float invDirection = 1.0 / localDirection.x;
		float t0 = (-halfExtents.x - localOrigin.x) * invDirection;
		float t1 = (halfExtents.x - localOrigin.x) * invDirection;
		if (t0 > t1) {
			float swap = t0;
			t0 = t1;
			t1 = swap;
		}
		tMin = max(tMin, t0);
		tMax = min(tMax, t1);
	}

	if (abs(localDirection.y) <= EPSILON) {
		if (localOrigin.y < -halfExtents.y || localOrigin.y > halfExtents.y) {
			return false;
		}
	} else {
		float invDirection = 1.0 / localDirection.y;
		float t0 = (-halfExtents.y - localOrigin.y) * invDirection;
		float t1 = (halfExtents.y - localOrigin.y) * invDirection;
		if (t0 > t1) {
			float swap = t0;
			t0 = t1;
			t1 = swap;
		}
		tMin = max(tMin, t0);
		tMax = min(tMax, t1);
	}

	if (abs(localDirection.z) <= EPSILON) {
		if (localOrigin.z < -halfExtents.z || localOrigin.z > halfExtents.z) {
			return false;
		}
	} else {
		float invDirection = 1.0 / localDirection.z;
		float t0 = (-halfExtents.z - localOrigin.z) * invDirection;
		float t1 = (halfExtents.z - localOrigin.z) * invDirection;
		if (t0 > t1) {
			float swap = t0;
			t0 = t1;
			t1 = swap;
		}
		tMin = max(tMin, t0);
		tMax = min(tMax, t1);
	}

	if (tMax < max(tMin, 0.0)) {
		return false;
	}
	float t = tMin > EPSILON ? tMin : tMax;
	if (t <= EPSILON) {
		return false;
	}
	localHit = localOrigin + localDirection * t;
	return true;
}

bool intersectReflectionProbeSphere(
	int probeIndex,
	vec3 localOrigin,
	vec3 localDirection,
	out vec3 localHit
) {
	float radius = 1.0 / max(uReflectionProbeDataA[probeIndex].w, 1e-5);
	float b = dot(localOrigin, localDirection);
	float c = dot(localOrigin, localOrigin) - radius * radius;
	float discriminant = b * b - c;
	if (discriminant < 0.0) {
		return false;
	}

	float sqrtDiscriminant = sqrt(discriminant);
	float t0 = -b - sqrtDiscriminant;
	float t1 = -b + sqrtDiscriminant;
	float t = 1e20;
	if (t0 > EPSILON) {
		t = min(t, t0);
	}
	if (t1 > EPSILON) {
		t = min(t, t1);
	}
	if (t >= 1e19) {
		return false;
	}

	localHit = localOrigin + localDirection * t;
	return true;
}

vec3 computeReflectionProbeParallaxDirection(
	int probeIndex,
	vec3 worldPosition,
	vec3 reflectionDirection
) {
	vec3 fallback = safeNormalize(reflectionDirection, vec3(0.0, 0.0, 1.0));
	int parallaxMode = int(floor(uReflectionProbeDataC[probeIndex].x + 0.5));
	if (parallaxMode <= 0) {
		return fallback;
	}

	vec3 localOrigin = worldToProbePoint(probeIndex, worldPosition);
	vec3 localDirection = safeNormalize(
		worldToProbeDirection(probeIndex, fallback),
		fallback
	);
	vec3 localHit = vec3(0.0);
	bool hasHit = false;
	if (parallaxMode == 1) {
		hasHit = intersectReflectionProbeBox(
			probeIndex,
			localOrigin,
			localDirection,
			localHit
		);
	} else if (parallaxMode == 2) {
		hasHit = intersectReflectionProbeSphere(
			probeIndex,
			localOrigin,
			localDirection,
			localHit
		);
	}
	if (!hasHit) {
		return fallback;
	}

	vec3 worldHit = probeToWorldPoint(probeIndex, localHit);
	vec3 corrected = worldHit - uReflectionProbeDataB[probeIndex].xyz;
	return dot(corrected, corrected) > EPSILON ?
		normalize(corrected)
	:	fallback;
}

vec3 sampleEnvironmentSpecular(vec3 worldPosition, vec3 direction, float roughness) {
	if (uHasEnvSpecularMap == 0) {
		return vec3(0.0);
	}

	int probeCount = clamp(uReflectionProbeCount, 0, MAX_REFLECTION_PROBES);
	vec3 normalizedDirection = safeNormalize(direction, vec3(0.0, 1.0, 0.0));
	if (probeCount <= 0) {
		return samplePrefilteredEnvSpecularLayer(normalizedDirection, roughness, 0.0, 1.0);
	}

	ivec2 indices;
	vec2 weights;
	selectTopTwoReflectionProbes(worldPosition, probeCount, indices, weights);
	vec3 fallbackSample = sampleFallbackEnvSpecular(normalizedDirection, roughness);
	if (indices.x < 0) {
		return fallbackSample;
	}

	int firstIndex = indices.x;
	float firstLayer = uReflectionProbeDataC[firstIndex].w;
	vec3 firstDirection = computeReflectionProbeParallaxDirection(
		firstIndex,
		worldPosition,
		normalizedDirection
	);
	vec3 firstSample = samplePrefilteredEnvSpecularLayer(
		firstDirection,
		roughness,
		firstLayer,
		float(probeCount)
	);
	float firstMetric = computeReflectionProbeMetric(firstIndex, worldPosition);
	float firstDepthOcclusion = computeReflectionProbeDepthOcclusion(
		firstIndex,
		firstMetric
	);
	float firstContribution = weights.x * firstDepthOcclusion;

	if (indices.y < 0 || weights.y <= 1e-6) {
		return firstSample * firstContribution +
			fallbackSample * (1.0 - clamp(firstContribution, 0.0, 1.0));
	}

	int secondIndex = indices.y;
	float secondLayer = uReflectionProbeDataC[secondIndex].w;
	vec3 secondDirection = computeReflectionProbeParallaxDirection(
		secondIndex,
		worldPosition,
		normalizedDirection
	);
	vec3 secondSample = samplePrefilteredEnvSpecularLayer(
		secondDirection,
		roughness,
		secondLayer,
		float(probeCount)
	);
	float secondMetric = computeReflectionProbeMetric(secondIndex, worldPosition);
	float secondDepthOcclusion = computeReflectionProbeDepthOcclusion(
		secondIndex,
		secondMetric
	);
	float secondContribution = weights.y * secondDepthOcclusion;
	float combinedContribution = clamp(
		firstContribution + secondContribution,
		0.0,
		1.0
	);

	return
		firstSample * firstContribution +
		secondSample * secondContribution +
		fallbackSample * (1.0 - combinedContribution);
}

float pointAttenuation(float distanceSq, float range) {
	float rangeSq = max(range * range, EPSILON);
	float rangeFactor = distanceSq / rangeSq;
	float smoothFactor = max(0.0, 1.0 - rangeFactor * rangeFactor);
	return (smoothFactor * smoothFactor) / (distanceSq + 1.0);
}

float spotAttenuation(float cosTheta, float outerCos, float innerCos) {
	if (cosTheta < outerCos) {
		return 0.0;
	}
	float cutoffRange = max(innerCos - outerCos, EPSILON);
	return clamp((cosTheta - outerCos) / cutoffRange, 0.0, 1.0);
}

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

float distributionGGX(vec3 n, vec3 h, float roughness) {
	float a = roughness * roughness;
	float a2 = a * a;
	float nDotH = max(dot(n, h), 0.0);
	float nDotH2 = nDotH * nDotH;
	float denom = nDotH2 * (a2 - 1.0) + 1.0;
	return a2 / max(PI * denom * denom, 0.0001);
}

float geometrySchlickGGX(float nDotValue, float roughness) {
	float r = roughness + 1.0;
	float k = (r * r) / 8.0;
	return nDotValue / max(nDotValue * (1.0 - k) + k, 0.0001);
}

float geometrySmith(float nDotV, float nDotL, float roughness) {
	return geometrySchlickGGX(nDotV, roughness) *
		geometrySchlickGGX(nDotL, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 f0) {
	return f0 + (vec3(1.0) - f0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

float fresnelSchlickScalar(float cosTheta, float f0) {
	return f0 + (1.0 - f0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

float iorToFresnel0(float transmittedIor, float incidentIor) {
	float value = (transmittedIor - incidentIor) /
		max(transmittedIor + incidentIor, EPSILON);
	return value * value;
}

vec3 fresnel0ToIor(vec3 f0) {
	vec3 sqrtF0 = sqrt(clamp(f0, vec3(0.0), vec3(0.9999)));
	return (vec3(1.0) + sqrtF0) / max(vec3(1.0) - sqrtF0, vec3(EPSILON));
}

vec3 evalIridescenceSensitivity(float opd, vec3 shift) {
	float phase = 2.0 * PI * opd * 1.0e-9;
	float phaseSq = phase * phase;
	vec3 val = vec3(5.4856e-13, 4.4201e-13, 5.2481e-13);
	vec3 pos = vec3(1.6810e6, 1.7953e6, 2.2084e6);
	vec3 variance = vec3(4.3278e9, 9.3046e9, 6.6121e9);
	vec3 xyz =
		val *
		sqrt(vec3(2.0 * PI) * variance) *
		cos(pos * phase + shift) *
		exp(-variance * phaseSq);
	xyz.x +=
		9.7470e-14 *
		sqrt(2.0 * PI * 4.5282e9) *
		cos(2.2399e6 * phase + shift.x) *
		exp(-4.5282e9 * phaseSq);
	xyz /= 1.0685e-7;

	const mat3 xyzToRec709 = mat3(
		3.2404542, -0.9692660, 0.0556434,
		-1.5371385, 1.8760108, -0.2040259,
		-0.4985314, 0.0415560, 1.0572252
	);
	return xyzToRec709 * xyz;
}

vec3 iridescentFresnel(
	float outsideIor,
	float iridescenceIor,
	vec3 baseF0,
	float iridescenceThickness,
	float cosTheta1
) {
	float filmIor = max(iridescenceIor, EPSILON);
	float cos1 = clamp(cosTheta1, 0.0, 1.0);
	float eta = outsideIor / filmIor;
	float sinTheta2Sq = eta * eta * (1.0 - cos1 * cos1);
	if (sinTheta2Sq > 1.0) {
		return vec3(1.0);
	}

	float cosTheta2 = sqrt(max(1.0 - sinTheta2Sq, 0.0));
	float r0 = iorToFresnel0(filmIor, outsideIor);
	float r12 = fresnelSchlickScalar(cos1, r0);
	float t121 = 1.0 - r12;
	vec3 baseIor = fresnel0ToIor(baseF0 + vec3(0.0001));
	vec3 r1 = vec3(
		iorToFresnel0(baseIor.x, filmIor),
		iorToFresnel0(baseIor.y, filmIor),
		iorToFresnel0(baseIor.z, filmIor)
	);
	vec3 r23 = fresnelSchlick(cosTheta2, r1);

	float phi12 = filmIor < outsideIor ? PI : 0.0;
	float phi21 = PI - phi12;
	vec3 phi23 = vec3(
		baseIor.x < filmIor ? PI : 0.0,
		baseIor.y < filmIor ? PI : 0.0,
		baseIor.z < filmIor ? PI : 0.0
	);
	vec3 phi = vec3(phi21) + phi23;
	float opd = 2.0 * filmIor * iridescenceThickness * cosTheta2;
	vec3 r123 = clamp(vec3(r12) * r23, vec3(1e-5), vec3(0.9999));
	vec3 sqrtR123 = sqrt(r123);
	vec3 rs = (t121 * t121) * r23 / (vec3(1.0) - r123);

	vec3 interference = vec3(r12) + rs;
	vec3 cm = rs - vec3(t121);
	for (int order = 1; order <= 2; order++) {
		cm *= sqrtR123;
		float orderValue = float(order);
		vec3 sensitivity = evalIridescenceSensitivity(
			orderValue * opd,
			orderValue * phi
		);
		interference += cm * 2.0 * sensitivity;
	}

	return max(interference, vec3(0.0));
}

vec3 resolveIridescenceFresnel(
	float cosTheta,
	vec3 baseF0,
	float iridescence,
	float iridescenceThickness,
	float iridescenceIor
) {
	vec3 base = fresnelSchlick(cosTheta, baseF0);
	float strength = clamp(iridescence, 0.0, 1.0);
	if (strength <= EPSILON || iridescenceThickness <= 0.0) {
		return base;
	}
	vec3 iridescent = iridescentFresnel(
		1.0,
		max(iridescenceIor, 1.0),
		clamp(baseF0, vec3(0.0), vec3(0.9999)),
		iridescenceThickness,
		cosTheta
	);
	return clamp(mix(base, iridescent, strength), vec3(0.0), vec3(1.0));
}

vec3 diffuseFresnelWeight(vec3 fresnel, float iridescence) {
	if (iridescence > EPSILON) {
		float fresnelMax = max(max(fresnel.r, fresnel.g), fresnel.b);
		return vec3(1.0 - fresnelMax);
	}
	return vec3(1.0) - fresnel;
}

float resolveTransmissionAlpha(
	float baseAlpha,
	float transmission,
	float fresnelAverage
) {
	float clampedTransmission = clamp(transmission, 0.0, 1.0);
	if (clampedTransmission <= EPSILON) {
		return clamp(baseAlpha, 0.0, 1.0);
	}
	float floorAlpha = max(
		TRANSMISSION_ALPHA_FLOOR,
		clamp(fresnelAverage, 0.0, 1.0)
	);
	float blended =
		baseAlpha * (1.0 - clampedTransmission) +
		floorAlpha * clampedTransmission;
	return clamp(max(floorAlpha, blended), 0.0, 1.0);
}

vec3 shadePhong(vec3 albedo, vec3 n, vec3 shadowNormal, vec3 v) {
	vec3 ambientBase = uAmbientColor;
	if (uEnableSH == 1) {
		ambientBase = sampleDiffuseProbeIrradiance(vWorldPos, n) / 255.0;
	} else if (ambientBase.x + ambientBase.y + ambientBase.z <= 0.0) {
		ambientBase = vec3(PBR_AMBIENT_FALLBACK_LINEAR);
	}
	vec3 lit = ambientBase * albedo;
	vec3 specular = vec3(0.0);
	float shininess = max(1.0, uPhong.x);

	for (int i = 0; i < MAX_DIRECTIONAL_LIGHTS; i++) {
		if (i >= uDirLightCount) break;
		vec3 l = safeNormalize(uDirLightDirection[i].xyz, vec3(0.0, 1.0, 0.0));
		float nDotL = max(dot(n, l), 0.0);
		vec3 shadow = sampleDirectionalShadowVisibility(
			i,
			vWorldPos,
			shadowNormal,
			l
		);
		lit += albedo * uDirLightColor[i].xyz * nDotL * shadow;
		if (nDotL > 0.0) {
			vec3 h = safeNormalize(l + v, v);
			specular +=
				uDirLightColor[i].xyz *
				pow(max(dot(n, h), 0.0), shininess) *
				shadow;
		}
	}

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
				vec3 l = toLight / distanceValue;
				float attenuation = pointAttenuation(distanceSq, lightRange);
				float nDotL = max(dot(n, l), 0.0);

				if (lightType == 0) {
					lit += albedo * lightC.xyz * nDotL * attenuation;
					if (nDotL > 0.0) {
						vec3 h = safeNormalize(l + v, v);
						specular +=
							lightC.xyz *
							pow(max(dot(n, h), 0.0), shininess) *
							attenuation;
					}
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
					lit +=
						albedo * lightC.xyz *
						nDotL * attenuation * coneFactor * shadow;
					if (nDotL > 0.0) {
						vec3 h = safeNormalize(l + v, v);
						specular +=
							lightC.xyz *
							pow(max(dot(n, h), 0.0), shininess) *
							attenuation *
							coneFactor *
							shadow;
					}
				}
			}
		}
	} else {
		for (int i = 0; i < MAX_POINT_LIGHTS; i++) {
			if (i >= uPointLightCount) break;
			vec3 toLight = uPointLightPositionRange[i].xyz - vWorldPos;
			float distanceSq = dot(toLight, toLight);
			float distanceValue = sqrt(max(distanceSq, EPSILON));
			float lightRange = max(uPointLightPositionRange[i].w, 0.001);
			if (distanceValue > lightRange) {
				continue;
			}
			vec3 l = toLight / distanceValue;
			float attenuation = pointAttenuation(distanceSq, lightRange);
			float nDotL = max(dot(n, l), 0.0);
			lit += albedo * uPointLightColor[i].xyz * nDotL * attenuation;
			if (nDotL > 0.0) {
				vec3 h = safeNormalize(l + v, v);
				specular +=
					uPointLightColor[i].xyz *
					pow(max(dot(n, h), 0.0), shininess) *
					attenuation;
			}
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
			vec3 l = toLight / distanceValue;
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
			float nDotL = max(dot(n, l), 0.0);
			vec3 shadow = sampleSpotShadowVisibility(
				i,
				vWorldPos,
				shadowNormal,
				l
			);
			lit +=
				albedo * uSpotLightColorInner[i].xyz *
				nDotL * attenuation * coneFactor * shadow;
			if (nDotL > 0.0) {
				vec3 h = safeNormalize(l + v, v);
				specular +=
					uSpotLightColorInner[i].xyz *
					pow(max(dot(n, h), 0.0), shininess) *
					attenuation *
					coneFactor *
					shadow;
			}
		}
	}

	return lit + specular * 0.25;
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
	float ior = max(uTransmissionVolume.x, 1.0);
	float thickness = max(uTransmissionVolume.y, 0.0);
	float attenuationDistance = uTransmissionVolume.z;
	vec3 attenuationColor = clamp(uAttenuationColor.rgb, vec3(0.0001), vec3(1.0));
	vec3 volumeAttenuation = vec3(1.0);
	if (thickness > 0.0 && attenuationDistance > 0.0) {
		vec3 absorb = -log(attenuationColor) / attenuationDistance;
		volumeAttenuation = exp(-absorb * thickness);
	}
	float dielectricF0 = 0.16 * reflectance * reflectance;
	vec3 f0 = mix(vec3(dielectricF0), albedo, metalness);
	float nDotV = max(dot(pbrNormal, viewDir), PBR_MIN_NDOTV);
	vec3 reflectionDir =
		anisotropyStrength > EPSILON ?
			resolveAnisotropicReflectionDirection(
				pbrNormal,
				viewDir,
				anisotropyBitangent,
				roughness,
				anisotropyStrength
			)
		:	reflect(-viewDir, pbrNormal);
	ivec2 localProbeIndices = ivec2(-1);
	vec2 localProbeWeights = vec2(0.0);

	vec3 ambientBase = uAmbientColor;
	vec3 specularAmbientBase = ambientBase;
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
	} else if (ambientBase.x + ambientBase.y + ambientBase.z <= 0.0) {
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
	} else {
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
			if (uHasEnvSpecularMap == 1) {
				transmissionRadiance = sampleEnvironmentSpecular(
					vWorldPos,
					transmissionDir,
					roughness
				);
			} else if (uEnableSH == 1) {
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
	} else {
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

void main() {
	vec2 baseUv = resolveMappedUV(
		uBaseMapUV,
		uBaseMapTransformA,
		uBaseMapTransformB
	);
	vec3 albedo = uBaseColor.rgb;
	float alpha = clamp(uBaseColor.a, 0.0, 1.0);
	if (uHasBaseMap == 1) {
		vec4 texel = texture(uBaseMap, baseUv);
		vec3 texColor = uBaseMapIsLinear == 1 ? texel.rgb : srgbToLinear(texel.rgb);
		albedo *= texColor;
		alpha *= texel.a;
	}

	if (uAlpha.y > 0.5 && alpha < uAlpha.x) {
		discard;
	}

	float roughness = clamp(uPBR.x, 0.04, 1.0);
	float metalness = clamp(uPBR.y, 0.0, 1.0);
	float reflectance = clamp(uPBR.z, 0.0, 1.0);
	float transmission = clamp(uPBR.w, 0.0, 1.0);
	if (uHasMetallicRoughnessMap == 1) {
		vec2 metallicRoughnessUv = resolveMappedUV(
			uMetallicRoughnessMapUV,
			uMetallicRoughnessMapTransformA,
			uMetallicRoughnessMapTransformB
		);
		vec4 metallicRoughnessTexel = texture(uMetallicRoughnessMap, metallicRoughnessUv);
		roughness = clamp(roughness * metallicRoughnessTexel.g, 0.04, 1.0);
		metalness = clamp(metalness * metallicRoughnessTexel.b, 0.0, 1.0);
	}
	float occlusion = 1.0;
	if (uHasOcclusionMap == 1) {
		vec2 occlusionUv = resolveMappedUV(
			uOcclusionMapUV,
			uOcclusionMapTransformA,
			uOcclusionMapTransformB
		);
		float occlusionTexel = texture(uOcclusionMap, occlusionUv).r;
		occlusion = clamp(
			1.0 + clamp(uOcclusionStrength, 0.0, 1.0) * (occlusionTexel - 1.0),
			0.0,
			1.0
		);
	}
	float iridescence = clamp(uIridescence.x, 0.0, 1.0);
	if (uHasIridescenceMap == 1) {
		vec2 iridescenceUv = resolveMappedUV(
			uIridescenceMapUV,
			uIridescenceMapTransformA,
			uIridescenceMapTransformB
		);
		iridescence *= texture(uIridescenceMap, iridescenceUv).r;
	}
	float iridescenceIor = max(uIridescence.y, 1.0);
	float iridescenceThickness = max(uIridescence.w, 0.0);
	if (uHasIridescenceThicknessMap == 1) {
		vec2 iridescenceThicknessUv = resolveMappedUV(
			uIridescenceThicknessMapUV,
			uIridescenceThicknessMapTransformA,
			uIridescenceThicknessMapTransformB
		);
		iridescenceThickness = max(
			mix(
				uIridescence.z,
				uIridescence.w,
				texture(uIridescenceThicknessMap, iridescenceThicknessUv).g
			),
			0.0
		);
	}

	vec3 normal = normalize(vNormal);
	vec3 shadowNormal = normal;
	vec3 viewDir = safeNormalize(uCameraPosition - vWorldPos, vec3(0.0, 0.0, 1.0));
	if (uDoubleSided == 1 && dot(normal, viewDir) < 0.0) {
		normal = -normal;
	}
	vec2 normalUv = resolveMappedUV(
		uNormalMapUV,
		uNormalMapTransformA,
		uNormalMapTransformB
	);
	if (uShadingModel == 1 && uHasNormalMap == 1) {
		normal = applyNormalMap(
			normal,
			normalUv,
			texture(uNormalMap, normalUv).xyz,
			max(uNormalScale, 0.0)
		);
	}
	float anisotropyStrength = clamp(uAnisotropy.x, 0.0, 1.0);
	vec2 anisotropyDirection = vec2(1.0, 0.0);
	vec2 anisotropyFrameUv = uHasNormalMap == 1 ? normalUv : baseUv;
	if (uHasAnisotropyMap == 1) {
		vec2 anisotropyUv = resolveMappedUV(
			uAnisotropyMapUV,
			uAnisotropyMapTransformA,
			uAnisotropyMapTransformB
		);
		vec3 anisotropyTexel = texture(uIridescenceThicknessMap, anisotropyUv).rgb;
		anisotropyDirection = anisotropyTexel.rg * 2.0 - vec2(1.0);
		float anisotropyDirectionLen = length(anisotropyDirection);
		anisotropyDirection =
			anisotropyDirectionLen > EPSILON ?
				anisotropyDirection / anisotropyDirectionLen
			:	vec2(1.0, 0.0);
		anisotropyStrength = clamp(
			anisotropyStrength * anisotropyTexel.b,
			0.0,
			1.0
		);
		anisotropyFrameUv = anisotropyUv;
	}
	anisotropyDirection = rotateAnisotropyDirection(anisotropyDirection);
	vec3 anisotropyBaseTangent;
	vec3 anisotropyBaseBitangent;
	resolveDerivativeTangentFrame(
		normal,
		anisotropyFrameUv,
		anisotropyBaseTangent,
		anisotropyBaseBitangent
	);
	vec3 anisotropyTangent = safeNormalize(
		anisotropyBaseTangent * anisotropyDirection.x +
			anisotropyBaseBitangent * anisotropyDirection.y,
		anisotropyBaseTangent
	);
	vec3 anisotropyBitangent = safeNormalize(
		cross(normal, anisotropyTangent),
		anisotropyBaseBitangent
	);
	vec3 emissive = uEmissive.rgb;
	if (uHasEmissiveMap == 1) {
		vec2 emissiveUv = resolveMappedUV(
			uEmissiveMapUV,
			uEmissiveMapTransformA,
			uEmissiveMapTransformB
		);
		vec3 emissiveTexel = texture(uEmissiveMap, emissiveUv).rgb;
		emissive *=
			uEmissiveMapIsLinear == 1 ? emissiveTexel : srgbToLinear(emissiveTexel);
	}
	vec3 color;
	if (uEnableLighting == 0 || uShadingModel == 2) {
		color = albedo;
	} else if (uShadingModel == 1) {
		color = shadePBR(
			albedo,
			normal,
			shadowNormal,
			viewDir,
			roughness,
			metalness,
			reflectance,
			transmission,
			anisotropyStrength,
			anisotropyTangent,
			anisotropyBitangent,
			iridescence,
			iridescenceThickness,
			iridescenceIor,
			occlusion
		);
		if (transmission > EPSILON) {
			float dielectricF0 = 0.16 * reflectance * reflectance;
			vec3 f0 = mix(vec3(dielectricF0), albedo, metalness);
			float nDotV = max(dot(normal, viewDir), PBR_MIN_NDOTV);
			vec3 fresnel = resolveIridescenceFresnel(
				nDotV,
				f0,
				iridescence,
				iridescenceThickness,
				iridescenceIor
			);
			float fresnelAverage = clamp(
				dot(fresnel, vec3(1.0 / 3.0)),
				0.0,
				1.0
			);
			alpha = resolveTransmissionAlpha(alpha, transmission, fresnelAverage);
		}
	} else {
		color = shadePhong(albedo, normal, shadowNormal, viewDir);
	}

	color += emissive;
	int fogMode = int(floor(uFogParams0.x + 0.5));
	float fogFactor = ignisComputeFogFactor(
		fogMode,
		max(vViewDepth, 0.0),
		uFogParams0.y,
		uFogParams0.z,
		uFogParams0.w,
		uFogParams1.w
	);
	color = max(mix(color, uFogParams1.rgb, fogFactor), vec3(0.0));
	vec3 finalColor = max(color, vec3(0.0));
	float finalAlpha = clamp(alpha, 0.0, 1.0);
	if (uOITPassMode == 1) {
		float weight = resolveOITWeight(finalAlpha, max(vViewDepth, 0.0));
		fragColor = vec4(finalColor * finalAlpha, finalAlpha) * weight;
		fragMotion = vec4(0.0);
		fragNormal = vec4(0.0);
		return;
	}
	if (uOITPassMode == 2) {
		fragColor = vec4(finalAlpha);
		fragMotion = vec4(0.0);
		fragNormal = vec4(0.0);
		return;
	}
	fragColor = vec4(finalColor, finalAlpha);
	fragNormal = vec4(normal * 0.5 + 0.5, 1.0);
	vec2 curUV = (vCurrentClip.xy / vCurrentClip.w) * 0.5 + 0.5;
	vec2 prevUV = (vPrevClip.xy / vPrevClip.w) * 0.5 + 0.5;
	fragMotion = vec4(curUV - prevUV, vViewDepth, 1.0);
}
