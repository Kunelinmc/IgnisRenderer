#version 300 es
precision highp float;

const int MAX_DIRECTIONAL_LIGHTS = __MAX_DIRECTIONAL_LIGHTS__;
const int MAX_POINT_LIGHTS = __MAX_POINT_LIGHTS__;
const int MAX_SPOT_LIGHTS = __MAX_SPOT_LIGHTS__;

const float PI = 3.14159265359;
const float EPSILON = 0.000001;
const float PBR_MIN_NDOTV = 0.001;
const float PBR_SPEC_FALLBACK = 0.02;
const float PBR_AMBIENT_FALLBACK_LINEAR = 0.05;

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUv;
in vec4 vCurrentClip;
in vec4 vPrevClip;
in float vViewDepth;

uniform vec3 uCameraPosition;
uniform vec3 uAmbientColor;
uniform int uEnableLighting;
uniform int uShadingModel;
uniform int uDoubleSided;
uniform vec4 uBaseColor;
uniform vec4 uEmissive;
uniform vec4 uPBR;
uniform vec4 uPhong;
uniform vec4 uAlpha;
uniform sampler2D uBaseMap;
uniform int uHasBaseMap;
uniform int uBaseMapIsLinear;

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
uniform mat4 uDirShadowViewProjection[MAX_DIRECTIONAL_LIGHTS];
uniform vec4 uDirShadowParamsA[MAX_DIRECTIONAL_LIGHTS];
uniform vec4 uDirShadowParamsB[MAX_DIRECTIONAL_LIGHTS];
uniform vec4 uDirShadowParamsC[MAX_DIRECTIONAL_LIGHTS];
uniform mat4 uSpotShadowViewProjection[MAX_SPOT_LIGHTS];
uniform vec4 uSpotShadowParamsA[MAX_SPOT_LIGHTS];
uniform vec4 uSpotShadowParamsB[MAX_SPOT_LIGHTS];
uniform vec4 uSpotShadowParamsC[MAX_SPOT_LIGHTS];

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragMotion;
layout(location = 2) out vec4 fragNormal;

vec3 srgbToLinear(vec3 c) {
	vec3 a = c / 12.92;
	vec3 b = pow((c + 0.055) / 1.055, vec3(2.4));
	return mix(b, a, lessThanEqual(c, vec3(0.04045)));
}

vec3 safeNormalize(vec3 value, vec3 fallback) {
	float len = length(value);
	return len > EPSILON ? value / len : fallback;
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

float sampleShadowVisibility(
	int shadowType,
	int index,
	mat4 shadowViewProjection,
	vec4 paramsA,
	vec4 paramsB,
	vec4 paramsC,
	vec3 worldPosition,
	vec3 normal,
	vec3 lightDirection
) {
	if (uEnableShadows == 0 || paramsA.x < 0.5) {
		return 1.0;
	}
	if (dot(normal, lightDirection) <= 0.0) {
		return 1.0;
	}

	float shadowSize = max(paramsB.z, 1.0);
	float atlasTileSize = max(paramsB.w, shadowSize);
	float slopeBias = max(paramsC.x, 0.0);
	float maxNormalBias = max(paramsA.z, 0.0);
	float minNormalBias = max(paramsA.w, 0.0);
	float cosTheta = max(dot(normal, lightDirection), 0.0);
	float bias = max(paramsA.y + slopeBias * (1.0 - cosTheta), 0.0);
	float normalBias = mix(minNormalBias, maxNormalBias, 1.0 - cosTheta);
	vec3 shadowWorldPosition = worldPosition + normal * normalBias;
	vec4 shadowClip = shadowViewProjection * vec4(shadowWorldPosition, 1.0);
	if (shadowClip.w <= EPSILON) {
		return 1.0;
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
		return 1.0;
	}

	float pcfRadius = max(paramsB.x, 1.0);
	vec2 texelPosition = shadowUv * vec2(shadowSize - 1.0);
	vec2 atlasExtent = vec2(textureSize(uShadowAtlas, 0));
	if (atlasExtent.x < 1.0 || atlasExtent.y < 1.0) {
		return 1.0;
	}
	float atlasColumns = max(floor(atlasExtent.x / max(atlasTileSize, 1.0)), 1.0);
	float tileIndex = shadowType == 1 ?
		float(MAX_DIRECTIONAL_LIGHTS + index) :
		float(index);
	float tileX = mod(tileIndex, atlasColumns);
	float tileY = floor(tileIndex / atlasColumns);
	vec2 tileOffset = vec2(tileX * atlasTileSize, tileY * atlasTileSize);
	float visible = 0.0;
	float sampleCount = 0.0;

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
			visible += currentDepth - bias <= sampleDepth ? 1.0 : 0.0;
			sampleCount += 1.0;
		}
	}

	if (sampleCount <= 0.0) {
		return 1.0;
	}

	float filteredVisibility = visible / sampleCount;
	float strength = clamp(paramsB.y, 0.0, 1.0);
	return 1.0 - strength + strength * filteredVisibility;
}

float sampleDirectionalShadowVisibility(
	int index,
	vec3 worldPosition,
	vec3 normal,
	vec3 lightDirection
) {
	return sampleShadowVisibility(
		0,
		index,
		uDirShadowViewProjection[index],
		uDirShadowParamsA[index],
		uDirShadowParamsB[index],
		uDirShadowParamsC[index],
		worldPosition,
		normal,
		lightDirection
	);
}

float sampleSpotShadowVisibility(
	int index,
	vec3 worldPosition,
	vec3 normal,
	vec3 lightDirection
) {
	return sampleShadowVisibility(
		1,
		index,
		uSpotShadowViewProjection[index],
		uSpotShadowParamsA[index],
		uSpotShadowParamsB[index],
		uSpotShadowParamsC[index],
		worldPosition,
		normal,
		lightDirection
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

vec3 shadePhong(vec3 albedo, vec3 n, vec3 shadowNormal, vec3 v) {
	vec3 lit = uAmbientColor * albedo;
	vec3 specular = vec3(0.0);
	float shininess = max(1.0, uPhong.x);

	for (int i = 0; i < MAX_DIRECTIONAL_LIGHTS; i++) {
		if (i >= uDirLightCount) break;
		vec3 l = safeNormalize(uDirLightDirection[i].xyz, vec3(0.0, 1.0, 0.0));
		float nDotL = max(dot(n, l), 0.0);
		float shadow = sampleDirectionalShadowVisibility(
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
		float shadow = sampleSpotShadowVisibility(
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
	vec3 f0,
	float nDotV
) {
	float nDotL = max(dot(pbrNormal, lightDir), 0.0);
	if (nDotL <= 0.0) {
		return vec3(0.0);
	}

	vec3 halfVector = safeNormalize(viewDir + lightDir, viewDir);
	float ndf = distributionGGX(pbrNormal, halfVector, roughness);
	float geometry = geometrySmith(nDotV, nDotL, roughness);
	vec3 fresnel = fresnelSchlick(max(dot(halfVector, viewDir), 0.0), f0);
	float denominator = max(4.0 * nDotV * nDotL, 0.0001);
	vec3 specular = (ndf * geometry * fresnel) / denominator;

	vec3 kd = (vec3(1.0) - fresnel) * (1.0 - metalness);
	vec3 diffuse = (kd * albedo) / PI;
	return (diffuse + specular) * radiance * nDotL;
}

vec3 shadePBR(
	vec3 albedo,
	vec3 pbrNormal,
	vec3 shadowNormal,
	vec3 viewDir
) {
	float roughness = clamp(uPBR.x, 0.04, 1.0);
	float metalness = clamp(uPBR.y, 0.0, 1.0);
	float reflectance = clamp(uPBR.z, 0.0, 1.0);
	float dielectricF0 = 0.16 * reflectance * reflectance;
	vec3 f0 = mix(vec3(dielectricF0), albedo, metalness);
	float nDotV = max(dot(pbrNormal, viewDir), PBR_MIN_NDOTV);

	vec3 ambientBase = uAmbientColor;
	if (ambientBase.x + ambientBase.y + ambientBase.z <= 0.0) {
		ambientBase = vec3(PBR_AMBIENT_FALLBACK_LINEAR);
	}
	vec3 ambientFresnel = fresnelSchlick(nDotV, f0);
	vec3 ambientDiffuse = ambientBase *
		albedo *
		(vec3(1.0) - ambientFresnel) *
		(1.0 - metalness);
	float specularAmbientFactor = max(PBR_SPEC_FALLBACK, (1.0 - roughness) * 0.5);
	vec3 ambientSpecular = ambientBase * ambientFresnel * specularAmbientFactor;

	vec3 directLight = vec3(0.0);

	for (int i = 0; i < MAX_DIRECTIONAL_LIGHTS; i++) {
		if (i >= uDirLightCount) break;
		vec3 lightDir = safeNormalize(
			uDirLightDirection[i].xyz,
			vec3(0.0, 1.0, 0.0)
		);
		float shadow = sampleDirectionalShadowVisibility(
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
			f0,
			nDotV
		) * shadow;
	}

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
			f0,
			nDotV
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
		float shadow = sampleSpotShadowVisibility(
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
			f0,
			nDotV
		) * shadow;
	}

	return ambientDiffuse + ambientSpecular + directLight;
}

void main() {
	vec3 albedo = uBaseColor.rgb;
	float alpha = clamp(uBaseColor.a, 0.0, 1.0);
	if (uHasBaseMap == 1) {
		vec4 texel = texture(uBaseMap, vUv);
		vec3 texColor = uBaseMapIsLinear == 1 ? texel.rgb : srgbToLinear(texel.rgb);
		albedo *= texColor;
		alpha *= texel.a;
	}

	if (uAlpha.y > 0.5 && alpha < uAlpha.x) {
		discard;
	}

	vec3 normal = normalize(vNormal);
	vec3 shadowNormal = normal;
	vec3 viewDir = safeNormalize(uCameraPosition - vWorldPos, vec3(0.0, 0.0, 1.0));
	if (uDoubleSided == 1 && dot(normal, viewDir) < 0.0) {
		normal = -normal;
	}
	vec3 color;
	if (uEnableLighting == 0 || uShadingModel == 2) {
		color = albedo;
	} else if (uShadingModel == 1) {
		color = shadePBR(albedo, normal, shadowNormal, viewDir);
	} else {
		color = shadePhong(albedo, normal, shadowNormal, viewDir);
	}

	color += uEmissive.rgb;
	fragColor = vec4(max(color, vec3(0.0)), alpha);
	fragNormal = vec4(normal * 0.5 + 0.5, 1.0);
	vec2 curUV = (vCurrentClip.xy / vCurrentClip.w) * 0.5 + 0.5;
	vec2 prevUV = (vPrevClip.xy / vPrevClip.w) * 0.5 + 0.5;
	fragMotion = vec4(curUV - prevUV, vViewDepth, 1.0);
}
