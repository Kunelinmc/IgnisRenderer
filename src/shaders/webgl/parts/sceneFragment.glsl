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

uniform vec3 uCameraPosition;
uniform vec3 uAmbientColor;
uniform int uEnableLighting;
uniform int uShadingModel;
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

out vec4 fragColor;

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

vec3 shadePhong(vec3 albedo, vec3 n, vec3 v) {
	vec3 lit = uAmbientColor * albedo;
	vec3 specular = vec3(0.0);
	float shininess = max(1.0, uPhong.x);

	for (int i = 0; i < MAX_DIRECTIONAL_LIGHTS; i++) {
		if (i >= uDirLightCount) break;
		vec3 l = safeNormalize(uDirLightDirection[i].xyz, vec3(0.0, 1.0, 0.0));
		float nDotL = max(dot(n, l), 0.0);
		lit += albedo * uDirLightColor[i].xyz * nDotL;
		if (nDotL > 0.0) {
			vec3 h = safeNormalize(l + v, v);
			specular +=
				uDirLightColor[i].xyz * pow(max(dot(n, h), 0.0), shininess);
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
		lit +=
			albedo * uSpotLightColorInner[i].xyz *
			nDotL * attenuation * coneFactor;
		if (nDotL > 0.0) {
			vec3 h = safeNormalize(l + v, v);
			specular +=
				uSpotLightColorInner[i].xyz *
				pow(max(dot(n, h), 0.0), shininess) *
				attenuation *
				coneFactor;
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

vec3 shadePBR(vec3 albedo, vec3 pbrNormal, vec3 viewDir) {
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
		);
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
	vec3 viewDir = safeNormalize(uCameraPosition - vWorldPos, vec3(0.0, 0.0, 1.0));
	vec3 color;
	if (uEnableLighting == 0 || uShadingModel == 2) {
		color = albedo;
	} else if (uShadingModel == 1) {
		color = shadePBR(albedo, normal, viewDir);
	} else {
		color = shadePhong(albedo, normal, viewDir);
	}

	color += uEmissive.rgb;
	fragColor = vec4(max(color, vec3(0.0)), alpha);
}