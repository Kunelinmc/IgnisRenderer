#version 300 es
precision highp float;
precision highp int;

#import <ignis/webgl/constants>
#import <ignis/color/srgb>

const float PREFILTER_EPSILON = 1e-6;
const float EQUIRECT_DISTORTION_EPSILON = 1e-4;
const int MAX_SAMPLE_COUNT = 256;

uniform sampler2D uEnvironmentMap;
uniform vec2 uOutputSize;
uniform vec2 uSourceSize;
uniform float uRoughness;
uniform int uSampleCount;
uniform int uSourceIsLinear;
uniform int uSourceMipLevelCount;

layout(location = 0) out vec4 fragColor;

float radicalInverseVdC(uint bits) {
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) |
		((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) |
		((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) |
		((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) |
		((bits & 0xFF00FF00u) >> 8u);
	return float(bits) * 2.3283064365386963e-10;
}

vec2 hammersley(uint index, uint count) {
	return vec2(
		float(index) / max(float(count), 1.0),
		radicalInverseVdC(index)
	);
}

vec3 importanceSampleGGX(vec2 xi, vec3 normal, float roughness) {
	float alpha = max(roughness * roughness, 1e-4);
	float alpha2 = alpha * alpha;
	float phi = TWO_PI * xi.x;
	float cosTheta = sqrt(
		(1.0 - xi.y) / (1.0 + (alpha2 - 1.0) * xi.y)
	);
	float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
	vec3 tangentHalf = vec3(
		cos(phi) * sinTheta,
		sin(phi) * sinTheta,
		cosTheta
	);
	vec3 up = abs(normal.y) > 0.999 ?
		vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
	vec3 tangent = normalize(cross(up, normal));
	vec3 bitangent = cross(normal, tangent);
	return normalize(
		tangent * tangentHalf.x +
		bitangent * tangentHalf.y +
		normal * tangentHalf.z
	);
}

float distributionGGX(float nDotH, float roughness) {
	float alpha = max(roughness * roughness, 1e-4);
	float alpha2 = alpha * alpha;
	float denominator = nDotH * nDotH * (alpha2 - 1.0) + 1.0;
	return alpha2 / max(
		PI * denominator * denominator,
		PREFILTER_EPSILON
	);
}

float computeGGXSamplePDF(float nDotH, float vDotH, float roughness) {
	if (nDotH <= 0.0 || vDotH <= 0.0) {
		return PREFILTER_EPSILON;
	}
	float distribution = distributionGGX(nDotH, roughness);
	return max(
		(distribution * nDotH) / max(4.0 * vDotH, PREFILTER_EPSILON),
		PREFILTER_EPSILON
	);
}

float computeEquirectTexelSolidAngle(float directionY) {
	float sinTheta = sqrt(max(1.0 - directionY * directionY, 0.0));
	return (
		2.0 * PI * PI * max(sinTheta, EQUIRECT_DISTORTION_EPSILON)
	) / max(uSourceSize.x * uSourceSize.y, 1.0);
}

float resolveSampleLevel(
	float roughness,
	int sampleCount,
	float pdf,
	float directionY
) {
	if (uSourceMipLevelCount <= 1 || roughness <= PREFILTER_EPSILON) {
		return 0.0;
	}
	float texelSolidAngle = computeEquirectTexelSolidAngle(directionY);
	float sampleSolidAngle = 1.0 / max(
		float(sampleCount) * pdf,
		PREFILTER_EPSILON
	);
	float lod = 0.5 * log2(
		sampleSolidAngle / max(texelSolidAngle, PREFILTER_EPSILON)
	);
	return clamp(lod, 0.0, float(uSourceMipLevelCount - 1));
}

vec2 directionToEquirectUV(vec3 direction) {
	vec3 normalized = normalize(direction);
	float phi = atan(normalized.x, normalized.z);
	float theta = acos(clamp(normalized.y, -1.0, 1.0));
	return vec2((phi + PI) / TWO_PI, theta / PI);
}

void main() {
	// `readPixels` is bottom-row first. Mapping the bottom row to v=0 keeps the
	// CPU-backed equirectangular result in the engine's north-first row order.
	vec2 uv = gl_FragCoord.xy / max(uOutputSize, vec2(1.0));
	float theta = uv.y * PI;
	float phi = uv.x * TWO_PI - PI;
	vec3 normal = vec3(
		sin(theta) * sin(phi),
		cos(theta),
		sin(theta) * cos(phi)
	);
	int sampleCount = max(uSampleCount, 1);
	float totalWeight = 0.0;
	vec3 accumulated = vec3(0.0);

	for (int i = 0; i < MAX_SAMPLE_COUNT; i++) {
		if (i >= sampleCount) {
			break;
		}
		vec2 xi = hammersley(uint(i), uint(sampleCount));
		vec3 halfVector = importanceSampleGGX(xi, normal, uRoughness);
		float nDotH = max(dot(normal, halfVector), 0.0);
		float vDotH = nDotH;
		if (vDotH <= PREFILTER_EPSILON) {
			continue;
		}
		vec3 lightDirection = normalize(
			2.0 * nDotH * halfVector - normal
		);
		float nDotL = max(dot(normal, lightDirection), 0.0);
		if (nDotL <= 0.0) {
			continue;
		}
		float pdf = computeGGXSamplePDF(nDotH, vDotH, uRoughness);
		float sampleLevel = resolveSampleLevel(
			uRoughness,
			sampleCount,
			pdf,
			lightDirection.y
		);
		vec3 sampleColor = textureLod(
			uEnvironmentMap,
			directionToEquirectUV(lightDirection),
			sampleLevel
		).rgb;
		if (uSourceIsLinear == 0) {
			sampleColor = srgbToLinear(sampleColor);
		}
		accumulated += sampleColor * nDotL;
		totalWeight += nDotL;
	}

	vec3 outputColor = totalWeight > PREFILTER_EPSILON ?
		accumulated / totalWeight : vec3(0.0);
	fragColor = vec4(max(outputColor, vec3(0.0)), 1.0);
}
