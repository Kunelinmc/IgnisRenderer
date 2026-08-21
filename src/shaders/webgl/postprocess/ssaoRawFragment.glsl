#version 300 es
precision highp float;
#import <ignis/webgl/constants>

in vec2 vUv;

uniform sampler2D uNormalMap;
uniform sampler2D uDepthMap;
uniform vec4 uInvSize; // fullInvW, fullInvH, aoInvW, aoInvH
uniform vec4 uGTAO; // radius, bias, intensity, samples
uniform vec4 uBlurProj; // blurRadius, blurSharpness, tanHalfFov, aspect
uniform vec4 uPass; // blurDirX, blurDirY, isOrthographic, frameJitter
uniform vec3 uCameraPosition;
uniform vec3 uBasisRight;
uniform vec3 uBasisUp;
uniform vec3 uBasisBackward;

out vec4 fragColor;

const int MAX_DIRECTION_COUNT = 8;
const int MAX_STEP_COUNT = 6;

float saturate(float value) {
	return clamp(value, 0.0, 1.0);
}

vec3 decodeNormal(vec3 encoded) {
	vec3 normal = encoded * 2.0 - vec3(1.0);
	float len = length(normal);
	return len > 1e-5 ? normal / len : vec3(0.0, 0.0, 1.0);
}

vec3 reconstructWorldPos(vec2 uv, float depth) {
	vec2 ndc = uv * 2.0 - 1.0;
	if (uPass.z > 0.5) {
		return
			uCameraPosition +
			uBasisRight * ndc.x +
			uBasisUp * ndc.y -
			uBasisBackward * depth;
	}
	float tanHalfFov = max(uBlurProj.z, 1e-4);
	float aspect = max(uBlurProj.w, 1e-4);
	float x = ndc.x * aspect * tanHalfFov * depth;
	float y = ndc.y * tanHalfFov * depth;
	return
		uCameraPosition +
		uBasisRight * x +
		uBasisUp * y -
		uBasisBackward * depth;
}

float interleavedGradientNoise(vec2 pixel, float frameJitter) {
	float seed = dot(pixel, vec2(0.06711056, 0.00583715));
	return fract(52.9829189 * fract(seed + frameJitter * 0.754877666));
}

void main() {
	vec2 uv = vUv;
	float depth = texture(uDepthMap, uv).z;
	if (depth <= 0.0) {
		fragColor = vec4(1.0);
		return;
	}

	vec3 normal = decodeNormal(texture(uNormalMap, uv).xyz);
	vec3 centerPos = reconstructWorldPos(uv, depth);

	int sampleBudget = clamp(int(uGTAO.w + 0.5), 4, 48);
	int directionCount = clamp((sampleBudget + 3) / 4, 2, MAX_DIRECTION_COUNT);
	int stepCount = clamp(
		(sampleBudget + directionCount * 2 - 1) / (directionCount * 2),
		1,
		MAX_STEP_COUNT
	);

	float radiusPixels = max(uGTAO.x, 1.0);
	vec2 radiusUv = radiusPixels * uInvSize.xy;
	float bias = max(uGTAO.y, 1e-4);
	float intensity = max(uGTAO.z, 0.0);
	float frameNoise = interleavedGradientNoise(gl_FragCoord.xy, uPass.w);

	float perspectiveRadiusView =
		radiusPixels *
		depth *
		max(uBlurProj.z, 0.05) *
		max(uInvSize.x, uInvSize.y) *
		2.0;
	float orthographicRadiusView = max(radiusUv.x, radiusUv.y) * 2.0;
	float radiusView = max(
		(uPass.z > 0.5) ? orthographicRadiusView : perspectiveRadiusView,
		1e-3
	);

	float occ = 0.0;
	for (int dirIdx = 0; dirIdx < MAX_DIRECTION_COUNT; dirIdx++) {
		if (dirIdx >= directionCount) {
			break;
		}
		float angle = ((float(dirIdx) + frameNoise) / float(directionCount)) * PI;
		vec2 dir2 = vec2(cos(angle), sin(angle));
		float horizonPos = 0.0;
		float horizonNeg = 0.0;

		for (int stepIdx = 1; stepIdx <= MAX_STEP_COUNT; stepIdx++) {
			if (stepIdx > stepCount) {
				break;
			}
			float jitter = fract(frameNoise + float(stepIdx) * GOLDEN_RATIO_CONJUGATE);
			float stepFrac =
				(float(stepIdx) - 0.35 + jitter * 0.6) / float(stepCount);
			vec2 stepUv = dir2 * stepFrac * radiusUv;

			vec2 sampleUvPos = uv + stepUv;
			if (all(greaterThanEqual(sampleUvPos, vec2(0.0))) &&
				all(lessThanEqual(sampleUvPos, vec2(1.0)))) {
				float sampleDepthPos = texture(uDepthMap, sampleUvPos).z;
				if (sampleDepthPos > 0.0 && sampleDepthPos < depth - bias) {
					vec3 samplePos = reconstructWorldPos(sampleUvPos, sampleDepthPos);
					vec3 delta = samplePos - centerPos;
					float distSq = dot(delta, delta);
					if (distSq > 1e-6) {
						float invDist = inversesqrt(distSq);
						float dist = distSq * invDist;
						float alignment = max(dot(normal, delta * invDist), 0.0);
						float distWeight = saturate(1.0 - dist / radiusView);
						horizonPos = max(horizonPos, alignment * distWeight);
					}
				}
			}

			vec2 sampleUvNeg = uv - stepUv;
			if (all(greaterThanEqual(sampleUvNeg, vec2(0.0))) &&
				all(lessThanEqual(sampleUvNeg, vec2(1.0)))) {
				float sampleDepthNeg = texture(uDepthMap, sampleUvNeg).z;
				if (sampleDepthNeg > 0.0 && sampleDepthNeg < depth - bias) {
					vec3 samplePos = reconstructWorldPos(sampleUvNeg, sampleDepthNeg);
					vec3 delta = samplePos - centerPos;
					float distSq = dot(delta, delta);
					if (distSq > 1e-6) {
						float invDist = inversesqrt(distSq);
						float dist = distSq * invDist;
						float alignment = max(dot(normal, delta * invDist), 0.0);
						float distWeight = saturate(1.0 - dist / radiusView);
						horizonNeg = max(horizonNeg, alignment * distWeight);
					}
				}
			}
		}

		occ += 0.5 * (horizonPos + horizonNeg);
	}

	float horizonOcclusion = occ / max(float(directionCount), 1.0);
	float ao = clamp(1.0 - horizonOcclusion * intensity, 0.0, 1.0);
	fragColor = vec4(vec3(ao), 1.0);
}
