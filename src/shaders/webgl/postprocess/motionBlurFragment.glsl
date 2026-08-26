#version 300 es
precision highp float;
#import <ignis/postprocess/luma-common>
#import <ignis/webgl/constants>

in vec2 vUv;

uniform sampler2D uSourceMap;
uniform sampler2D uMotionDepthMap;
uniform vec2 uTexelSize;
uniform vec4 uMotionParams;
// x=shutterScale, y=maxSamples, z=velocityClamp, w=depthReject
uniform float uCenterWeight;

out vec4 fragColor;

float depthConfidence(float centerDepth, float sampleDepth) {
	if (centerDepth <= 0.0 || sampleDepth <= 0.0) {
		return 0.0;
	}
	float rel = abs(centerDepth - sampleDepth) /
		max(max(centerDepth, sampleDepth), 1e-4);
	float reject = max(uMotionParams.w, 1e-5);
	return 1.0 - smoothstep(reject, reject * 4.0, rel);
}

void main() {
	vec4 source = texture(uSourceMap, vUv);
	vec3 motionDepth = texture(uMotionDepthMap, vUv).xyz;
	float centerDepth = motionDepth.z;

	vec2 velocity =
		vec2(motionDepth.x * 0.5, -motionDepth.y * 0.5) * max(uMotionParams.x, 0.0);
	float velocityMag = length(velocity);
	float minInv = max(max(uTexelSize.x, uTexelSize.y), EPSILON);
	if (velocityMag <= minInv * 0.35) {
		fragColor = source;
		return;
	}

	float maxVelocity = max(uMotionParams.z, minInv);
	float clampScale = min(1.0, maxVelocity / max(velocityMag, EPSILON));
	velocity *= clampScale;

	float pixelVelocity = length(velocity / vec2(minInv, minInv));
	int sampleCount = int(
		clamp(ceil(pixelVelocity), 1.0, max(uMotionParams.y, 1.0))
	);

	vec4 accum = source * max(uCenterWeight, 0.0);
	float weight = max(uCenterWeight, 0.0);
	float sourceLuma = ignisLuma(source.rgb, IGNIS_LUMA_WEIGHTS_BT709, true);
	vec2 uvMin = uTexelSize * 0.5;
	vec2 uvMax = vec2(1.0) - uTexelSize * 0.5;

	for (int i = 1; i <= 64; i++) {
		if (i > sampleCount) {
			break;
		}
		float t = float(i) / float(max(sampleCount, 1));
		vec2 offset = velocity * t;
		vec2 uvA = clamp(vUv - offset, uvMin, uvMax);
		vec2 uvB = clamp(vUv + offset, uvMin, uvMax);

		vec4 sampleA = texture(uSourceMap, uvA);
		vec4 sampleB = texture(uSourceMap, uvB);
		float depthA = texture(uMotionDepthMap, uvA).z;
		float depthB = texture(uMotionDepthMap, uvB).z;

		float motionWeight = 1.0 - t * 0.85;
		float lumaWeightA = 0.5 + 0.5 * clamp(
			ignisLuma(sampleA.rgb, IGNIS_LUMA_WEIGHTS_BT709, true) /
				max(sourceLuma, 1e-4),
			0.0,
			1.5
		);
		float lumaWeightB = 0.5 + 0.5 * clamp(
			ignisLuma(sampleB.rgb, IGNIS_LUMA_WEIGHTS_BT709, true) /
				max(sourceLuma, 1e-4),
			0.0,
			1.5
		);
		float weightA =
			motionWeight * depthConfidence(centerDepth, depthA) * lumaWeightA;
		float weightB =
			motionWeight * depthConfidence(centerDepth, depthB) * lumaWeightB;

		accum += sampleA * weightA;
		accum += sampleB * weightB;
		weight += weightA + weightB;
	}

	vec4 filtered = accum / max(weight, 1e-4);
	fragColor = vec4(max(filtered.rgb, vec3(0.0)), clamp(filtered.a, 0.0, 1.0));
}
