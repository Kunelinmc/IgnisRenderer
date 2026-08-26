#version 300 es
precision highp float;
#import <ignis/postprocess/luma-common>
#import <ignis/webgl/constants>

in vec2 vUv;

uniform sampler2D uSourceMap;
uniform sampler2D uMotionDepthMap;
uniform vec2 uTexelSize;
uniform vec4 uFocusParams;
// x=focusDistance, y=focusRange, z=nearStrength, w=farStrength
uniform vec4 uDOFParams;
// x=maxBlurRadius, y=depthCurve, z=highlightThreshold, w=highlightGain
uniform float uChromaticAberration;

out vec4 fragColor;

float computeCoC(float depth) {
	if (depth <= 0.0) {
		return 0.0;
	}
	float normalized = abs(depth - uFocusParams.x) / max(uFocusParams.y, 1e-4);
	float shaped = pow(clamp(normalized, 0.0, 1.0), max(uDOFParams.y, 0.25));
	float strength = depth > uFocusParams.x ? max(uFocusParams.w, 0.0) :
		max(uFocusParams.z, 0.0);
	return clamp(shaped * strength, 0.0, 1.0);
}

float depthGate(float centerDepth, float sampleDepth, bool isFar) {
	if (centerDepth <= 0.0 || sampleDepth <= 0.0) {
		return 0.0;
	}
	float rel = abs(centerDepth - sampleDepth) /
		max(max(centerDepth, sampleDepth), 1e-4);
	float gate = 1.0 - smoothstep(0.015, 0.08, rel);
	if (isFar && sampleDepth + 0.001 < centerDepth) {
		gate *= 0.15;
	}
	if (!isFar && sampleDepth - 0.001 > centerDepth) {
		gate *= 0.2;
	}
	return gate;
}

const vec2 DOF_OFFSETS[12] = vec2[](
	vec2(1.0, 0.0),
	vec2(-1.0, 0.0),
	vec2(0.0, 1.0),
	vec2(0.0, -1.0),
	vec2(0.707, 0.707),
	vec2(-0.707, 0.707),
	vec2(0.707, -0.707),
	vec2(-0.707, -0.707),
	vec2(2.0, 0.0),
	vec2(-2.0, 0.0),
	vec2(0.0, 2.0),
	vec2(0.0, -2.0)
);

void main() {
	vec4 source = texture(uSourceMap, vUv);
	float centerDepth = texture(uMotionDepthMap, vUv).z;
	float coc = computeCoC(centerDepth);
	float radiusPx = coc * max(uDOFParams.x, 0.0);
	if (radiusPx <= 0.25) {
		fragColor = source;
		return;
	}

	vec2 blurScale = uTexelSize * radiusPx;
	bool isFar = centerDepth > uFocusParams.x;
	vec2 uvMin = uTexelSize * 0.5;
	vec2 uvMax = vec2(1.0) - uTexelSize * 0.5;

	vec4 accum = source;
	float weight = 1.0;
	for (int i = 0; i < 12; i++) {
		vec2 sampleUv = clamp(vUv + DOF_OFFSETS[i] * blurScale, uvMin, uvMax);
		vec4 sampleColor = texture(uSourceMap, sampleUv);
		float sampleDepth = texture(uMotionDepthMap, sampleUv).z;
		float gate = depthGate(centerDepth, sampleDepth, isFar);
		float highlight = max(
			ignisLuma(sampleColor.rgb, IGNIS_LUMA_WEIGHTS_BT709, true) -
				uDOFParams.z,
			0.0
		) * max(uDOFParams.w, 0.0);
		float sampleWeight = gate * (1.0 + highlight);
		accum += sampleColor * sampleWeight;
		weight += sampleWeight;
	}

	vec4 filtered = mix(source, accum / max(weight, 1e-4), coc);
	if (uChromaticAberration > 0.0) {
		vec2 radial = vUv - vec2(0.5);
		float radialLenSq = dot(radial, radial);
		if (radialLenSq < EPSILON) {
			radial = vec2(1.0, 0.0);
		} else {
			radial = normalize(radial);
		}
		vec2 chromaOffset =
			uTexelSize * coc * uChromaticAberration * radiusPx * 0.15;
		vec2 redUv = clamp(vUv + radial * chromaOffset, uvMin, uvMax);
		vec2 blueUv = clamp(vUv - radial * chromaOffset, uvMin, uvMax);
		float red = texture(uSourceMap, redUv).r;
		float blue = texture(uSourceMap, blueUv).b;
		filtered = vec4(red, filtered.g, blue, filtered.a);
	}

	fragColor = vec4(max(filtered.rgb, vec3(0.0)), clamp(filtered.a, 0.0, 1.0));
}
