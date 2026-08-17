#version 300 es
precision highp float;

in vec2 vUv;
uniform sampler2D uSourceMap;
uniform float uExposure;
uniform float uHdrHeadroom;
uniform float uHdrEnabled;

out vec4 fragColor;

vec3 acesFitted(vec3 color) {
	const float a = 2.51;
	const float b = 0.03;
	const float c = 2.43;
	const float d = 0.59;
	const float e = 0.14;
	vec3 mapped =
		(color * (a * color + vec3(b))) /
		(color * (c * color + vec3(d)) + vec3(e));
	return clamp(mapped, vec3(0.0), vec3(1.0));
}

vec3 hdrSoftShoulder(vec3 color, float headroom) {
	vec3 positive = max(color, vec3(0.0));
	float peak = max(positive.r, max(positive.g, positive.b));
	if (peak <= 1.0) {
		return positive;
	}
	if (headroom <= 1.0001) {
		return clamp(positive, vec3(0.0), vec3(1.0));
	}
	float mappedPeak =
		1.0 + (headroom - 1.0) *
		(1.0 - exp(-(peak - 1.0) / (headroom - 1.0)));
	return positive * (mappedPeak / max(peak, 1e-6));
}

void main() {
	vec4 sampled = texture(uSourceMap, vUv);
	vec3 exposed = max(sampled.rgb * uExposure, vec3(0.0));
	vec3 mapped = uHdrEnabled > 0.5 ?
		hdrSoftShoulder(exposed, uHdrHeadroom) : acesFitted(exposed);
	fragColor = vec4(mapped, sampled.a);
}
