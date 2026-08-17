#version 300 es
precision highp float;
#import <ignis/color/srgb>

in vec2 vUv;

uniform sampler2D uSourceMap;
uniform float uExposure;
uniform float uHdrHeadroom;
uniform float uHdrEnabled;
uniform float uColorDomain;

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
	if (peak <= 1.0) return positive;
	if (headroom <= 1.0001) return clamp(positive, vec3(0.0), vec3(1.0));
	float mappedPeak =
		1.0 + (headroom - 1.0) *
		(1.0 - exp(-(peak - 1.0) / (headroom - 1.0)));
	return positive * (mappedPeak / max(peak, 1e-6));
}

vec3 linearSrgbToDisplayP3(vec3 color) {
	return vec3(
		0.82259287 * color.r + 0.17753395 * color.g,
		0.03319951 * color.r + 0.96678350 * color.g,
		0.01708535 * color.r + 0.07239572 * color.g + 0.91030148 * color.b
	);
}

void main() {
	vec4 sampled = texture(uSourceMap, vUv);
	bool hdr = uHdrEnabled > 0.5;
	int domain = int(uColorDomain + 0.5);
	vec3 color = max(sampled.rgb, vec3(0.0));
	if (domain == 0) {
		vec3 exposed = color * uExposure;
		color = hdr ? hdrSoftShoulder(exposed, uHdrHeadroom) : acesFitted(exposed);
	}
	if (domain < 2) {
		if (hdr) {
			color = linearToSrgb(clamp(
				linearSrgbToDisplayP3(color),
				vec3(0.0),
				vec3(uHdrHeadroom)
			));
		} else {
			color = linearToSrgb(clamp(color, vec3(0.0), vec3(1.0)));
		}
	}
	color = hdr ? max(color, vec3(0.0)) : clamp(color, vec3(0.0), vec3(1.0));
	fragColor = vec4(color, clamp(sampled.a, 0.0, 1.0));
}
