#version 300 es
precision highp float;
#import <ignis/color/srgb>
#import <ignis/webgl/constants>

in vec2 vUv;

uniform sampler2D uSourceMap;
uniform float uHdrEnabled;
uniform float uHdrHeadroom;

out vec4 fragColor;

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
	float alpha = clamp(sampled.a, 0.0, 1.0);
	vec3 linear = alpha > EPSILON ? max(sampled.rgb, vec3(0.0)) / alpha : vec3(0.0);
	if (hdr) {
		linear = clamp(
			linearSrgbToDisplayP3(linear),
			vec3(0.0),
			vec3(uHdrHeadroom)
		);
	}
	vec3 color = linearToSrgb(linear);
	vec3 encoded = hdr ? max(color, vec3(0.0)) : clamp(color, 0.0, 1.0);
	fragColor = vec4(encoded * alpha, alpha);
}
