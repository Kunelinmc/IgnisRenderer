#version 300 es
precision highp float;

in vec2 vUv;

uniform sampler2D uSourceMap;
uniform int uApplyGamma;

out vec4 fragColor;

vec3 linearToSrgb(vec3 c) {
	vec3 a = c * 12.92;
	vec3 b = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
	return mix(b, a, lessThanEqual(c, vec3(0.0031308)));
}

void main() {
	vec4 sampled = texture(uSourceMap, vUv);
	vec3 color = sampled.rgb;
	if (uApplyGamma == 1) {
		color = linearToSrgb(max(color, vec3(0.0)));
	}
	fragColor = vec4(clamp(color, 0.0, 1.0), sampled.a);
}