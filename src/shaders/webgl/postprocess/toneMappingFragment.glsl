#version 300 es
precision highp float;

in vec2 vUv;
uniform sampler2D uSourceMap;
uniform float uExposure;

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

void main() {
	vec4 sampled = texture(uSourceMap, vUv);
	vec3 mapped = acesFitted(max(sampled.rgb * uExposure, vec3(0.0)));
	fragColor = vec4(mapped, sampled.a);
}
