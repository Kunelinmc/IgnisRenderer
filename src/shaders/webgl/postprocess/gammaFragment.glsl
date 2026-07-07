#version 300 es
precision highp float;
#import <ignis/color/srgb>

in vec2 vUv;

uniform sampler2D uSourceMap;

out vec4 fragColor;

void main() {
	vec4 sampled = texture(uSourceMap, vUv);
	vec3 color = linearToSrgb(max(sampled.rgb, vec3(0.0)));
	fragColor = vec4(clamp(color, 0.0, 1.0), sampled.a);
}
