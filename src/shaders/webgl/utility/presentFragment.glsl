#version 300 es
precision highp float;
#import <ignis/webgl/constants>

in vec2 vUv;

uniform sampler2D uSourceMap;
uniform float uHdrEnabled;
uniform float uTransparentOutput;

out vec4 fragColor;

void main() {
	vec4 sampled = texture(uSourceMap, vUv);
	bool hdr = uHdrEnabled > 0.5;
	bool transparent = uTransparentOutput > 0.5;
	float alpha = transparent ? clamp(sampled.a, 0.0, 1.0) : 1.0;
	vec3 color = transparent ?
		(alpha > EPSILON ? max(sampled.rgb, vec3(0.0)) / alpha : vec3(0.0))
	:	max(sampled.rgb, vec3(0.0));
	color = hdr ? max(color, vec3(0.0)) : clamp(color, vec3(0.0), vec3(1.0));
	fragColor = vec4(transparent ? color * alpha : color, alpha);
}
