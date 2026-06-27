#version 300 es
precision highp float;

in vec2 vUv;

uniform sampler2D uSceneColor;
uniform sampler2D uOITAccumMap;
uniform sampler2D uOITRevealMap;

out vec4 fragColor;

void main() {
	vec4 base = texture(uSceneColor, vUv);
	vec4 accum = texture(uOITAccumMap, vUv);
	float reveal = clamp(texture(uOITRevealMap, vUv).r, 0.0, 1.0);
	vec3 weightedColor = accum.rgb / max(accum.a, 1e-5);
	float alpha = clamp(1.0 - reveal, 0.0, 1.0);
	vec3 color = weightedColor * alpha + base.rgb * reveal;
	fragColor = vec4(max(color, vec3(0.0)), base.a);
}
