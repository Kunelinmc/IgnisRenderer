#version 300 es
precision highp float;

in vec2 vUv;

uniform sampler2D uSceneColor;
uniform sampler2D uAoMap;
uniform vec4 uInvSize; // fullInvW, fullInvH, aoInvW, aoInvH

out vec4 fragColor;

void main() {
	vec4 color = texture(uSceneColor, vUv);
	float ao = texture(uAoMap, vUv).x;
	fragColor = vec4(
		max(color.rgb * clamp(ao, 0.0, 1.0), vec3(0.0)),
		color.a
	);
}
