#version 300 es
precision highp float;

layout(location = 0) in vec3 aPosition;
layout(location = 2) in vec2 aUv0;
layout(location = 3) in vec2 aUv1;
layout(location = 4) in vec2 aUv2;
layout(location = 5) in vec2 aUv3;

uniform mat4 uModel;
uniform mat4 uViewProjection;
uniform vec4 uTaaJitter;

out vec2 vUv;
out vec2 vUv1;
out vec2 vUv2;
out vec2 vUv3;

void main() {
	vec4 worldPos = uModel * vec4(aPosition, 1.0);
	vUv = aUv0;
	vUv1 = aUv1;
	vUv2 = aUv2;
	vUv3 = aUv3;

	vec4 clipPos = uViewProjection * worldPos;
	clipPos.xy += uTaaJitter.xy * clipPos.w;
	gl_Position = clipPos;
}
