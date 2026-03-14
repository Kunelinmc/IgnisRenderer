#version 300 es
precision highp float;

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;

uniform mat4 uModel;
uniform mat4 uViewProjection;
uniform mat3 uNormalMatrix;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUv;

void main() {
	vec4 worldPos = uModel * vec4(aPosition, 1.0);
	vWorldPos = worldPos.xyz;
	vNormal = normalize(uNormalMatrix * aNormal);
	vUv = aUv;
	gl_Position = uViewProjection * worldPos;
}