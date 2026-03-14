#version 300 es
precision highp float;

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;

uniform mat4 uModel;
uniform mat4 uViewProjection;
uniform mat3 uNormalMatrix;
uniform vec4 uTaaJitter;
uniform mat4 uPrevViewProjection;
uniform mat4 uPrevModel;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUv;
out vec4 vCurrentClip;
out vec4 vPrevClip;

void main() {
	vec4 worldPos = uModel * vec4(aPosition, 1.0);
	vWorldPos = worldPos.xyz;
	vNormal = normalize(uNormalMatrix * aNormal);
	vUv = aUv;
	
	vec4 clipPos = uViewProjection * worldPos;
	clipPos.xy += uTaaJitter.xy * clipPos.w;
	vCurrentClip = clipPos;
	gl_Position = clipPos;
	
	vec4 prevWorldPos = uPrevModel * vec4(aPosition, 1.0);
	vec4 prevClipPos = uPrevViewProjection * prevWorldPos;
	prevClipPos.xy += uTaaJitter.zw * prevClipPos.w;
	vPrevClip = prevClipPos;
}