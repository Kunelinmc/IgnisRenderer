#version 300 es
precision highp float;

layout(location = 0) in vec2 aQuadPosition;
layout(location = 1) in vec2 aQuadUv;
layout(location = 2) in vec4 aInstancePositionSize;
layout(location = 3) in vec4 aInstanceColor;
layout(location = 4) in vec4 aInstanceUvRect;
layout(location = 5) in float aInstanceRotation;

uniform mat4 uViewProjection;
uniform vec3 uBasisRight;
uniform vec3 uBasisUp;
uniform vec3 uCameraPosition;

out vec2 vUv;
out vec4 vColor;
out vec2 vLocalUv;
out float vViewDepth;

void main() {
	float c = cos(aInstanceRotation);
	float s = sin(aInstanceRotation);
	vec2 rotated = vec2(
		aQuadPosition.x * c - aQuadPosition.y * s,
		aQuadPosition.x * s + aQuadPosition.y * c
	);
	vec3 worldPosition =
		aInstancePositionSize.xyz +
		(uBasisRight * rotated.x + uBasisUp * rotated.y) *
			aInstancePositionSize.w;

	gl_Position = uViewProjection * vec4(worldPosition, 1.0);
	vUv = vec2(
		mix(aInstanceUvRect.x, aInstanceUvRect.z, aQuadUv.x),
		mix(aInstanceUvRect.y, aInstanceUvRect.w, aQuadUv.y)
	);
	vColor = aInstanceColor;
	vLocalUv = aQuadUv;
	vViewDepth = length(uCameraPosition - worldPosition);
}
