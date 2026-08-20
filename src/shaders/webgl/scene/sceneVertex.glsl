#version 300 es
precision highp float;
__IGNIS_WEBGL_ANIMATION_DEFINES__

#if IGNIS_WEBGL_DEFORMATION_ACTIVE
#import <ignis/webgl/animation>
#endif

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv0;
layout(location = 3) in vec2 aUv1;
layout(location = 4) in vec2 aUv2;
layout(location = 5) in vec2 aUv3;
layout(location = 6) in vec4 aTangent;
#if IGNIS_WEBGL_DEFORMATION_ACTIVE
layout(location = 7) in vec4 aJoints0;
layout(location = 8) in vec4 aWeights0;
#if IGNIS_WEBGL_SKIN_INFLUENCES == 8
layout(location = 9) in vec4 aJoints1;
layout(location = 10) in vec4 aWeights1;
#endif
#endif

uniform mat4 uModel;
uniform mat4 uViewMatrix;
uniform mat4 uViewProjection;
uniform mat3 uNormalMatrix;
uniform vec4 uTaaJitter;
uniform mat4 uPrevViewProjection;
uniform mat4 uPrevModel;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUv;
out vec2 vUv1;
out vec2 vUv2;
out vec2 vUv3;
out vec4 vTangent;
out vec4 vCurrentClip;
out vec4 vPrevClip;
out float vViewDepth;

void main() {
	vec3 localPosition = aPosition;
	vec3 localNormal = aNormal;
	vec3 localTangent = aTangent.xyz;
	vec3 previousLocalPosition = aPosition;
#if IGNIS_WEBGL_DEFORMATION_ACTIVE
	vec4 joints1 = vec4(0.0);
	vec4 weights1 = vec4(0.0);
#if IGNIS_WEBGL_SKIN_INFLUENCES == 8
	joints1 = aJoints1;
	weights1 = aWeights1;
#endif
	IgnisAnimationVertex currentVertex = ignisApplyAnimationVertex(
		aPosition, aNormal, aTangent.xyz,
		aJoints0, aWeights0, joints1, weights1,
		uint(gl_VertexID), false
	);
	localPosition = currentVertex.position;
	localNormal = currentVertex.normal;
	localTangent = currentVertex.tangent;
	previousLocalPosition = ignisApplyAnimationPosition(
		aPosition, aJoints0, aWeights0, joints1, weights1,
		uint(gl_VertexID), true
	);
#endif
	vec4 worldPos = uModel * vec4(localPosition, 1.0);
	vWorldPos = worldPos.xyz;
	vNormal = normalize(uNormalMatrix * localNormal);
	vUv = aUv0;
	vUv1 = aUv1;
	vUv2 = aUv2;
	vUv3 = aUv3;
	vec3 worldTangent = uNormalMatrix * localTangent;
	float tangentLen = length(worldTangent);
	vTangent = vec4(
		tangentLen > 0.000001 ? worldTangent / tangentLen : vec3(0.0),
		aTangent.w
	);
	vViewDepth = max(-(uViewMatrix * worldPos).z, 0.0);
	
	vec4 clipPos = uViewProjection * worldPos;
	clipPos.xy += uTaaJitter.xy * clipPos.w;
	vCurrentClip = clipPos;
	gl_Position = clipPos;
	
	vec4 prevWorldPos = uPrevModel * vec4(previousLocalPosition, 1.0);
	vec4 prevClipPos = uPrevViewProjection * prevWorldPos;
	prevClipPos.xy += uTaaJitter.zw * prevClipPos.w;
	vPrevClip = prevClipPos;
}
