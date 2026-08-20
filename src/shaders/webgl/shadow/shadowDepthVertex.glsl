#version 300 es
precision highp float;
__IGNIS_WEBGL_ANIMATION_DEFINES__

#if IGNIS_WEBGL_DEFORMATION_ACTIVE
#import <ignis/webgl/animation>
#endif

layout(location = 0) in vec3 aPosition;
#if IGNIS_WEBGL_DEFORMATION_ACTIVE
layout(location = 7) in vec4 aJoints0;
layout(location = 8) in vec4 aWeights0;
#if IGNIS_WEBGL_SKIN_INFLUENCES == 8
layout(location = 9) in vec4 aJoints1;
layout(location = 10) in vec4 aWeights1;
#endif
#endif

uniform mat4 uMvp;

void main() {
	vec3 localPosition = aPosition;
#if IGNIS_WEBGL_DEFORMATION_ACTIVE
	vec4 joints1 = vec4(0.0);
	vec4 weights1 = vec4(0.0);
#if IGNIS_WEBGL_SKIN_INFLUENCES == 8
	joints1 = aJoints1;
	weights1 = aWeights1;
#endif
	localPosition = ignisApplyAnimationPosition(
		aPosition, aJoints0, aWeights0, joints1, weights1,
		uint(gl_VertexID), false
	);
#endif
	gl_Position = uMvp * vec4(localPosition, 1.0);
}
