#ifndef IGNIS_WEBGL_SKIN_INFLUENCES
#define IGNIS_WEBGL_SKIN_INFLUENCES 8
#endif

uniform highp sampler2D uAnimationPayload;
uniform highp sampler2D uMorphPositionDeltas;
uniform highp sampler2D uMorphNormalDeltas;
uniform ivec4 uAnimationCounts;
uniform ivec4 uAnimationOffsets;
uniform ivec4 uAnimationTextureWidths;

struct IgnisAnimationVertex {
	vec3 position;
	vec3 normal;
	vec3 tangent;
};

vec4 ignisFetchLinear(sampler2D source, int index, int width) {
	int safeWidth = max(width, 1);
	return texelFetch(source, ivec2(index % safeWidth, index / safeWidth), 0);
}

mat4 ignisFetchJointMatrix(int jointIndex, bool previousPose) {
	int base = previousPose ? uAnimationOffsets.y : uAnimationOffsets.x;
	int texel = base + jointIndex * 4;
	int width = uAnimationTextureWidths.x;
	return mat4(
		ignisFetchLinear(uAnimationPayload, texel, width),
		ignisFetchLinear(uAnimationPayload, texel + 1, width),
		ignisFetchLinear(uAnimationPayload, texel + 2, width),
		ignisFetchLinear(uAnimationPayload, texel + 3, width)
	);
}

float ignisFetchMorphWeight(int targetIndex, bool previousPose) {
	int base = previousPose ? uAnimationOffsets.w : uAnimationOffsets.z;
	return ignisFetchLinear(
		uAnimationPayload,
		base + targetIndex,
		uAnimationTextureWidths.x
	).r;
}

float ignisJointComponent(vec4 joints0, vec4 joints1, int influence) {
	if (influence == 0) return joints0.x;
	if (influence == 1) return joints0.y;
	if (influence == 2) return joints0.z;
	if (influence == 3) return joints0.w;
	if (influence == 4) return joints1.x;
	if (influence == 5) return joints1.y;
	if (influence == 6) return joints1.z;
	return joints1.w;
}

float ignisWeightComponent(vec4 weights0, vec4 weights1, int influence) {
	if (influence == 0) return weights0.x;
	if (influence == 1) return weights0.y;
	if (influence == 2) return weights0.z;
	if (influence == 3) return weights0.w;
	if (influence == 4) return weights1.x;
	if (influence == 5) return weights1.y;
	if (influence == 6) return weights1.z;
	return weights1.w;
}

vec3 ignisApplyMorphPosition(vec3 position, uint vertexIndex, bool previousPose) {
	int targetCount = min(uAnimationCounts.y, 8);
	if (targetCount <= 0 || (uAnimationCounts.w & 1) == 0) return position;
	vec3 result = position;
	for (int targetIndex = 0; targetIndex < 8; targetIndex++) {
		if (targetIndex >= targetCount) break;
		float weight = ignisFetchMorphWeight(targetIndex, previousPose);
		if (abs(weight) <= 0.000001) continue;
		int deltaIndex = targetIndex * uAnimationCounts.z + int(vertexIndex);
		result += ignisFetchLinear(
			uMorphPositionDeltas,
			deltaIndex,
			uAnimationTextureWidths.y
		).xyz * weight;
	}
	return result;
}

vec3 ignisApplyMorphNormal(vec3 normal, uint vertexIndex, bool previousPose) {
	int targetCount = min(uAnimationCounts.y, 8);
	if (targetCount <= 0 || (uAnimationCounts.w & 2) == 0) return normal;
	vec3 result = normal;
	for (int targetIndex = 0; targetIndex < 8; targetIndex++) {
		if (targetIndex >= targetCount) break;
		float weight = ignisFetchMorphWeight(targetIndex, previousPose);
		if (abs(weight) <= 0.000001) continue;
		int deltaIndex = targetIndex * uAnimationCounts.z + int(vertexIndex);
		result += ignisFetchLinear(
			uMorphNormalDeltas,
			deltaIndex,
			uAnimationTextureWidths.z
		).xyz * weight;
	}
	return result;
}

vec3 ignisSkinDirection(
	vec3 direction,
	vec4 joints0,
	vec4 weights0,
	vec4 joints1,
	vec4 weights1,
	bool previousPose
) {
	if (uAnimationCounts.x <= 0) return direction;
	vec3 result = vec3(0.0);
	float weightSum = 0.0;
	for (int influence = 0; influence < 8; influence++) {
		if (influence >= IGNIS_WEBGL_SKIN_INFLUENCES) break;
		float weight = ignisWeightComponent(weights0, weights1, influence);
		int jointIndex = int(max(ignisJointComponent(joints0, joints1, influence), 0.0) + 0.5);
		if (weight <= 0.000001 || jointIndex >= uAnimationCounts.x) continue;
		result += (ignisFetchJointMatrix(jointIndex, previousPose) * vec4(direction, 0.0)).xyz * weight;
		weightSum += weight;
	}
	return weightSum > 0.000001 ? result / weightSum : direction;
}

vec3 ignisSkinPosition(
	vec3 position,
	vec4 joints0,
	vec4 weights0,
	vec4 joints1,
	vec4 weights1,
	bool previousPose
) {
	if (uAnimationCounts.x <= 0) return position;
	vec3 result = vec3(0.0);
	float weightSum = 0.0;
	for (int influence = 0; influence < 8; influence++) {
		if (influence >= IGNIS_WEBGL_SKIN_INFLUENCES) break;
		float weight = ignisWeightComponent(weights0, weights1, influence);
		int jointIndex = int(max(ignisJointComponent(joints0, joints1, influence), 0.0) + 0.5);
		if (weight <= 0.000001 || jointIndex >= uAnimationCounts.x) continue;
		result += (ignisFetchJointMatrix(jointIndex, previousPose) * vec4(position, 1.0)).xyz * weight;
		weightSum += weight;
	}
	return weightSum > 0.000001 ? result / weightSum : position;
}

vec3 ignisApplyAnimationPosition(
	vec3 position,
	vec4 joints0,
	vec4 weights0,
	vec4 joints1,
	vec4 weights1,
	uint vertexIndex,
	bool previousPose
) {
	return ignisSkinPosition(
		ignisApplyMorphPosition(position, vertexIndex, previousPose),
		joints0,
		weights0,
		joints1,
		weights1,
		previousPose
	);
}

IgnisAnimationVertex ignisApplyAnimationVertex(
	vec3 position,
	vec3 normal,
	vec3 tangent,
	vec4 joints0,
	vec4 weights0,
	vec4 joints1,
	vec4 weights1,
	uint vertexIndex,
	bool previousPose
) {
	vec3 morphedPosition = ignisApplyMorphPosition(position, vertexIndex, previousPose);
	vec3 morphedNormal = ignisApplyMorphNormal(normal, vertexIndex, previousPose);
	vec3 skinnedNormal = ignisSkinDirection(
		morphedNormal, joints0, weights0, joints1, weights1, previousPose
	);
	vec3 skinnedTangent = ignisSkinDirection(
		tangent, joints0, weights0, joints1, weights1, previousPose
	);
	return IgnisAnimationVertex(
		ignisSkinPosition(
			morphedPosition, joints0, weights0, joints1, weights1, previousPose
		),
		length(skinnedNormal) > 0.000001 ? normalize(skinnedNormal) : normal,
		length(skinnedTangent) > 0.000001 ? normalize(skinnedTangent) : tangent
	);
}
