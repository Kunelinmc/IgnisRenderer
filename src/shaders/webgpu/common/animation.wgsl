#import <ignis/webgpu/constants>

#ifndef IGNIS_WEBGPU_ANIMATION_POSITION_ONLY
#define IGNIS_WEBGPU_ANIMATION_POSITION_ONLY 0
#endif

// Consumers own jointMatrices, morphWeights, and morphPositionDeltas bindings.
// Full vertex deformation also requires morphNormalDeltas and safeNormalize.
struct MorphVertex {
	position: vec3<f32>,
	normal: vec3<f32>,
}

struct SkinnedVertex {
	position: vec3<f32>,
	normal: vec3<f32>,
	tangent: vec3<f32>,
}

fn applyMorphDeltas(
	basePosition: vec3<f32>,
	baseNormal: vec3<f32>,
	vertexIndex: u32,
	morphTargetCount: u32,
	morphWeightOffset: u32,
	morphDeltaOffset: u32,
	vertexCount: u32,
	semanticMask: u32
) -> MorphVertex {
	if (morphTargetCount == 0u || vertexCount == 0u) {
		return MorphVertex(basePosition, baseNormal);
	}

	let morphDeltaCount = arrayLength(&morphPositionDeltas);
	let morphWeightCount = arrayLength(&morphWeights);
	if (morphWeightCount == 0u) {
		return MorphVertex(basePosition, baseNormal);
	}
#if IGNIS_WEBGPU_ANIMATION_POSITION_ONLY
	if ((semanticMask & 1u) == 0u || morphDeltaCount == 0u) {
		return MorphVertex(basePosition, baseNormal);
	}
#endif

	// Offsets count float3 deltas; storage arrays contain packed scalar values.
	let deltaBase = min(morphDeltaOffset * 3u, morphDeltaCount);
	var position = basePosition;
	var normal = baseNormal;
	for (var targetIndex: u32 = 0u; targetIndex < morphTargetCount; targetIndex = targetIndex + 1u) {
		let weightIndex = morphWeightOffset + targetIndex;
		if (weightIndex >= morphWeightCount) {
			continue;
		}

		let weight = morphWeights[weightIndex];
		if (abs(weight) <= EPSILON) {
			continue;
		}

		let deltaIndex = deltaBase + (targetIndex * vertexCount + vertexIndex) * 3u;
		if ((semanticMask & 1u) != 0u && deltaIndex + 2u < morphDeltaCount) {
			position += vec3<f32>(
				morphPositionDeltas[deltaIndex],
				morphPositionDeltas[deltaIndex + 1u],
				morphPositionDeltas[deltaIndex + 2u]
			) * weight;
		}
#if !IGNIS_WEBGPU_ANIMATION_POSITION_ONLY
		if (
			(semanticMask & 2u) != 0u &&
			deltaIndex + 2u < arrayLength(&morphNormalDeltas)
		) {
			normal += vec3<f32>(
				morphNormalDeltas[deltaIndex],
				morphNormalDeltas[deltaIndex + 1u],
				morphNormalDeltas[deltaIndex + 2u]
			) * weight;
		}
#endif
	}

	return MorphVertex(position, normal);
}

fn applySkinning(
	basePosition: vec3<f32>,
	baseNormal: vec3<f32>,
	baseTangent: vec3<f32>,
	jointIndices: array<f32, 8>,
	jointWeights: array<f32, 8>,
	jointCount: u32,
	jointOffset: u32
) -> SkinnedVertex {
	if (jointCount == 0u) {
		return SkinnedVertex(basePosition, baseNormal, baseTangent);
	}

	let matrixCount = arrayLength(&jointMatrices);
	if (matrixCount == 0u) {
		return SkinnedVertex(basePosition, baseNormal, baseTangent);
	}

	var skinnedPosition = vec3<f32>(0.0);
#if !IGNIS_WEBGPU_ANIMATION_POSITION_ONLY
	var skinnedNormal = vec3<f32>(0.0);
	var skinnedTangent = vec3<f32>(0.0);
#endif
	var weightSum = 0.0;
	for (var influence: u32 = 0u; influence < 8u; influence = influence + 1u) {
		let weight = jointWeights[influence];
		if (weight <= EPSILON) {
			continue;
		}

		let rawJoint = max(jointIndices[influence], 0.0);
		let jointIndex = u32(rawJoint + 0.5);
		if (jointIndex >= jointCount) {
			continue;
		}

		let matrixIndex = jointOffset + jointIndex;
		if (matrixIndex >= matrixCount) {
			continue;
		}

		let skinMatrix = jointMatrices[matrixIndex];
		skinnedPosition += (skinMatrix * vec4<f32>(basePosition, 1.0)).xyz * weight;
#if !IGNIS_WEBGPU_ANIMATION_POSITION_ONLY
		skinnedNormal += (skinMatrix * vec4<f32>(baseNormal, 0.0)).xyz * weight;
		skinnedTangent += (skinMatrix * vec4<f32>(baseTangent, 0.0)).xyz * weight;
#endif
		weightSum += weight;
	}

	if (weightSum <= EPSILON) {
		return SkinnedVertex(basePosition, baseNormal, baseTangent);
	}

#if IGNIS_WEBGPU_ANIMATION_POSITION_ONLY
	return SkinnedVertex(skinnedPosition / weightSum, baseNormal, baseTangent);
#else
	let invWeight = 1.0 / weightSum;
	return SkinnedVertex(
		skinnedPosition * invWeight,
		safeNormalize(skinnedNormal, baseNormal),
		safeNormalize(skinnedTangent, baseTangent)
	);
#endif
}
