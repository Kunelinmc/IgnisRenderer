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
	vertexCount: u32,
	semanticMask: u32
) -> MorphVertex {
	if (morphTargetCount == 0u || vertexCount == 0u) {
		return MorphVertex(basePosition, baseNormal);
	}

	let morphWeightCount = arrayLength(&morphWeights);
	if (morphWeightCount == 0u) {
		return MorphVertex(basePosition, baseNormal);
	}

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

		let deltaIndex = (targetIndex * vertexCount + vertexIndex) * 3u;
		if (
			(semanticMask & 1u) != 0u &&
			deltaIndex + 2u < arrayLength(&morphPositionDeltas)
		) {
			position += vec3<f32>(
				morphPositionDeltas[deltaIndex],
				morphPositionDeltas[deltaIndex + 1u],
				morphPositionDeltas[deltaIndex + 2u]
			) * weight;
		}
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
	var skinnedNormal = vec3<f32>(0.0);
	var skinnedTangent = vec3<f32>(0.0);
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
		skinnedNormal += (skinMatrix * vec4<f32>(baseNormal, 0.0)).xyz * weight;
		skinnedTangent += (skinMatrix * vec4<f32>(baseTangent, 0.0)).xyz * weight;
		weightSum += weight;
	}

	if (weightSum <= EPSILON) {
		return SkinnedVertex(basePosition, baseNormal, baseTangent);
	}

	let invWeight = 1.0 / weightSum;
	return SkinnedVertex(
		skinnedPosition * invWeight,
		safeNormalize(skinnedNormal, baseNormal),
		safeNormalize(skinnedTangent, baseTangent)
	);
}

@vertex
fn vsMain(input: VertexInput, @builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
	var output: VertexOutput;
	let jointCount = u32(animationParams.jointCount + 0.5);
	let morphTargetCount = u32(animationParams.morphTargetCount + 0.5);
	let prevJointOffset = u32(animationParams.prevJointOffset + 0.5);
	let prevMorphOffset = u32(animationParams.prevMorphOffset + 0.5);
	let vertexCount = u32(animationParams.vertexCount + 0.5);
	let morphSemanticMask = u32(animationParams.morphSemanticMask + 0.5);
	let baseTangent = safeNormalize(input.tangent.xyz, vec3<f32>(1.0, 0.0, 0.0));
	let joints = array<f32, 8>(
		input.joints0.x,
		input.joints0.y,
		input.joints0.z,
		input.joints0.w,
		input.joints1.x,
		input.joints1.y,
		input.joints1.z,
		input.joints1.w
	);
	let weights = array<f32, 8>(
		input.weights0.x,
		input.weights0.y,
		input.weights0.z,
		input.weights0.w,
		input.weights1.x,
		input.weights1.y,
		input.weights1.z,
		input.weights1.w
	);

	let morphedCurrent = applyMorphDeltas(
		input.position,
		input.normal,
		vertexIndex,
		morphTargetCount,
		0u,
		vertexCount,
		morphSemanticMask
	);
	let morphedPrev = applyMorphDeltas(
		input.position,
		input.normal,
		vertexIndex,
		morphTargetCount,
		prevMorphOffset,
		vertexCount,
		morphSemanticMask
	);
	let skinnedCurrent = applySkinning(
		morphedCurrent.position,
		morphedCurrent.normal,
		baseTangent,
		joints,
		weights,
		jointCount,
		0u
	);
	let skinnedPrev = applySkinning(
		morphedPrev.position,
		morphedPrev.normal,
		baseTangent,
		joints,
		weights,
		jointCount,
		prevJointOffset
	);

	var resolvedModelMatrix = model.modelMatrix;
	var resolvedPrevModelMatrix = model.prevModelMatrix;
	var resolvedNormalMatrix = model.normalMatrix;
	var resolvedNodeRenderLayers = model.nodeRenderLayers;
	if (model.instanceParams.x > 0.5) {
		let instance = staticInstances[input.instanceIndex];
		resolvedModelMatrix = instance.modelMatrix;
		resolvedPrevModelMatrix = instance.prevModelMatrix;
		resolvedNormalMatrix = instance.normalMatrix;
		resolvedNodeRenderLayers = instance.nodeRenderLayers;
	}
	let worldPosition = resolvedModelMatrix * vec4<f32>(skinnedCurrent.position, 1.0);
	let worldNormal = safeNormalize(
		(resolvedNormalMatrix * vec4<f32>(skinnedCurrent.normal, 0.0)).xyz,
		vec3<f32>(0.0, 0.0, 1.0)
	);
	let worldTangent =
		(resolvedNormalMatrix * vec4<f32>(skinnedCurrent.tangent, 0.0)).xyz;
	var clipPosition = frame.viewProjection * worldPosition;
	let currJitter = frame.taaJitterCurrentPrev.xy * clipPosition.w;
	clipPosition = vec4<f32>(
		clipPosition.x + currJitter.x,
		clipPosition.y + currJitter.y,
		clipPosition.z,
		clipPosition.w
	);
	clipPosition.z = clipPosition.z * 0.5 + clipPosition.w * 0.5;
	let prevWorldPosition =
		resolvedPrevModelMatrix * vec4<f32>(skinnedPrev.position, 1.0);
	var prevClipPosition = frame.prevViewProjection * prevWorldPosition;
	let prevJitter = frame.taaJitterCurrentPrev.zw * prevClipPosition.w;
	prevClipPosition = vec4<f32>(
		prevClipPosition.x + prevJitter.x,
		prevClipPosition.y + prevJitter.y,
		prevClipPosition.z,
		prevClipPosition.w
	);
	prevClipPosition.z = prevClipPosition.z * 0.5 + prevClipPosition.w * 0.5;

	output.position = clipPosition;
	output.worldPosition = worldPosition.xyz;
	output.worldNormal = worldNormal;
	output.uv0 = input.uv0;
	output.worldTangent = vec4<f32>(
		safeNormalize(worldTangent, vec3<f32>(1.0, 0.0, 0.0)),
		input.tangent.w
	);
	output.uv1 = input.uv1;
	output.uv2 = input.uv2;
	output.uv3 = input.uv3;
	output.currentClip = clipPosition;
	output.prevClip = prevClipPosition;
	output.instanceMeta = resolvedNodeRenderLayers.xy;
	return output;
}
