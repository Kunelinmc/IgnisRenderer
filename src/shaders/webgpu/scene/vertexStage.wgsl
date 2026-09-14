#import <ignis/webgpu/animation>

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
		0u,
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

	var resolvedModelMatrix = object.modelMatrix;
	var resolvedPrevModelMatrix = object.prevModelMatrix;
	var resolvedNormalMatrix = object.normalMatrix;
	var resolvedInstanceData = object.instanceData;
	if (object.instanceData.z > 0.5) {
		let instance = staticInstances[input.instanceIndex];
		resolvedModelMatrix = instance.modelMatrix;
		resolvedPrevModelMatrix = instance.prevModelMatrix;
		resolvedNormalMatrix = instance.normalMatrix;
		resolvedInstanceData = instance.instanceData;
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
	output.instanceMeta = resolvedInstanceData.xy;
	return output;
}
