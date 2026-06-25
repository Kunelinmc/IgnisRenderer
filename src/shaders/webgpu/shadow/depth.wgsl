const EPSILON: f32 = 1e-6;

struct AnimationParams {
	jointCount: u32,
	morphTargetCount: u32,
	jointStride: u32,
	morphWeightStride: u32,
}

struct ShadowInstanceData {
	instanceBaseOffset: u32,
	vertexBaseOffset: u32,
	jointBaseOffset: u32,
	morphWeightBaseOffset: u32,
	morphDeltaBaseOffset: u32,
	atlasOffsetX: u32,
	atlasOffsetY: u32,
	atlasPageSize: u32,
	atlasSize: u32,
	flags: u32,
	_pad0: u32,
	_pad1: u32,
}

struct ShadowVertexInput {
	@location(0) position: vec3<f32>,
	@location(5) joints0: vec4<f32>,
	@location(6) weights0: vec4<f32>,
	@location(7) joints1: vec4<f32>,
	@location(8) weights1: vec4<f32>,
}

struct ShadowVertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) transmittance: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> shadowMvps: array<mat4x4<f32>>;
@group(0) @binding(1) var<storage, read> shadowInstances: array<ShadowInstanceData>;
@group(0) @binding(2) var<storage, read> shadowTransmittance: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> animationParams: AnimationParams;
@group(1) @binding(1) var<storage, read> jointMatrices: array<mat4x4<f32>>;
@group(1) @binding(2) var<storage, read> morphWeights: array<f32>;
@group(1) @binding(3) var<storage, read> morphPositionDeltas: array<vec4<f32>>;

fn applyMorphPosition(
	basePosition: vec3<f32>,
	vertexIndex: u32,
	morphTargetCount: u32,
	morphWeightOffset: u32,
	morphDeltaOffset: u32
) -> vec3<f32> {
	if (morphTargetCount == 0u) {
		return basePosition;
	}

	let morphDeltaCount = arrayLength(&morphPositionDeltas);
	let morphWeightCount = arrayLength(&morphWeights);
	if (morphDeltaCount == 0u || morphWeightCount == 0u) {
		return basePosition;
	}

	let deltaBase = min(morphDeltaOffset, morphDeltaCount);
	let deltaRange = morphDeltaCount - deltaBase;
	let vertexCount = max(deltaRange / morphTargetCount, 1u);
	var position = basePosition;
	for (
		var targetIndex: u32 = 0u;
		targetIndex < morphTargetCount;
		targetIndex = targetIndex + 1u
	) {
		let weightIndex = morphWeightOffset + targetIndex;
		if (weightIndex >= morphWeightCount) {
			continue;
		}

		let weight = morphWeights[weightIndex];
		if (abs(weight) <= EPSILON) {
			continue;
		}

		let deltaIndex = deltaBase + targetIndex * vertexCount + vertexIndex;
		if (deltaIndex >= morphDeltaCount) {
			continue;
		}

		position += morphPositionDeltas[deltaIndex].xyz * weight;
	}

	return position;
}

fn applySkinningPosition(
	basePosition: vec3<f32>,
	jointIndices: array<f32, 8>,
	jointWeights: array<f32, 8>,
	jointCount: u32,
	jointOffset: u32
) -> vec3<f32> {
	if (jointCount == 0u) {
		return basePosition;
	}

	let matrixCount = arrayLength(&jointMatrices);
	if (matrixCount == 0u) {
		return basePosition;
	}

	var skinnedPosition = vec3<f32>(0.0);
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
		weightSum += weight;
	}

	if (weightSum <= EPSILON) {
		return basePosition;
	}

	return skinnedPosition / weightSum;
}

@vertex
fn vsMain(
	input: ShadowVertexInput,
	@builtin(vertex_index) vertexIndex: u32,
	@builtin(instance_index) instanceIndex: u32
) -> ShadowVertexOutput {
	var output: ShadowVertexOutput;
	output.position = vec4<f32>(0.0);
	output.transmittance = vec4<f32>(1.0);
	let morphTargetCount = animationParams.morphTargetCount;
	let jointCount = animationParams.jointCount;

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

	let mvpCount = arrayLength(&shadowMvps);
	let instanceDataCount = arrayLength(&shadowInstances);
	if (mvpCount == 0u || instanceDataCount == 0u) {
		return output;
	}
	let safeInstanceIndex = min(instanceIndex, min(mvpCount, instanceDataCount) - 1u);
	let transmittanceCount = arrayLength(&shadowTransmittance);
	if (transmittanceCount > 0u) {
		output.transmittance =
			shadowTransmittance[min(safeInstanceIndex, transmittanceCount - 1u)];
	}
	let instanceData = shadowInstances[safeInstanceIndex];
	var localInstanceIndex = 0u;
	if (safeInstanceIndex >= instanceData.instanceBaseOffset) {
		localInstanceIndex = safeInstanceIndex - instanceData.instanceBaseOffset;
	}
	let jointOffset =
		instanceData.jointBaseOffset +
		localInstanceIndex * animationParams.jointStride;
	let morphWeightOffset =
		instanceData.morphWeightBaseOffset +
		localInstanceIndex * animationParams.morphWeightStride;
	var localVertexIndex = 0u;
	if (vertexIndex >= instanceData.vertexBaseOffset) {
		localVertexIndex = vertexIndex - instanceData.vertexBaseOffset;
	}

	let morphedPosition = applyMorphPosition(
		input.position,
		localVertexIndex,
		morphTargetCount,
		morphWeightOffset,
		instanceData.morphDeltaBaseOffset
	);
	let skinnedPosition = applySkinningPosition(
		morphedPosition,
		joints,
		weights,
		jointCount,
		jointOffset
	);
	output.position =
		shadowMvps[safeInstanceIndex] * vec4<f32>(skinnedPosition, 1.0);
	if (instanceData.atlasSize > 0u && instanceData.atlasPageSize > 0u) {
		let atlasSize = f32(instanceData.atlasSize);
		let pageSize = f32(instanceData.atlasPageSize);
		let atlasOffset = vec2<f32>(
			f32(instanceData.atlasOffsetX),
			f32(instanceData.atlasOffsetY)
		);
		let ndc = output.position.xy / vec2<f32>(max(abs(output.position.w), EPSILON));
		let pageUv = ndc * 0.5 + vec2<f32>(0.5);
		let atlasUv = (atlasOffset + pageUv * pageSize) / vec2<f32>(atlasSize);
		let remappedNdc = atlasUv * 2.0 - vec2<f32>(1.0);
		output.position.x = remappedNdc.x * output.position.w;
		output.position.y = remappedNdc.y * output.position.w;
	}
	return output;
}

@fragment
fn fsTransmittance(input: ShadowVertexOutput) -> @location(0) vec4<f32> {
	return clamp(input.transmittance, vec4<f32>(0.0), vec4<f32>(1.0));
}
