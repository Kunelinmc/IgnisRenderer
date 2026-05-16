#import <ignis/webgpu/constants>

const EPSILON: f32 = 1e-6;

struct FrameUniforms {
	viewProjection: mat4x4<f32>,
}

struct ModelUniforms {
	modelMatrix: mat4x4<f32>,
	prevModelMatrix: mat4x4<f32>,
	normalMatrix: mat4x4<f32>,
	baseColorFactor: vec4<f32>,
	emissiveFactor: vec4<f32>,
	surfaceParams0: vec4<f32>,
	surfaceParams1: vec4<f32>,
	surfaceParams2: vec4<f32>,
	surfaceParams3: vec4<f32>,
	specularColorFactor: vec4<f32>,
	phongAmbientShininess: vec4<f32>,
	phongSpecularShading: vec4<f32>,
	sheenColorClearcoatNormalScale: vec4<f32>,
	attenuationColor: vec4<f32>,
	anisotropyParams: vec4<f32>,
	anisotropyTextureTransformA: vec4<f32>,
	anisotropyTextureTransformB: vec4<f32>,
	materialFlags: vec4<f32>,
	textureTransformA: array<vec4<f32>, __WEBGPU_TEXTURE_SLOT_COUNT__>,
	textureTransformB: array<vec4<f32>, __WEBGPU_TEXTURE_SLOT_COUNT__>,
}

struct AnimationParams {
	jointCount: f32,
	morphTargetCount: f32,
	prevJointOffset: f32,
	prevMorphOffset: f32,
}

struct VertexInput {
	@location(0) position: vec3<f32>,
	@location(1) uv0: vec2<f32>,
	@location(2) normal: vec3<f32>,
	@location(3) tangent: vec4<f32>,
	@location(4) uv1: vec2<f32>,
	@location(5) joints0: vec4<f32>,
	@location(6) weights0: vec4<f32>,
	@location(7) joints1: vec4<f32>,
	@location(8) weights1: vec4<f32>,
	@location(9) uv2: vec2<f32>,
	@location(10) uv3: vec2<f32>,
}

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) screenUv: vec2<f32>,
	@location(1) uv0: vec2<f32>,
	@location(2) uv1: vec2<f32>,
	@location(3) uv2: vec2<f32>,
	@location(4) uv3: vec2<f32>,
}

struct FragmentOutput {
	@location(0) sceneColor: vec4<f32>,
	@location(1) mask: vec4<f32>,
}

struct MorphVertex {
	position: vec3<f32>,
	normal: vec3<f32>,
}

struct SkinnedVertex {
	position: vec3<f32>,
	normal: vec3<f32>,
	tangent: vec3<f32>,
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;

@group(1) @binding(0) var<uniform> model: ModelUniforms;
@group(1) @binding(1) var baseColorTexture: texture_2d<f32>;
@group(1) @binding(2) var baseColorSampler: sampler;
@group(1) @binding(30) var<uniform> animationParams: AnimationParams;
@group(1) @binding(32) var<storage, read> jointMatrices: array<mat4x4<f32>>;
@group(1) @binding(33) var<storage, read> morphWeights: array<f32>;
@group(1) @binding(34) var<storage, read> morphPositionDeltas: array<vec4<f32>>;
@group(1) @binding(35) var<storage, read> morphNormalDeltas: array<vec4<f32>>;

@group(2) @binding(0) var reflectionTexture: texture_2d<f32>;

fn safeNormalize(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
	let len = length(value);
	return select(fallback, value / max(len, EPSILON), len > EPSILON);
}

fn transformUV(
	slotIndex: u32,
	uv0: vec2<f32>,
	uv1: vec2<f32>,
	uv2: vec2<f32>,
	uv3: vec2<f32>
) -> vec2<f32> {
	let transformA = model.textureTransformA[slotIndex];
	let transformB = model.textureTransformB[slotIndex];
	let uvSet = u32(clamp(floor(transformB.y + 0.5), 0.0, 3.0));
	var uv = uv0;
	if (uvSet == 1u) {
		uv = uv1;
	} else if (uvSet == 2u) {
		uv = uv2;
	} else if (uvSet >= 3u) {
		uv = uv3;
	}
	uv = uv * transformA.zw;

	let rotation = transformB.x;
	if (abs(rotation) > EPSILON) {
		let c = cos(rotation);
		let s = sin(rotation);
		uv = vec2<f32>(uv.x * c - uv.y * s, uv.x * s + uv.y * c);
	}

	return uv + transformA.xy;
}

fn sampleReflection(uv: vec2<f32>) -> vec4<f32> {
	let dimensions = vec2<f32>(textureDimensions(reflectionTexture));
	let maxTexel = dimensions - vec2<f32>(1.0);
	let texel =
		clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) *
		dimensions -
		vec2<f32>(0.5);
	let base = floor(texel);
	let next = base + vec2<f32>(1.0);
	let baseCoord = vec2<i32>(clamp(base, vec2<f32>(0.0), maxTexel));
	let nextCoord = vec2<i32>(clamp(next, vec2<f32>(0.0), maxTexel));
	let blend = texel - base;

	let c00 = textureLoad(reflectionTexture, baseCoord, 0);
	let c10 = textureLoad(
		reflectionTexture,
		vec2<i32>(nextCoord.x, baseCoord.y),
		0
	);
	let c01 = textureLoad(
		reflectionTexture,
		vec2<i32>(baseCoord.x, nextCoord.y),
		0
	);
	let c11 = textureLoad(reflectionTexture, nextCoord, 0);
	let cx0 = mix(c00, c10, blend.x);
	let cx1 = mix(c01, c11, blend.x);
	return mix(cx0, cx1, blend.y);
}

fn applyMorphDeltas(
	basePosition: vec3<f32>,
	baseNormal: vec3<f32>,
	vertexIndex: u32,
	morphTargetCount: u32,
	morphWeightOffset: u32
) -> MorphVertex {
	if (morphTargetCount == 0u) {
		return MorphVertex(basePosition, baseNormal);
	}

	let morphDeltaCount = arrayLength(&morphPositionDeltas);
	let morphNormalCount = arrayLength(&morphNormalDeltas);
	let morphWeightCount = arrayLength(&morphWeights);
	if (morphDeltaCount == 0u || morphWeightCount == 0u) {
		return MorphVertex(basePosition, baseNormal);
	}

	let vertexCount = max(morphDeltaCount / morphTargetCount, 1u);
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

		let deltaIndex = targetIndex * vertexCount + vertexIndex;
		if (deltaIndex >= morphDeltaCount) {
			continue;
		}

		position += morphPositionDeltas[deltaIndex].xyz * weight;
		if (deltaIndex < morphNormalCount) {
			normal += morphNormalDeltas[deltaIndex].xyz * weight;
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
	let jointCount = u32(animationParams.jointCount + 0.5);
	let morphTargetCount = u32(animationParams.morphTargetCount + 0.5);
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
	let morphed = applyMorphDeltas(
		input.position,
		input.normal,
		vertexIndex,
		morphTargetCount,
		0u
	);
	let skinned = applySkinning(
		morphed.position,
		morphed.normal,
		baseTangent,
		joints,
		weights,
		jointCount,
		0u
	);
	let worldPosition = model.modelMatrix * vec4<f32>(skinned.position, 1.0);
	var clipPosition = frame.viewProjection * worldPosition;
	let invW = 1.0 / max(abs(clipPosition.w), EPSILON);
	let ndc = clipPosition.xy * invW;
	clipPosition.z = clipPosition.z * 0.5 + clipPosition.w * 0.5;

	var output: VertexOutput;
	output.position = clipPosition;
	output.screenUv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
	output.uv0 = input.uv0;
	output.uv1 = input.uv1;
	output.uv2 = input.uv2;
	output.uv3 = input.uv3;
	return output;
}

@fragment
fn fsMain(input: VertexOutput) -> FragmentOutput {
	let reflectivity = clamp(model.anisotropyParams.w, 0.0, 1.0);
	if (reflectivity <= 0.0) {
		discard;
	}

	let alphaModeMask = model.materialFlags.y > 0.5;
	if (alphaModeMask) {
		let baseSample = textureSample(
			baseColorTexture,
			baseColorSampler,
			transformUV(0u, input.uv0, input.uv1, input.uv2, input.uv3)
		);
		let alpha = clamp(model.baseColorFactor.a * baseSample.a, 0.0, 1.0);
		if (alpha < model.surfaceParams0.w) {
			discard;
		}
	}

	let reflection = sampleReflection(input.screenUv);

	var output: FragmentOutput;
	output.sceneColor = vec4<f32>(reflection.rgb, reflectivity);
	output.mask = vec4<f32>(1.0, 0.0, 0.0, 1.0);
	return output;
}
