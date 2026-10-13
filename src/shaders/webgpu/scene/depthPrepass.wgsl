#define IGNIS_WEBGPU_ANIMATION_POSITION_ONLY 1
#import <ignis/webgpu/animation>

// Shared scene buffer owners retain their full ABI even when depth uses fewer fields.
struct FrameCameraUniforms {
	viewProjection: mat4x4<f32>,
	prevViewProjection: mat4x4<f32>,
	cameraPosition: vec4<f32>,
	environmentBasisRight: vec4<f32>,
	environmentBasisUp: vec4<f32>,
	environmentBasisBackward: vec4<f32>,
	ambientColor: vec4<f32>,
	lightCounts: vec4<f32>,
	options: vec4<f32>,
	environmentOptionsA: vec4<f32>,
	environmentOptionsB: vec4<f32>,
	taaJitterCurrentPrev: vec4<f32>,
}

struct ObjectUniforms {
	modelMatrix: mat4x4<f32>,
	prevModelMatrix: mat4x4<f32>,
	normalMatrix: mat4x4<f32>,
	instanceData: vec4<f32>,
}

struct AnimationParams {
	jointCount: f32,
	morphTargetCount: f32,
	prevJointOffset: f32,
	prevMorphOffset: f32,
	vertexCount: f32,
	morphSemanticMask: f32,
	_pad0: f32,
	_pad1: f32,
}

struct StaticInstance {
	modelMatrix: mat4x4<f32>,
	prevModelMatrix: mat4x4<f32>,
	normalMatrix: mat4x4<f32>,
	instanceData: vec4<f32>,
}

struct DepthMaterialUniforms {
	params: vec4<f32>,
	transformA: vec4<f32>,
	transformB: vec4<f32>,
}

struct DepthVertexInput {
	@builtin(instance_index) instanceIndex: u32,
	@location(0) position: vec3<f32>,
	@location(1) uv0: vec2<f32>,
	@location(4) uv1: vec2<f32>,
	@location(5) joints0: vec4<f32>,
	@location(6) weights0: vec4<f32>,
	@location(7) joints1: vec4<f32>,
	@location(8) weights1: vec4<f32>,
	@location(9) uv2: vec2<f32>,
	@location(10) uv3: vec2<f32>,
}

struct DepthVertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) uv0: vec2<f32>,
	@location(1) uv1: vec2<f32>,
	@location(2) uv2: vec2<f32>,
	@location(3) uv3: vec2<f32>,
}

@group(0) @binding(0) var<uniform> frame: FrameCameraUniforms;
@group(1) @binding(0) var<uniform> object: ObjectUniforms;
@group(1) @binding(34) var<uniform> animationParams: AnimationParams;
@group(1) @binding(35) var<storage, read> jointMatrices: array<mat4x4<f32>>;
@group(1) @binding(36) var<storage, read> morphWeights: array<f32>;
@group(1) @binding(37) var<storage, read> morphPositionDeltas: array<f32>;
@group(1) @binding(40) var<storage, read> staticInstances: array<StaticInstance>;
@group(2) @binding(0) var<uniform> depthMaterial: DepthMaterialUniforms;
@group(2) @binding(1) var baseTexture: texture_2d<f32>;
@group(2) @binding(2) var baseSampler: sampler;

@vertex
fn vsMain(input: DepthVertexInput, @builtin(vertex_index) vertexIndex: u32) -> DepthVertexOutput {
	let joints = array<f32, 8>(
		input.joints0.x, input.joints0.y, input.joints0.z, input.joints0.w,
		input.joints1.x, input.joints1.y, input.joints1.z, input.joints1.w
	);
	let weights = array<f32, 8>(
		input.weights0.x, input.weights0.y, input.weights0.z, input.weights0.w,
		input.weights1.x, input.weights1.y, input.weights1.z, input.weights1.w
	);
	let morphedCurrent = applyMorphDeltas(
		input.position, vec3<f32>(0.0), vertexIndex,
		u32(animationParams.morphTargetCount + 0.5), 0u, 0u,
		u32(animationParams.vertexCount + 0.5), u32(animationParams.morphSemanticMask + 0.5)
	);
	let skinnedCurrent = applySkinning(
		morphedCurrent.position, vec3<f32>(0.0), vec3<f32>(0.0), joints, weights,
		u32(animationParams.jointCount + 0.5), 0u
	);
	var resolvedModelMatrix = object.modelMatrix;
	if (object.instanceData.z > 0.5) {
		resolvedModelMatrix = staticInstances[input.instanceIndex].modelMatrix;
	}
	let worldPosition = resolvedModelMatrix * vec4<f32>(skinnedCurrent.position, 1.0);
	var clipPosition = frame.viewProjection * worldPosition;
	let currJitter = frame.taaJitterCurrentPrev.xy * clipPosition.w;
	clipPosition = vec4<f32>(
		clipPosition.x + currJitter.x,
		clipPosition.y + currJitter.y,
		clipPosition.z,
		clipPosition.w
	);
	clipPosition.z = clipPosition.z * 0.5 + clipPosition.w * 0.5;
	var output: DepthVertexOutput;
	output.position = clipPosition;
	output.uv0 = input.uv0;
	output.uv1 = input.uv1;
	output.uv2 = input.uv2;
	output.uv3 = input.uv3;
	return output;
}

fn transformDepthUV(input: DepthVertexOutput) -> vec2<f32> {
	let uvSet = u32(clamp(floor(depthMaterial.transformB.y + 0.5), 0.0, 3.0));
	var uv = input.uv0;
	if (uvSet == 1u) {
		uv = input.uv1;
	} else if (uvSet == 2u) {
		uv = input.uv2;
	} else if (uvSet >= 3u) {
		uv = input.uv3;
	}
	uv = uv * depthMaterial.transformA.zw;
	let rotation = depthMaterial.transformB.x;
	if (abs(rotation) > EPSILON) {
		let c = cos(rotation);
		let s = sin(rotation);
		uv = vec2<f32>(uv.x * c - uv.y * s, uv.x * s + uv.y * c);
	}
	return uv + depthMaterial.transformA.xy;
}

@fragment
fn fsMainDepthMask(input: DepthVertexOutput) {
	let baseAlpha = textureSample(baseTexture, baseSampler, transformDepthUV(input)).a;
	let alpha = clamp(depthMaterial.params.x * baseAlpha, 0.0, 1.0);
	if (alpha < depthMaterial.params.y) {
		discard;
	}
}
