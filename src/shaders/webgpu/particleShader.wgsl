#import <ignis/webgpu/constants>
#import <ignis/postprocess/fog>
struct FrameUniforms {
	viewProjection: mat4x4<f32>,
	prevViewProjection: mat4x4<f32>,
	cameraPosition: vec4<f32>,
	skyboxBasisRight: vec4<f32>,
	skyboxBasisUp: vec4<f32>,
	skyboxBasisBackward: vec4<f32>,
	ambientColor: vec4<f32>,
	lightCounts: vec4<f32>,
	options: vec4<f32>,
	environmentOptionsA: vec4<f32>,
	environmentOptionsB: vec4<f32>,
	taaJitterCurrentPrev: vec4<f32>,
	directionalLights: array<vec4<f32>, __WEBGPU_FRAME_DIRECTIONAL_LIGHT_VEC4_COUNT__>,
	pointLights: array<vec4<f32>, __WEBGPU_FRAME_POINT_LIGHT_VEC4_COUNT__>,
	spotLights: array<vec4<f32>, __WEBGPU_FRAME_SPOT_LIGHT_VEC4_COUNT__>,
	directionalShadows: array<ShadowData, __WEBGPU_MAX_DIRECTIONAL_LIGHTS__>,
	spotShadows: array<ShadowData, __WEBGPU_MAX_SPOT_LIGHTS__>,
	shAmbientCoeffs: array<vec4<f32>, __WEBGPU_SH_COEFFICIENT_COUNT__>,
}

struct ParticleVertexInput {
	@location(0) quadPosition: vec2<f32>,
	@location(1) quadUV: vec2<f32>,
	@location(2) instancePositionSize: vec4<f32>,
	@location(3) instanceColor: vec4<f32>,
	@location(4) instanceUVRect: vec4<f32>,
	@location(5) instanceRotation: f32,
	@location(6) instanceReceiveShadow: f32,
}

struct ParticleVertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) uv: vec2<f32>,
	@location(1) color: vec4<f32>,
	@location(2) worldPosition: vec3<f32>,
	@location(3) receiveShadow: f32,
	@location(4) localUV: vec2<f32>,
}

struct ParticleUVTransform {
	transformA: vec4<f32>, // xy: repeat, zw: offset
	transformB: vec4<f32>, // x: cos(rotation), y: sin(rotation)
}

struct ParticleOITOutput {
	@location(0) accum: vec4<f32>,
	@location(1) reveal: vec4<f32>,
}

struct FogUniforms {
	fogParams0: vec4<f32>,
	fogParams1: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var shadowAtlas: texture_depth_2d;
@group(0) @binding(2) var envSpecularTexture: texture_2d<f32>;
@group(0) @binding(3) var envSpecularSampler: sampler;
@group(0) @binding(6) var<uniform> fog: FogUniforms;

@group(1) @binding(0) var particleTexture: texture_2d<f32>;
@group(1) @binding(1) var particleSampler: sampler;
@group(1) @binding(2) var<uniform> particleUVTransform: ParticleUVTransform;

fn resolveParticleOITWeight(alpha: f32, viewDepth: f32) -> f32 {
	let clampedAlpha = clamp(alpha, 0.0, 1.0);
	let normalizedDepth = clamp(viewDepth / 400.0, 0.0, 1.0);
	let depthWeight = clamp(1.0 - normalizedDepth, 0.05, 1.0);
	let alphaWeight = max(clampedAlpha * 8.0 + 0.01, 0.01);
	let weight = alphaWeight * alphaWeight * alphaWeight * depthWeight;
	return clamp(weight, 1e-2, 3e3);
}

@vertex
fn vsMain(input: ParticleVertexInput) -> ParticleVertexOutput {
	var output: ParticleVertexOutput;
	let right = frame.skyboxBasisRight.xyz;
	let up = frame.skyboxBasisUp.xyz;
	let c = cos(input.instanceRotation);
	let s = sin(input.instanceRotation);
	let rotated = vec2<f32>(
		input.quadPosition.x * c - input.quadPosition.y * s,
		input.quadPosition.x * s + input.quadPosition.y * c
	);
	let worldPosition = input.instancePositionSize.xyz +
		(right * rotated.x + up * rotated.y) * input.instancePositionSize.w;

	var clipPosition = frame.viewProjection * vec4<f32>(worldPosition, 1.0);
	let currJitter = frame.taaJitterCurrentPrev.xy * clipPosition.w;
	clipPosition = vec4<f32>(
		clipPosition.x + currJitter.x,
		clipPosition.y + currJitter.y,
		clipPosition.z,
		clipPosition.w
	);
	clipPosition.z = clipPosition.z * 0.5 + clipPosition.w * 0.5;

	output.position = clipPosition;
	output.uv = vec2<f32>(
		mix(input.instanceUVRect.x, input.instanceUVRect.z, input.quadUV.x),
		mix(input.instanceUVRect.y, input.instanceUVRect.w, input.quadUV.y)
	);
	output.color = input.instanceColor;
	output.worldPosition = worldPosition;
	output.receiveShadow = input.instanceReceiveShadow;
	output.localUV = input.quadUV;
	return output;
}

fn sampleDirectionalShadowVisibility(worldPosition: vec3<f32>) -> f32 {
	if (frame.options.z < 0.5) {
		return 1.0;
	}

	let shadow = frame.directionalShadows[0];
	if (shadow.paramsA.x < 0.5) {
		return 1.0;
	}

	let lightClip = shadow.viewProjection * vec4<f32>(worldPosition, 1.0);
	if (lightClip.w <= 1e-6) {
		return 1.0;
	}

	let ndc = lightClip.xyz / lightClip.w;
	let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
	if (any(uv < vec2<f32>(0.0, 0.0)) || any(uv > vec2<f32>(1.0, 1.0))) {
		return 1.0;
	}

	let dimensions = vec2<f32>(textureDimensions(shadowAtlas));
	let texel = vec2<i32>(
		clamp(
			uv * dimensions,
			vec2<f32>(0.0, 0.0),
			dimensions - vec2<f32>(1.0, 1.0)
		)
	);
	let sampledDepth = textureLoad(shadowAtlas, texel, 0);
	let depthBias = shadow.paramsA.y;
	let currentDepth = ndc.z * 0.5 + 0.5;
	let occluded = (currentDepth - depthBias) > sampledDepth;
	let strength = clamp(shadow.paramsB.y, 0.0, 1.0);
	return select(1.0, 1.0 - strength, occluded);
}

fn shadeParticle(input: ParticleVertexOutput) -> vec4<f32> {
	// Procedural soft radial falloff (circle mask)
	let dist = distance(input.localUV, vec2<f32>(0.5, 0.5));
	let radialMask = 1.0 - smoothstep(0.4, 0.5, dist);
	
	let scaledUV = vec2<f32>(
		input.uv.x * particleUVTransform.transformA.x,
		input.uv.y * particleUVTransform.transformA.y
	);
	let rotatedUV = vec2<f32>(
		scaledUV.x * particleUVTransform.transformB.x -
			scaledUV.y * particleUVTransform.transformB.y,
		scaledUV.x * particleUVTransform.transformB.y +
			scaledUV.y * particleUVTransform.transformB.x
	);
	let sampled = textureSample(
		particleTexture,
		particleSampler,
		rotatedUV + particleUVTransform.transformA.zw
	);
	
	var color = sampled * input.color;
	color.a = color.a * radialMask;
	
	if (color.a <= 0.001) {
		discard;
	}
	if (input.receiveShadow > 0.5) {
		let visibility = sampleDirectionalShadowVisibility(input.worldPosition);
		color = vec4<f32>(color.rgb * visibility, color.a);
	}

	let viewDepth = length(frame.cameraPosition.xyz - input.worldPosition);
	let fogMode = i32(floor(fog.fogParams0.x + 0.5));
	let fogFactor = ignisComputeFogFactor(
		fogMode,
		max(viewDepth, 0.0),
		fog.fogParams0.y,
		fog.fogParams0.z,
		fog.fogParams0.w,
		fog.fogParams1.w
	);
	color = vec4<f32>(
		max(mix(color.rgb, fog.fogParams1.rgb, fogFactor), vec3<f32>(0.0)),
		color.a
	);
	return color;
}

@fragment
fn fsMain(input: ParticleVertexOutput) -> @location(0) vec4<f32> {
	return shadeParticle(input);
}

@fragment
fn fsMainOIT(input: ParticleVertexOutput) -> ParticleOITOutput {
	let color = shadeParticle(input);
	let alpha = clamp(color.a, 0.0, 1.0);
	let viewDepth = length(frame.cameraPosition.xyz - input.worldPosition);
	let weight = resolveParticleOITWeight(alpha, viewDepth);
	var output: ParticleOITOutput;
	output.accum = vec4<f32>(color.rgb * alpha, alpha) * weight;
	output.reveal = vec4<f32>(alpha, alpha, alpha, alpha);
	return output;
}
