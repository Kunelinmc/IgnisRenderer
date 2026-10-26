#import <ignis/webgpu/frame-camera>

struct ParticleVertexInput {
	@location(0) quadPosition: vec2<f32>,
	@location(1) quadUV: vec2<f32>,
	@location(2) instancePositionSize: vec4<f32>,
	@location(3) instanceColor: vec4<f32>,
	@location(4) instanceUVRect: vec4<f32>,
	@location(5) instanceRotation: f32,
}

struct ParticleVertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) uv: vec2<f32>,
	@location(1) color: vec4<f32>,
	@location(2) worldPosition: vec3<f32>,
	@location(3) localUV: vec2<f32>,
}

struct ParticleUVTransform {
	transformA: vec4<f32>, // xy: repeat, zw: offset
	transformB: vec4<f32>, // x: cos(rotation), y: sin(rotation)
}

struct ParticleOITOutput {
	@location(0) accum: vec4<f32>,
	@location(1) reveal: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: FrameCameraUniforms;

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
	let right = cameraRight(frame.inverseView);
	let up = cameraUp(frame.inverseView);
	let c = cos(input.instanceRotation);
	let s = sin(input.instanceRotation);
	let rotated = vec2<f32>(
		input.quadPosition.x * c - input.quadPosition.y * s,
		input.quadPosition.x * s + input.quadPosition.y * c
	);
	let worldPosition = input.instancePositionSize.xyz +
		(right * rotated.x + up * rotated.y) * input.instancePositionSize.w;

	var clipPosition = frame.viewProjection * vec4<f32>(worldPosition, 1.0);
	let currJitter = frame.jitterNdc * clipPosition.w;
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
	output.localUV = input.quadUV;
	return output;
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

	color = vec4<f32>(
		max(color.rgb, vec3<f32>(0.0)),
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
	let viewDepth = length(cameraPosition(frame.inverseView) - input.worldPosition);
	let weight = resolveParticleOITWeight(alpha, viewDepth);
	var output: ParticleOITOutput;
	output.accum = vec4<f32>(color.rgb * alpha, alpha) * weight;
	output.reveal = vec4<f32>(alpha, alpha, alpha, alpha);
	return output;
}
