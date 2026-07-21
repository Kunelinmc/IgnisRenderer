#import <ignis/webgpu/constants>
#import <ignis/color/srgb>
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

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) ndc: vec2<f32>,
}

struct EnvironmentBackgroundParams {
	tintExposureStrength: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: FrameCameraUniforms;
@group(0) @binding(1) var environmentTexture: texture_2d<f32>;
@group(0) @binding(2) var environmentSampler: sampler;
@group(0) @binding(3) var<uniform> environmentBackground: EnvironmentBackgroundParams;

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
	var positions = array<vec2<f32>, 3>(
		vec2<f32>(-1.0, -3.0),
		vec2<f32>(3.0, 1.0),
		vec2<f32>(-1.0, 1.0)
	);

	var output: VertexOutput;
	let pos = positions[vertexIndex];
	output.position = vec4<f32>(pos, 1.0, 1.0);
	output.ndc = pos;
	return output;
}

@fragment
fn fsMain(input: VertexOutput) -> @location(0) vec4<f32> {
	let right = frame.environmentBasisRight.xyz;
	let up = frame.environmentBasisUp.xyz;
	let backward = frame.environmentBasisBackward.xyz;
	let tanHalfFov = frame.environmentBasisRight.w;
	let aspect = frame.environmentBasisUp.w;
	let jitteredNdc = input.ndc + frame.taaJitterCurrentPrev.xy;

	let cx = jitteredNdc.x * aspect * tanHalfFov;
	let cy = jitteredNdc.y * tanHalfFov;
	let perspectiveDirection = normalize(right * cx + up * cy - backward);
	let direction = select(
		perspectiveDirection,
		-backward,
		frame.environmentBasisBackward.w > 0.5
	);

	let phi = atan2(direction.x, direction.z);
	let theta = acos(clamp(direction.y, -1.0, 1.0));
	let uv = vec2<f32>((phi + PI) / (2.0 * PI), theta / PI);
	var skyColor = textureSample(environmentTexture, environmentSampler, uv).rgb;
	if (frame.environmentOptionsB.z < 0.5) {
		skyColor = srgbToLinear(skyColor);
	}
	skyColor *=
		environmentBackground.tintExposureStrength.xyz *
		environmentBackground.tintExposureStrength.w;

	return vec4<f32>(max(skyColor, vec3<f32>(0.0)), 1.0);
}
