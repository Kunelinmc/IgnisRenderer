#import <ignis/webgpu/constants>
#import <ignis/color/srgb>
#import <ignis/webgpu/frame-camera>

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) ndc: vec2<f32>,
}

struct EnvironmentBackgroundParams {
	tintExposureStrength: vec4<f32>,
	inputIsLinear: u32,
	_pad0: u32,
	_pad1: u32,
	_pad2: u32,
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
	let rasterUv = vec2<f32>(input.ndc.x * 0.5 + 0.5, 0.5 - input.ndc.y * 0.5);
	let direction = cameraWorldRayDirection(
		frame.inverseView, frame.inverseProjection, frame.jitterNdc, rasterUv
	);

	let phi = atan2(direction.x, direction.z);
	let theta = acos(clamp(direction.y, -1.0, 1.0));
	let uv = vec2<f32>((phi + PI) / (2.0 * PI), theta / PI);
	var skyColor = textureSample(environmentTexture, environmentSampler, uv).rgb;
	if (environmentBackground.inputIsLinear == 0u) {
		skyColor = srgbToLinear(skyColor);
	}
	skyColor *=
		environmentBackground.tintExposureStrength.xyz *
		environmentBackground.tintExposureStrength.w;

	return vec4<f32>(max(skyColor, vec3<f32>(0.0)), 1.0);
}
