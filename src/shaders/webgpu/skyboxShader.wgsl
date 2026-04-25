#import <ignis/color/srgb>
const PI: f32 = 3.14159265359;

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
	directionalLights: array<DirectionalLightData, __WEBGPU_MAX_DIRECTIONAL_LIGHTS__>,
	pointLights: array<PointLightData, __WEBGPU_MAX_POINT_LIGHTS__>,
	spotLights: array<SpotLightData, __WEBGPU_MAX_SPOT_LIGHTS__>,
	directionalShadows: array<ShadowData, __WEBGPU_MAX_DIRECTIONAL_LIGHTS__>,
	spotShadows: array<ShadowData, __WEBGPU_MAX_SPOT_LIGHTS__>,
	shAmbientCoeffs: array<vec4<f32>, __WEBGPU_SH_COEFFICIENT_COUNT__>,
}

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) ndc: vec2<f32>,
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var skyboxTexture: texture_2d<f32>;
@group(0) @binding(2) var skyboxSampler: sampler;

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
	let right = frame.skyboxBasisRight.xyz;
	let up = frame.skyboxBasisUp.xyz;
	let backward = frame.skyboxBasisBackward.xyz;
	let tanHalfFov = frame.skyboxBasisRight.w;
	let aspect = frame.skyboxBasisUp.w;
	let jitteredNdc = input.ndc + frame.taaJitterCurrentPrev.xy;

	let cx = jitteredNdc.x * aspect * tanHalfFov;
	let cy = jitteredNdc.y * tanHalfFov;
	let direction = normalize(right * cx + up * cy - backward);

	let phi = atan2(direction.x, direction.z);
	let theta = acos(clamp(direction.y, -1.0, 1.0));
	let uv = vec2<f32>((phi + PI) / (2.0 * PI), theta / PI);
	var skyColor = textureSample(skyboxTexture, skyboxSampler, uv).rgb;
	if (frame.environmentOptionsB.z < 0.5) {
		skyColor = srgbToLinear(skyColor);
	}

	if (frame.options.w > 0.5) {
		skyColor = linearToSrgb(skyColor);
	}

	return vec4<f32>(clamp(skyColor, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
