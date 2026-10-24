#import <ignis/webgpu/constants>

struct PresentVSOut {
	@builtin(position) position: vec4<f32>,
	@location(0) uv: vec2<f32>,
}

struct PresentParams {
	hdrEnabled: f32,
	transparentOutput: f32,
}

@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<uniform> params: PresentParams;

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> PresentVSOut {
	var positions = array<vec2<f32>, 3>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>(3.0, -1.0),
		vec2<f32>(-1.0, 3.0)
	);

	let pos = positions[vertexIndex];
	var output: PresentVSOut;
	output.position = vec4<f32>(pos, 0.0, 1.0);
	output.uv = vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
	return output;
}

@fragment
fn fsMain(input: PresentVSOut) -> @location(0) vec4<f32> {
	let sampled = textureSample(srcTexture, srcSampler, input.uv);
	let hdr = params.hdrEnabled > 0.5;
	let transparent = params.transparentOutput > 0.5;
	let alpha = select(1.0, clamp(sampled.a, 0.0, 1.0), transparent);
	var color = select(
		max(sampled.rgb, vec3<f32>(0.0)),
		select(
			vec3<f32>(0.0),
			max(sampled.rgb, vec3<f32>(0.0)) / max(alpha, EPSILON),
			alpha > EPSILON
		),
		transparent
	);
	color = select(
		clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)),
		max(color, vec3<f32>(0.0)),
		hdr
	);
	let outputColor = select(color, color * alpha, transparent);
	return vec4<f32>(outputColor, alpha);
}
