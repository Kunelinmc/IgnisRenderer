#import <ignis/color/srgb>
#import <ignis/webgpu/constants>

struct PresentVSOut {
	@builtin(position) position: vec4<f32>,
	@location(0) uv: vec2<f32>,
}

struct PresentParams {
	exposure: f32,
	hdrHeadroom: f32,
	hdrEnabled: f32,
	colorDomain: f32,
}

@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<uniform> params: PresentParams;

fn acesFitted(color: vec3<f32>) -> vec3<f32> {
	let a = 2.51;
	let b = 0.03;
	let c = 2.43;
	let d = 0.59;
	let e = 0.14;
	let mapped =
		(color * (a * color + vec3<f32>(b))) /
		(color * (c * color + vec3<f32>(d)) + vec3<f32>(e));
	return clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn hdrSoftShoulder(color: vec3<f32>, headroom: f32) -> vec3<f32> {
	let positive = max(color, vec3<f32>(0.0));
	let peak = max(positive.r, max(positive.g, positive.b));
	if (peak <= 1.0) {
		return positive;
	}
	if (headroom <= 1.0001) {
		return clamp(positive, vec3<f32>(0.0), vec3<f32>(1.0));
	}
	let mappedPeak =
		1.0 + (headroom - 1.0) *
		(1.0 - exp(-(peak - 1.0) / (headroom - 1.0)));
	return positive * (mappedPeak / max(peak, EPSILON));
}

fn linearSrgbToDisplayP3(color: vec3<f32>) -> vec3<f32> {
	return vec3<f32>(
		0.82259287 * color.r + 0.17753395 * color.g,
		0.03319951 * color.r + 0.96678350 * color.g,
		0.01708535 * color.r + 0.07239572 * color.g + 0.91030148 * color.b
	);
}

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
	let encodedDomain = i32(params.colorDomain + 0.5);
	let transparent = encodedDomain >= 4;
	let domain = encodedDomain % 4;
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
	if (domain == 0) {
		let exposed = color * params.exposure;
		color = select(
			acesFitted(exposed),
			hdrSoftShoulder(exposed, params.hdrHeadroom),
			hdr
		);
	}
	if (domain < 2) {
		if (hdr) {
			color = linearToSrgb(clamp(
				linearSrgbToDisplayP3(color),
				vec3<f32>(0.0),
				vec3<f32>(params.hdrHeadroom)
			));
		} else {
			color = linearToSrgb(clamp(
				color,
				vec3<f32>(0.0),
				vec3<f32>(1.0)
			));
		}
	}
	color = select(
		clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)),
		max(color, vec3<f32>(0.0)),
		hdr
	);
	let outputColor = select(color, color * alpha, transparent);
	return vec4<f32>(outputColor, alpha);
}
