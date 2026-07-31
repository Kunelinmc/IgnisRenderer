#import <ignis/color/srgb>

struct GammaParams {
	hdrEnabled: f32,
	hdrHeadroom: f32,
	_pad1: f32,
	_pad2: f32,
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: GammaParams;
@group(0) @binding(2) var outTex: texture_storage_2d<rgba16float, write>;

fn linearSrgbToDisplayP3(color: vec3<f32>) -> vec3<f32> {
	return vec3<f32>(
		0.82259287 * color.r + 0.17753395 * color.g,
		0.03319951 * color.r + 0.96678350 * color.g,
		0.01708535 * color.r + 0.07239572 * color.g + 0.91030148 * color.b
	);
}

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) {
		return;
	}

	let coord = vec2<i32>(gid.xy);
	let sampled = textureLoad(srcTex, coord, 0);
	let linearColor = max(sampled.rgb, vec3<f32>(0.0));
	var encoded: vec3<f32>;
	if (params.hdrEnabled > 0.5) {
		let displayP3 = clamp(
			linearSrgbToDisplayP3(linearColor),
			vec3<f32>(0.0),
			vec3<f32>(params.hdrHeadroom)
		);
		encoded = max(linearToSrgb(displayP3), vec3<f32>(0.0));
	} else {
		encoded = clamp(
			linearToSrgb(clamp(
				linearColor,
				vec3<f32>(0.0),
				vec3<f32>(1.0)
			)),
			vec3<f32>(0.0),
			vec3<f32>(1.0)
		);
	}
	textureStore(outTex, coord, vec4<f32>(encoded, sampled.a));
}
