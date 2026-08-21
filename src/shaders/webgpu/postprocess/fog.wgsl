#import <ignis/postprocess/fog>

struct Params {
	fogParams0: vec4<f32>,
	fogParams1: vec4<f32>,
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var gMotionDepth: texture_2d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var outTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) {
		return;
	}

	let coord = vec2<i32>(gid.xy);
	let invSize = vec2<f32>(1.0 / f32(size.x), 1.0 / f32(size.y));
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * invSize;
	let source = textureSampleLevel(srcTex, linearSampler, uv, 0.0);
	let depth = max(textureSampleLevel(gMotionDepth, linearSampler, uv, 0.0).z, 0.0);
	let mode = i32(floor(params.fogParams0.x + 0.5));
	let fogFactor = ignisComputeFogFactor(
		mode,
		depth,
		params.fogParams0.y,
		params.fogParams0.z,
		params.fogParams0.w,
		params.fogParams1.w
	);
	let fogged = max(
		mix(source.rgb, params.fogParams1.rgb, fogFactor),
		vec3<f32>(0.0)
	);
	let outputAlpha = clamp(source.a, 0.0, 1.0) +
		fogFactor * (1.0 - clamp(source.a, 0.0, 1.0));
	textureStore(outTex, coord, vec4<f32>(fogged, outputAlpha));
}
