struct Params {
	filterParams0: vec4<f32>,
	filterParams1: vec4<f32>,
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var outTex: texture_storage_2d<rgba16float, write>;

fn applyColorFilter(color: vec3<f32>) -> vec3<f32> {
	let brightness = params.filterParams0.x;
	let saturation = params.filterParams0.y;
	let contrast = params.filterParams0.z;
	let temperature = params.filterParams0.w;
	let tint = params.filterParams1.x;

	var filtered = color + vec3<f32>(brightness);
	let luma = dot(filtered, vec3<f32>(0.2126, 0.7152, 0.0722));
	filtered = mix(vec3<f32>(luma), filtered, saturation);
	filtered = (filtered - vec3<f32>(0.5)) * contrast + vec3<f32>(0.5);
	filtered += vec3<f32>(
		temperature * 0.1 + tint * 0.05,
		-tint * 0.1,
		-temperature * 0.1 + tint * 0.05
	);
	return clamp(filtered, vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) {
		return;
	}

	let uv =
		(vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5, 0.5)) /
		vec2<f32>(f32(size.x), f32(size.y));
	let sampled = textureSampleLevel(srcTex, linearSampler, uv, 0.0);
	let filtered = applyColorFilter(max(sampled.rgb, vec3<f32>(0.0)));
	textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(filtered, sampled.a));
}
