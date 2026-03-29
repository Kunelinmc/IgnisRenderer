struct Params {
	texelSize: vec2<f32>,
	filterRadius: f32,
	_pad0: f32,
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var blendTex: texture_2d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var dstTex: texture_storage_2d<rgba16float, write>;

fn tentFilter(uv: vec2<f32>, texelSize: vec2<f32>, filterRadius: f32) -> vec3<f32> {
	let d = texelSize * filterRadius;

	var result = textureSampleLevel(srcTex, linearSampler, uv + vec2<f32>(-d.x,  d.y), 0.0).rgb;
	result    += textureSampleLevel(srcTex, linearSampler, uv + vec2<f32>( 0.0,  d.y), 0.0).rgb * 2.0;
	result    += textureSampleLevel(srcTex, linearSampler, uv + vec2<f32>( d.x,  d.y), 0.0).rgb;
	result    += textureSampleLevel(srcTex, linearSampler, uv + vec2<f32>(-d.x,  0.0), 0.0).rgb * 2.0;
	result    += textureSampleLevel(srcTex, linearSampler, uv,                         0.0).rgb * 4.0;
	result    += textureSampleLevel(srcTex, linearSampler, uv + vec2<f32>( d.x,  0.0), 0.0).rgb * 2.0;
	result    += textureSampleLevel(srcTex, linearSampler, uv + vec2<f32>(-d.x, -d.y), 0.0).rgb;
	result    += textureSampleLevel(srcTex, linearSampler, uv + vec2<f32>( 0.0, -d.y), 0.0).rgb * 2.0;
	result    += textureSampleLevel(srcTex, linearSampler, uv + vec2<f32>( d.x, -d.y), 0.0).rgb;

	return result * (1.0 / 16.0);
}

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(dstTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }

	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
	let upsampled = tentFilter(uv, params.texelSize, params.filterRadius);
	let existing = textureSampleLevel(blendTex, linearSampler, uv, 0.0).rgb;
	let combined = existing + upsampled;

	textureStore(dstTex, vec2<i32>(gid.xy), vec4<f32>(combined, 1.0));
}
