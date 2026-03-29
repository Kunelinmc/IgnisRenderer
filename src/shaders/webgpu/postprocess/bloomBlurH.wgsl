struct Params {
	texelSize: vec2<f32>,
	direction: vec2<f32>,
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var dstTex: texture_storage_2d<rgba16float, write>;

// 9-tap Gaussian weights (sigma ~= 2.5, normalized)
const W0: f32 = 0.227027;
const W1: f32 = 0.194594;
const W2: f32 = 0.121621;
const W3: f32 = 0.054054;
const W4: f32 = 0.016216;

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(dstTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }

	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
	let offset = vec2<f32>(params.texelSize.x, 0.0);

	var color = textureSampleLevel(srcTex, linearSampler, uv, 0.0).rgb * W0;
	color += textureSampleLevel(srcTex, linearSampler, uv + offset * 1.0, 0.0).rgb * W1;
	color += textureSampleLevel(srcTex, linearSampler, uv - offset * 1.0, 0.0).rgb * W1;
	color += textureSampleLevel(srcTex, linearSampler, uv + offset * 2.0, 0.0).rgb * W2;
	color += textureSampleLevel(srcTex, linearSampler, uv - offset * 2.0, 0.0).rgb * W2;
	color += textureSampleLevel(srcTex, linearSampler, uv + offset * 3.0, 0.0).rgb * W3;
	color += textureSampleLevel(srcTex, linearSampler, uv - offset * 3.0, 0.0).rgb * W3;
	color += textureSampleLevel(srcTex, linearSampler, uv + offset * 4.0, 0.0).rgb * W4;
	color += textureSampleLevel(srcTex, linearSampler, uv - offset * 4.0, 0.0).rgb * W4;

	textureStore(dstTex, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}
