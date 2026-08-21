struct Params {
	srcInvSize: vec2<f32>,
	threshold: f32,
	softKnee: f32,
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var dstTex: texture_storage_2d<rgba16float, write>;

fn luminance(color: vec3<f32>) -> f32 {
	return dot(max(color, vec3<f32>(0.0)), vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn extractBloom(color: vec4<f32>) -> vec4<f32> {
	if (params.threshold < 0.0) {
		return color;
	}
	let luma = luminance(color.rgb);
	let knee = max(params.softKnee, 1e-4);
	let soft = clamp(
		(luma - params.threshold + knee) / (2.0 * knee),
		0.0,
		1.0
	);
	let contribution = soft * soft * knee + max(luma - params.threshold, 0.0);
	let scale = clamp(contribution / max(luma, 1e-4), 0.0, 1.0);
	return vec4<f32>(color.rgb * scale, clamp(color.a, 0.0, 1.0) * scale);
}

// 13-tap box filter for downsample — matches the Call-of-Duty approach.
// Produces a high-quality anti-aliased downsample that reduces firefly
// artifacts from HDR sources.
fn downsample13Tap(uv: vec2<f32>, texelSize: vec2<f32>) -> vec4<f32> {
	let a = textureSampleLevel(srcTex, linearSampler, uv + texelSize * vec2<f32>(-1.0, -1.0), 0.0);
	let b = textureSampleLevel(srcTex, linearSampler, uv + texelSize * vec2<f32>( 0.0, -1.0), 0.0);
	let c = textureSampleLevel(srcTex, linearSampler, uv + texelSize * vec2<f32>( 1.0, -1.0), 0.0);
	let d = textureSampleLevel(srcTex, linearSampler, uv + texelSize * vec2<f32>(-0.5, -0.5), 0.0);
	let e = textureSampleLevel(srcTex, linearSampler, uv + texelSize * vec2<f32>( 0.5, -0.5), 0.0);
	let f = textureSampleLevel(srcTex, linearSampler, uv + texelSize * vec2<f32>(-1.0,  0.0), 0.0);
	let g = textureSampleLevel(srcTex, linearSampler, uv,                                     0.0);
	let h = textureSampleLevel(srcTex, linearSampler, uv + texelSize * vec2<f32>( 1.0,  0.0), 0.0);
	let i = textureSampleLevel(srcTex, linearSampler, uv + texelSize * vec2<f32>(-0.5,  0.5), 0.0);
	let j = textureSampleLevel(srcTex, linearSampler, uv + texelSize * vec2<f32>( 0.5,  0.5), 0.0);
	let k = textureSampleLevel(srcTex, linearSampler, uv + texelSize * vec2<f32>(-1.0,  1.0), 0.0);
	let l = textureSampleLevel(srcTex, linearSampler, uv + texelSize * vec2<f32>( 0.0,  1.0), 0.0);
	let m = textureSampleLevel(srcTex, linearSampler, uv + texelSize * vec2<f32>( 1.0,  1.0), 0.0);

	// Weighted average that suppresses fireflies
	var result = g * 0.125;
	result += (d + e + i + j) * 0.125;
	result += (a + b + f + g) * 0.03125;
	result += (b + c + g + h) * 0.03125;
	result += (f + g + k + l) * 0.03125;
	result += (g + h + l + m) * 0.03125;
	return result;
}

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(dstTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }

	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
	let color = downsample13Tap(uv, params.srcInvSize);
	let bloom = extractBloom(color);
	textureStore(dstTex, vec2<i32>(gid.xy), bloom);
}
