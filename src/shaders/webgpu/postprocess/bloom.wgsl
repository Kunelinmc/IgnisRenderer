struct DownsampleParams {
	srcInvSize: vec2<f32>,
	threshold: f32,
	softKnee: f32,
}

struct BlurParams {
	texelSize: vec2<f32>,
}

struct UpsampleParams {
	texelSize: vec2<f32>,
	filterRadius: f32,
	_pad0: f32,
}

struct CompositeParams {
	invSize: vec2<f32>,
	intensity: f32,
	_pad0: f32,
}

// Entry-point reachability limits each pipeline to the resources it uses.
@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var<uniform> downsampleParams: DownsampleParams;
@group(0) @binding(3) var dstTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> blurParams: BlurParams;
@group(0) @binding(5) var secondaryTex: texture_2d<f32>;
@group(0) @binding(6) var<uniform> upsampleParams: UpsampleParams;
@group(0) @binding(7) var<uniform> compositeParams: CompositeParams;

// 9-tap Gaussian weights shared by both blur axes.
const W0: f32 = 0.227027;
const W1: f32 = 0.194594;
const W2: f32 = 0.121621;
const W3: f32 = 0.054054;
const W4: f32 = 0.016216;

fn luminance(color: vec3<f32>) -> f32 {
	return dot(max(color, vec3<f32>(0.0)), vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn extractBloom(color: vec4<f32>) -> vec4<f32> {
	if (downsampleParams.threshold < 0.0) {
		return color;
	}
	let luma = luminance(color.rgb);
	let knee = max(downsampleParams.softKnee, 1e-4);
	let soft = clamp(
		(luma - downsampleParams.threshold + knee) / (2.0 * knee),
		0.0,
		1.0
	);
	let contribution = soft * soft * knee + max(luma - downsampleParams.threshold, 0.0);
	let scale = clamp(contribution / max(luma, 1e-4), 0.0, 1.0);
	return vec4<f32>(color.rgb * scale, clamp(color.a, 0.0, 1.0) * scale);
}

// 13-tap downsample filter suppresses fireflies from HDR sources.
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

	var result = g * 0.125;
	result += (d + e + i + j) * 0.125;
	result += (a + b + f + g) * 0.03125;
	result += (b + c + g + h) * 0.03125;
	result += (f + g + k + l) * 0.03125;
	result += (g + h + l + m) * 0.03125;
	return result;
}

@compute @workgroup_size(8, 8, 1)
fn csDownsample(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(dstTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }

	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
	let color = downsample13Tap(uv, downsampleParams.srcInvSize);
	textureStore(dstTex, vec2<i32>(gid.xy), extractBloom(color));
}

fn blur(gid: vec3<u32>, direction: vec2<f32>) {
	let size = textureDimensions(dstTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }

	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
	let offset = blurParams.texelSize * direction;

	var color = textureSampleLevel(srcTex, linearSampler, uv, 0.0) * W0;
	color += textureSampleLevel(srcTex, linearSampler, uv + offset * 1.0, 0.0) * W1;
	color += textureSampleLevel(srcTex, linearSampler, uv - offset * 1.0, 0.0) * W1;
	color += textureSampleLevel(srcTex, linearSampler, uv + offset * 2.0, 0.0) * W2;
	color += textureSampleLevel(srcTex, linearSampler, uv - offset * 2.0, 0.0) * W2;
	color += textureSampleLevel(srcTex, linearSampler, uv + offset * 3.0, 0.0) * W3;
	color += textureSampleLevel(srcTex, linearSampler, uv - offset * 3.0, 0.0) * W3;
	color += textureSampleLevel(srcTex, linearSampler, uv + offset * 4.0, 0.0) * W4;
	color += textureSampleLevel(srcTex, linearSampler, uv - offset * 4.0, 0.0) * W4;

	textureStore(dstTex, vec2<i32>(gid.xy), color);
}

@compute @workgroup_size(8, 8, 1)
fn csBlurH(@builtin(global_invocation_id) gid: vec3<u32>) {
	blur(gid, vec2<f32>(1.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn csBlurV(@builtin(global_invocation_id) gid: vec3<u32>) {
	blur(gid, vec2<f32>(0.0, 1.0));
}

fn tentFilter(uv: vec2<f32>, texelSize: vec2<f32>, filterRadius: f32) -> vec4<f32> {
	let d = texelSize * filterRadius;

	var result = textureSampleLevel(srcTex, linearSampler, uv + vec2<f32>(-d.x,  d.y), 0.0);
	result    += textureSampleLevel(srcTex, linearSampler, uv + vec2<f32>( 0.0,  d.y), 0.0) * 2.0;
	result    += textureSampleLevel(srcTex, linearSampler, uv + vec2<f32>( d.x,  d.y), 0.0);
	result    += textureSampleLevel(srcTex, linearSampler, uv + vec2<f32>(-d.x,  0.0), 0.0) * 2.0;
	result    += textureSampleLevel(srcTex, linearSampler, uv,                         0.0) * 4.0;
	result    += textureSampleLevel(srcTex, linearSampler, uv + vec2<f32>( d.x,  0.0), 0.0) * 2.0;
	result    += textureSampleLevel(srcTex, linearSampler, uv + vec2<f32>(-d.x, -d.y), 0.0);
	result    += textureSampleLevel(srcTex, linearSampler, uv + vec2<f32>( 0.0, -d.y), 0.0) * 2.0;
	result    += textureSampleLevel(srcTex, linearSampler, uv + vec2<f32>( d.x, -d.y), 0.0);

	return result * (1.0 / 16.0);
}

@compute @workgroup_size(8, 8, 1)
fn csUpsample(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(dstTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }

	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
	let upsampled = tentFilter(uv, upsampleParams.texelSize, upsampleParams.filterRadius);
	let existing = textureSampleLevel(secondaryTex, linearSampler, uv, 0.0);
	let combined = vec4<f32>(
		existing.rgb + upsampled.rgb,
		1.0 - (1.0 - clamp(existing.a, 0.0, 1.0)) *
			(1.0 - clamp(upsampled.a, 0.0, 1.0))
	);

	textureStore(dstTex, vec2<i32>(gid.xy), combined);
}

@compute @workgroup_size(8, 8, 1)
fn csComposite(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(dstTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }

	let coord = vec2<i32>(gid.xy);
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * compositeParams.invSize;
	let scene = textureLoad(srcTex, coord, 0);
	let bloom = textureSampleLevel(secondaryTex, linearSampler, uv, 0.0);
	let intensity = max(compositeParams.intensity, 0.0);
	let outColor = max(scene.rgb + bloom.rgb * intensity, vec3<f32>(0.0));
	let bloomCoverage = clamp(bloom.a * intensity, 0.0, 1.0);
	let outputAlpha = clamp(scene.a, 0.0, 1.0) +
		bloomCoverage * (1.0 - clamp(scene.a, 0.0, 1.0));

	textureStore(dstTex, coord, vec4<f32>(outColor, outputAlpha));
}
