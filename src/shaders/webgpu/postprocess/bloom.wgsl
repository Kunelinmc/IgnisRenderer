struct Params {
	invSize: vec2<f32>,
	threshold: f32,
	softKnee: f32,
	intensity: f32,
	radius: f32,
	_pad0: f32,
	_pad1: f32,
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var outTex: texture_storage_2d<rgba16float, write>;

fn luminance(color: vec3<f32>) -> f32 {
	return dot(max(color, vec3<f32>(0.0)), vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn extractBloom(color: vec3<f32>) -> vec3<f32> {
	let luma = luminance(color);
	let softKnee = max(params.softKnee, 1e-4);
	let soft = clamp(
		(luma - params.threshold + softKnee) / (2.0 * softKnee),
		0.0,
		1.0
	);
	let softContribution = soft * soft * softKnee;
	let hardContribution = max(luma - params.threshold, 0.0);
	let contribution = softContribution + hardContribution;
	return color * (contribution / max(luma, 1e-4));
}

fn sampleBloom(uv: vec2<f32>, offset: vec2<f32>) -> vec3<f32> {
	let sampleUv = uv + offset * params.invSize * params.radius;
	return extractBloom(textureSampleLevel(srcTex, linearSampler, sampleUv, 0.0).rgb);
}

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }

	let coord = vec2<i32>(gid.xy);
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * params.invSize;
	let source = textureLoad(srcTex, coord, 0);

	var bloom = sampleBloom(uv, vec2<f32>(0.0, 0.0)) * 0.204164;
	bloom += sampleBloom(uv, vec2<f32>(1.0, 0.0)) * 0.123841;
	bloom += sampleBloom(uv, vec2<f32>(-1.0, 0.0)) * 0.123841;
	bloom += sampleBloom(uv, vec2<f32>(0.0, 1.0)) * 0.123841;
	bloom += sampleBloom(uv, vec2<f32>(0.0, -1.0)) * 0.123841;
	bloom += sampleBloom(uv, vec2<f32>(1.0, 1.0)) * 0.07488;
	bloom += sampleBloom(uv, vec2<f32>(-1.0, 1.0)) * 0.07488;
	bloom += sampleBloom(uv, vec2<f32>(1.0, -1.0)) * 0.07488;
	bloom += sampleBloom(uv, vec2<f32>(-1.0, -1.0)) * 0.07488;

	let outColor = max(source.rgb + bloom * max(params.intensity, 0.0), vec3<f32>(0.0));
	textureStore(outTex, coord, vec4<f32>(outColor, source.a));
}
