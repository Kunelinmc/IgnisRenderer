struct Params {
	invSize: vec2<f32>,
	intensity: f32,
	_pad0: f32,
}

@group(0) @binding(0) var sceneTex: texture_2d<f32>;
@group(0) @binding(1) var bloomTex: texture_2d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var dstTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(dstTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }

	let coord = vec2<i32>(gid.xy);
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * params.invSize;
	let scene = textureLoad(sceneTex, coord, 0);
	let bloom = textureSampleLevel(bloomTex, linearSampler, uv, 0.0);
	let intensity = max(params.intensity, 0.0);
	let outColor = max(scene.rgb + bloom.rgb * intensity, vec3<f32>(0.0));
	let bloomCoverage = clamp(bloom.a * intensity, 0.0, 1.0);
	let outputAlpha = clamp(scene.a, 0.0, 1.0) +
		bloomCoverage * (1.0 - clamp(scene.a, 0.0, 1.0));

	textureStore(dstTex, coord, vec4<f32>(outColor, outputAlpha));
}
