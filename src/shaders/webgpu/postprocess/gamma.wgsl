struct GammaParams {
	gamma: f32,
	_pad0: f32,
	_pad1: f32,
	_pad2: f32,
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: GammaParams;
@group(0) @binding(2) var outTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) {
		return;
	}

	let coord = vec2<i32>(gid.xy);
	let sampled = textureLoad(srcTex, coord, 0);
	let gamma = max(params.gamma, 0.01);
	let linearColor = max(sampled.rgb, vec3<f32>(0.0));
	let gammaColor = pow(linearColor, vec3<f32>(1.0 / gamma));
	textureStore(
		outTex,
		coord,
		vec4<f32>(
			clamp(gammaColor, vec3<f32>(0.0), vec3<f32>(1.0)),
			sampled.a
		)
	);
}
