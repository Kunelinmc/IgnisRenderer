@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var outTex: texture_storage_2d<rgba16float, write>;

fn acesFitted(color: vec3<f32>) -> vec3<f32> {
	let a = 2.51;
	let b = 0.03;
	let c = 2.43;
	let d = 0.59;
	let e = 0.14;
	let mapped =
		(color * (a * color + vec3<f32>(b))) /
		(color * (c * color + vec3<f32>(d)) + vec3<f32>(e));
	return clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) {
		return;
	}

	let coord = vec2<i32>(gid.xy);
	let src = textureLoad(srcTex, coord, 0);
	let mapped = acesFitted(max(src.rgb, vec3<f32>(0.0)));
	textureStore(outTex, coord, vec4<f32>(mapped, src.a));
}
