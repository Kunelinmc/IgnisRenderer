@group(0) @binding(0) var depthTex: texture_2d<f32>;
@group(0) @binding(1) var outTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn csInit(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let d = max(textureLoad(depthTex, vec2<i32>(gid.xy), 0).z, 0.0);
	textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(d, 0.0, 0.0, 1.0));
}

@group(1) @binding(0) var srcTex: texture_2d<f32>;
@group(1) @binding(1) var dstTex: texture_storage_2d<rgba16float, write>;

fn minPos(a: f32, b: f32) -> f32 {
	if (a <= 0.0) { return max(b, 0.0); }
	if (b <= 0.0) { return max(a, 0.0); }
	return min(a, b);
}

@compute @workgroup_size(8, 8, 1)
fn csReduce(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(dstTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let srcSize = textureDimensions(srcTex);
	let base = vec2<i32>(gid.xy) * 2;
	let p00 = clamp(base + vec2<i32>(0, 0), vec2<i32>(0, 0), vec2<i32>(srcSize) - vec2<i32>(1, 1));
	let p10 = clamp(base + vec2<i32>(1, 0), vec2<i32>(0, 0), vec2<i32>(srcSize) - vec2<i32>(1, 1));
	let p01 = clamp(base + vec2<i32>(0, 1), vec2<i32>(0, 0), vec2<i32>(srcSize) - vec2<i32>(1, 1));
	let p11 = clamp(base + vec2<i32>(1, 1), vec2<i32>(0, 0), vec2<i32>(srcSize) - vec2<i32>(1, 1));
	let d = minPos(minPos(textureLoad(srcTex, p00, 0).x, textureLoad(srcTex, p10, 0).x), minPos(textureLoad(srcTex, p01, 0).x, textureLoad(srcTex, p11, 0).x));
	textureStore(dstTex, vec2<i32>(gid.xy), vec4<f32>(d, 0.0, 0.0, 1.0));
}
