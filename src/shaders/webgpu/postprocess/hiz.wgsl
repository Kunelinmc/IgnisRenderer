@group(0) @binding(0) var depthTex: texture_2d<f32>;
@group(0) @binding(1) var outTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn csInit(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let d = max(textureLoad(depthTex, vec2<i32>(gid.xy), 0).z, 0.0);
	// .x = minDepth, .y = maxDepth — both channels for HiZ traversal
	textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(d, d, 0.0, 1.0));
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var dstTex: texture_storage_2d<rgba16float, write>;

fn minPos(a: f32, b: f32) -> f32 {
	if (a <= 0.0) { return max(b, 0.0); }
	if (b <= 0.0) { return max(a, 0.0); }
	return min(a, b);
}

fn maxPos(a: f32, b: f32) -> f32 {
	return max(a, b);
}

@compute @workgroup_size(8, 8, 1)
fn csReduce(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(dstTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let srcSize = textureDimensions(srcTex);
	let base = vec2<i32>(gid.xy) * 2;
	let maxCoord = vec2<i32>(srcSize) - vec2<i32>(1, 1);
	let p00 = clamp(base + vec2<i32>(0, 0), vec2<i32>(0, 0), maxCoord);
	let p10 = clamp(base + vec2<i32>(1, 0), vec2<i32>(0, 0), maxCoord);
	let p01 = clamp(base + vec2<i32>(0, 1), vec2<i32>(0, 0), maxCoord);
	let p11 = clamp(base + vec2<i32>(1, 1), vec2<i32>(0, 0), maxCoord);
	let s00 = textureLoad(srcTex, p00, 0);
	let s10 = textureLoad(srcTex, p10, 0);
	let s01 = textureLoad(srcTex, p01, 0);
	let s11 = textureLoad(srcTex, p11, 0);
	let dMin = minPos(minPos(s00.x, s10.x), minPos(s01.x, s11.x));
	let dMax = maxPos(maxPos(s00.y, s10.y), maxPos(s01.y, s11.y));
	textureStore(dstTex, vec2<i32>(gid.xy), vec4<f32>(dMin, dMax, 0.0, 1.0));
}
