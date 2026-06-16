@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;

struct Params {
	strength: f32,
	invertX: f32,
	invertY: f32,
	heightSource: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

fn getHeight(p: vec2<i32>) -> f32 {
	let color = textureLoad(src, p, 0);
	let source = i32(params.heightSource + 0.5);
	if (source == 1) {
		return color.r;
	}
	if (source == 2) {
		return color.g;
	}
	if (source == 3) {
		return color.b;
	}
	if (source == 4) {
		return color.a;
	}
	return dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let dims = textureDimensions(src);
	let x = i32(gid.x);
	let y = i32(gid.y);

	if (x >= i32(dims.x) || y >= i32(dims.y)) { return; }

	// Sample 3x3 neighborhood
	let tl = getHeight(vec2<i32>(max(x - 1, 0), max(y - 1, 0)));
	let t  = getHeight(vec2<i32>(x,             max(y - 1, 0)));
	let tr = getHeight(vec2<i32>(min(x + 1, i32(dims.x) - 1), max(y - 1, 0)));
	
	let l  = getHeight(vec2<i32>(max(x - 1, 0), y));
	let r  = getHeight(vec2<i32>(min(x + 1, i32(dims.x) - 1), y));
	
	let bl = getHeight(vec2<i32>(max(x - 1, 0), min(y + 1, i32(dims.y) - 1)));
	let b  = getHeight(vec2<i32>(x,             min(y + 1, i32(dims.y) - 1)));
	let br = getHeight(vec2<i32>(min(x + 1, i32(dims.x) - 1), min(y + 1, i32(dims.y) - 1)));

	// Sobel operators
	let dx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
	let dy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);

	// Calculate normal
	var n = normalize(vec3<f32>(
		-dx * params.strength * params.invertX,
		-dy * params.strength * params.invertY,
		1.0
	));

	// Pack to [0, 1] range for rgba8unorm
	textureStore(dst, vec2<i32>(x, y), vec4<f32>(n * 0.5 + 0.5, 1.0));
}
