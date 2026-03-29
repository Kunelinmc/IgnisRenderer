struct OutlineParams {
	invSize: vec2<f32>,
	opacity: f32,
	thickness: f32,
	color: vec4<f32>,
	circleCount: f32,
	shape: f32,
	_pad0: vec3<f32>,
	circles: array<vec4<f32>, 64>,
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: OutlineParams;
@group(0) @binding(3) var outTex: texture_storage_2d<rgba16float, write>;

fn outlineShapeDistance(pixelDelta: vec2<f32>, shape: i32) -> f32 {
	let absDelta = abs(pixelDelta);
	if (shape == 1) {
		return max(absDelta.x, absDelta.y) * 1.41421356237;
	}
	if (shape == 2) {
		return absDelta.x + absDelta.y;
	}
	if (shape == 3) {
		return max(max(absDelta.x, absDelta.y), (absDelta.x + absDelta.y) * 0.70710678118);
	}
	return length(pixelDelta);
}

fn outlineMask(pixel: vec2<f32>) -> f32 {
	let thickness = max(1.0, params.thickness);
	let feather = max(1.0, thickness * 0.75);
	let shapeId = i32(params.shape + 0.5);
	var mask = 0.0;
	for (var index: u32 = 0u; index < 64u; index = index + 1u) {
		if (f32(index) >= params.circleCount) {
			break;
		}
		let circle = params.circles[index];
		let radius = max(0.0, circle.z);
		if (radius <= 0.0) {
			continue;
		}
		let shapeDistance = outlineShapeDistance(pixel - circle.xy, shapeId);
		let edgeDistance = abs(shapeDistance - radius);
		let edgeAlpha = 1.0 - smoothstep(thickness, thickness + feather, edgeDistance);
		mask = max(mask, edgeAlpha);
	}
	return clamp(mask, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) {
		return;
	}

	let coord = vec2<i32>(gid.xy);
	let srcColor = textureLoad(srcTex, coord, 0);
	let pixel = vec2<f32>(f32(coord.x) + 0.5, f32(coord.y) + 0.5);
	let mask = outlineMask(pixel);
	let alpha = clamp(mask * params.opacity, 0.0, 1.0);
	let blended = mix(srcColor.rgb, params.color.rgb, vec3<f32>(alpha));
	textureStore(outTex, coord, vec4<f32>(blended, srcColor.a));
}
