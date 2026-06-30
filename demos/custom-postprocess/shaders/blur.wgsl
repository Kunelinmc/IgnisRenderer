struct BlurParams {
	radius: f32,
	sigma: f32,
	width: f32,
	height: f32,
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: BlurParams;
@group(0) @binding(2) var outTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) {
		return;
	}

	let coord = vec2<i32>(gid.xy);

	let radius = i32(params.radius);
	let sigma = params.sigma;

	var colorSum = vec3<f32>(0.0);
	var weightSum = 0.0;

	// Loop boundaries are static to avoid WGSL compiler issues, checking radius dynamically inside
	for (var dy = -5; dy <= 5; dy++) {
		if (dy < -radius || dy > radius) { continue; }
		for (var dx = -5; dx <= 5; dx++) {
			if (dx < -radius || dx > radius) { continue; }

			let distSq = f32(dx * dx + dy * dy);
			let weight = exp(-distSq / (2.0 * sigma * sigma));

			let sampleCoord = clamp(
				coord + vec2<i32>(dx, dy),
				vec2<i32>(0),
				vec2<i32>(size) - vec2<i32>(1)
			);
			
			let sampled = textureLoad(srcTex, sampleCoord, 0).rgb;
			colorSum += sampled * weight;
			weightSum += weight;
		}
	}

	let originalAlpha = textureLoad(srcTex, coord, 0).a;
	
	var finalColor = colorSum;
	if (weightSum > 0.0) {
		finalColor = colorSum / weightSum;
	}

	textureStore(
		outTex,
		coord,
		vec4<f32>(finalColor, originalAlpha)
	);
}
