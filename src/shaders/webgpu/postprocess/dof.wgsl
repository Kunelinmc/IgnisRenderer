struct Params {
	invSize: vec2<f32>,
	focusDistance: f32,
	focusRange: f32,
	nearStrength: f32,
	farStrength: f32,
	maxBlurRadius: f32,
	depthCurve: f32,
	highlightThreshold: f32,
	highlightGain: f32,
	chromaticAberration: f32,
	_pad0: f32,
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var gMotionDepth: texture_2d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var outTex: texture_storage_2d<rgba16float, write>;

fn luma(color: vec3<f32>) -> f32 {
	return dot(max(color, vec3<f32>(0.0)), vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn computeCoC(depth: f32) -> f32 {
	if (depth <= 0.0) {
		return 0.0;
	}
	let normalized = abs(depth - params.focusDistance) /
		max(params.focusRange, 1e-4);
	let shaped = pow(clamp(normalized, 0.0, 1.0), max(params.depthCurve, 0.25));
	let strength = select(
		max(params.nearStrength, 0.0),
		max(params.farStrength, 0.0),
		depth > params.focusDistance
	);
	return clamp(shaped * strength, 0.0, 1.0);
}

fn depthGate(centerDepth: f32, sampleDepth: f32, isFar: bool) -> f32 {
	if (centerDepth <= 0.0 || sampleDepth <= 0.0) {
		return 0.0;
	}
	let rel = abs(centerDepth - sampleDepth) /
		max(max(centerDepth, sampleDepth), 1e-4);
	var gate = 1.0 - smoothstep(0.015, 0.08, rel);
	if (isFar && sampleDepth + 0.001 < centerDepth) {
		gate *= 0.15;
	}
	if (!isFar && sampleDepth - 0.001 > centerDepth) {
		gate *= 0.2;
	}
	return gate;
}

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }

	let coord = vec2<i32>(gid.xy);
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * params.invSize;
	let source = textureLoad(srcTex, coord, 0);
	let centerDepth = textureLoad(gMotionDepth, coord, 0).z;
	let coc = computeCoC(centerDepth);
	let radiusPx = coc * max(params.maxBlurRadius, 0.0);
	if (radiusPx <= 0.25) {
		textureStore(outTex, coord, source);
		return;
	}

	let blurScale = params.invSize * radiusPx;
	let isFar = centerDepth > params.focusDistance;
	let offsets = array<vec2<f32>, 12>(
		vec2<f32>(1.0, 0.0),
		vec2<f32>(-1.0, 0.0),
		vec2<f32>(0.0, 1.0),
		vec2<f32>(0.0, -1.0),
		vec2<f32>(0.707, 0.707),
		vec2<f32>(-0.707, 0.707),
		vec2<f32>(0.707, -0.707),
		vec2<f32>(-0.707, -0.707),
		vec2<f32>(2.0, 0.0),
		vec2<f32>(-2.0, 0.0),
		vec2<f32>(0.0, 2.0),
		vec2<f32>(0.0, -2.0)
	);

	var accum = source.rgb;
	var weight = 1.0;
	for (var i: i32 = 0; i < 12; i = i + 1) {
		let sampleUv = clamp(
			uv + offsets[i] * blurScale,
			vec2<f32>(params.invSize * 0.5),
			vec2<f32>(1.0) - params.invSize * 0.5
		);
		let sampleColor = textureSampleLevel(srcTex, linearSampler, sampleUv, 0.0);
		let sampleDepth = textureSampleLevel(
			gMotionDepth,
			linearSampler,
			sampleUv,
			0.0
		).z;
		let gate = depthGate(centerDepth, sampleDepth, isFar);
		let highlight =
			max(luma(sampleColor.rgb) - params.highlightThreshold, 0.0) *
			max(params.highlightGain, 0.0);
		let sampleWeight = gate * (1.0 + highlight);
		accum += sampleColor.rgb * sampleWeight;
		weight += sampleWeight;
	}

	var color = mix(source.rgb, accum / max(weight, 1e-4), coc);
	if (params.chromaticAberration > 0.0) {
		var radial = uv - vec2<f32>(0.5, 0.5);
		if (dot(radial, radial) < 1e-6) {
			radial = vec2<f32>(1.0, 0.0);
		} else {
			radial = normalize(radial);
		}
		let chromaOffset =
			params.invSize * coc * params.chromaticAberration * radiusPx * 0.15;
		let redUv = clamp(
			uv + radial * chromaOffset,
			vec2<f32>(params.invSize * 0.5),
			vec2<f32>(1.0) - params.invSize * 0.5
		);
		let blueUv = clamp(
			uv - radial * chromaOffset,
			vec2<f32>(params.invSize * 0.5),
			vec2<f32>(1.0) - params.invSize * 0.5
		);
		let red = textureSampleLevel(srcTex, linearSampler, redUv, 0.0).r;
		let blue = textureSampleLevel(srcTex, linearSampler, blueUv, 0.0).b;
		color = vec3<f32>(red, color.g, blue);
	}

	textureStore(outTex, coord, vec4<f32>(max(color, vec3<f32>(0.0)), source.a));
}
