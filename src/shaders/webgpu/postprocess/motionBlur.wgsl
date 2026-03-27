struct Params {
	invSize: vec2<f32>,
	shutterScale: f32,
	maxSamples: f32,
	velocityClamp: f32,
	depthReject: f32,
	centerWeight: f32,
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

fn depthConfidence(centerDepth: f32, sampleDepth: f32) -> f32 {
	if (centerDepth <= 0.0 || sampleDepth <= 0.0) {
		return 0.0;
	}
	let rel = abs(centerDepth - sampleDepth) /
		max(max(centerDepth, sampleDepth), 1e-4);
	let reject = max(params.depthReject, 1e-5);
	return 1.0 - smoothstep(reject, reject * 4.0, rel);
}

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }

	let coord = vec2<i32>(gid.xy);
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * params.invSize;
	let source = textureLoad(srcTex, coord, 0);
	let centerDepth = textureLoad(gMotionDepth, coord, 0).z;

	let rawVelocity = textureLoad(gMotionDepth, coord, 0).xy;
	var velocity = vec2<f32>(rawVelocity.x * 0.5, -rawVelocity.y * 0.5) *
		max(params.shutterScale, 0.0);
	let velocityMag = length(velocity);
	let minInv = max(max(params.invSize.x, params.invSize.y), 1e-6);
	if (velocityMag <= minInv * 0.35) {
		textureStore(outTex, coord, source);
		return;
	}

	let maxVelocity = max(params.velocityClamp, minInv);
	let clampScale = min(1.0, maxVelocity / max(velocityMag, 1e-6));
	velocity = velocity * clampScale;

	let pixelVelocity = length(velocity / vec2<f32>(minInv, minInv));
	let sampleCount = i32(
		clamp(ceil(pixelVelocity), 1.0, max(params.maxSamples, 1.0))
	);

	var accum = source.rgb * max(params.centerWeight, 0.0);
	var weight = max(params.centerWeight, 0.0);
	let sourceLuma = luma(source.rgb);

	for (var i: i32 = 1; i <= 64; i = i + 1) {
		if (i > sampleCount) { break; }
		let t = f32(i) / f32(max(sampleCount, 1));
		let offset = velocity * t;

		let uvA = clamp(
			uv - offset,
			vec2<f32>(params.invSize * 0.5),
			vec2<f32>(1.0) - params.invSize * 0.5
		);
		let uvB = clamp(
			uv + offset,
			vec2<f32>(params.invSize * 0.5),
			vec2<f32>(1.0) - params.invSize * 0.5
		);

		let sampleA = textureSampleLevel(srcTex, linearSampler, uvA, 0.0);
		let sampleB = textureSampleLevel(srcTex, linearSampler, uvB, 0.0);
		let depthA = textureSampleLevel(gMotionDepth, linearSampler, uvA, 0.0).z;
		let depthB = textureSampleLevel(gMotionDepth, linearSampler, uvB, 0.0).z;

		let motionWeight = 1.0 - t * 0.85;
		let lumaWeightA = 0.5 +
			0.5 * clamp(luma(sampleA.rgb) / max(sourceLuma, 1e-4), 0.0, 1.5);
		let lumaWeightB = 0.5 +
			0.5 * clamp(luma(sampleB.rgb) / max(sourceLuma, 1e-4), 0.0, 1.5);
		let weightA =
			motionWeight * depthConfidence(centerDepth, depthA) * lumaWeightA;
		let weightB =
			motionWeight * depthConfidence(centerDepth, depthB) * lumaWeightB;

		accum += sampleA.rgb * weightA;
		accum += sampleB.rgb * weightB;
		weight += weightA + weightB;
	}

	let outColor = max(accum / max(weight, 1e-4), vec3<f32>(0.0));
	textureStore(outTex, coord, vec4<f32>(outColor, source.a));
}
