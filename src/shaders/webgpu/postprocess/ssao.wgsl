#import <ignis/webgpu/constants>

struct Params {
	invSize: vec4<f32>,
	gtao: vec4<f32>,
	blurProj: vec4<f32>,
	passParams: vec4<f32>,
}

@group(0) @binding(0) var texA: texture_2d<f32>;
@group(0) @binding(1) var texB: texture_2d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var outTex: texture_storage_2d<rgba16float, write>;

const MAX_DIRECTION_COUNT: i32 = 8;
const MAX_STEP_COUNT: i32 = 6;

fn saturate(value: f32) -> f32 {
	return clamp(value, 0.0, 1.0);
}

fn signNotZero2(value: vec2<f32>) -> vec2<f32> {
	return vec2<f32>(
		select(-1.0, 1.0, value.x >= 0.0),
		select(-1.0, 1.0, value.y >= 0.0)
	);
}

fn octahedralWrap(value: vec2<f32>) -> vec2<f32> {
	return (vec2<f32>(1.0) - abs(value.yx)) * signNotZero2(value);
}

fn decodeNormal(encoded: vec2<f32>) -> vec3<f32> {
	let oct = encoded * 2.0 - vec2<f32>(1.0);
	var n = vec3<f32>(oct.x, oct.y, 1.0 - abs(oct.x) - abs(oct.y));
	if (n.z < 0.0) {
		n = vec3<f32>(octahedralWrap(n.xy), n.z);
	}
	return normalize(n);
}

fn reconstructViewPos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
	let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
	if (params.passParams.z > 0.5) {
		return vec3<f32>(ndc, -depth);
	}
	let tanHalfFov = max(params.blurProj.z, 1e-4);
	let aspect = max(params.blurProj.w, 1e-4);
	let x = ndc.x * aspect * tanHalfFov * depth;
	let y = ndc.y * tanHalfFov * depth;
	return vec3<f32>(x, y, -depth);
}

fn interleavedGradientNoise(pixel: vec2<f32>, frameJitter: f32) -> f32 {
	let seed = dot(pixel, vec2<f32>(0.06711056, 0.00583715));
	return fract(52.9829189 * fract(seed + frameJitter * 0.754877666));
}

fn isUvInside01(uv: vec2<f32>) -> bool {
	return !any(uv < vec2<f32>(0.0)) && !any(uv > vec2<f32>(1.0));
}

fn accumulateDirectionalHorizon(
	sampleUv: vec2<f32>,
	depth: f32,
	bias: f32,
	centerPos: vec3<f32>,
	normal: vec3<f32>,
	radiusView: f32,
	currentHorizon: f32
) -> f32 {
	if (!isUvInside01(sampleUv)) {
		return currentHorizon;
	}
	let sampleDepth = textureSampleLevel(texB, linearSampler, sampleUv, 0.0).z;
	if (sampleDepth <= 0.0 || sampleDepth >= depth - bias) {
		return currentHorizon;
	}
	let samplePos = reconstructViewPos(sampleUv, sampleDepth);
	let delta = samplePos - centerPos;
	let distSq = dot(delta, delta);
	if (distSq <= 1e-6) {
		return currentHorizon;
	}
	let invDist = inverseSqrt(distSq);
	let dist = distSq * invDist;
	let alignment = max(dot(normal, delta * invDist), 0.0);
	let distWeight = saturate(1.0 - dist / radiusView);
	return max(currentHorizon, alignment * distWeight);
}

@compute @workgroup_size(8, 8, 1)
fn csRaw(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * params.invSize.zw;
	let center = textureSampleLevel(texB, linearSampler, uv, 0.0);
	let depth = center.z;
	if (depth <= 0.0) {
		textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(1.0, 1.0, 1.0, 1.0));
		return;
	}
	let normal = decodeNormal(textureSampleLevel(texA, linearSampler, uv, 0.0).xy);
	let centerPos = reconstructViewPos(uv, depth);

	let sampleBudget = clamp(i32(params.gtao.w + 0.5), i32(4), i32(48));
	let directionCount = clamp(
		(sampleBudget + 3) / 4,
		i32(2),
		i32(MAX_DIRECTION_COUNT)
	);
	let stepCount = clamp(
		(sampleBudget + directionCount * 2 - 1) / (directionCount * 2),
		i32(1),
		i32(MAX_STEP_COUNT)
	);
	let radiusPixels = max(params.gtao.x, 1.0);
	let radiusUv = radiusPixels * params.invSize.xy;
	let bias = max(params.gtao.y, 1e-4);
	let intensity = max(params.gtao.z, 0.0);
	let frameNoise = interleavedGradientNoise(
		vec2<f32>(gid.xy),
		params.passParams.w
	);

	let perspectiveRadiusView = radiusPixels
		* depth
		* max(params.blurProj.z, 0.05)
		* max(params.invSize.x, params.invSize.y)
		* 2.0;
	let orthographicRadiusView = max(radiusUv.x, radiusUv.y) * 2.0;
	let radiusView = max(
		select(
			perspectiveRadiusView,
			orthographicRadiusView,
			params.passParams.z > 0.5
		),
		1e-3
	);

	var occ = 0.0;
	for (var dirIdx: i32 = 0; dirIdx < MAX_DIRECTION_COUNT; dirIdx = dirIdx + 1) {
		if (dirIdx >= directionCount) { break; }
		let angle = ((f32(dirIdx) + frameNoise) / f32(directionCount)) * PI;
		let dir2 = vec2<f32>(cos(angle), sin(angle));

		var horizonPos = 0.0;
		var horizonNeg = 0.0;

		for (var stepIdx: i32 = 1; stepIdx <= MAX_STEP_COUNT; stepIdx = stepIdx + 1) {
			if (stepIdx > stepCount) { break; }

			let jitter = fract(frameNoise + f32(stepIdx) * GOLDEN_RATIO_CONJUGATE);
			let stepFrac = (f32(stepIdx) - 0.35 + jitter * 0.6) / f32(stepCount);
			let stepUv = dir2 * stepFrac * radiusUv;

			horizonPos = accumulateDirectionalHorizon(
				uv + stepUv,
				depth,
				bias,
				centerPos,
				normal,
				radiusView,
				horizonPos
			);
			horizonNeg = accumulateDirectionalHorizon(
				uv - stepUv,
				depth,
				bias,
				centerPos,
				normal,
				radiusView,
				horizonNeg
			);
		}

		occ += 0.5 * (horizonPos + horizonNeg);
	}

	let horizonOcclusion = occ / max(f32(directionCount), 1.0);
	let ao = clamp(1.0 - horizonOcclusion * intensity, 0.0, 1.0);
	textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(ao, ao, ao, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn csCombine(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let coord = vec2<i32>(gid.xy);
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * params.invSize.xy;
	let color = textureLoad(texA, coord, 0);
	let ao = textureSampleLevel(texB, linearSampler, uv, 0.0).x;
	textureStore(
		outTex,
		coord,
		vec4<f32>(max(color.rgb * clamp(ao, 0.0, 1.0), vec3<f32>(0.0)), color.a)
	);
}
