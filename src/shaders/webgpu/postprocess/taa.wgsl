struct Params {
	invSize: vec2<f32>,
	historyWeight: f32,
	depthThreshold: f32,
	motionFactor: f32,
	varianceClampGamma: f32,
	sharpen: f32,
	historyValid: f32,
}

@group(0) @binding(0) var currentColor: texture_2d<f32>;
@group(0) @binding(1) var historyColor: texture_2d<f32>;
@group(0) @binding(2) var motionDepth: texture_2d<f32>;
@group(0) @binding(3) var motionHistory: texture_2d<f32>;
@group(0) @binding(4) var linearSampler: sampler;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var outColor: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var outHistory: texture_storage_2d<rgba16float, write>;

fn rgbToYCoCg(c: vec3<f32>) -> vec3<f32> {
	let co = c.r - c.b;
	let t = c.b + co * 0.5;
	let cg = c.g - t;
	let y = t + cg * 0.5;
	return vec3<f32>(y, co, cg);
}

fn yCoCgToRgb(c: vec3<f32>) -> vec3<f32> {
	let t = c.x - c.z * 0.5;
	let g = c.z + t;
	let b = t - c.y * 0.5;
	let r = b + c.y;
	return vec3<f32>(r, g, b);
}

fn luma(c: vec3<f32>) -> f32 {
	return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn minPositiveDepth(a: f32, b: f32) -> f32 {
	if (a <= 0.0) { return max(0.0, b); }
	if (b <= 0.0) { return max(0.0, a); }
	return min(a, b);
}

fn loadDepthMinCross(
	textureData: texture_2d<f32>,
	coord: vec2<i32>,
	size: vec2<u32>
) -> f32 {
	var minDepth = textureLoad(textureData, coord, 0).z;
	let leftCoord = vec2<i32>(max(coord.x - 1, 0), coord.y);
	let rightCoord = vec2<i32>(min(coord.x + 1, i32(size.x) - 1), coord.y);
	let upCoord = vec2<i32>(coord.x, max(coord.y - 1, 0));
	let downCoord = vec2<i32>(coord.x, min(coord.y + 1, i32(size.y) - 1));
	minDepth = minPositiveDepth(minDepth, textureLoad(textureData, leftCoord, 0).z);
	minDepth = minPositiveDepth(minDepth, textureLoad(textureData, rightCoord, 0).z);
	minDepth = minPositiveDepth(minDepth, textureLoad(textureData, upCoord, 0).z);
	minDepth = minPositiveDepth(minDepth, textureLoad(textureData, downCoord, 0).z);
	return minDepth;
}

fn sampleDepthMinCross(
	textureData: texture_2d<f32>,
	uv: vec2<f32>,
	texel: vec2<f32>
) -> f32 {
	var minDepth = textureSampleLevel(textureData, linearSampler, uv, 0.0).z;
	minDepth = minPositiveDepth(
		minDepth,
		textureSampleLevel(textureData, linearSampler, uv + vec2<f32>(texel.x, 0.0), 0.0).z
	);
	minDepth = minPositiveDepth(
		minDepth,
		textureSampleLevel(textureData, linearSampler, uv - vec2<f32>(texel.x, 0.0), 0.0).z
	);
	minDepth = minPositiveDepth(
		minDepth,
		textureSampleLevel(textureData, linearSampler, uv + vec2<f32>(0.0, texel.y), 0.0).z
	);
	minDepth = minPositiveDepth(
		minDepth,
		textureSampleLevel(textureData, linearSampler, uv - vec2<f32>(0.0, texel.y), 0.0).z
	);
	return minDepth;
}

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outColor);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let coord = vec2<i32>(gid.xy);
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * params.invSize;
	let safeInvSize = vec2<f32>(
		max(params.invSize.x, 1e-6),
		max(params.invSize.y, 1e-6)
	);
	let curr = textureLoad(currentColor, coord, 0);
	let motion = textureLoad(motionDepth, coord, 0).xy;
	let prevUv = uv - vec2<f32>(motion.x * 0.5, -motion.y * 0.5);
	let inside = prevUv.x >= 0.0 && prevUv.x <= 1.0 && prevUv.y >= 0.0 && prevUv.y <= 1.0;

	var minYCoCg = vec3<f32>(1e9, 1e9, 1e9);
	var maxYCoCg = vec3<f32>(-1e9, -1e9, -1e9);
	var sumYCoCg = vec3<f32>(0.0);
	var sumSqYCoCg = vec3<f32>(0.0);
	let offsets = array<vec2<i32>, 5>(
		vec2<i32>(0, 0),
		vec2<i32>(-1, 0),
		vec2<i32>(1, 0),
		vec2<i32>(0, -1),
		vec2<i32>(0, 1)
	);
	for (var i: i32 = 0; i < 5; i = i + 1) {
		let sampleCoord = clamp(coord + offsets[i], vec2<i32>(0, 0), vec2<i32>(i32(size.x) - 1, i32(size.y) - 1));
		let ycocg = rgbToYCoCg(textureLoad(currentColor, sampleCoord, 0).rgb);
		minYCoCg = min(minYCoCg, ycocg);
		maxYCoCg = max(maxYCoCg, ycocg);
		sumYCoCg = sumYCoCg + ycocg;
		sumSqYCoCg = sumSqYCoCg + ycocg * ycocg;
	}
	let meanYCoCg = sumYCoCg / 5.0;
	let varianceYCoCg = max(sumSqYCoCg / 5.0 - meanYCoCg * meanYCoCg, vec3<f32>(0.0));
	let sigmaYCoCg = sqrt(varianceYCoCg);
	let gamma = max(params.varianceClampGamma, 0.0);
	let varianceMin = meanYCoCg - sigmaYCoCg * gamma;
	let varianceMax = meanYCoCg + sigmaYCoCg * gamma;
	let intersectionMin = max(minYCoCg, varianceMin);
	let intersectionMax = min(maxYCoCg, varianceMax);
	let clampMin = min(intersectionMin, intersectionMax);
	let clampMax = max(intersectionMin, intersectionMax);

	var hist = textureSampleLevel(historyColor, linearSampler, prevUv, 0.0);
	let histYCoCg = clamp(rgbToYCoCg(hist.rgb), clampMin, clampMax);
	hist = vec4<f32>(max(yCoCgToRgb(histYCoCg), vec3<f32>(0.0)), hist.a);

	let currDepth = loadDepthMinCross(motionDepth, coord, size);
	let prevDepth = sampleDepthMinCross(motionHistory, prevUv, safeInvSize);
	let hasDepth = currDepth > 0.0 && prevDepth > 0.0;
	let relDepthDiff = select(
		1e6,
		abs(currDepth - prevDepth) / max(max(currDepth, prevDepth), 1e-4),
		hasDepth
	);
	let safeDepthThreshold = max(params.depthThreshold, 1e-4);
	let depthConfidence = select(
		0.0,
		1.0 - smoothstep(
			safeDepthThreshold * 0.5,
			safeDepthThreshold * 2.5,
			relDepthDiff
		),
		hasDepth
	);

	let prevMotion = textureSampleLevel(motionHistory, linearSampler, prevUv, 0.0).xy;
	let forwardUv = prevUv + vec2<f32>(prevMotion.x * 0.5, -prevMotion.y * 0.5);
	let reprojectionError = abs(forwardUv - uv) / safeInvSize;
	let reprojectionErrorPx = max(reprojectionError.x, reprojectionError.y);
	let reprojectionConfidence = 1.0 - smoothstep(0.75, 3.0, reprojectionErrorPx);

	let currLuma = luma(curr.rgb);
	let histLuma = luma(hist.rgb);
	let lumaDiff = abs(currLuma - histLuma) / max(max(currLuma, histLuma), 1e-3);
	let colorConfidence = 1.0 - smoothstep(0.12, 0.7, lumaDiff);

	let validBase = params.historyValid > 0.5 && inside;
	let historyConfidence = select(
		0.0,
		clamp(depthConfidence * reprojectionConfidence * colorConfidence, 0.0, 1.0),
		validBase
	);
	let motionMag = length(motion);
	let adaptive = clamp(
		params.historyWeight * exp(-motionMag * params.motionFactor),
		0.0,
		0.96
	);
	let blend = clamp(adaptive * historyConfidence, 0.0, 0.96);
	let temporalColor = mix(curr, hist, blend);

	let left = textureLoad(currentColor, vec2<i32>(max(coord.x - 1, 0), coord.y), 0);
	let right = textureLoad(currentColor, vec2<i32>(min(coord.x + 1, i32(size.x) - 1), coord.y), 0);
	let up = textureLoad(currentColor, vec2<i32>(coord.x, max(coord.y - 1, 0)), 0);
	let down = textureLoad(currentColor, vec2<i32>(coord.x, min(coord.y + 1, i32(size.y) - 1)), 0);
	let blur = (left + right + up + down) * 0.25;
	let sharpenStrength = max(params.sharpen, 0.0) * (1.0 - blend * 0.5);
	let outC = vec4<f32>(
		max(
			temporalColor.rgb + (temporalColor.rgb - blur.rgb) * sharpenStrength,
			vec3<f32>(0.0)
		),
		temporalColor.a
	);

	textureStore(outColor, coord, outC);
	textureStore(outHistory, coord, temporalColor);
}
