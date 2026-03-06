struct Params {
	invSize: vec2<f32>,
	historyWeight: f32,
	depthThreshold: f32,
	motionFactor: f32,
	varianceClampGamma: f32,
	sharpen: f32,
	historyValid: f32,
	_pad0: f32,
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

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outColor);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let coord = vec2<i32>(gid.xy);
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * params.invSize;
	let curr = textureLoad(currentColor, coord, 0);
	let motion = textureLoad(motionDepth, coord, 0).xy;
	let prevUv = uv - vec2<f32>(motion.x * 0.5, -motion.y * 0.5);
	var minYCoCg = vec3<f32>(1e9, 1e9, 1e9);
	var maxYCoCg = vec3<f32>(-1e9, -1e9, -1e9);
	for (var y: i32 = -1; y <= 1; y = y + 1) {
		for (var x: i32 = -1; x <= 1; x = x + 1) {
			let sampleCoord = clamp(coord + vec2<i32>(x, y), vec2<i32>(0, 0), vec2<i32>(i32(size.x) - 1, i32(size.y) - 1));
			let ycocg = rgbToYCoCg(textureLoad(currentColor, sampleCoord, 0).rgb);
			minYCoCg = min(minYCoCg, ycocg);
			maxYCoCg = max(maxYCoCg, ycocg);
		}
	}
	var hist = textureSampleLevel(historyColor, linearSampler, prevUv, 0.0);
	let histYCoCg = clamp(rgbToYCoCg(hist.rgb), minYCoCg - vec3<f32>(params.varianceClampGamma), maxYCoCg + vec3<f32>(params.varianceClampGamma));
	hist.rgb = max(yCoCgToRgb(histYCoCg), vec3<f32>(0.0));
	let currDepth = textureLoad(motionDepth, coord, 0).z;
	let prevDepth = textureSampleLevel(motionHistory, linearSampler, prevUv, 0.0).z;
	let relDepthDiff = abs(currDepth - prevDepth) / max(max(currDepth, prevDepth), 1e-4);
	let inside = prevUv.x >= 0.0 && prevUv.x <= 1.0 && prevUv.y >= 0.0 && prevUv.y <= 1.0;
	let valid = params.historyValid > 0.5 && inside && relDepthDiff <= params.depthThreshold;
	let motionMag = length(motion);
	let adaptive = clamp(params.historyWeight * exp(-motionMag * params.motionFactor), 0.0, 0.95);
	let blend = select(0.0, adaptive, valid);
	var outC = mix(curr, hist, blend);
	let left = textureLoad(currentColor, vec2<i32>(max(coord.x - 1, 0), coord.y), 0);
	let right = textureLoad(currentColor, vec2<i32>(min(coord.x + 1, i32(size.x) - 1), coord.y), 0);
	let up = textureLoad(currentColor, vec2<i32>(coord.x, max(coord.y - 1, 0)), 0);
	let down = textureLoad(currentColor, vec2<i32>(coord.x, min(coord.y + 1, i32(size.y) - 1)), 0);
	let blur = (left + right + up + down) * 0.25;
	outC.rgb = max(outC.rgb + (outC.rgb - blur.rgb) * params.sharpen, vec3<f32>(0.0));
	textureStore(outColor, coord, outC);
	textureStore(outHistory, coord, outC);
}
