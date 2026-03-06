import { CameraType } from "../../../cameras/Camera";
import type {
	FrameContext,
	SSAOOptions,
	SSROptions,
	TAAOptions,
} from "../../pipeline/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import {
	AddressMode,
	BufferUsage,
	FilterMode,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type ISampler,
	type IShaderModule,
} from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
import type { WebGPUFrameTargets } from "./WebGPUPostProcessGraph";

interface InternalTexture extends IRenderTexture {
	_gpuTexture?: any;
	_gpuResource?: any;
}

const WORKGROUP_SIZE = 8;

const DEFAULT_SSAO: Required<
	Pick<
		SSAOOptions,
		| "samples"
		| "radius"
		| "bias"
		| "intensity"
		| "downsample"
		| "blurRadius"
		| "blurSharpness"
	>
> = {
	samples: 16,
	radius: 8,
	bias: 0.1,
	intensity: 1,
	downsample: 2,
	blurRadius: 2,
	blurSharpness: 8,
};

const DEFAULT_TAA: Required<
	Pick<
		TAAOptions,
		| "historyWeight"
		| "disocclusionDepthThreshold"
		| "motionFactor"
		| "varianceClampGamma"
		| "sharpen"
	>
> = {
	historyWeight: 0.9,
	disocclusionDepthThreshold: 0.02,
	motionFactor: 80,
	varianceClampGamma: 1,
	sharpen: 0.1,
};

const DEFAULT_SSR: Required<
	Pick<
		SSROptions,
		| "downsample"
		| "maxSteps"
		| "binarySearchSteps"
		| "maxDistance"
		| "thickness"
		| "stride"
		| "intensity"
		| "historyWeight"
		| "edgeFade"
		| "maxRoughness"
	>
> = {
	downsample: 2,
	maxSteps: 64,
	binarySearchSteps: 6,
	maxDistance: 100,
	thickness: 0.2,
	stride: 1,
	intensity: 1,
	historyWeight: 0.85,
	edgeFade: 0.12,
	maxRoughness: 0.85,
};

const DEFAULT_FXAA = {
	edgeThresholdMin: 0.03125,
	edgeThresholdMultiplier: 0.166,
	subpixQuality: 0.75,
};

const SSAO_SHADER = /* wgsl */ `
struct Params {
	fullInvSize: vec2<f32>,
	aoInvSize: vec2<f32>,
	radius: f32,
	bias: f32,
	intensity: f32,
	blurRadius: f32,
	blurSharpness: f32,
	_pad0: f32,
}

@group(0) @binding(0) var texA: texture_2d<f32>;
@group(0) @binding(1) var texB: texture_2d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var outTex: texture_storage_2d<rgba16float, write>;

fn decodeNormal(encoded: vec2<f32>) -> vec3<f32> {
	let xy = encoded * 2.0 - vec2<f32>(1.0, 1.0);
	let z2 = max(1.0 - dot(xy, xy), 0.0);
	return normalize(vec3<f32>(xy, sqrt(z2)));
}

@compute @workgroup_size(8, 8, 1)
fn csRaw(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * params.aoInvSize;
	let center = textureSampleLevel(texB, linearSampler, uv, 0.0);
	let depth = center.z;
	if (depth <= 0.0) {
		textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(1.0, 1.0, 1.0, 1.0));
		return;
	}
	let normal = decodeNormal(textureSampleLevel(texA, linearSampler, uv, 0.0).xy);
	let kernel = array<vec2<f32>, 8>(
		vec2<f32>(1.0, 0.0),
		vec2<f32>(-1.0, 0.0),
		vec2<f32>(0.0, 1.0),
		vec2<f32>(0.0, -1.0),
		vec2<f32>(0.7071, 0.7071),
		vec2<f32>(-0.7071, 0.7071),
		vec2<f32>(0.7071, -0.7071),
		vec2<f32>(-0.7071, -0.7071)
	);
	var occ = 0.0;
	for (var i: i32 = 0; i < 8; i = i + 1) {
		let offset = normalize(vec3<f32>(kernel[i], 0.25) + normal * 0.2).xy;
		let sampleUv = uv + offset * params.radius * params.fullInvSize;
		let sampleDepth = textureSampleLevel(texB, linearSampler, sampleUv, 0.0).z;
		occ += select(0.0, 1.0, sampleDepth > 0.0 && sampleDepth < depth - params.bias);
	}
	let ao = clamp(1.0 - (occ / 8.0) * params.intensity, 0.0, 1.0);
	textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(ao, ao, ao, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn csBlur(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let coord = vec2<i32>(gid.xy);
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * params.aoInvSize;
	let centerDepth = textureSampleLevel(texB, linearSampler, uv, 0.0).z;
	let radius = clamp(i32(params.blurRadius + 0.5), 1, 4);
	var sum = 0.0;
	var weightSum = 0.0;
	for (var y: i32 = -radius; y <= radius; y = y + 1) {
		for (var x: i32 = -radius; x <= radius; x = x + 1) {
			let sampleCoord = clamp(
				coord + vec2<i32>(x, y),
				vec2<i32>(0, 0),
				vec2<i32>(i32(size.x) - 1, i32(size.y) - 1)
			);
			let sampleUv = (vec2<f32>(sampleCoord) + vec2<f32>(0.5)) * params.aoInvSize;
			let sampleDepth = textureSampleLevel(texB, linearSampler, sampleUv, 0.0).z;
			let depthDelta = abs(sampleDepth - centerDepth);
			let bilateral = exp(-depthDelta * max(params.blurSharpness, 1e-3));
			let sampleAo = textureLoad(texA, sampleCoord, 0).x;
			sum += sampleAo * bilateral;
			weightSum += bilateral;
		}
	}
	let ao = select(textureLoad(texA, coord, 0).x, sum / max(weightSum, 1e-4), weightSum > 0.0);
	textureStore(outTex, coord, vec4<f32>(ao, ao, ao, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn csCombine(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let coord = vec2<i32>(gid.xy);
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * params.fullInvSize;
	let color = textureLoad(texA, coord, 0);
	let ao = textureSampleLevel(texB, linearSampler, uv, 0.0).x;
	textureStore(
		outTex,
		coord,
		vec4<f32>(max(color.rgb * clamp(ao, 0.0, 1.0), vec3<f32>(0.0)), color.a)
	);
}
`;

const TAA_SHADER = /* wgsl */ `
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
`;
const HIZ_SHADER = /* wgsl */ `
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
`;

const SSR_SHADER = /* wgsl */ `
struct TraceParams {
	invHalfSize: vec2<f32>,
	maxDistance: f32,
	thickness: f32,
	stride: f32,
	intensity: f32,
	maxRoughness: f32,
	edgeFade: f32,
	maxSteps: f32,
	binarySearchSteps: f32,
	maxMip: f32,
	historyWeight: f32,
	historyValid: f32,
	depthThreshold: f32,
}

@group(0) @binding(0) var sceneColor: texture_2d<f32>;
@group(0) @binding(1) var gNormalRoughMetal: texture_2d<f32>;
@group(0) @binding(2) var gMotionDepth: texture_2d<f32>;
@group(0) @binding(3) var hiZ: texture_2d<f32>;
@group(0) @binding(4) var ssrHistory: texture_2d<f32>;
@group(0) @binding(5) var motionHistory: texture_2d<f32>;
@group(0) @binding(6) var linearSampler: sampler;
@group(0) @binding(7) var<uniform> traceParams: TraceParams;
@group(0) @binding(8) var outSSR: texture_storage_2d<rgba16float, write>;

struct ComposeParams {
	invFullSize: vec2<f32>,
	_pad0: vec2<f32>,
}

@group(1) @binding(0) var composeScene: texture_2d<f32>;
@group(1) @binding(1) var composeSSR: texture_2d<f32>;
@group(1) @binding(2) var composeMotionDepth: texture_2d<f32>;
@group(1) @binding(3) var composeSampler: sampler;
@group(1) @binding(4) var<uniform> composeParams: ComposeParams;
@group(1) @binding(5) var composeOut: texture_storage_2d<rgba16float, write>;

fn decodeNormal(encoded: vec2<f32>) -> vec3<f32> {
	let xy = encoded * 2.0 - vec2<f32>(1.0, 1.0);
	let z2 = max(1.0 - dot(xy, xy), 0.0);
	return normalize(vec3<f32>(xy, sqrt(z2)));
}

fn viewToUv(viewPos: vec3<f32>) -> vec2<f32> {
	let z = max(-viewPos.z, 1e-4);
	let p = vec2<f32>(viewPos.x / z, viewPos.y / z);
	return vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
}

@compute @workgroup_size(8, 8, 1)
fn csTrace(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outSSR);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * traceParams.invHalfSize;
	let g = textureSampleLevel(gNormalRoughMetal, linearSampler, uv, 0.0);
	let depth = textureSampleLevel(gMotionDepth, linearSampler, uv, 0.0).z;
	let roughness = clamp(g.z, 0.0, 1.0);
	let metalness = clamp(g.w, 0.0, 1.0);
	if (depth <= 0.0 || roughness > traceParams.maxRoughness) { textureStore(outSSR, vec2<i32>(gid.xy), vec4<f32>(0.0)); return; }
	let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
	let viewPos = vec3<f32>(ndc * depth, -depth);
	let reflectionDir = normalize(reflect(normalize(-viewPos), decodeNormal(g.xy)));
	if (reflectionDir.z >= -1e-4) { textureStore(outSSR, vec2<i32>(gid.xy), vec4<f32>(0.0)); return; }
	var hitUv = uv;
	var hit = false;
	var t = max(traceParams.stride, 0.25);
	var hitT = t;
	var missT = 0.0;
	let maxSteps = i32(clamp(traceParams.maxSteps, 1.0, 128.0));
	for (var i: i32 = 0; i < 128; i = i + 1) {
		if (i >= maxSteps) { break; }
		let samplePos = viewPos + reflectionDir * t;
		if (samplePos.z >= -1e-4) { break; }
		let suv = viewToUv(samplePos);
		if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) { break; }
		let rayDepth = -samplePos.z;
		let mip = clamp(i32(log2(max(t * 0.25, 1.0))), 0, i32(traceParams.maxMip));
		let sceneDepth = textureSampleLevel(hiZ, linearSampler, suv, f32(mip)).x;
		if (sceneDepth > 0.0 && rayDepth >= sceneDepth - traceParams.thickness) {
			hit = true;
			hitUv = suv;
			hitT = t;
			break;
		}
		missT = t;
		t += traceParams.stride;
		if (t > traceParams.maxDistance) { break; }
	}
	if (!hit) { textureStore(outSSR, vec2<i32>(gid.xy), vec4<f32>(0.0)); return; }
	let refineCount = i32(clamp(traceParams.binarySearchSteps, 0.0, 16.0));
	for (var j: i32 = 0; j < 16; j = j + 1) {
		if (j >= refineCount) { break; }
		let midT = (missT + hitT) * 0.5;
		let refinePos = viewPos + reflectionDir * midT;
		let refineUv = viewToUv(refinePos);
		if (refineUv.x < 0.0 || refineUv.x > 1.0 || refineUv.y < 0.0 || refineUv.y > 1.0) {
			hitT = midT;
			continue;
		}
		let rayDepth = -refinePos.z;
		let sceneDepth = textureSampleLevel(hiZ, linearSampler, refineUv, 0.0).x;
		if (sceneDepth > 0.0 && rayDepth >= sceneDepth - traceParams.thickness) {
			hitT = midT;
			hitUv = refineUv;
		} else {
			missT = midT;
		}
	}
	let hitColor = textureSampleLevel(sceneColor, linearSampler, hitUv, 0.0).rgb;
	let edgeDistance = min(min(hitUv.x, 1.0 - hitUv.x), min(hitUv.y, 1.0 - hitUv.y));
	let edge = clamp(edgeDistance / max(traceParams.edgeFade, 1e-4), 0.0, 1.0);
	let weight = traceParams.intensity * edge * max(metalness, 0.04) * (1.0 - roughness);
	let motion = textureSampleLevel(gMotionDepth, linearSampler, uv, 0.0).xy;
	let prevUv = uv - vec2<f32>(motion.x * 0.5, -motion.y * 0.5);
	let hist = textureSampleLevel(ssrHistory, linearSampler, prevUv, 0.0);
	let currDepth = textureSampleLevel(gMotionDepth, linearSampler, uv, 0.0).z;
	let prevDepth = textureSampleLevel(motionHistory, linearSampler, prevUv, 0.0).z;
	let relDepth = abs(currDepth - prevDepth) / max(max(currDepth, prevDepth), 1e-4);
	let historyOk = traceParams.historyValid > 0.5 && prevUv.x >= 0.0 && prevUv.x <= 1.0 && prevUv.y >= 0.0 && prevUv.y <= 1.0 && relDepth <= traceParams.depthThreshold;
	let blend = select(0.0, clamp(traceParams.historyWeight, 0.0, 0.95), historyOk);
	textureStore(outSSR, vec2<i32>(gid.xy), mix(vec4<f32>(max(hitColor * weight, vec3<f32>(0.0)), weight), hist, blend));
}

@compute @workgroup_size(8, 8, 1)
fn csCompose(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(composeOut);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let coord = vec2<i32>(gid.xy);
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * composeParams.invFullSize;
	let scene = textureLoad(composeScene, coord, 0);
	let centerDepth = textureSampleLevel(composeMotionDepth, composeSampler, uv, 0.0).z;
	let step = composeParams.invFullSize;
	let taps = array<vec2<f32>, 5>(
		vec2<f32>(0.0, 0.0),
		vec2<f32>(step.x, 0.0),
		vec2<f32>(-step.x, 0.0),
		vec2<f32>(0.0, step.y),
		vec2<f32>(0.0, -step.y)
	);
	var ssrSum = vec4<f32>(0.0);
	var weightSum = 0.0;
	for (var i: i32 = 0; i < 5; i = i + 1) {
		let sampleUv = uv + taps[i];
		let sampleDepth = textureSampleLevel(composeMotionDepth, composeSampler, sampleUv, 0.0).z;
		let depthWeight = exp(-abs(sampleDepth - centerDepth) * 48.0);
		let ssrTap = textureSampleLevel(composeSSR, composeSampler, sampleUv, 0.0);
		ssrSum += ssrTap * depthWeight;
		weightSum += depthWeight;
	}
	let ssr = select(textureSampleLevel(composeSSR, composeSampler, uv, 0.0), ssrSum / max(weightSum, 1e-4), weightSum > 0.0);
	textureStore(
		composeOut,
		coord,
		vec4<f32>(max(scene.rgb + ssr.rgb, vec3<f32>(0.0)), scene.a)
	);
}
`;

const FXAA_SHADER = /* wgsl */ `
struct Params {
	invSize: vec2<f32>,
	edgeThresholdMin: f32,
	edgeThresholdMultiplier: f32,
	subpixQuality: f32,
	_pad0: f32,
}

const FXAA_QUALITY = array<f32, 11>(
	1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0
);

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var outTex: texture_storage_2d<rgba16float, write>;

fn perceptualLuma(rgb: vec3<f32>) -> f32 {
	let linearLuma = max(
		dot(max(rgb, vec3<f32>(0.0)), vec3<f32>(0.2126, 0.7152, 0.0722)),
		0.0
	);
	return sqrt(linearLuma);
}

fn loadColorClamped(coord: vec2<i32>) -> vec4<f32> {
	let size = vec2<i32>(textureDimensions(srcTex));
	let clamped = clamp(coord, vec2<i32>(0, 0), size - vec2<i32>(1, 1));
	return textureLoad(srcTex, clamped, 0);
}

fn loadLuma(coord: vec2<i32>) -> f32 {
	return perceptualLuma(loadColorClamped(coord).rgb);
}

fn sampleColor(pixelPos: vec2<f32>) -> vec4<f32> {
	let uv = (pixelPos + vec2<f32>(0.5, 0.5)) * params.invSize;
	return textureSampleLevel(srcTex, linearSampler, uv, 0.0);
}

fn sampleLuma(pixelPos: vec2<f32>) -> f32 {
	return perceptualLuma(sampleColor(pixelPos).rgb);
}

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }

	let coord = vec2<i32>(gid.xy);
	let centerPos = vec2<f32>(coord);
	let centerColor = textureLoad(srcTex, coord, 0);
	let lumaCenter = perceptualLuma(centerColor.rgb);
	let lumaNorth = loadLuma(coord + vec2<i32>(0, -1));
	let lumaSouth = loadLuma(coord + vec2<i32>(0, 1));
	let lumaEast = loadLuma(coord + vec2<i32>(1, 0));
	let lumaWest = loadLuma(coord + vec2<i32>(-1, 0));

	let lumaMin = min(
		lumaCenter,
		min(min(lumaNorth, lumaSouth), min(lumaEast, lumaWest))
	);
	let lumaMax = max(
		lumaCenter,
		max(max(lumaNorth, lumaSouth), max(lumaEast, lumaWest))
	);
	let lumaRange = lumaMax - lumaMin;

	if (
		lumaRange <
		max(params.edgeThresholdMin, lumaMax * params.edgeThresholdMultiplier)
	) {
		textureStore(outTex, coord, centerColor);
		return;
	}

	let lumaNorthWest = loadLuma(coord + vec2<i32>(-1, -1));
	let lumaNorthEast = loadLuma(coord + vec2<i32>(1, -1));
	let lumaSouthWest = loadLuma(coord + vec2<i32>(-1, 1));
	let lumaSouthEast = loadLuma(coord + vec2<i32>(1, 1));

	var filteredLuma = 2.0 * (lumaNorth + lumaSouth + lumaEast + lumaWest);
	filteredLuma += lumaNorthEast + lumaNorthWest + lumaSouthEast + lumaSouthWest;
	filteredLuma /= 12.0;

	let subpixOffset1 = abs(filteredLuma - lumaCenter);
	let subpixOffset2 = clamp(subpixOffset1 / max(lumaRange, 1e-4), 0.0, 1.0);
	let subpixOffset3 =
		(-2.0 * subpixOffset2 + 3.0) * subpixOffset2 * subpixOffset2;
	let subpixOffset =
		subpixOffset3 * subpixOffset3 * params.subpixQuality;

	let edgeHorz =
		abs(-2.0 * lumaWest + lumaNorthWest + lumaSouthWest) +
		abs(-2.0 * lumaCenter + lumaNorth + lumaSouth) * 2.0 +
		abs(-2.0 * lumaEast + lumaNorthEast + lumaSouthEast);
	let edgeVert =
		abs(-2.0 * lumaNorth + lumaNorthWest + lumaNorthEast) +
		abs(-2.0 * lumaCenter + lumaWest + lumaEast) * 2.0 +
		abs(-2.0 * lumaSouth + lumaSouthWest + lumaSouthEast);
	let isHorz = edgeHorz >= edgeVert;

	let lumaA = select(lumaWest, lumaNorth, isHorz);
	let lumaB = select(lumaEast, lumaSouth, isHorz);
	let gradientA = abs(lumaA - lumaCenter);
	let gradientB = abs(lumaB - lumaCenter);
	let isASteeper = gradientA >= gradientB;
	let gradientScaled = 0.25 * max(gradientA, gradientB);
	let stepSign = select(1.0, -1.0, isASteeper);
	let edgeLuma = select(
		(lumaB + lumaCenter) * 0.5,
		(lumaA + lumaCenter) * 0.5,
		isASteeper
	);

	var posN = centerPos;
	if (isHorz) {
		posN.y += stepSign * 0.5;
	} else {
		posN.x += stepSign * 0.5;
	}
	var posP = posN;
	let axis = select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), isHorz);

	var doneN = false;
	var doneP = false;
	var lumaEndN = 0.0;
	var lumaEndP = 0.0;
	for (var i: i32 = 0; i < 11; i = i + 1) {
		if (!doneN) {
			lumaEndN = sampleLuma(posN);
			doneN = abs(lumaEndN - edgeLuma) >= gradientScaled;
		}
		if (!doneP) {
			lumaEndP = sampleLuma(posP);
			doneP = abs(lumaEndP - edgeLuma) >= gradientScaled;
		}
		if (doneN && doneP) {
			break;
		}
		if (!doneN) {
			posN -= axis * FXAA_QUALITY[i];
		}
		if (!doneP) {
			posP += axis * FXAA_QUALITY[i];
		}
	}

	let distN = select(centerPos.y - posN.y, centerPos.x - posN.x, isHorz);
	let distP = select(posP.y - centerPos.y, posP.x - centerPos.x, isHorz);
	let isNCloser = distN < distP;
	let distMin = min(distN, distP);
	let lumaEndMin = select(lumaEndP, lumaEndN, isNCloser);
	let isCenterBright = (lumaCenter - edgeLuma) >= 0.0;
	let isEndBright = (lumaEndMin - edgeLuma) >= 0.0;
	let reachedProperly = isEndBright != isCenterBright;

	var edgeOffset = -distMin / max(distN + distP, 1e-4) + 0.5;
	if (!reachedProperly) {
		edgeOffset = 0.0;
	}
	let pixelOffset = max(subpixOffset, edgeOffset);

	var finalPos = centerPos;
	if (isHorz) {
		finalPos.y += stepSign * pixelOffset;
	} else {
		finalPos.x += stepSign * pixelOffset;
	}

	textureStore(outTex, coord, sampleColor(finalPos));
}
`;

const COPY_SHADER = /* wgsl */ `
@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var dstTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(dstTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	textureStore(dstTex, vec2<i32>(gid.xy), textureLoad(srcTex, vec2<i32>(gid.xy), 0));
}
`;

export class WebGPUPostProcessRuntime {
	private _backend: WebGPUBackend;
	private _warn: (key: string, message: string) => void;
	private _sampler: ISampler | null = null;
	private _ssaoModule: IShaderModule | null = null;
	private _ssaoRawPipeline: IComputePipeline | null = null;
	private _ssaoBlurPipeline: IComputePipeline | null = null;
	private _ssaoCombinePipeline: IComputePipeline | null = null;
	private _ssaoParams: IRenderBuffer | null = null;
	private _taaModule: IShaderModule | null = null;
	private _taaPipeline: IComputePipeline | null = null;
	private _taaParams: IRenderBuffer | null = null;
	private _hizModule: IShaderModule | null = null;
	private _hizInitPipeline: IComputePipeline | null = null;
	private _hizReducePipeline: IComputePipeline | null = null;
	private _ssrModule: IShaderModule | null = null;
	private _ssrTracePipeline: IComputePipeline | null = null;
	private _ssrComposePipeline: IComputePipeline | null = null;
	private _ssrTraceParams: IRenderBuffer | null = null;
	private _ssrComposeParams: IRenderBuffer | null = null;
	private _fxaaModule: IShaderModule | null = null;
	private _fxaaPipeline: IComputePipeline | null = null;
	private _fxaaParams: IRenderBuffer | null = null;
	private _copyModule: IShaderModule | null = null;
	private _copyPipeline: IComputePipeline | null = null;
	private _hizViewCache = new WeakMap<object, any[]>();

	constructor(
		backend: WebGPUBackend,
		warn: (key: string, message: string) => void
	) {
		this._backend = backend;
		this._warn = warn;
	}
	public async executeSSAO(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets,
		frameContext: FrameContext
	): Promise<void> {
		await this._ensureSSAOResources();
		if (
			!this._sampler ||
			!this._ssaoRawPipeline ||
			!this._ssaoBlurPipeline ||
			!this._ssaoCombinePipeline ||
			!this._ssaoParams
		) {
			return;
		}
		const options = frameContext.features.ssaoOptions ?? {};
		const radius = finiteOr(options.radius, DEFAULT_SSAO.radius);
		const bias = finiteOr(options.bias, DEFAULT_SSAO.bias);
		const intensity = finiteOr(options.intensity, DEFAULT_SSAO.intensity);
		const blurRadius = finiteOr(options.blurRadius, DEFAULT_SSAO.blurRadius);
		const blurSharpness = finiteOr(
			options.blurSharpness,
			DEFAULT_SSAO.blurSharpness
		);
		const fullInvW = 1 / Math.max(targets.sceneColor.width, 1);
		const fullInvH = 1 / Math.max(targets.sceneColor.height, 1);
		const aoInvW = 1 / Math.max(targets.aoRaw.width, 1);
		const aoInvH = 1 / Math.max(targets.aoRaw.height, 1);
		this._backend.writeBuffer(
			this._ssaoParams,
			new Float32Array([
				fullInvW,
				fullInvH,
				aoInvW,
				aoInvH,
				radius,
				bias,
				intensity,
				blurRadius,
				blurSharpness,
				0,
				0,
				0,
			])
		);
		let binding = this._backend.createBindingGroup({
			pipeline: this._ssaoRawPipeline,
			layoutIndex: 0,
			entries: [
				{ binding: 0, resource: targets.gNormalRoughMetal },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: this._sampler },
				{ binding: 3, resource: this._ssaoParams },
				{ binding: 4, resource: targets.aoRaw },
			],
			label: "WebGPUSSAO_RawBinding",
		});
		encoder.beginComputePass({ label: "WebGPUSSAO_Raw" });
		encoder.setComputePipeline(this._ssaoRawPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(targets.aoRaw.width, WORKGROUP_SIZE),
			ceilDiv(targets.aoRaw.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		binding = this._backend.createBindingGroup({
			pipeline: this._ssaoBlurPipeline,
			layoutIndex: 0,
			entries: [
				{ binding: 0, resource: targets.aoRaw },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: this._sampler },
				{ binding: 3, resource: this._ssaoParams },
				{ binding: 4, resource: targets.aoBlur },
			],
			label: "WebGPUSSAO_BlurBinding",
		});
		encoder.beginComputePass({ label: "WebGPUSSAO_Blur" });
		encoder.setComputePipeline(this._ssaoBlurPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(targets.aoBlur.width, WORKGROUP_SIZE),
			ceilDiv(targets.aoBlur.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();

		const combineTarget =
			targets.sceneColor === targets.postPing
				? targets.postPong
				: targets.postPing;
		binding = this._backend.createBindingGroup({
			pipeline: this._ssaoCombinePipeline,
			layoutIndex: 0,
			entries: [
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.aoBlur },
				{ binding: 2, resource: this._sampler },
				{ binding: 3, resource: this._ssaoParams },
				{ binding: 4, resource: combineTarget },
			],
			label: "WebGPUSSAO_CombineBinding",
		});
		encoder.beginComputePass({ label: "WebGPUSSAO_Combine" });
		encoder.setComputePipeline(this._ssaoCombinePipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(combineTarget.width, WORKGROUP_SIZE),
			ceilDiv(combineTarget.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		targets.sceneColor = combineTarget;
	}

	public async executeTAA(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets,
		frameContext: FrameContext,
		historyValid: boolean
	): Promise<boolean> {
		await this._ensureTAAResources();
		if (!this._sampler || !this._taaPipeline || !this._taaParams) return false;
		const options = frameContext.features.taaOptions ?? {};
		const taaTarget =
			targets.sceneColor === targets.postPong
				? targets.postPing
				: targets.postPong;
		const invW = 1 / Math.max(taaTarget.width, 1);
		const invH = 1 / Math.max(taaTarget.height, 1);
		this._backend.writeBuffer(
			this._taaParams,
			new Float32Array([
				invW,
				invH,
				finiteOr(options.historyWeight, DEFAULT_TAA.historyWeight),
				finiteOr(
					options.disocclusionDepthThreshold,
					DEFAULT_TAA.disocclusionDepthThreshold
				),
				finiteOr(options.motionFactor, DEFAULT_TAA.motionFactor),
				finiteOr(options.varianceClampGamma, DEFAULT_TAA.varianceClampGamma),
				finiteOr(options.sharpen, DEFAULT_TAA.sharpen),
				historyValid ? 1 : 0,
			])
		);
		const binding = this._backend.createBindingGroup({
			pipeline: this._taaPipeline,
			layoutIndex: 0,
			entries: [
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.historyRead },
				{ binding: 2, resource: targets.gMotionDepth },
				{ binding: 3, resource: targets.motionHistoryRead },
				{ binding: 4, resource: this._sampler },
				{ binding: 5, resource: this._taaParams },
				{ binding: 6, resource: taaTarget },
				{ binding: 7, resource: targets.historyWrite },
			],
			label: "WebGPUTAA_Binding",
		});
		encoder.beginComputePass({ label: "WebGPUTAA" });
		encoder.setComputePipeline(this._taaPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(taaTarget.width, WORKGROUP_SIZE),
			ceilDiv(taaTarget.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		targets.sceneColor = taaTarget;
		return true;
	}

	public async executeSSR(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets,
		frameContext: FrameContext,
		historyValid: boolean
	): Promise<boolean> {
		if (frameContext.camera.type === CameraType.Orthographic) {
			this._warn(
				"webgpu-ssr-orthographic-disabled",
				"WebGPU SSR is disabled for OrthographicCamera in v1"
			);
			return false;
		}
		await this._ensureSSRResources();
		if (
			!this._sampler ||
			!this._hizInitPipeline ||
			!this._hizReducePipeline ||
			!this._ssrTracePipeline ||
			!this._ssrComposePipeline ||
			!this._ssrTraceParams ||
			!this._ssrComposeParams
		)
			return false;
		const options = frameContext.features.ssrOptions ?? {};
		const hiZMips = this._getHiZMipViews(targets.hiZ);
		if (hiZMips.length === 0) return false;
		let binding = this._backend.createBindingGroup({
			pipeline: this._hizInitPipeline,
			layoutIndex: 0,
			entries: [
				{ binding: 0, resource: targets.gMotionDepth },
				{ binding: 1, resource: hiZMips[0] },
			],
			label: "WebGPUSSR_HiZInitBinding",
		});
		encoder.beginComputePass({ label: "WebGPUSSR_HiZInit" });
		encoder.setComputePipeline(this._hizInitPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(targets.hiZ.width, WORKGROUP_SIZE),
			ceilDiv(targets.hiZ.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		let srcW = targets.hiZ.width;
		let srcH = targets.hiZ.height;
		for (let mip = 1; mip < hiZMips.length; mip++) {
			const dstW = Math.max(1, srcW >> 1);
			const dstH = Math.max(1, srcH >> 1);
			binding = this._backend.createBindingGroup({
				pipeline: this._hizReducePipeline,
				layoutIndex: 1,
				entries: [
					{ binding: 0, resource: hiZMips[mip - 1] },
					{ binding: 1, resource: hiZMips[mip] },
				],
				label: `WebGPUSSR_HiZReduceBinding_${mip}`,
			});
			encoder.beginComputePass({ label: `WebGPUSSR_HiZReduce_${mip}` });
			encoder.setComputePipeline(this._hizReducePipeline);
			encoder.setBindingGroup(0, binding);
			encoder.dispatchWorkgroups(
				ceilDiv(dstW, WORKGROUP_SIZE),
				ceilDiv(dstH, WORKGROUP_SIZE),
				1
			);
			encoder.endComputePass();
			srcW = dstW;
			srcH = dstH;
		}
		this._backend.writeBuffer(
			this._ssrTraceParams,
			new Float32Array([
				1 / Math.max(targets.ssrRaw.width, 1),
				1 / Math.max(targets.ssrRaw.height, 1),
				finiteOr(options.maxDistance, DEFAULT_SSR.maxDistance),
				finiteOr(options.thickness, DEFAULT_SSR.thickness),
				finiteOr(options.stride, DEFAULT_SSR.stride),
				finiteOr(options.intensity, DEFAULT_SSR.intensity),
				finiteOr(options.maxRoughness, DEFAULT_SSR.maxRoughness),
				finiteOr(options.edgeFade, DEFAULT_SSR.edgeFade),
				finiteOr(options.maxSteps, DEFAULT_SSR.maxSteps),
				finiteOr(options.binarySearchSteps, DEFAULT_SSR.binarySearchSteps),
				hiZMips.length - 1,
				finiteOr(options.historyWeight, DEFAULT_SSR.historyWeight),
				historyValid ? 1 : 0,
				0.02,
				0,
				0,
			])
		);
		binding = this._backend.createBindingGroup({
			pipeline: this._ssrTracePipeline,
			layoutIndex: 0,
			entries: [
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.gNormalRoughMetal },
				{ binding: 2, resource: targets.gMotionDepth },
				{ binding: 3, resource: targets.hiZ },
				{ binding: 4, resource: targets.ssrHistoryRead },
				{ binding: 5, resource: targets.motionHistoryRead },
				{ binding: 6, resource: this._sampler },
				{ binding: 7, resource: this._ssrTraceParams },
				{ binding: 8, resource: targets.ssrRaw },
			],
			label: "WebGPUSSR_TraceBinding",
		});
		encoder.beginComputePass({ label: "WebGPUSSR_TraceTemporal" });
		encoder.setComputePipeline(this._ssrTracePipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(targets.ssrRaw.width, WORKGROUP_SIZE),
			ceilDiv(targets.ssrRaw.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		await this._copyTexture(encoder, targets.ssrRaw, targets.ssrHistoryWrite);
		const composeTarget =
			targets.sceneColor === targets.postPing
				? targets.postPong
				: targets.postPing;
		this._backend.writeBuffer(
			this._ssrComposeParams,
			new Float32Array([
				1 / Math.max(composeTarget.width, 1),
				1 / Math.max(composeTarget.height, 1),
				0,
				0,
			])
		);
		binding = this._backend.createBindingGroup({
			pipeline: this._ssrComposePipeline,
			layoutIndex: 1,
			entries: [
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.ssrRaw },
				{ binding: 2, resource: targets.gMotionDepth },
				{ binding: 3, resource: this._sampler },
				{ binding: 4, resource: this._ssrComposeParams },
				{ binding: 5, resource: composeTarget },
			],
			label: "WebGPUSSR_ComposeBinding",
		});
		encoder.beginComputePass({ label: "WebGPUSSR_Compose" });
		encoder.setComputePipeline(this._ssrComposePipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(composeTarget.width, WORKGROUP_SIZE),
			ceilDiv(composeTarget.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		targets.sceneColor = composeTarget;
		return true;
	}

	public async executeFXAA(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets
	): Promise<void> {
		await this._ensureFXAAResources();
		if (!this._sampler || !this._fxaaPipeline || !this._fxaaParams) return;
		const target =
			targets.sceneColor === targets.postPong
				? targets.postPing
				: targets.postPong;
		this._backend.writeBuffer(
			this._fxaaParams,
			new Float32Array([
				1 / Math.max(target.width, 1),
				1 / Math.max(target.height, 1),
				DEFAULT_FXAA.edgeThresholdMin,
				DEFAULT_FXAA.edgeThresholdMultiplier,
				DEFAULT_FXAA.subpixQuality,
				0,
			])
		);
		const binding = this._backend.createBindingGroup({
			pipeline: this._fxaaPipeline,
			layoutIndex: 0,
			entries: [
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: this._sampler },
				{ binding: 2, resource: this._fxaaParams },
				{ binding: 3, resource: target },
			],
			label: "WebGPUFXAA_Binding",
		});
		encoder.beginComputePass({ label: "WebGPUFXAA" });
		encoder.setComputePipeline(this._fxaaPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(target.width, WORKGROUP_SIZE),
			ceilDiv(target.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		targets.sceneColor = target;
	}

	private async _copyTexture(
		encoder: ICommandEncoder,
		src: IRenderTexture,
		dst: IRenderTexture
	): Promise<void> {
		if (src === dst) return;
		await this._ensureCopyResources();
		if (!this._copyPipeline) return;
		const binding = this._backend.createBindingGroup({
			pipeline: this._copyPipeline,
			layoutIndex: 0,
			entries: [
				{ binding: 0, resource: src },
				{ binding: 1, resource: dst },
			],
			label: "WebGPUPost_CopyBinding",
		});
		encoder.beginComputePass({ label: "WebGPUPost_Copy" });
		encoder.setComputePipeline(this._copyPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(dst.width, WORKGROUP_SIZE),
			ceilDiv(dst.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
	}

	private async _ensureCommonResources(): Promise<void> {
		if (this._sampler) return;
		this._sampler = this._backend.createSampler({
			label: "WebGPUPost_LinearSampler",
			magFilter: FilterMode.Linear,
			minFilter: FilterMode.Linear,
			mipmapFilter: FilterMode.Linear,
			addressModeU: AddressMode.ClampToEdge,
			addressModeV: AddressMode.ClampToEdge,
		});
	}

	private async _ensureSSAOResources(): Promise<void> {
		await this._ensureCommonResources();
		if (!this._ssaoModule)
			this._ssaoModule = await this._backend.createShaderModule({
				label: "WebGPUSSAOShader",
				code: SSAO_SHADER,
			});
		if (!this._ssaoRawPipeline)
			this._ssaoRawPipeline = this._backend.createComputePipeline({
				label: "WebGPUSSAORawPipeline",
				compute: { module: this._ssaoModule, entryPoint: "csRaw" },
			});
		if (!this._ssaoBlurPipeline)
			this._ssaoBlurPipeline = this._backend.createComputePipeline({
				label: "WebGPUSSAOBlurPipeline",
				compute: { module: this._ssaoModule, entryPoint: "csBlur" },
			});
		if (!this._ssaoCombinePipeline)
			this._ssaoCombinePipeline = this._backend.createComputePipeline({
				label: "WebGPUSSAOCombinePipeline",
				compute: { module: this._ssaoModule, entryPoint: "csCombine" },
			});
		if (!this._ssaoParams)
			this._ssaoParams = this._backend.createBuffer({
				label: "WebGPUSSAOParams",
				size: 12 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
	}

	private async _ensureTAAResources(): Promise<void> {
		await this._ensureCommonResources();
		if (!this._taaModule)
			this._taaModule = await this._backend.createShaderModule({
				label: "WebGPUTAAShader",
				code: TAA_SHADER,
			});
		if (!this._taaPipeline)
			this._taaPipeline = this._backend.createComputePipeline({
				label: "WebGPUTAAPipeline",
				compute: { module: this._taaModule, entryPoint: "csMain" },
			});
		if (!this._taaParams)
			this._taaParams = this._backend.createBuffer({
				label: "WebGPUTAAParams",
				size: 8 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
	}

	private async _ensureSSRResources(): Promise<void> {
		await this._ensureCommonResources();
		if (!this._hizModule)
			this._hizModule = await this._backend.createShaderModule({
				label: "WebGPUHiZShader",
				code: HIZ_SHADER,
			});
		if (!this._ssrModule)
			this._ssrModule = await this._backend.createShaderModule({
				label: "WebGPUSSRShader",
				code: SSR_SHADER,
			});
		if (!this._hizInitPipeline)
			this._hizInitPipeline = this._backend.createComputePipeline({
				label: "WebGPUHiZInitPipeline",
				compute: { module: this._hizModule, entryPoint: "csInit" },
			});
		if (!this._hizReducePipeline)
			this._hizReducePipeline = this._backend.createComputePipeline({
				label: "WebGPUHiZReducePipeline",
				compute: { module: this._hizModule, entryPoint: "csReduce" },
			});
		if (!this._ssrTracePipeline)
			this._ssrTracePipeline = this._backend.createComputePipeline({
				label: "WebGPUSSRTracePipeline",
				compute: { module: this._ssrModule, entryPoint: "csTrace" },
			});
		if (!this._ssrComposePipeline)
			this._ssrComposePipeline = this._backend.createComputePipeline({
				label: "WebGPUSSRComposePipeline",
				compute: { module: this._ssrModule, entryPoint: "csCompose" },
			});
		if (!this._ssrTraceParams)
			this._ssrTraceParams = this._backend.createBuffer({
				label: "WebGPUSSRTraceParams",
				size: 16 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		if (!this._ssrComposeParams)
			this._ssrComposeParams = this._backend.createBuffer({
				label: "WebGPUSSRComposeParams",
				size: 4 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
	}

	private async _ensureFXAAResources(): Promise<void> {
		await this._ensureCommonResources();
		if (!this._fxaaModule)
			this._fxaaModule = await this._backend.createShaderModule({
				label: "WebGPUFXAAShader",
				code: FXAA_SHADER,
			});
		if (!this._fxaaPipeline)
			this._fxaaPipeline = this._backend.createComputePipeline({
				label: "WebGPUFXAAPipeline",
				compute: { module: this._fxaaModule, entryPoint: "csMain" },
			});
		if (!this._fxaaParams)
			this._fxaaParams = this._backend.createBuffer({
				label: "WebGPUFXAAParams",
				size: 6 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
	}

	private async _ensureCopyResources(): Promise<void> {
		if (!this._copyModule)
			this._copyModule = await this._backend.createShaderModule({
				label: "WebGPUCopyShader",
				code: COPY_SHADER,
			});
		if (!this._copyPipeline)
			this._copyPipeline = this._backend.createComputePipeline({
				label: "WebGPUCopyPipeline",
				compute: { module: this._copyModule, entryPoint: "csMain" },
			});
	}

	private _getHiZMipViews(texture: IRenderTexture): any[] {
		const cached = this._hizViewCache.get(texture as object);
		if (cached) return cached;
		const gpuTexture =
			(texture as InternalTexture)._gpuTexture ??
			(texture as InternalTexture)._gpuResource;
		if (!gpuTexture?.createView) return [];
		const mipCount =
			Math.floor(Math.log2(Math.max(texture.width, texture.height))) + 1;
		const views: any[] = [];
		for (let i = 0; i < mipCount; i++) {
			views.push(gpuTexture.createView({ baseMipLevel: i, mipLevelCount: 1 }));
		}
		this._hizViewCache.set(texture as object, views);
		return views;
	}
}

function finiteOr(value: unknown, fallback: number): number {
	return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function ceilDiv(value: number, divisor: number): number {
	return Math.max(1, Math.ceil(value / Math.max(divisor, 1)));
}
