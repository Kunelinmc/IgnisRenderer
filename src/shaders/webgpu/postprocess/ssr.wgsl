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

@group(0) @binding(0) var composeScene: texture_2d<f32>;
@group(0) @binding(1) var composeSSR: texture_2d<f32>;
@group(0) @binding(2) var composeMotionDepth: texture_2d<f32>;
@group(0) @binding(3) var composeSampler: sampler;
@group(0) @binding(4) var<uniform> composeParams: ComposeParams;
@group(0) @binding(5) var composeOut: texture_storage_2d<rgba16float, write>;

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
