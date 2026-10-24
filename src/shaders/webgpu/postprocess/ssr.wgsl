#import <ignis/webgpu/constants>
#import <ignis/webgpu/frame-camera>

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
	frameIndex: f32,
	raysPerPixel: f32,
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
@group(0) @binding(9) var ssrMomentHistory: texture_2d<f32>;
@group(0) @binding(10) var outSSRMoments: texture_storage_2d<rgba16float, write>;

@group(1) @binding(0) var<uniform> frame: FrameCameraUniforms;

struct ComposeParams {
	invFullSize: vec2<f32>,
}

@group(0) @binding(0) var composeScene: texture_2d<f32>;
@group(0) @binding(1) var composeSSR: texture_2d<f32>;
@group(0) @binding(2) var composeMotionDepth: texture_2d<f32>;
@group(0) @binding(3) var composeSampler: sampler;
@group(0) @binding(4) var<uniform> composeParams: ComposeParams;
@group(0) @binding(5) var composeOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var composePlanarReflectionMask: texture_2d<f32>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Blue-noise hash for stochastic jitter
// Interleaved gradient noise — compact, high-frequency, low-discrepancy
// Roberts R2 quasi-random sequence for multi-sample offsets
// ---------------------------------------------------------------------------

fn interleavedGradientNoise(pixel: vec2<f32>, frameIndex: f32) -> f32 {
	let frame = frameIndex * 5.588238;
	let uv = pixel + frame;
	return fract(52.9829189 * fract(0.06711056 * uv.x + 0.00583715 * uv.y));
}

fn r2Sequence(index: f32) -> vec2<f32> {
	// Generalized golden ratio for 2D
	let a1 = 1.0 / 1.3247179572;
	let a2 = a1 * a1;
	return fract(vec2<f32>(a1, a2) * (index + 1.0));
}

// ---------------------------------------------------------------------------
// GGX importance sampling — generates microfacet half-vector
// ---------------------------------------------------------------------------

fn importanceSampleGGX(xi: vec2<f32>, roughness: f32, N: vec3<f32>) -> vec3<f32> {
	let a = roughness * roughness;
	let a2 = a * a;

	let phi = TWO_PI * xi.x;
	let cosTheta = sqrt(max((1.0 - xi.y) / max(1.0 + (a2 - 1.0) * xi.y, 1e-6), 0.0));
	let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));

	// Tangent-space half vector
	let H_t = vec3<f32>(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

	// Build orthonormal basis around N
	var up: vec3<f32>;
	if (abs(N.y) < 0.999) {
		up = vec3<f32>(0.0, 1.0, 0.0);
	} else {
		up = vec3<f32>(1.0, 0.0, 0.0);
	}
	let T = normalize(cross(up, N));
	let B = cross(N, T);

	return normalize(T * H_t.x + B * H_t.y + N * H_t.z);
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

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
	var vn = vec3<f32>(oct.x, oct.y, 1.0 - abs(oct.x) - abs(oct.y));
	if (vn.z < 0.0) {
		vn = vec3<f32>(octahedralWrap(vn.xy), vn.z);
	}
	vn = normalize(vn);
	return normalize(cameraViewToWorldDirection(frame.inverseView, vn));
}

fn getPosition(uv: vec2<f32>, depth: f32) -> vec3<f32> {
	return cameraReconstructWorldPosition(
		frame.inverseView, frame.inverseProjection, frame.jitterNdc, uv, depth
	);
}

fn worldToUv(worldPos: vec3<f32>) -> vec2<f32> {
	if (cameraLinearDepth(frame.inverseView, worldPos) <= 1e-4) {
		return vec2<f32>(-1.0);
	}
	return cameraWorldToRasterUv(frame.viewProjection, frame.jitterNdc, worldPos);
}

fn worldToLinearDepth(worldPos: vec3<f32>) -> f32 {
	return cameraLinearDepth(frame.inverseView, worldPos);
}

// ---------------------------------------------------------------------------
// HiZ cell traversal helpers
// ---------------------------------------------------------------------------

fn getHiZCellSize(mip: i32) -> vec2<f32> {
	let baseSize = vec2<f32>(textureDimensions(hiZ));
	return baseSize / pow(2.0, f32(mip));
}

fn isInsideScreen(uv: vec2<f32>) -> bool {
	return uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
}

// ---------------------------------------------------------------------------
// HiZ ray march — steps through the hierarchical depth buffer
// Uses min-depth (.x) for conservative front and max-depth (.y) for back
// ---------------------------------------------------------------------------

struct HiZTraceResult {
	hitUv: vec2<f32>,
	hit: bool,
	t: f32,
}

fn ssrDepthCrossing(previousDelta: f32, currentDelta: f32) -> bool {
	return previousDelta < 0.0 && currentDelta >= 0.0;
}

fn ssrAcceptRefinedHit(rayDepth: f32, sceneDepth: f32, thickness: f32) -> bool {
	return sceneDepth > 0.0 && abs(rayDepth - sceneDepth) <= max(thickness, 0.01);
}

fn ssrBlendTemporal(
	current: vec4<f32>, history: vec4<f32>, historyWeight: f32, historyValid: bool
) -> vec4<f32> {
	let weight = select(0.0, clamp(historyWeight, 0.0, 0.98), historyValid);
	return mix(current, history, weight);
}

fn ssrBlendMoments(
	current: vec2<f32>, history: vec4<f32>, historyWeight: f32, historyValid: bool
) -> vec4<f32> {
	let weight = select(0.0, clamp(historyWeight, 0.0, 0.98), historyValid);
	let resolved = mix(current, history.xy, weight);
	let effectiveFrames = 1.0 / max((1.0 - weight) * (1.0 - weight) +
		weight * weight / max(history.z, 1.0), 1.0 / 128.0);
	return vec4<f32>(resolved, min(effectiveFrames, 128.0), 1.0);
}

fn loadHiZDepth(uv: vec2<f32>, mip: i32) -> vec2<f32> {
	let size = textureDimensions(hiZ, mip);
	let coord = clamp(
		vec2<i32>(uv * vec2<f32>(size)), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1)
	);
	// Depth bounds must remain conservative; filtering across silhouettes
	// invents surfaces between unrelated foreground and background depths.
	return textureLoad(hiZ, coord, mip).xy;
}

fn traceHiZ(
	worldPos: vec3<f32>,
	reflectionDir: vec3<f32>,
	startT: f32,
	jitter: f32,
) -> HiZTraceResult {
	var result: HiZTraceResult;
	result.hit = false;
	result.hitUv = vec2<f32>(0.0);
	result.t = 0.0;

	let maxSteps = i32(clamp(traceParams.maxSteps, 1.0, 256.0));
	let maxMip = i32(clamp(traceParams.maxMip, 0.0, 12.0));
	let thickness = max(traceParams.thickness, 0.01);
	let stride = max(traceParams.stride, 0.1);

	var t = startT + jitter * stride;
	var mip: i32 = 0;
	var missT = 0.0;
	var previousDelta = 0.0;
	var hasFrontSample = false;

	for (var step: i32 = 0; step < 256; step = step + 1) {
		if (step >= maxSteps) { break; }
		if (t > traceParams.maxDistance) { break; }

		let samplePos = worldPos + reflectionDir * t;
		let suv = worldToUv(samplePos);

		// Boundary check — abort if outside screen
		if (!isInsideScreen(suv)) { break; }

		let rayDepth = worldToLinearDepth(samplePos);
		if (rayDepth <= 0.0) { break; }

		// Clamp mip level to stay in valid range
		let safeMip = clamp(mip, 0, maxMip);
		let hizSample = loadHiZDepth(suv, safeMip);
		let minDepth = hizSample.x;
		let maxDepth = max(hizSample.y, hizSample.x); // fallback if .y wasn't written
		let depthDelta = rayDepth - minDepth;

		// No scene geometry at this location?
		if (minDepth <= 0.0) {
			hasFrontSample = false;
			t += stride * pow(2.0, f32(safeMip));
			continue;
		}

		// A step can cross the entire thickness interval. Keep its front/back
		// bracket and refine instead of advancing farther behind the surface.
		if (rayDepth > maxDepth + thickness) {
			if (hasFrontSample && ssrDepthCrossing(previousDelta, depthDelta)) {
				if (mip > 0) {
					mip = mip - 1;
					continue;
				}
				result.hit = true;
				result.t = t;
				break;
			}
			mip = min(mip + 1, maxMip);
			t += stride * pow(2.0, f32(safeMip));
			continue;
		}

		if (depthDelta < 0.0) {
			missT = t;
			previousDelta = depthDelta;
			hasFrontSample = true;
			t += stride * pow(2.0, f32(safeMip));
			continue;
		}

		// Ray is within the depth range [minDepth - thickness, maxDepth + thickness]
		if (mip > 0) {
			// Refine at finer mip level
			mip = mip - 1;
			continue;
		}

		if (ssrDepthCrossing(previousDelta, depthDelta) ||
			ssrAcceptRefinedHit(rayDepth, minDepth, thickness)) {
			result.hit = true;
			result.t = t;
			break;
		}

		t += stride;
	}

	// Binary refinement if we got a hit
	if (result.hit) {
		var refineMin = select(result.t, missT, hasFrontSample);
		var refineMax = result.t;
		let refineSteps = i32(clamp(traceParams.binarySearchSteps, 0.0, 16.0));

		for (var j: i32 = 0; j < 16; j = j + 1) {
			if (j >= refineSteps) { break; }

			let midT = (refineMin + refineMax) * 0.5;
			let refinePos = worldPos + reflectionDir * midT;
			let refineUv = worldToUv(refinePos);

			if (!isInsideScreen(refineUv)) {
				refineMax = midT;
				continue;
			}

			let rayDepth = worldToLinearDepth(refinePos);
			let sceneDepth = loadHiZDepth(refineUv, 0).x;

			if (sceneDepth > 0.0 && rayDepth >= sceneDepth) {
				refineMax = midT;
			} else {
				refineMin = midT;
			}
		}
		result.t = refineMax;
		let refinedPosition = worldPos + reflectionDir * refineMax;
		let refinedUv = worldToUv(refinedPosition);
		result.hitUv = refinedUv;
		result.hit = isInsideScreen(refinedUv) && ssrAcceptRefinedHit(
			worldToLinearDepth(refinedPosition), loadHiZDepth(refinedUv, 0).x, thickness
		);
	}

	return result;
}

// ---------------------------------------------------------------------------
// Fresnel-Schlick for reflectance weighting
// ---------------------------------------------------------------------------

fn fresnelSchlickSSR(cosTheta: f32, f0: vec3<f32>) -> vec3<f32> {
	return f0 + (vec3<f32>(1.0) - f0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

// ---------------------------------------------------------------------------
// Compute entry: stochastic SSR trace + temporal accumulation
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn csTrace(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outSSR);
	if (gid.x >= size.x || gid.y >= size.y) { return; }

	let pixel = vec2<f32>(gid.xy);
	let uv = (pixel + vec2<f32>(0.5)) * traceParams.invHalfSize;

	// Read G-buffer
	let g = textureSampleLevel(gNormalRoughMetal, linearSampler, uv, 0.0);
	let motionDepth = textureSampleLevel(gMotionDepth, linearSampler, uv, 0.0);
	let depth = motionDepth.z;
	let roughness = clamp(g.z, 0.0, 1.0);
	let metalness = clamp(g.w, 0.0, 1.0);

	// Early-out: no geometry or too rough
	if (depth <= 0.0 || roughness > traceParams.maxRoughness || traceParams.intensity <= 0.0) {
		textureStore(outSSR, vec2<i32>(gid.xy), vec4<f32>(0.0));
		textureStore(outSSRMoments, vec2<i32>(gid.xy), vec4<f32>(0.0));
		return;
	}

	let worldPos = getPosition(uv, depth);
	let worldNormal = decodeNormal(g.xy);
	let worldViewDir = normalize(worldPos - cameraPosition(frame.inverseView));
	let nDotV = max(dot(worldNormal, -worldViewDir), 0.001);

	// Physically-based F0 — dielectrics ≈ 0.04, metals use their albedo
	// (We don't have albedo here so approximate with vec3(1) for metals)
	let f0 = mix(vec3<f32>(0.04), vec3<f32>(1.0), metalness);

	// -----------------------------------------------------------------------
	// Blue-noise jitter from interleaved gradient noise
	// -----------------------------------------------------------------------
	let noise = interleavedGradientNoise(pixel, traceParams.frameIndex);

	// -----------------------------------------------------------------------
	// Stochastic reflection direction
	// For smooth surfaces (roughness < 0.05), use mirror reflection.
	// For rough surfaces, importance-sample a GGX microfacet normal to
	// approximate roughness cone tracing.
	// -----------------------------------------------------------------------
	let sampleRoughness = select(roughness, 0.0, roughness < 0.05);
	let sampleCount = select(i32(clamp(traceParams.raysPerPixel, 1.0, 8.0)), 1,
		roughness < 0.05);
	let startT = max(traceParams.stride, 0.25);
	var current = vec4<f32>(0.0);
	var sampleLogSum = 0.0;
	var sampleLogSquareSum = 0.0;
	for (var sampleIndex = 0; sampleIndex < 8; sampleIndex++) {
		if (sampleIndex >= sampleCount) { break; }
		var reflectionDir = reflect(worldViewDir, worldNormal);
		let sampleNoise = fract(noise + f32(sampleIndex) * GOLDEN_RATIO_CONJUGATE);
		if (roughness >= 0.05) {
			let r2 = r2Sequence(traceParams.frameIndex * f32(sampleCount) +
				f32(sampleIndex) + noise * 64.0);
			let xi = vec2<f32>(fract(r2.x + sampleNoise),
				fract(r2.y + sampleNoise * 0.7071));
			reflectionDir = reflect(worldViewDir,
				importanceSampleGGX(xi, roughness, worldNormal));
		}
		if (dot(reflectionDir, worldNormal) < 0.001) { continue; }
		let traceResult = traceHiZ(worldPos, reflectionDir, startT, sampleNoise);
		if (!traceResult.hit) { continue; }
		// -----------------------------------------------------------------------
		// Sample the radiance of each hit before temporal/spatial reconstruction.
		// -----------------------------------------------------------------------
		let hitUv = traceResult.hitUv;
		// Roughness is integrated by ray directions and multiscale reconstruction.
		// Scene color has one mip; requesting a nonexistent LOD cannot prefilter it.
		let hitColor = clamp(textureSampleLevel(sceneColor, linearSampler, hitUv, 0.0).rgb,
			vec3<f32>(0.0), vec3<f32>(65504.0));

		// -----------------------------------------------------------------------
		// Screen-edge fade
		// -----------------------------------------------------------------------
		let edgeDistance = min(
			min(hitUv.x, 1.0 - hitUv.x),
			min(hitUv.y, 1.0 - hitUv.y)
		);
		let edge = clamp(edgeDistance / max(traceParams.edgeFade, 1e-4), 0.0, 1.0);

		// -----------------------------------------------------------------------
		// Distance fade — reflections far from origin are less reliable
		// -----------------------------------------------------------------------
		let distFade = 1.0 - clamp(traceResult.t / max(traceParams.maxDistance, 1.0), 0.0, 1.0);

		// -----------------------------------------------------------------------
		// Roughness attenuation — rougher surfaces get dimmer SSR
		// (IBL / fallback should cover rough reflections better)
		// -----------------------------------------------------------------------
		let roughFade = 1.0 - clamp(sampleRoughness / max(traceParams.maxRoughness, 0.01), 0.0, 1.0);

		// -----------------------------------------------------------------------
		// Fresnel-weighted reflectance
		// -----------------------------------------------------------------------
		let fresnel = fresnelSchlickSSR(nDotV, f0);
		let fresnelWeight = max(max(fresnel.r, fresnel.g), fresnel.b);

		// -----------------------------------------------------------------------
		// Final weight
		// -----------------------------------------------------------------------
		let weight = clamp(
			traceParams.intensity * edge * distFade * roughFade * fresnelWeight,
			0.0,
			1.0
		);
		let radiance = max(hitColor * weight, vec3<f32>(0.0));
		current += vec4<f32>(radiance, weight);
		let sampleLog = log(1.0 + dot(radiance, vec3<f32>(0.2126, 0.7152, 0.0722)));
		sampleLogSum += sampleLog;
		sampleLogSquareSum += sampleLog * sampleLog;
	}
	// Include misses in the denominator so sparse bright hits do not gain energy.
	current /= f32(sampleCount);

	// -----------------------------------------------------------------------
	// Temporal reprojection
	// -----------------------------------------------------------------------
	let motion = motionDepth.xy;
	let prevUv = uv - vec2<f32>(motion.x * 0.5, -motion.y * 0.5);
	let hist = textureSampleLevel(ssrHistory, linearSampler, prevUv, 0.0);
	let momentHistory = textureSampleLevel(ssrMomentHistory, linearSampler, prevUv, 0.0);

	let currDepth = depth;
	let prevDepth = textureSampleLevel(motionHistory, linearSampler, prevUv, 0.0).z;
	let relDepth = abs(currDepth - prevDepth) / max(max(currDepth, prevDepth), 1e-4);

	let historyOk = traceParams.historyValid > 0.5
		&& momentHistory.z >= 1.0 && momentHistory.w > 0.5
		&& prevUv.x >= 0.0 && prevUv.x <= 1.0
		&& prevUv.y >= 0.0 && prevUv.y <= 1.0
		&& relDepth <= traceParams.depthThreshold;

	// For stochastic SSR, blend more aggressively with history to converge
	// Rough surfaces need more temporal blending, smooth surfaces less
	let baseBlend = clamp(traceParams.historyWeight, 0.0, 0.98);
	let roughnessBlend = mix(baseBlend, min(baseBlend + 0.05, 0.98), sampleRoughness);
	textureStore(outSSR, vec2<i32>(gid.xy),
		ssrBlendTemporal(current, hist, roughnessBlend, historyOk));
	// Log moments remain finite in rgba16float for bright HDR emitters. Include
	// within-frame uncertainty so the first few frames can also be reconstructed.
	let frameLog = log(1.0 + dot(current.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)));
	let sampleLogMean = sampleLogSum / f32(sampleCount);
	let frameVariance = max(sampleLogSquareSum / f32(sampleCount) -
		sampleLogMean * sampleLogMean, 0.0) / max(f32(sampleCount - 1), 1.0);
	let currentMoments = vec2<f32>(frameLog, frameLog * frameLog + frameVariance);
	textureStore(outSSRMoments, vec2<i32>(gid.xy),
		ssrBlendMoments(currentMoments, momentHistory, roughnessBlend, historyOk));
}

// ---------------------------------------------------------------------------
// Compose pass — bilateral upscale + blend SSR onto scene
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn csCompose(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(composeOut);
	if (gid.x >= size.x || gid.y >= size.y) { return; }

	let coord = vec2<i32>(gid.xy);
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * composeParams.invFullSize;
	let scene = textureLoad(composeScene, coord, 0);
	if (textureLoad(composePlanarReflectionMask, coord, 0).r > 0.5) {
		textureStore(composeOut, coord, scene);
		return;
	}
	let centerDepth = textureSampleLevel(composeMotionDepth, composeSampler, uv, 0.0).z;
	let step = composeParams.invFullSize;

	// 5-tap depth-aware bilateral filter for upscaling
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
		let sampleUv = clamp(
			uv + taps[i],
			vec2<f32>(0.0),
			vec2<f32>(1.0) - step * 0.5
		);
		let sampleDepth = textureSampleLevel(composeMotionDepth, composeSampler, sampleUv, 0.0).z;
		let depthWeight = exp(-abs(sampleDepth - centerDepth) * 48.0);
		let ssrTap = textureSampleLevel(composeSSR, composeSampler, sampleUv, 0.0);
		ssrSum += ssrTap * depthWeight;
		weightSum += depthWeight;
	}

	let ssr = select(
		textureSampleLevel(composeSSR, composeSampler, uv, 0.0),
		ssrSum / max(weightSum, 1e-4),
		weightSum > 0.0
	);

	textureStore(
		composeOut,
		coord,
		vec4<f32>(max(scene.rgb + ssr.rgb, vec3<f32>(0.0)), scene.a)
	);
}
