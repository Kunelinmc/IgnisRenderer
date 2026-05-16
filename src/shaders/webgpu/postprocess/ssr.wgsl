#import <ignis/webgpu/constants>
struct FrameUniforms {
	viewProjection: mat4x4<f32>,
	prevViewProjection: mat4x4<f32>,
	cameraPosition: vec4<f32>,
	environmentBasisRight: vec4<f32>,
	environmentBasisUp: vec4<f32>,
	environmentBasisBackward: vec4<f32>,
	ambientColor: vec4<f32>,
	lightCounts: vec4<f32>,
	options: vec4<f32>,
	environmentOptionsA: vec4<f32>,
	environmentOptionsB: vec4<f32>,
	taaJitterCurrentPrev: vec4<f32>,
	directionalLights: array<DirectionalLightData, __WEBGPU_MAX_DIRECTIONAL_LIGHTS__>,
	pointLights: array<PointLightData, __WEBGPU_MAX_POINT_LIGHTS__>,
	spotLights: array<SpotLightData, __WEBGPU_MAX_SPOT_LIGHTS__>,
	directionalShadows: array<ShadowData, __WEBGPU_MAX_DIRECTIONAL_LIGHTS__>,
	spotShadows: array<ShadowData, __WEBGPU_MAX_SPOT_LIGHTS__>,
	shAmbientCoeffs: array<vec4<f32>, __WEBGPU_SH_COEFFICIENT_COUNT__>,
}

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
	_pad0: f32,
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

@group(1) @binding(0) var<uniform> frame: FrameUniforms;

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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PI: f32 = 3.14159265359;

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

	let phi = 2.0 * PI * xi.x;
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

struct CameraBasis {
	right: vec3<f32>,
	up: vec3<f32>,
	backward: vec3<f32>,
	tanHalfFov: f32,
	aspect: f32,
	orthographic: f32,
}

fn getCameraBasis() -> CameraBasis {
	return CameraBasis(
		frame.environmentBasisRight.xyz,
		frame.environmentBasisUp.xyz,
		frame.environmentBasisBackward.xyz,
		frame.environmentBasisRight.w,
		frame.environmentBasisUp.w,
		frame.environmentBasisBackward.w
	);
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
	var vn = vec3<f32>(oct.x, oct.y, 1.0 - abs(oct.x) - abs(oct.y));
	if (vn.z < 0.0) {
		vn = vec3<f32>(octahedralWrap(vn.xy), vn.z);
	}
	vn = normalize(vn);
	let basis = getCameraBasis();
	return normalize(
		basis.right * vn.x + basis.up * vn.y + basis.backward * vn.z
	);
}

fn getPosition(uv: vec2<f32>, depth: f32) -> vec3<f32> {
	let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
	let basis = getCameraBasis();
	if (basis.orthographic > 0.5) {
		return frame.cameraPosition.xyz
			+ basis.right * ndc.x * basis.tanHalfFov
			+ basis.up * ndc.y * basis.aspect
			- basis.backward * depth;
	}
	let cx = ndc.x * basis.aspect * basis.tanHalfFov * depth;
	let cy = ndc.y * basis.tanHalfFov * depth;
	return frame.cameraPosition.xyz
		+ basis.right * cx
		+ basis.up * cy
		- basis.backward * depth;
}

fn worldToUv(worldPos: vec3<f32>) -> vec2<f32> {
	let rel = worldPos - frame.cameraPosition.xyz;
	let basis = getCameraBasis();
	let depth = dot(rel, -basis.backward);
	if (depth <= 1e-4) { return vec2<f32>(-1.0); }

	if (basis.orthographic > 0.5) {
		let cx = dot(rel, basis.right) / max(basis.tanHalfFov, 1e-6);
		let cy = dot(rel, basis.up) / max(basis.aspect, 1e-6);
		return vec2<f32>(cx * 0.5 + 0.5, 0.5 - cy * 0.5);
	}

	let cx =
		dot(rel, basis.right) /
		(depth * basis.aspect * basis.tanHalfFov);
	let cy = dot(rel, basis.up) / (depth * basis.tanHalfFov);

	return vec2<f32>(cx * 0.5 + 0.5, 0.5 - cy * 0.5);
}

fn worldToLinearDepth(worldPos: vec3<f32>) -> f32 {
	let rel = worldPos - frame.cameraPosition.xyz;
	return dot(rel, -frame.environmentBasisBackward.xyz);
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
		let hizSample = textureSampleLevel(hiZ, linearSampler, suv, f32(safeMip));
		let minDepth = hizSample.x;
		let maxDepth = max(hizSample.y, hizSample.x); // fallback if .y wasn't written

		// No scene geometry at this location?
		if (minDepth <= 0.0) {
			t += stride * pow(2.0, f32(safeMip));
			continue;
		}

		// Is the ray behind the back surface? (completely occluded)
		if (rayDepth > maxDepth + thickness) {
			// Ray is past this cell — advance and try coarser
			missT = t;
			mip = min(mip + 1, maxMip);
			t += stride * pow(2.0, f32(safeMip));
			continue;
		}

		// Is the ray in front of the front surface?
		if (rayDepth < minDepth - thickness) {
			// Ray hasn't reached this cell yet — advance at current mip
			missT = t;
			t += stride * pow(2.0, f32(safeMip));
			continue;
		}

		// Ray is within the depth range [minDepth - thickness, maxDepth + thickness]
		if (mip > 0) {
			// Refine at finer mip level
			mip = mip - 1;
			continue;
		}

		// At mip 0 and within thickness — we have a hit!
		if (rayDepth >= minDepth - thickness && rayDepth <= minDepth + thickness * 2.0) {
			result.hit = true;
			result.hitUv = suv;
			result.t = t;
			break;
		}

		// Continue marching at mip 0
		missT = t;
		t += stride;
	}

	// Binary refinement if we got a hit
	if (result.hit) {
		var refineMin = missT;
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
			let sceneDepth = textureSampleLevel(hiZ, linearSampler, refineUv, 0.0).x;

			if (sceneDepth > 0.0 && rayDepth >= sceneDepth - thickness) {
				refineMax = midT;
				result.hitUv = refineUv;
			} else {
				refineMin = midT;
			}
		}
		result.t = refineMax;
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
	if (depth <= 0.0 || roughness > traceParams.maxRoughness) {
		textureStore(outSSR, vec2<i32>(gid.xy), vec4<f32>(0.0));
		return;
	}

	let worldPos = getPosition(uv, depth);
	let worldNormal = decodeNormal(g.xy);
	let worldViewDir = normalize(worldPos - frame.cameraPosition.xyz);
	let nDotV = max(dot(worldNormal, -worldViewDir), 0.001);

	// Physically-based F0 — dielectrics ≈ 0.04, metals use their albedo
	// (We don't have albedo here so approximate with vec3(1) for metals)
	let f0 = mix(vec3<f32>(0.04), vec3<f32>(1.0), metalness);

	// -----------------------------------------------------------------------
	// Blue-noise jitter from interleaved gradient noise
	// -----------------------------------------------------------------------
	let noise = interleavedGradientNoise(pixel, traceParams.frameIndex);
	let r2 = r2Sequence(traceParams.frameIndex + noise * 64.0);

	// -----------------------------------------------------------------------
	// Stochastic reflection direction
	// For smooth surfaces (roughness < 0.05), use mirror reflection.
	// For rough surfaces, importance-sample a GGX microfacet normal to
	// approximate roughness cone tracing.
	// -----------------------------------------------------------------------
	var reflectionDir: vec3<f32>;
	var sampleRoughness: f32;

	if (roughness < 0.05) {
		// Mirror reflection — no stochastic sampling needed
		reflectionDir = reflect(worldViewDir, worldNormal);
		sampleRoughness = 0.0;
	} else {
		// Importance-sample a GGX half-vector
		let xi = vec2<f32>(
			fract(r2.x + noise),
			fract(r2.y + noise * 0.7071)
		);
		let H = importanceSampleGGX(xi, roughness, worldNormal);
		reflectionDir = reflect(worldViewDir, H);
		sampleRoughness = roughness;
	}

	// Safety: skip if reflection points INTO the surface
	if (dot(reflectionDir, worldNormal) < 0.001) {
		textureStore(outSSR, vec2<i32>(gid.xy), vec4<f32>(0.0));
		return;
	}

	// -----------------------------------------------------------------------
	// HiZ ray march with blue-noise jitter for per-pixel t-offset
	// -----------------------------------------------------------------------
	let startT = max(traceParams.stride, 0.25);
	let traceResult = traceHiZ(worldPos, reflectionDir, startT, noise);

	if (!traceResult.hit) {
		textureStore(outSSR, vec2<i32>(gid.xy), vec4<f32>(0.0));
		return;
	}

	// -----------------------------------------------------------------------
	// Sample hit color at appropriate mip for roughness cone approximation
	// Wider cone (higher roughness) → coarser mip → smoother reflections
	// -----------------------------------------------------------------------
	let hitUv = traceResult.hitUv;
	let coneMip = clamp(sampleRoughness * 4.0, 0.0, 3.0);
	let hitColor = textureSampleLevel(sceneColor, linearSampler, hitUv, coneMip).rgb;

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

	// -----------------------------------------------------------------------
	// Temporal reprojection
	// -----------------------------------------------------------------------
	let motion = motionDepth.xy;
	let prevUv = uv - vec2<f32>(motion.x * 0.5, -motion.y * 0.5);
	let hist = textureSampleLevel(ssrHistory, linearSampler, prevUv, 0.0);

	let currDepth = depth;
	let prevDepth = textureSampleLevel(motionHistory, linearSampler, prevUv, 0.0).z;
	let relDepth = abs(currDepth - prevDepth) / max(max(currDepth, prevDepth), 1e-4);

	let historyOk = traceParams.historyValid > 0.5
		&& prevUv.x >= 0.0 && prevUv.x <= 1.0
		&& prevUv.y >= 0.0 && prevUv.y <= 1.0
		&& relDepth <= traceParams.depthThreshold;

	// For stochastic SSR, blend more aggressively with history to converge
	// Rough surfaces need more temporal blending, smooth surfaces less
	let baseBlend = clamp(traceParams.historyWeight, 0.0, 0.98);
	let roughnessBlend = mix(baseBlend, min(baseBlend + 0.05, 0.98), sampleRoughness);
	let blend = select(0.0, roughnessBlend, historyOk);

	let current = vec4<f32>(max(hitColor * weight, vec3<f32>(0.0)), weight);
	textureStore(outSSR, vec2<i32>(gid.xy), mix(current, hist, blend));
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
