#import <ignis/postprocess/luma-common>
#define IGNIS_LUMA_PROFILE bt709
#define IGNIS_LUMA_CLAMP false
#inject <ignis/postprocess/luma>(profile=IGNIS_LUMA_PROFILE, clamp=IGNIS_LUMA_CLAMP)
struct DirectionalLightData {
	direction: vec4<f32>,
	color: vec4<f32>,
}

struct PointLightData {
	positionRange: vec4<f32>,
	color: vec4<f32>,
}

struct SpotLightData {
	positionRange: vec4<f32>,
	directionOuter: vec4<f32>,
	colorInner: vec4<f32>,
}

struct ShadowData {
	viewProjection: mat4x4<f32>,
	paramsA: vec4<f32>,
	paramsB: vec4<f32>,
	paramsC: vec4<f32>,
}

struct VolumetricLightData {
	positionRange: vec4<f32>,
	directionOuter: vec4<f32>,
	colorInner: vec4<f32>,
}

struct VolumetricLightBuffer {
	lights: array<VolumetricLightData>,
}

struct FrameUniforms {
	viewProjection: mat4x4<f32>,
	prevViewProjection: mat4x4<f32>,
	cameraPosition: vec4<f32>,
	skyboxBasisRight: vec4<f32>,
	skyboxBasisUp: vec4<f32>,
	skyboxBasisBackward: vec4<f32>,
	ambientColor: vec4<f32>,
	lightCounts: vec4<f32>,
	options: vec4<f32>,
	environmentOptionsA: vec4<f32>,
	environmentOptionsB: vec4<f32>,
	taaJitterCurrentPrev: vec4<f32>,
	directionalLights: array<DirectionalLightData, 4>,
	pointLights: array<PointLightData, 4>,
	spotLights: array<SpotLightData, 8>,
	directionalShadows: array<ShadowData, 4>,
	spotShadows: array<ShadowData, 8>,
	shAmbientCoeffs: array<vec4<f32>, 16>,
}

struct VolumetricParams {
	invSize: vec2<f32>,
	samples: f32,
	weight: f32,
	exposure: f32,
	airDensity: f32,
	anisotropy: f32,
	maxRayDistance: f32,
	scatteringAlbedo: f32,
	shadowSampleInterval: f32,
	adaptiveSteps: f32,
	depthThickness: f32,
	maxMip: f32,
	jitterStrength: f32,
	historyValid: f32,
	lightCount: f32,
	restirCandidates: f32,
	restirTemporalWeight: f32,
	restirScaleClamp: f32,
	frameIndex: f32,
}

struct Reservoir {
	lightIndex: i32,
	selectedWeight: f32,
	weightSum: f32,
	sampleCount: f32,
}

struct CameraBasis {
	right: vec3<f32>,
	up: vec3<f32>,
	backward: vec3<f32>,
	tanHalfFov: f32,
	aspect: f32,
}

@group(0) @binding(0) var sceneColor: texture_2d<f32>;
@group(0) @binding(1) var gMotionDepth: texture_2d<f32>;
@group(0) @binding(2) var hiZ: texture_2d<f32>;
@group(0) @binding(3) var volumetricHistory: texture_2d<f32>;
@group(0) @binding(4) var motionHistory: texture_2d<f32>;
@group(0) @binding(5) var linearSampler: sampler;
@group(0) @binding(6) var<uniform> params: VolumetricParams;
@group(0) @binding(7) var outColor: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var outHistory: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var volumetricReservoirHistory: texture_2d<f32>;
@group(0) @binding(10) var outReservoirHistory: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var<storage, read> volumetricLightBuffer: VolumetricLightBuffer;

@group(1) @binding(0) var<uniform> frame: FrameUniforms;

const PI: f32 = 3.14159265359;
const MAX_VIEW_STEPS: i32 = 96;
const MAX_SHADOW_STEPS: i32 = 24;
const MAX_RESTIR_CANDIDATES: i32 = 64;
const SIGMA_T_SCALE: f32 = 0.02;
const TEMPORAL_HISTORY_WEIGHT: f32 = 0.75;
const TEMPORAL_DEPTH_THRESHOLD: f32 = 0.04;
const SHADOW_CONE_SLOPE: f32 = 0.02;
const SKY_STEP_SCALE: f32 = 0.55;
const TEMPORAL_MOTION_FACTOR: f32 = 28.0;
const TEMPORAL_DISOCCLUSION_FACTOR: f32 = 6.0;

fn safeNormalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
	let len = length(v);
	return select(fallback, v / max(len, 1e-6), len > 1e-6);
}

fn interleavedGradientNoise(pixel: vec2<f32>, frameIndex: f32) -> f32 {
	let frameOffset = frameIndex * 5.588238;
	let uv = pixel + frameOffset;
	return fract(52.9829189 * fract(0.06711056 * uv.x + 0.00583715 * uv.y));
}

fn randomSample(pixel: vec2<f32>, frameIndex: f32, salt: f32) -> f32 {
	let shifted = pixel + vec2<f32>(salt * 1.13, salt * 2.71);
	return interleavedGradientNoise(shifted, frameIndex + salt * 0.77);
}

fn pointAttenuation(distanceSq: f32, range: f32) -> f32 {
	let rangeSq = max(range * range, 1e-6);
	let rangeFactor = distanceSq / rangeSq;
	let smoothFactor = max(0.0, 1.0 - rangeFactor * rangeFactor);
	return (smoothFactor * smoothFactor) / (distanceSq + 1.0);
}

fn spotAttenuation(cosTheta: f32, outerCos: f32, innerCos: f32) -> f32 {
	if (cosTheta < outerCos) {
		return 0.0;
	}
	let cutoffRange = max(innerCos - outerCos, 1e-6);
	return clamp((cosTheta - outerCos) / cutoffRange, 0.0, 1.0);
}

fn henyeyGreenstein(cosTheta: f32, g: f32) -> f32 {
	let gg = g * g;
	let denom = pow(max(1.0 + gg - 2.0 * g * cosTheta, 1e-6), 1.5);
	return (1.0 - gg) / (4.0 * PI * denom);
}

fn getCameraBasis() -> CameraBasis {
	return CameraBasis(
		frame.skyboxBasisRight.xyz,
		frame.skyboxBasisUp.xyz,
		frame.skyboxBasisBackward.xyz,
		frame.skyboxBasisRight.w,
		frame.skyboxBasisUp.w
	);
}

fn getWorldRayDirection(uv: vec2<f32>) -> vec3<f32> {
	let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
	let basis = getCameraBasis();
	let camRay = vec3<f32>(
		ndc.x * basis.aspect * basis.tanHalfFov,
		ndc.y * basis.tanHalfFov,
		-1.0
	);
	return safeNormalize(
		basis.right * camRay.x + basis.up * camRay.y + basis.backward * camRay.z,
		-basis.backward
	);
}

fn worldToUv(worldPos: vec3<f32>) -> vec2<f32> {
	let rel = worldPos - frame.cameraPosition.xyz;
	let basis = getCameraBasis();
	let depth = dot(rel, -basis.backward);
	if (depth <= 1e-4) { return vec2<f32>(-1.0); }
	let cx =
		dot(rel, basis.right) /
		(depth * basis.aspect * basis.tanHalfFov);
	let cy = dot(rel, basis.up) / (depth * basis.tanHalfFov);
	return vec2<f32>(cx * 0.5 + 0.5, 0.5 - cy * 0.5);
}

fn worldToLinearDepth(worldPos: vec3<f32>) -> f32 {
	let rel = worldPos - frame.cameraPosition.xyz;
	let basis = getCameraBasis();
	return dot(rel, -basis.backward);
}

fn isInsideScreen(uv: vec2<f32>) -> bool {
	return uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
}

fn sampleDepthRange(uv: vec2<f32>, mip: i32) -> vec2<f32> {
	let safeMip = i32(clamp(f32(mip), 0.0, 12.0));
	let s = textureSampleLevel(hiZ, linearSampler, uv, f32(safeMip));
	return vec2<f32>(s.x, max(s.y, s.x));
}

fn traceHiZShadowCone(
	startPos: vec3<f32>,
	lightDir: vec3<f32>,
	maxDistance: f32,
	interval: f32
) -> f32 {
	let safeMaxDistance = max(maxDistance, 0.0);
	if (safeMaxDistance <= 0.0) {
		return 1.0;
	}
	let stepBase = max(interval, 0.2);
	let maxMip = i32(clamp(params.maxMip, 0.0, 12.0));
	let thickness = max(params.depthThickness, 0.01);
	var t = max(stepBase, thickness * 2.0);

	for (var step: i32 = 0; step < MAX_SHADOW_STEPS; step = step + 1) {
		if (t >= safeMaxDistance) { break; }

		let samplePos = startPos + lightDir * t;
		let uv = worldToUv(samplePos);
		if (!isInsideScreen(uv)) {
			t += stepBase;
			continue;
		}

		let rayDepth = worldToLinearDepth(samplePos);
		if (rayDepth <= 0.0) {
			t += stepBase;
			continue;
		}

		let coneRadius = max(t * SHADOW_CONE_SLOPE, thickness);
		let distanceMip =
			clamp((t / max(safeMaxDistance, 1.0)) * f32(maxMip), 0.0, f32(maxMip));
		let coneMip = clamp(log2(1.0 + coneRadius * 8.0), 0.0, f32(maxMip));
		let mipFloat = clamp(max(distanceMip, coneMip), 0.0, f32(maxMip));
		let mip = i32(mipFloat);
		let depthRange = sampleDepthRange(uv, mip);
		let minDepth = depthRange.x;
		let maxDepth = depthRange.y;

		if (minDepth > 0.0 && rayDepth > maxDepth + coneRadius * 2.0) {
			return 0.0;
		}

		let stepScale = max(1.0, pow(2.0, mipFloat * 0.7));
		t += stepBase * stepScale;
	}

	return 1.0;
}

fn emptyReservoir() -> Reservoir {
	return Reservoir(-1, 0.0, 0.0, 0.0);
}

fn updateReservoir(
	reservoir: Reservoir,
	candidateIndex: i32,
	candidateWeight: f32,
	candidateSampleCount: f32,
	randValue: f32
) -> Reservoir {
	var updated = reservoir;
	let safeWeight = max(candidateWeight, 0.0);
	let safeSampleCount = max(candidateSampleCount, 0.0);
	if (safeWeight <= 0.0 || safeSampleCount <= 0.0 || candidateIndex < 0) {
		return updated;
	}
	updated.weightSum += safeWeight;
	updated.sampleCount += safeSampleCount;
	let replaceChance = safeWeight / max(updated.weightSum, 1e-6);
	if (randValue < replaceChance) {
		updated.lightIndex = candidateIndex;
		updated.selectedWeight = safeWeight;
	}
	return updated;
}

fn isDirectional(light: VolumetricLightData) -> bool {
	return light.positionRange.w < 0.0;
}

fn isPoint(light: VolumetricLightData) -> bool {
	return light.positionRange.w >= 0.0 && light.directionOuter.w < -1.0;
}

fn sampleCandidateLightIndex(
	candidate: i32,
	candidateCount: i32,
	totalLights: i32,
	pixel: vec2<f32>,
	frameIndex: f32
) -> i32 {
	if (totalLights <= 1) {
		return 0;
	}
	if (candidateCount >= totalLights) {
		let scramble = i32(
			min(
				floor(randomSample(pixel, frameIndex, 2.0) * f32(totalLights)),
				f32(totalLights - 1)
			)
		);
		return (candidate + scramble) % totalLights;
	}
	let jitter = randomSample(pixel, frameIndex, f32(candidate) + 23.0);
	let u = (f32(candidate) + jitter) / f32(max(candidateCount, 1));
	return i32(min(floor(u * f32(totalLights)), f32(totalLights - 1)));
}

fn estimateLightWeight(
	light: VolumetricLightData,
	samplePos: vec3<f32>,
	rayDir: vec3<f32>,
	anisotropy: f32
) -> f32 {
	var contribution = vec3<f32>(0.0);

	if (isDirectional(light)) {
		let lightDir = safeNormalize(light.directionOuter.xyz, vec3<f32>(0.0, 1.0, 0.0));
		let phase = henyeyGreenstein(dot(rayDir, lightDir), anisotropy);
		contribution = light.colorInner.xyz * phase;
	} else {
		let toLight = light.positionRange.xyz - samplePos;
		let distanceSq = dot(toLight, toLight);
		let distanceValue = sqrt(max(distanceSq, 1e-6));
		let lightRange = max(light.positionRange.w, 0.001);
		if (distanceValue > lightRange) {
			return 0.0;
		}
		let lightDir = toLight / distanceValue;
		var attenuation = pointAttenuation(distanceSq, lightRange);
		if (!isPoint(light)) {
			let coneDirection = safeNormalize(light.directionOuter.xyz, vec3<f32>(0.0, -1.0, 0.0));
			let cone = spotAttenuation(
				dot(-lightDir, coneDirection),
				light.directionOuter.w,
				light.colorInner.w
			);
			if (cone <= 0.0) {
				return 0.0;
			}
			attenuation *= cone;
		}
		let phase = henyeyGreenstein(dot(rayDir, lightDir), anisotropy);
		contribution = light.colorInner.xyz * attenuation * phase;
	}

	return max(luma(contribution), 0.0);
}

fn evaluateLightAtSample(
	light: VolumetricLightData,
	samplePos: vec3<f32>,
	rayDir: vec3<f32>,
	anisotropy: f32,
	doShadowSample: bool,
	cachedVisibility: f32,
	maxDistance: f32,
	shadowStepWorld: f32
) -> vec4<f32> {
	var visibility = cachedVisibility;
	var inScatter = vec3<f32>(0.0);

	if (isDirectional(light)) {
		let lightDir = safeNormalize(light.directionOuter.xyz, vec3<f32>(0.0, 1.0, 0.0));
		if (doShadowSample) {
			visibility = traceHiZShadowCone(
				samplePos + lightDir * params.depthThickness,
				lightDir,
				maxDistance,
				shadowStepWorld
			);
		}
		let phase = henyeyGreenstein(dot(rayDir, lightDir), anisotropy);
		inScatter = light.colorInner.xyz * visibility * phase;
	} else {
		let toLight = light.positionRange.xyz - samplePos;
		let distanceSq = dot(toLight, toLight);
		let distanceValue = sqrt(max(distanceSq, 1e-6));
		let lightRange = max(light.positionRange.w, 0.001);
		if (distanceValue > lightRange) {
			return vec4<f32>(vec3<f32>(0.0), visibility);
		}

		let lightDir = toLight / distanceValue;
		var attenuation = pointAttenuation(distanceSq, lightRange);
		if (!isPoint(light)) {
			let coneDirection = safeNormalize(light.directionOuter.xyz, vec3<f32>(0.0, -1.0, 0.0));
			let cone = spotAttenuation(
				dot(-lightDir, coneDirection),
				light.directionOuter.w,
				light.colorInner.w
			);
			if (cone <= 0.0) {
				return vec4<f32>(vec3<f32>(0.0), visibility);
			}
			attenuation *= cone;
		}

		if (doShadowSample) {
			visibility = traceHiZShadowCone(
				samplePos + lightDir * params.depthThickness,
				lightDir,
				distanceValue,
				shadowStepWorld
			);
		}

		let phase = henyeyGreenstein(dot(rayDir, lightDir), anisotropy);
		inScatter = light.colorInner.xyz * attenuation * visibility * phase;
	}

	return vec4<f32>(inScatter, visibility);
}

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outColor);
	if (gid.x >= size.x || gid.y >= size.y) { return; }

	let coord = vec2<i32>(gid.xy);
	let scene = textureLoad(sceneColor, coord, 0);
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * params.invSize;
	let motionDepth = textureSampleLevel(gMotionDepth, linearSampler, uv, 0.0);
	let depth = motionDepth.z;
	let hasSurface = depth > 0.0;

	let rayDir = getWorldRayDirection(uv);
	let maxRayDistance = max(params.maxRayDistance, 0.1);
	let maxDistance = select(maxRayDistance, min(depth, maxRayDistance), hasSurface);
	if (maxDistance <= 0.0) {
		textureStore(outColor, coord, scene);
		textureStore(outHistory, coord, vec4<f32>(0.0));
		textureStore(outReservoirHistory, coord, vec4<f32>(-1.0, 0.0, 0.0, 0.0));
		return;
	}

	let safeInvSize = vec2<f32>(
		max(params.invSize.x, 1e-6),
		max(params.invSize.y, 1e-6)
	);
	let motion = motionDepth.xy;
	let prevUv = uv - vec2<f32>(motion.x * 0.5, -motion.y * 0.5);
	let insidePrev =
		prevUv.x >= 0.0 && prevUv.x <= 1.0 && prevUv.y >= 0.0 && prevUv.y <= 1.0;

	let baseSteps = i32(clamp(params.samples, 1.0, f32(MAX_VIEW_STEPS)));
	let adaptiveScale = select(
		1.0,
		clamp(maxDistance / max(params.maxRayDistance, 1.0), 0.35, 1.0),
		params.adaptiveSteps > 0.5
	);
	let skyStepScale = select(SKY_STEP_SCALE, 1.0, hasSurface);
	let steps = max(4, i32(f32(baseSteps) * adaptiveScale * skyStepScale));
	let stepSize = max(maxDistance / f32(max(steps, 1)), 1e-3);
	let sigmaT = max(params.airDensity, 0.001) * SIGMA_T_SCALE;
	let sigmaS = sigmaT * clamp(params.scatteringAlbedo, 0.0, 1.0);
	let weight = max(params.weight, 0.0);
	let shadowIntervalScale = select(2.0, 1.0, hasSurface);
	let shadowInterval = i32(
		clamp(params.shadowSampleInterval * shadowIntervalScale, 1.0, 32.0)
	);
	let shadowStepWorld = max(stepSize * 2.0, params.depthThickness * 6.0);
	let anisotropy = clamp(params.anisotropy, -0.95, 0.95);
	let pixel = vec2<f32>(gid.xy);
	let jitter =
		(randomSample(pixel, params.frameIndex, 1.0) - 0.5) *
		params.jitterStrength;

	let totalLights = i32(clamp(params.lightCount, 0.0, 65000.0));
	let candidateCount = i32(clamp(params.restirCandidates, 1.0, f32(MAX_RESTIR_CANDIDATES)));
	let effectiveCandidateCount = min(candidateCount, max(totalLights, 1));
	let referenceT = clamp(maxDistance * 0.45 + stepSize, stepSize, maxDistance);
	let referencePos = frame.cameraPosition.xyz + rayDir * referenceT;
	var reservoir = emptyReservoir();

	if (totalLights > 0) {
		for (var candidate: i32 = 0; candidate < MAX_RESTIR_CANDIDATES; candidate = candidate + 1) {
			if (candidate >= effectiveCandidateCount) { break; }
			let salt = f32(candidate) + 3.0;
			let lightIndex = sampleCandidateLightIndex(
				candidate,
				candidateCount,
				totalLights,
				pixel,
				params.frameIndex
			);
			let light = volumetricLightBuffer.lights[u32(lightIndex)];
			let candidateWeight = estimateLightWeight(light, referencePos, rayDir, anisotropy);
			let r = randomSample(pixel, params.frameIndex, salt + 0.5);
			reservoir = updateReservoir(
				reservoir,
				lightIndex,
				candidateWeight,
				1.0,
				r
			);
		}

		if (params.historyValid > 0.5 && insidePrev) {
			let prevCoord = vec2<i32>(
				i32(clamp(prevUv.x * f32(size.x), 0.0, f32(max(size.x, 1u) - 1u))),
				i32(clamp(prevUv.y * f32(size.y), 0.0, f32(max(size.y, 1u) - 1u)))
			);
			let prevReservoir = textureLoad(volumetricReservoirHistory, prevCoord, 0);
			let prevIndex = i32(prevReservoir.x + 0.5);
			if (prevIndex >= 0 && prevIndex < totalLights) {
				let prevWeightSum = max(prevReservoir.y, 0.0);
				let prevSelectedWeight = max(prevReservoir.z, 0.0);
				let prevSampleCount = max(prevReservoir.w, 0.0);
				if (
					prevWeightSum > 0.0 &&
					prevSelectedWeight > 0.0 &&
					prevSampleCount > 0.0
				) {
					let prevLight = volumetricLightBuffer.lights[u32(prevIndex)];
					let prevCurrentWeight = estimateLightWeight(
						prevLight,
						referencePos,
						rayDir,
						anisotropy
					);
					let temporalWeight = clamp(params.restirTemporalWeight, 0.0, 1.0);
					let temporalCandidateWeight =
						prevCurrentWeight *
						(prevWeightSum / max(prevSelectedWeight, 1e-6)) *
						temporalWeight;
					let temporalSampleCount = prevSampleCount * temporalWeight;
					if (temporalCandidateWeight > 0.0 && temporalSampleCount > 0.0) {
						let r = randomSample(pixel, params.frameIndex, 103.0);
						reservoir = updateReservoir(
							reservoir,
							prevIndex,
							temporalCandidateWeight,
							temporalSampleCount,
							r
						);
					}
				}
			}
		}
	}

	var selectedLightIndex: u32 = 0u;
	var hasSelectedLight = false;
	var restirScale = 0.0;
	if (reservoir.lightIndex >= 0 && reservoir.sampleCount > 0.0 && totalLights > 0) {
		hasSelectedLight = true;
		selectedLightIndex = u32(reservoir.lightIndex);
		let denom = max(reservoir.selectedWeight * reservoir.sampleCount, 1e-6);
		let normalization = reservoir.weightSum / denom;
		restirScale = clamp(
			normalization * f32(totalLights),
			0.0,
			max(params.restirScaleClamp, 1.0)
		);
	}

	var transmittance = 1.0;
	var accum = vec3<f32>(0.0);
	var selectedVisibility = 1.0;
	var selectedVisibilityValid = false;
	let stepTransmittance = exp(-sigmaT * stepSize);
	let stepMidTransmittance = exp(-sigmaT * stepSize * 0.5);

	for (var step: i32 = 0; step < MAX_VIEW_STEPS; step = step + 1) {
		if (step >= steps) { break; }

		let sampleT = clamp((f32(step) + 0.5 + jitter) * stepSize, 0.0, maxDistance);
		if (sampleT >= maxDistance) { break; }

		let samplePos = frame.cameraPosition.xyz + rayDir * sampleT;
		let doShadowSample = (step % shadowInterval) == 0;
		var inScatter = vec3<f32>(0.0);

		if (hasSelectedLight) {
			let selectedLight = volumetricLightBuffer.lights[selectedLightIndex];
			let canCacheVisibility = isDirectional(selectedLight);
			let shouldTraceShadow =
				doShadowSample ||
				!canCacheVisibility ||
				!selectedVisibilityValid;
			let cachedVisibility = select(
				1.0,
				selectedVisibility,
				canCacheVisibility && selectedVisibilityValid
			);
			let sampled = evaluateLightAtSample(
				selectedLight,
				samplePos,
				rayDir,
				anisotropy,
				shouldTraceShadow,
				cachedVisibility,
				maxDistance,
				shadowStepWorld
			);
			selectedVisibility = sampled.w;
			selectedVisibilityValid = true;
			inScatter = sampled.rgb * restirScale;
		}

		accum +=
			inScatter *
			sigmaS *
			weight *
			(transmittance * stepMidTransmittance) *
			stepSize;
		transmittance *= stepTransmittance;

		if (transmittance < 0.001) { break; }
	}

	let volumetricCurrent = max(accum * max(params.exposure, 0.0), vec3<f32>(0.0));
	let prevVolumetric = textureSampleLevel(
		volumetricHistory,
		linearSampler,
		prevUv,
		0.0
	);
	let currDepthForHistory = select(maxDistance, depth, hasSurface);
	let prevDepth = textureSampleLevel(motionHistory, linearSampler, prevUv, 0.0).z;
	var depthConfidence = 1.0;
	if (hasSurface) {
		let hasPrevDepth = prevDepth > 0.0;
		let relDepth =
			abs(currDepthForHistory - prevDepth) /
			max(max(currDepthForHistory, prevDepth), 1e-4);
		depthConfidence = select(
			0.0,
			1.0 - smoothstep(
				TEMPORAL_DEPTH_THRESHOLD * 0.5,
				TEMPORAL_DEPTH_THRESHOLD * 2.0,
				relDepth
			),
			hasPrevDepth
		);
	}
	let prevMotion = textureSampleLevel(motionHistory, linearSampler, prevUv, 0.0).xy;
	let forwardUv = prevUv + vec2<f32>(prevMotion.x * 0.5, -prevMotion.y * 0.5);
	let reprojectionError = abs(forwardUv - uv) / safeInvSize;
	let reprojectionErrorPx = max(reprojectionError.x, reprojectionError.y);
	let reprojectionConfidence = 1.0 - smoothstep(0.5, 2.0, reprojectionErrorPx);

	let currLuma = luma(volumetricCurrent);
	let prevLuma = luma(prevVolumetric.rgb);
	let relLumaDiff = abs(currLuma - prevLuma) / max(max(currLuma, prevLuma), 1e-3);
	let colorConfidence = 1.0 - smoothstep(0.08, 0.5, relLumaDiff);

	// Neighborhood clamp: restrict history luminance to a tolerance around
	// the current value so stale bright/dark artifacts cannot persist.
	let lumaTolerance = max(currLuma * 0.6, 0.005);
	let clampedPrevLuma = clamp(prevLuma, currLuma - lumaTolerance, currLuma + lumaTolerance);
	let lumaScale = select(clampedPrevLuma / max(prevLuma, 1e-6), 1.0, prevLuma < 1e-6);
	let clampedPrevVolumetric = prevVolumetric.rgb * lumaScale;

	let validBase = params.historyValid > 0.5 && insidePrev;
	let motionMag = length(motion);
	// Velocity-magnitude disocclusion: fast motion quickly invalidates history
	let velocityDisocclusion = exp(-motionMag * TEMPORAL_DISOCCLUSION_FACTOR);
	let historyConfidence = select(
		0.0,
		clamp(
			depthConfidence * reprojectionConfidence * colorConfidence * velocityDisocclusion,
			0.0,
			1.0
		),
		validBase
	);
	let adaptiveHistory = clamp(
		TEMPORAL_HISTORY_WEIGHT * exp(-motionMag * TEMPORAL_MOTION_FACTOR),
		0.0,
		TEMPORAL_HISTORY_WEIGHT
	);
	let blend = clamp(adaptiveHistory * historyConfidence, 0.0, TEMPORAL_HISTORY_WEIGHT);
	let volumetric = mix(volumetricCurrent, clampedPrevVolumetric, blend);

	let storedIndex = select(-1.0, f32(reservoir.lightIndex), reservoir.lightIndex >= 0);
	let storedWeightSum = min(max(reservoir.weightSum, 0.0), 65000.0);
	let storedSelectedWeight = min(max(reservoir.selectedWeight, 0.0), 65000.0);
	let storedSampleCount = min(max(reservoir.sampleCount, 0.0), 65000.0);

	textureStore(outReservoirHistory, coord, vec4<f32>(
		storedIndex,
		storedWeightSum,
		storedSelectedWeight,
		storedSampleCount
	));
	textureStore(outHistory, coord, vec4<f32>(volumetric, 1.0));
	textureStore(
		outColor,
		coord,
		vec4<f32>(max(scene.rgb + volumetric, vec3<f32>(0.0)), scene.a)
	);
}
