#import <ignis/webgpu/constants>
struct FrameCameraUniforms {
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
}

struct TraceParams {
	invTraceSize: vec2<f32>,
	maxDistance: f32,
	thickness: f32,
	normalBias: f32,
	distanceFalloffExponent: f32,
	edgeFade: f32,
	raysPerPixel: f32,
	maxSteps: f32,
	binarySearchSteps: f32,
	maxMip: f32,
	frameIndex: f32,
	historyWeight: f32,
	depthThreshold: f32,
	historyClamp: f32,
	historyValid: f32,
	_pad0: vec4<f32>,
}

struct DenoiseParams {
	invTraceSize: vec2<f32>,
	radius: f32,
	depthPhi: f32,
	normalPhi: f32,
	_pad0: f32,
	_pad1: f32,
	_pad2: f32,
}

struct ComposeParams {
	invFullSize: vec2<f32>,
	invTraceSize: vec2<f32>,
	intensity: f32,
	depthPhi: f32,
	normalPhi: f32,
	_pad0: f32,
}

@group(0) @binding(0) var traceSceneColor: texture_2d<f32>;
@group(0) @binding(1) var traceNormalRoughMetal: texture_2d<f32>;
@group(0) @binding(2) var traceMotionDepth: texture_2d<f32>;
@group(0) @binding(3) var traceHiZTexture: texture_2d<f32>;
@group(0) @binding(4) var traceHistory: texture_2d<f32>;
@group(0) @binding(5) var traceMotionHistory: texture_2d<f32>;
@group(0) @binding(6) var traceSampler: sampler;
@group(0) @binding(7) var<uniform> traceParams: TraceParams;
@group(0) @binding(8)
var traceOut: texture_storage_2d<rgba16float, write>;

@group(1) @binding(0) var<uniform> frame: FrameCameraUniforms;

@group(0) @binding(0) var denoiseSource: texture_2d<f32>;
@group(0) @binding(1) var denoiseNormalRoughMetal: texture_2d<f32>;
@group(0) @binding(2) var denoiseMotionDepth: texture_2d<f32>;
@group(0) @binding(3) var denoiseSampler: sampler;
@group(0) @binding(4) var<uniform> denoiseParams: DenoiseParams;
@group(0) @binding(5)
var denoiseOut: texture_storage_2d<rgba16float, write>;

@group(0) @binding(0) var composeSceneColor: texture_2d<f32>;
@group(0) @binding(1) var composeSSGI: texture_2d<f32>;
@group(0) @binding(2) var composeAlbedo: texture_2d<f32>;
@group(0) @binding(3) var composeNormalRoughMetal: texture_2d<f32>;
@group(0) @binding(4) var composeMotionDepth: texture_2d<f32>;
@group(0) @binding(5) var composeSampler: sampler;
@group(0) @binding(6) var<uniform> composeParams: ComposeParams;
@group(0) @binding(7)
var composeOut: texture_storage_2d<rgba16float, write>;

const MAX_RAYS_PER_PIXEL = 4;
const MAX_TRACE_STEPS = 64;
const MAX_BINARY_SEARCH_STEPS = 8;
const MAX_DENOISE_RADIUS = 4;
const LUMINANCE = vec3<f32>(0.2126, 0.7152, 0.0722);

struct CameraBasis {
	right: vec3<f32>,
	up: vec3<f32>,
	backward: vec3<f32>,
	tanHalfFov: f32,
	aspect: f32,
}

struct HiZTraceResult {
	hitUv: vec2<f32>,
	t: f32,
	hit: bool,
}

fn getCameraBasis() -> CameraBasis {
	return CameraBasis(
		frame.environmentBasisRight.xyz,
		frame.environmentBasisUp.xyz,
		frame.environmentBasisBackward.xyz,
		frame.environmentBasisRight.w,
		frame.environmentBasisUp.w
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

fn decodeWorldNormal(encoded: vec2<f32>) -> vec3<f32> {
	let oct = encoded * 2.0 - vec2<f32>(1.0);
	var normal = vec3<f32>(
		oct.x,
		oct.y,
		1.0 - abs(oct.x) - abs(oct.y)
	);
	if (normal.z < 0.0) {
		normal = vec3<f32>(octahedralWrap(normal.xy), normal.z);
	}
	return normalize(normal);
}

fn isInsideScreen(uv: vec2<f32>) -> bool {
	return uv.x >= 0.0 && uv.x <= 1.0 &&
		uv.y >= 0.0 && uv.y <= 1.0;
}

fn reconstructWorldPosition(uv: vec2<f32>, depth: f32) -> vec3<f32> {
	let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
	let basis = getCameraBasis();
	let cameraX = ndc.x * basis.aspect * basis.tanHalfFov * depth;
	let cameraY = ndc.y * basis.tanHalfFov * depth;
	return frame.cameraPosition.xyz +
		basis.right * cameraX +
		basis.up * cameraY -
		basis.backward * depth;
}

fn worldToUv(worldPosition: vec3<f32>) -> vec2<f32> {
	let relative = worldPosition - frame.cameraPosition.xyz;
	let basis = getCameraBasis();
	let depth = dot(relative, -basis.backward);
	if (depth <= 1e-4) {
		return vec2<f32>(-1.0);
	}
	let cameraX =
		dot(relative, basis.right) /
		max(depth * basis.aspect * basis.tanHalfFov, 1e-6);
	let cameraY =
		dot(relative, basis.up) /
		max(depth * basis.tanHalfFov, 1e-6);
	return vec2<f32>(cameraX * 0.5 + 0.5, 0.5 - cameraY * 0.5);
}

fn worldToLinearDepth(worldPosition: vec3<f32>) -> f32 {
	return dot(
		worldPosition - frame.cameraPosition.xyz,
		-frame.environmentBasisBackward.xyz
	);
}

fn interleavedGradientNoise(pixel: vec2<f32>, frameIndex: f32) -> f32 {
	let frameOffset = frameIndex * 5.588238;
	let shiftedPixel = pixel + frameOffset;
	return fract(
		52.9829189 *
		fract(0.06711056 * shiftedPixel.x + 0.00583715 * shiftedPixel.y)
	);
}

fn r2Sequence(index: f32) -> vec2<f32> {
	let reciprocalPlastic = 1.0 / 1.3247179572;
	return fract(
		vec2<f32>(reciprocalPlastic, reciprocalPlastic * reciprocalPlastic) *
		(index + 1.0)
	);
}

fn cosineHemisphereDirection(
	xi: vec2<f32>,
	worldNormal: vec3<f32>
) -> vec3<f32> {
	let radius = sqrt(xi.x);
	let phi = TWO_PI * xi.y;
	let localDirection = vec3<f32>(
		radius * cos(phi),
		radius * sin(phi),
		sqrt(max(1.0 - xi.x, 0.0))
	);
	let up = select(
		vec3<f32>(0.0, 1.0, 0.0),
		vec3<f32>(1.0, 0.0, 0.0),
		abs(worldNormal.y) > 0.999
	);
	let tangent = normalize(cross(up, worldNormal));
	let bitangent = cross(worldNormal, tangent);
	return normalize(
		tangent * localDirection.x +
		bitangent * localDirection.y +
		worldNormal * localDirection.z
	);
}

fn traceHiZ(
	origin: vec3<f32>,
	direction: vec3<f32>,
	jitter: f32
) -> HiZTraceResult {
	var result: HiZTraceResult;
	result.hitUv = vec2<f32>(0.0);
	result.t = 0.0;
	result.hit = false;

	let maxSteps = i32(clamp(
		traceParams.maxSteps,
		4.0,
		f32(MAX_TRACE_STEPS)
	));
	let maxMip = i32(clamp(traceParams.maxMip, 0.0, 12.0));
	let thickness = max(traceParams.thickness, 0.001);
	let baseStride = max(
		traceParams.maxDistance / max(traceParams.maxSteps, 1.0),
		thickness
	);
	var t = baseStride * mix(0.5, 1.5, jitter);
	var missT = 0.0;
	var mip = maxMip;

	for (var step = 0; step < MAX_TRACE_STEPS; step = step + 1) {
		if (step >= maxSteps || t > traceParams.maxDistance) {
			break;
		}
		let samplePosition = origin + direction * t;
		let sampleUv = worldToUv(samplePosition);
		if (!isInsideScreen(sampleUv)) {
			break;
		}
		let rayDepth = worldToLinearDepth(samplePosition);
		if (rayDepth <= 0.0) {
			break;
		}
		let safeMip = clamp(mip, 0, maxMip);
		let depthRange = textureSampleLevel(
			traceHiZTexture,
			traceSampler,
			sampleUv,
			f32(safeMip)
		).xy;
		let minDepth = depthRange.x;
		let maxDepth = max(depthRange.y, minDepth);
		if (minDepth <= 0.0) {
			missT = t;
			t += baseStride * exp2(f32(safeMip));
			continue;
		}
		if (
			rayDepth >= minDepth - thickness &&
			rayDepth <= maxDepth + thickness
		) {
			if (safeMip > 0) {
				mip = safeMip - 1;
				continue;
			}
			result.hitUv = sampleUv;
			result.t = t;
			result.hit = true;
			break;
		}
		missT = t;
		if (rayDepth < minDepth - thickness) {
			mip = min(safeMip + 1, maxMip);
			t += baseStride * exp2(f32(safeMip));
		} else {
			if (safeMip > 0) {
				mip = safeMip - 1;
			} else {
				t += baseStride;
			}
		}
	}

	if (result.hit) {
		var refineMin = missT;
		var refineMax = result.t;
		let refineSteps = i32(clamp(
			traceParams.binarySearchSteps,
			0.0,
			f32(MAX_BINARY_SEARCH_STEPS)
		));
		for (
			var step = 0;
			step < MAX_BINARY_SEARCH_STEPS;
			step = step + 1
		) {
			if (step >= refineSteps) {
				break;
			}
			let midpoint = (refineMin + refineMax) * 0.5;
			let samplePosition = origin + direction * midpoint;
			let sampleUv = worldToUv(samplePosition);
			if (!isInsideScreen(sampleUv)) {
				refineMax = midpoint;
				continue;
			}
			let rayDepth = worldToLinearDepth(samplePosition);
			let sceneDepth = textureSampleLevel(
				traceHiZTexture,
				traceSampler,
				sampleUv,
				0.0
			).x;
			if (
				sceneDepth > 0.0 &&
				rayDepth >= sceneDepth - thickness
			) {
				refineMax = midpoint;
				result.hitUv = sampleUv;
			} else {
				refineMin = midpoint;
			}
		}
		result.t = refineMax;
	}
	return result;
}

fn clampHistoryLuminance(
	historyRadiance: vec3<f32>,
	currentRadiance: vec3<f32>,
	clampRatio: f32
) -> vec3<f32> {
	let currentLuminance = max(dot(currentRadiance, LUMINANCE), 1e-4);
	let historyLuminance = max(dot(historyRadiance, LUMINANCE), 1e-4);
	let clampedLuminance = clamp(
		historyLuminance,
		currentLuminance / clampRatio,
		currentLuminance * clampRatio
	);
	return historyRadiance * (clampedLuminance / historyLuminance);
}

@compute @workgroup_size(8, 8, 1)
fn csTraceTemporal(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(traceOut);
	if (gid.x >= size.x || gid.y >= size.y) {
		return;
	}
	let coord = vec2<i32>(gid.xy);
	let pixel = vec2<f32>(gid.xy);
	let uv = (pixel + vec2<f32>(0.5)) * traceParams.invTraceSize;
	let motionDepth = textureSampleLevel(
		traceMotionDepth,
		traceSampler,
		uv,
		0.0
	);
	let depth = motionDepth.z;
	if (depth <= 0.0) {
		textureStore(traceOut, coord, vec4<f32>(0.0));
		return;
	}

	let worldNormal = decodeWorldNormal(
		textureSampleLevel(
			traceNormalRoughMetal,
			traceSampler,
			uv,
			0.0
		).xy
	);
	let worldPosition = reconstructWorldPosition(uv, depth);
	let origin = worldPosition +
		worldNormal * max(traceParams.normalBias, 0.0);
	let rayCount = i32(clamp(
		traceParams.raysPerPixel,
		1.0,
		f32(MAX_RAYS_PER_PIXEL)
	));
	let baseNoise = interleavedGradientNoise(pixel, traceParams.frameIndex);
	var radiance = vec3<f32>(0.0);
	var confidence = 0.0;

	for (
		var rayIndex = 0;
		rayIndex < MAX_RAYS_PER_PIXEL;
		rayIndex = rayIndex + 1
	) {
		if (rayIndex >= rayCount) {
			break;
		}
		let sequenceIndex =
			traceParams.frameIndex * f32(MAX_RAYS_PER_PIXEL) +
			f32(rayIndex) +
			baseNoise;
		let sequence = r2Sequence(sequenceIndex);
		let xi = fract(sequence + vec2<f32>(
			baseNoise,
			baseNoise * 0.754877666
		));
		let rayDirection = cosineHemisphereDirection(xi, worldNormal);
		let hit = traceHiZ(origin, rayDirection, fract(baseNoise + sequence.x));
		if (!hit.hit || !isInsideScreen(hit.hitUv)) {
			continue;
		}
		let hitNormal = decodeWorldNormal(
			textureSampleLevel(
				traceNormalRoughMetal,
				traceSampler,
				hit.hitUv,
				0.0
			).xy
		);
		let frontFacing = dot(hitNormal, -rayDirection);
		if (frontFacing <= 0.0) {
			continue;
		}
		let hitDepth = textureSampleLevel(
			traceMotionDepth,
			traceSampler,
			hit.hitUv,
			0.0
		).z;
		let refinedRayDepth = worldToLinearDepth(
			origin + rayDirection * hit.t
		);
		if (
			hitDepth <= 0.0 ||
			abs(refinedRayDepth - hitDepth) >
				max(traceParams.thickness, 0.001) * 2.0
		) {
			continue;
		}
		let edgeDistance = min(
			min(hit.hitUv.x, 1.0 - hit.hitUv.x),
			min(hit.hitUv.y, 1.0 - hit.hitUv.y)
		);
		let edgeWeight = select(
			1.0,
			clamp(edgeDistance / max(traceParams.edgeFade, 1e-4), 0.0, 1.0),
			traceParams.edgeFade > 0.0
		);
		let normalizedDistance = clamp(
			hit.t / max(traceParams.maxDistance, 1e-4),
			0.0,
			1.0
		);
		let distanceWeight = pow(
			1.0 - normalizedDistance,
			max(traceParams.distanceFalloffExponent, 0.25)
		);
		let hitConfidence = edgeWeight *
			distanceWeight *
			sqrt(clamp(frontFacing, 0.0, 1.0));
		let hitRadiance = max(
			textureSampleLevel(
				traceSceneColor,
				traceSampler,
				hit.hitUv,
				0.0
			).rgb,
			vec3<f32>(0.0)
		);
		radiance += hitRadiance * hitConfidence;
		confidence += hitConfidence;
	}

	let inverseRayCount = 1.0 / f32(rayCount);
	var current = vec4<f32>(
		radiance * inverseRayCount,
		clamp(confidence * inverseRayCount, 0.0, 1.0)
	);
	let previousUv = uv - vec2<f32>(
		motionDepth.x * 0.5,
		-motionDepth.y * 0.5
	);
	let insideHistory = isInsideScreen(previousUv);
	var history = vec4<f32>(0.0);
	var previousDepth = 0.0;
	if (insideHistory) {
		history = textureSampleLevel(
			traceHistory,
			traceSampler,
			previousUv,
			0.0
		);
		previousDepth = textureSampleLevel(
			traceMotionHistory,
			traceSampler,
			previousUv,
			0.0
		).z;
	}
	let relativeDepthDifference =
		abs(depth - previousDepth) /
		max(max(depth, previousDepth), 1e-4);
	let historyAccepted =
		traceParams.historyValid > 0.5 &&
		insideHistory &&
		previousDepth > 0.0 &&
		relativeDepthDifference <= traceParams.depthThreshold;
	if (historyAccepted) {
		var historyRadiance = max(history.rgb, vec3<f32>(0.0));
		if (current.a > 0.0) {
			historyRadiance = clampHistoryLuminance(
				historyRadiance,
				current.rgb,
				max(traceParams.historyClamp, 1.0)
			);
		}
		let blend = clamp(traceParams.historyWeight, 0.0, 0.98);
		current = mix(
			current,
			vec4<f32>(historyRadiance, clamp(history.a, 0.0, 1.0)),
			blend
		);
	}
	textureStore(
		traceOut,
		coord,
		vec4<f32>(max(current.rgb, vec3<f32>(0.0)), current.a)
	);
}

fn denoise(direction: vec2<f32>, coord: vec2<i32>, uv: vec2<f32>) {
	let centerDepth = textureSampleLevel(
		denoiseMotionDepth,
		denoiseSampler,
		uv,
		0.0
	).z;
	if (centerDepth <= 0.0) {
		textureStore(denoiseOut, coord, vec4<f32>(0.0));
		return;
	}
	let centerNormal = decodeWorldNormal(
		textureSampleLevel(
			denoiseNormalRoughMetal,
			denoiseSampler,
			uv,
			0.0
		).xy
	);
	let radius = i32(clamp(
		denoiseParams.radius,
		1.0,
		f32(MAX_DENOISE_RADIUS)
	));
	var radiance = vec3<f32>(0.0);
	var confidence = 0.0;
	var weightSum = 0.0;
	for (
		var offset = -MAX_DENOISE_RADIUS;
		offset <= MAX_DENOISE_RADIUS;
		offset = offset + 1
	) {
		if (abs(offset) > radius) {
			continue;
		}
		let sampleUv = clamp(
			uv +
				direction *
				f32(offset) *
				denoiseParams.invTraceSize,
			vec2<f32>(0.0),
			vec2<f32>(1.0)
		);
		let sampleDepth = textureSampleLevel(
			denoiseMotionDepth,
			denoiseSampler,
			sampleUv,
			0.0
		).z;
		if (sampleDepth <= 0.0) {
			continue;
		}
		let sampleNormal = decodeWorldNormal(
			textureSampleLevel(
				denoiseNormalRoughMetal,
				denoiseSampler,
				sampleUv,
				0.0
			).xy
		);
		let sampleValue = max(
			textureSampleLevel(
				denoiseSource,
				denoiseSampler,
				sampleUv,
				0.0
			),
			vec4<f32>(0.0)
		);
		let normalizedOffset = f32(offset) / f32(radius);
		let spatialWeight = exp(-0.5 * normalizedOffset * normalizedOffset);
		let relativeDepth =
			abs(sampleDepth - centerDepth) /
			max(max(sampleDepth, centerDepth), 1e-4);
		let depthWeight = exp(
			-relativeDepth * max(denoiseParams.depthPhi, 0.001)
		);
		let normalWeight = pow(
			max(dot(centerNormal, sampleNormal), 0.0),
			max(denoiseParams.normalPhi, 0.001)
		);
		let confidenceWeight = mix(0.05, 1.0, clamp(sampleValue.a, 0.0, 1.0));
		let weight =
			spatialWeight * depthWeight * normalWeight * confidenceWeight;
		radiance += sampleValue.rgb * weight;
		confidence += sampleValue.a * weight;
		weightSum += weight;
	}
	let inverseWeight = select(
		0.0,
		1.0 / max(weightSum, 1e-6),
		weightSum > 0.0
	);
	textureStore(
		denoiseOut,
		coord,
		vec4<f32>(
			max(radiance * inverseWeight, vec3<f32>(0.0)),
			clamp(confidence * inverseWeight, 0.0, 1.0)
		)
	);
}

@compute @workgroup_size(8, 8, 1)
fn csDenoiseHorizontal(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(denoiseOut);
	if (gid.x >= size.x || gid.y >= size.y) {
		return;
	}
	let coord = vec2<i32>(gid.xy);
	let uv =
		(vec2<f32>(gid.xy) + vec2<f32>(0.5)) *
		denoiseParams.invTraceSize;
	denoise(vec2<f32>(1.0, 0.0), coord, uv);
}

@compute @workgroup_size(8, 8, 1)
fn csDenoiseVertical(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(denoiseOut);
	if (gid.x >= size.x || gid.y >= size.y) {
		return;
	}
	let coord = vec2<i32>(gid.xy);
	let uv =
		(vec2<f32>(gid.xy) + vec2<f32>(0.5)) *
		denoiseParams.invTraceSize;
	denoise(vec2<f32>(0.0, 1.0), coord, uv);
}

@compute @workgroup_size(8, 8, 1)
fn csCompose(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(composeOut);
	if (gid.x >= size.x || gid.y >= size.y) {
		return;
	}
	let coord = vec2<i32>(gid.xy);
	let uv =
		(vec2<f32>(gid.xy) + vec2<f32>(0.5)) *
		composeParams.invFullSize;
	let scene = textureLoad(composeSceneColor, coord, 0);
	let centerDepth = textureSampleLevel(
		composeMotionDepth,
		composeSampler,
		uv,
		0.0
	).z;
	if (centerDepth <= 0.0) {
		textureStore(composeOut, coord, scene);
		return;
	}
	let centerNormalRoughMetal = textureSampleLevel(
		composeNormalRoughMetal,
		composeSampler,
		uv,
		0.0
	);
	let centerNormal = decodeWorldNormal(centerNormalRoughMetal.xy);
	let offsets = array<vec2<f32>, 5>(
		vec2<f32>(0.0, 0.0),
		vec2<f32>(1.0, 0.0),
		vec2<f32>(-1.0, 0.0),
		vec2<f32>(0.0, 1.0),
		vec2<f32>(0.0, -1.0)
	);
	var radiance = vec3<f32>(0.0);
	var weightSum = 0.0;
	for (var index = 0; index < 5; index = index + 1) {
		let sampleUv = clamp(
			uv + offsets[index] * composeParams.invTraceSize,
			vec2<f32>(0.0),
			vec2<f32>(1.0)
		);
		let sampleDepth = textureSampleLevel(
			composeMotionDepth,
			composeSampler,
			sampleUv,
			0.0
		).z;
		if (sampleDepth <= 0.0) {
			continue;
		}
		let sampleNormal = decodeWorldNormal(
			textureSampleLevel(
				composeNormalRoughMetal,
				composeSampler,
				sampleUv,
				0.0
			).xy
		);
		let relativeDepth =
			abs(sampleDepth - centerDepth) /
			max(max(sampleDepth, centerDepth), 1e-4);
		let depthWeight = exp(
			-relativeDepth * max(composeParams.depthPhi, 0.001)
		);
		let normalWeight = pow(
			max(dot(centerNormal, sampleNormal), 0.0),
			max(composeParams.normalPhi, 0.001)
		);
		let spatialWeight = select(0.75, 1.0, index == 0);
		let weight = spatialWeight * depthWeight * normalWeight;
		let sampleRadiance = textureSampleLevel(
			composeSSGI,
			composeSampler,
			sampleUv,
			0.0
		).rgb;
		radiance += max(sampleRadiance, vec3<f32>(0.0)) * weight;
		weightSum += weight;
	}
	let filteredRadiance = select(
		vec3<f32>(0.0),
		radiance / max(weightSum, 1e-6),
		weightSum > 0.0
	);
	let receiverAlbedo = max(
		textureSampleLevel(
			composeAlbedo,
			composeSampler,
			uv,
			0.0
		).rgb,
		vec3<f32>(0.0)
	);
	let metallic = clamp(centerNormalRoughMetal.w, 0.0, 1.0);
	let indirect = filteredRadiance *
		receiverAlbedo *
		(1.0 - metallic) *
		max(composeParams.intensity, 0.0);
	textureStore(
		composeOut,
		coord,
		vec4<f32>(max(scene.rgb + indirect, vec3<f32>(0.0)), scene.a)
	);
}
