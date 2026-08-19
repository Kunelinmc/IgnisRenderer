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
	stride: f32,
	intensity: f32,
	edgeFade: f32,
	maxSteps: f32,
	binarySearchSteps: f32,
	maxMip: f32,
	roughnessMipScale: f32,
	planeRefinementSteps: f32,
	_pad1: vec4<f32>,
}

@group(0) @binding(0) var backgroundColor: texture_2d<f32>;
@group(0) @binding(1) var transmissionSurface0: texture_2d<f32>;
@group(0) @binding(2) var transmissionSurface1: texture_2d<f32>;
@group(0) @binding(3) var transmissionSurface2: texture_2d<f32>;
@group(0) @binding(4) var opaqueMotionDepth: texture_2d<f32>;
@group(0) @binding(5) var hiZ: texture_2d<f32>;
@group(0) @binding(6) var linearSampler: sampler;
@group(0) @binding(7) var<uniform> traceParams: TraceParams;
@group(0) @binding(8) var outRefraction: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var opaqueNormal: texture_2d<f32>;

@group(1) @binding(0) var<uniform> frame: FrameCameraUniforms;

struct ComposeParams {
	invFullSize: vec2<f32>,
	intensity: f32,
	_pad0: f32,
}

@group(0) @binding(0) var composeScene: texture_2d<f32>;
@group(0) @binding(1) var composeRefraction: texture_2d<f32>;
@group(0) @binding(2) var composeTransmissionLighting: texture_2d<f32>;
@group(0) @binding(3) var composeSampler: sampler;
@group(0) @binding(4) var<uniform> composeParams: ComposeParams;
@group(0) @binding(5) var composeOut: texture_storage_2d<rgba16float, write>;

struct CameraBasis {
	right: vec3<f32>,
	up: vec3<f32>,
	backward: vec3<f32>,
	tanHalfFov: f32,
	aspect: f32,
	orthographic: f32,
}

struct RefractionResult {
	direction: vec3<f32>,
	valid: bool,
}

struct HiZTraceResult {
	hitUv: vec2<f32>,
	hit: bool,
	t: f32,
}

const ROUGH_TRANSMISSION_OFFSETS = array<vec2<f32>, 8>(
	vec2<f32>(1.0, 0.0),
	vec2<f32>(-1.0, 0.0),
	vec2<f32>(0.0, 1.0),
	vec2<f32>(0.0, -1.0),
	vec2<f32>(0.7071, 0.7071),
	vec2<f32>(-0.7071, 0.7071),
	vec2<f32>(0.7071, -0.7071),
	vec2<f32>(-0.7071, -0.7071)
);

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
	if (depth <= 1e-4) {
		return vec2<f32>(-1.0);
	}
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

fn isInsideScreen(uv: vec2<f32>) -> bool {
	return uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
}

fn refractViewDirection(v: vec3<f32>, n: vec3<f32>, ior: f32) -> RefractionResult {
	let cosThetaI = dot(v, n);
	let outside = cosThetaI > 0.0;
	let eta = select(ior, 1.0 / max(ior, 1.0), outside);
	let refractNormal = select(-n, n, outside);
	let absCosThetaI = abs(cosThetaI);
	let sin2ThetaT = eta * eta * (1.0 - absCosThetaI * absCosThetaI);
	if (sin2ThetaT > 1.0) {
		return RefractionResult(vec3<f32>(0.0), false);
	}
	let cosThetaT = sqrt(max(1.0 - sin2ThetaT, 0.0));
	let refraction = eta * -v + (eta * absCosThetaI - cosThetaT) * refractNormal;
	return RefractionResult(normalize(refraction), true);
}

fn sampleRoughTransmissionBackground(
	uv: vec2<f32>,
	roughness: f32,
	travelDistance: f32
) -> vec3<f32> {
	let alpha = roughness * roughness;
	let distanceScale = 1.0 + clamp(
		travelDistance / max(traceParams.maxDistance, 1.0),
		0.0,
		1.0
	) * 8.0;
	let radiusPixels = alpha * max(traceParams.roughnessMipScale, 0.0) * distanceScale;
	let radiusUv = traceParams.invTraceSize * radiusPixels;
	var color = textureSampleLevel(backgroundColor, linearSampler, uv, 0.0).rgb * 2.0;
	var weight = 2.0;
	for (var i: i32 = 0; i < 8; i = i + 1) {
		let sampleUv = clamp(
			uv + ROUGH_TRANSMISSION_OFFSETS[i] * radiusUv,
			vec2<f32>(0.0),
			vec2<f32>(1.0)
		);
		color += textureSampleLevel(backgroundColor, linearSampler, sampleUv, 0.0).rgb;
		weight += 1.0;
	}
	return color / weight;
}

fn refineHitWithPlane(
	worldPos: vec3<f32>,
	rayDir: vec3<f32>,
	initialUv: vec2<f32>
) -> HiZTraceResult {
	var result: HiZTraceResult;
	result.hit = false;
	result.hitUv = initialUv;
	result.t = 0.0;

	let refineSteps = i32(clamp(traceParams.planeRefinementSteps, 0.0, 8.0));
	let thickness = max(traceParams.thickness, 0.01);
	var uv = initialUv;

	for (var i: i32 = 0; i < 8; i = i + 1) {
		if (i >= refineSteps) { break; }
		if (!isInsideScreen(uv)) { break; }

		let motionDepth = textureSampleLevel(opaqueMotionDepth, linearSampler, uv, 0.0);
		let sceneDepth = motionDepth.z;
		if (sceneDepth <= 0.0) { break; }

		let normalSample = textureSampleLevel(opaqueNormal, linearSampler, uv, 0.0);
		let planeNormal = decodeNormal(normalSample.xy);
		let planePoint = getPosition(uv, sceneDepth);
		let denom = dot(rayDir, planeNormal);
		if (abs(denom) < 1e-4) { break; }

		let candidateT = dot(planePoint - worldPos, planeNormal) / denom;
		if (candidateT <= 0.0 || candidateT > traceParams.maxDistance) { break; }

		let candidatePos = worldPos + rayDir * candidateT;
		let candidateUv = worldToUv(candidatePos);
		if (!isInsideScreen(candidateUv)) { break; }

		let candidateDepth =
			textureSampleLevel(opaqueMotionDepth, linearSampler, candidateUv, 0.0).z;
		if (candidateDepth <= 0.0) { break; }

		let rayDepth = worldToLinearDepth(candidatePos);
		if (abs(rayDepth - candidateDepth) > thickness) { break; }

		result.hit = true;
		result.hitUv = candidateUv;
		result.t = candidateT;
		uv = candidateUv;
	}

	return result;
}

fn traceHiZ(worldPos: vec3<f32>, rayDir: vec3<f32>) -> HiZTraceResult {
	var result: HiZTraceResult;
	result.hit = false;
	result.hitUv = vec2<f32>(0.0);
	result.t = 0.0;

	let maxSteps = i32(clamp(traceParams.maxSteps, 1.0, 256.0));
	let maxMip = i32(clamp(traceParams.maxMip, 0.0, 12.0));
	let thickness = max(traceParams.thickness, 0.01);
	let stride = max(traceParams.stride, 0.1);

	var t = max(thickness, stride);
	var mip: i32 = 0;
	var missT = 0.0;

	for (var step: i32 = 0; step < 256; step = step + 1) {
		if (step >= maxSteps) { break; }
		if (t > traceParams.maxDistance) { break; }

		let samplePos = worldPos + rayDir * t;
		let suv = worldToUv(samplePos);
		if (!isInsideScreen(suv)) { break; }

		let rayDepth = worldToLinearDepth(samplePos);
		if (rayDepth <= 0.0) { break; }

		let safeMip = clamp(mip, 0, maxMip);
		let hizSample = textureSampleLevel(hiZ, linearSampler, suv, f32(safeMip));
		let minDepth = hizSample.x;
		let maxDepth = max(hizSample.y, hizSample.x);

		if (minDepth <= 0.0) {
			t += stride * pow(2.0, f32(safeMip));
			continue;
		}

		if (rayDepth > maxDepth + thickness) {
			missT = t;
			mip = min(mip + 1, maxMip);
			t += stride * pow(2.0, f32(safeMip));
			continue;
		}

		if (rayDepth < minDepth - thickness) {
			missT = t;
			t += stride * pow(2.0, f32(safeMip));
			continue;
		}

		if (mip > 0) {
			mip = mip - 1;
			continue;
		}

		result.hit = true;
		result.hitUv = suv;
		result.t = t;
		break;
	}

	if (result.hit) {
		let planeResult = refineHitWithPlane(worldPos, rayDir, result.hitUv);
		if (planeResult.hit) {
			return planeResult;
		}

		var refineMin = missT;
		var refineMax = result.t;
		let refineSteps = i32(clamp(traceParams.binarySearchSteps, 0.0, 16.0));
		for (var j: i32 = 0; j < 16; j = j + 1) {
			if (j >= refineSteps) { break; }
			let midT = (refineMin + refineMax) * 0.5;
			let refinePos = worldPos + rayDir * midT;
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

@compute @workgroup_size(8, 8, 1)
fn csTrace(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outRefraction);
	if (gid.x >= size.x || gid.y >= size.y) { return; }

	let pixel = vec2<f32>(gid.xy);
	let uv = (pixel + vec2<f32>(0.5)) * traceParams.invTraceSize;
	let surface0 = textureSampleLevel(transmissionSurface0, linearSampler, uv, 0.0);
	let surface1 = textureSampleLevel(transmissionSurface1, linearSampler, uv, 0.0);
	let surface2 = textureSampleLevel(transmissionSurface2, linearSampler, uv, 0.0);

	let depth = surface0.z;
	let transmission = clamp(surface0.w, 0.0, 1.0);
	let coverage = clamp(surface2.w, 0.0, 1.0);
	if (depth <= 0.0 || transmission <= 0.0001 || coverage <= 0.0001) {
		textureStore(outRefraction, vec2<i32>(gid.xy), vec4<f32>(0.0));
		return;
	}

	let worldPos = getPosition(uv, depth);
	let normal = decodeNormal(surface0.xy);
	let viewDir = normalize(frame.cameraPosition.xyz - worldPos);
	let ior = max(surface1.x, 1.0);
	let materialThickness = max(surface1.y, 0.0);
	let roughness = clamp(surface1.z, 0.0, 1.0);
	let tint = clamp(surface2.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
	let fresnelR = clamp(surface1.w, 0.0, 1.0);
	let incidentDirection = -viewDir;
	let frontSide = dot(viewDir, normal) > 0.0;
	let entryRefraction = refractViewDirection(viewDir, normal, ior);
	var rayOrigin = worldPos;
	var rayDirection = incidentDirection;
	var pathLength = 0.0;
	var totalInternalReflection = !entryRefraction.valid;

	if (entryRefraction.valid) {
		if (frontSide && materialThickness > 1e-5) {
			let insideCos = max(-dot(entryRefraction.direction, normal), 1e-4);
			pathLength = materialThickness / insideCos;
			rayOrigin = worldPos + entryRefraction.direction * pathLength;
			let exitRefraction = refractViewDirection(
				-entryRefraction.direction,
				-normal,
				ior
			);
			totalInternalReflection = !exitRefraction.valid;
			rayDirection = select(
				reflect(entryRefraction.direction, -normal),
				exitRefraction.direction,
				exitRefraction.valid
			);
		} else if (frontSide) {
			// Coincident parallel interfaces cancel angular refraction.
			rayDirection = incidentDirection;
		} else {
			rayDirection = entryRefraction.direction;
		}
	} else {
		// Recover the transmission lobe as reflection during TIR.
		rayDirection = normalize(reflect(incidentDirection, normal));
	}

	let traceResult = traceHiZ(rayOrigin, rayDirection);
	var sampleUv = uv;
	var hitConfidence = 0.0;
	var traceDistance = 0.0;
	if (traceResult.hit) {
		sampleUv = traceResult.hitUv;
		traceDistance = traceResult.t;
		let distanceConfidence =
			1.0 - clamp(traceResult.t / max(traceParams.maxDistance, 1.0), 0.0, 1.0);
		let edgeDistance = min(
			min(sampleUv.x, 1.0 - sampleUv.x),
			min(sampleUv.y, 1.0 - sampleUv.y)
		);
		let edgeConfidence =
			clamp(edgeDistance / max(traceParams.edgeFade, 1e-4), 0.0, 1.0);
		hitConfidence = edgeConfidence * distanceConfidence;
	}

	let fallbackBackground = sampleRoughTransmissionBackground(
		uv,
		roughness,
		pathLength
	);
	let tracedBackground = sampleRoughTransmissionBackground(
		sampleUv,
		roughness,
		pathLength + traceDistance
	);
	let background = mix(fallbackBackground, tracedBackground, hitConfidence);
	let opticalWeight = clamp(
		transmission * (1.0 - fresnelR) * traceParams.intensity,
		0.0,
		1.0
	);
	textureStore(
		outRefraction,
		vec2<i32>(gid.xy),
		vec4<f32>(
			max(background * tint * opticalWeight, vec3<f32>(0.0)),
			coverage * select(max(hitConfidence, 0.05), 1.0, totalInternalReflection)
		)
	);
}

@compute @workgroup_size(8, 8, 1)
fn csCompose(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(composeOut);
	if (gid.x >= size.x || gid.y >= size.y) { return; }

	let coord = vec2<i32>(gid.xy);
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * composeParams.invFullSize;
	let scene = textureLoad(composeScene, coord, 0);
	let raw = textureSampleLevel(composeRefraction, composeSampler, uv, 0.0);
	let lighting = textureLoad(composeTransmissionLighting, coord, 0);
	let coverage = clamp(lighting.a, 0.0, 1.0);
	let color = scene.rgb * (1.0 - coverage) + lighting.rgb + raw.rgb * coverage;
	textureStore(composeOut, coord, vec4<f32>(max(color, vec3<f32>(0.0)), scene.a));
}
