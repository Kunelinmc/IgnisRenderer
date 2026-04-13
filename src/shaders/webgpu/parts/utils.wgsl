#import <ignis/color/srgb>
#import <ignis/postprocess/fog>
fn saturate(value: f32) -> f32 {
	return clamp(value, 0.0, 1.0);
}

fn isFiniteF32(value: f32) -> bool {
	return value == value && abs(value) <= 3.402823466e+38;
}

fn safeNormalize(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
	let len = length(value);
	return select(fallback, value / max(len, EPSILON), len > EPSILON);
}

fn transformUV(slotIndex: u32, uv0: vec2<f32>, uv1: vec2<f32>) -> vec2<f32> {
	let transformA = model.textureTransformA[slotIndex];
	let transformB = model.textureTransformB[slotIndex];
	var uv = select(uv0, uv1, transformB.y > 0.5);
	uv = uv * transformA.zw;

	let rotation = transformB.x;
	if (abs(rotation) > EPSILON) {
		let c = cos(rotation);
		let s = sin(rotation);
		uv = vec2<f32>(uv.x * c - uv.y * s, uv.x * s + uv.y * c);
	}

	return uv + transformA.xy;
}

fn sampleLinearTexture(
	textureRef: texture_2d<f32>,
	samplerRef: sampler,
	slotIndex: u32,
	uv0: vec2<f32>,
	uv1: vec2<f32>
) -> vec4<f32> {
	return textureSample(textureRef, samplerRef, transformUV(slotIndex, uv0, uv1));
}

fn sampleColorTexture(
	textureRef: texture_2d<f32>,
	samplerRef: sampler,
	slotIndex: u32,
	uv0: vec2<f32>,
	uv1: vec2<f32>
) -> vec4<f32> {
	let sampled = sampleLinearTexture(textureRef, samplerRef, slotIndex, uv0, uv1);
	let isLinear = model.textureTransformB[slotIndex].z > 0.5;
	return select(vec4<f32>(srgbToLinear(sampled.rgb), sampled.a), sampled, isLinear);
}

fn applyNormalMap(
	baseNormal: vec3<f32>,
	tangent: vec4<f32>,
	normalSample: vec3<f32>,
	scale: f32
) -> vec3<f32> {
	let n = safeNormalize(baseNormal, vec3<f32>(0.0, 0.0, 1.0));
	let tangentLenSq = dot(tangent.xyz, tangent.xyz);
	let hasValidTangent = tangentLenSq > 1e-12 && abs(tangent.w) > EPSILON;

	if (!hasValidTangent) {
		return n;
	}

	var t = tangent.xyz - n * dot(n, tangent.xyz);
	let tLen = length(t);
	if (tLen <= EPSILON) {
		return n;
	}

	t = t / tLen;
	let handedness = select(1.0, -1.0, tangent.w < 0.0);
	let b = cross(n, t) * handedness;
	let tangentNormal = vec3<f32>(
		(normalSample.x * 2.0 - 1.0) * scale,
		(normalSample.y * 2.0 - 1.0) * scale,
		normalSample.z * 2.0 - 1.0
	);

	return safeNormalize(
		t * tangentNormal.x + b * tangentNormal.y + n * tangentNormal.z,
		n
	);
}

fn pointAttenuation(distanceSq: f32, range: f32) -> f32 {
	let rangeSq = max(range * range, EPSILON);
	let rangeFactor = distanceSq / rangeSq;
	let smoothFactor = max(0.0, 1.0 - rangeFactor * rangeFactor);
	return (smoothFactor * smoothFactor) / (distanceSq + 1.0);
}

fn spotAttenuation(cosTheta: f32, outerCos: f32, innerCos: f32) -> f32 {
	if (cosTheta < outerCos) {
		return 0.0;
	}

	let cutoffRange = max(innerCos - outerCos, EPSILON);
	return saturate((cosTheta - outerCos) / cutoffRange);
}

fn distributionGGX(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
	let a = roughness * roughness;
	let a2 = a * a;
	let nDotH = max(dot(n, h), 0.0);
	let nDotH2 = nDotH * nDotH;
	let denom = nDotH2 * (a2 - 1.0) + 1.0;
	return a2 / max(PI * denom * denom, 0.0001);
}

fn geometrySchlickGGX(nDotValue: f32, roughness: f32) -> f32 {
	let r = roughness + 1.0;
	let k = (r * r) / 8.0;
	return nDotValue / max(nDotValue * (1.0 - k) + k, 0.0001);
}

fn geometrySmith(nDotV: f32, nDotL: f32, roughness: f32) -> f32 {
	return geometrySchlickGGX(nDotV, roughness) * geometrySchlickGGX(nDotL, roughness);
}

fn geometrySmithClearcoat(nDotV: f32, nDotL: f32, roughness: f32) -> f32 {
	let a = roughness * roughness;
	let k = a / 2.0;
	let g1V = nDotV / max(nDotV * (1.0 - k) + k, 0.0001);
	let g1L = nDotL / max(nDotL * (1.0 - k) + k, 0.0001);
	return g1V * g1L;
}

fn fresnelSchlick(cosTheta: f32, f0: vec3<f32>) -> vec3<f32> {
	return f0 + (vec3<f32>(1.0) - f0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

fn fresnelSchlickScalar(cosTheta: f32, f0: f32) -> f32 {
	return f0 + (1.0 - f0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

fn resolveTransmissionAlpha(
	baseAlpha: f32,
	transmission: f32,
	nDotV: f32,
	f0: vec3<f32>
) -> f32 {
	let clampedTransmission = clamp(transmission, 0.0, 1.0);
	if (clampedTransmission <= EPSILON) {
		return clamp(baseAlpha, 0.0, 1.0);
	}

	let fresnel = fresnelSchlick(nDotV, f0);
	let fresnelAverage = clamp(
		(fresnel.x + fresnel.y + fresnel.z) * (1.0 / 3.0),
		0.0,
		1.0
	);
	let floorAlpha = max(0.12, fresnelAverage);
	let blended =
		baseAlpha * (1.0 - clampedTransmission) +
		floorAlpha * clampedTransmission;
	return clamp(max(floorAlpha, blended), 0.0, 1.0);
}

fn distributionCharlie(nDotH: f32, roughness: f32) -> f32 {
	let invAlpha = 1.0 / max(roughness * roughness, 1e-6);
	let cos2h = nDotH * nDotH;
	let sin2h = max(1.0 - cos2h, 0.0078125);
	return ((2.0 + invAlpha) * pow(sin2h, invAlpha * 0.5)) / (2.0 * PI);
}

fn visibilityAshikhmin(nDotL: f32, nDotV: f32) -> f32 {
	return 1.0 / max(4.0 * (nDotL + nDotV - nDotL * nDotV), 0.0001);
}

fn reflectViewDirection(n: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
	return safeNormalize(reflect(-v, n), n);
}

fn refractViewDirection(v: vec3<f32>, n: vec3<f32>, ior: f32) -> RefractionResult {
	let cosThetaI = dot(v, n);
	let outside = cosThetaI > 0.0;
	let eta = select(ior, 1.0 / max(ior, 1.0), outside);
	let refractNormal = select(-n, n, outside);
	let absCosThetaI = abs(cosThetaI);
	let sin2ThetaT = eta * eta * (1.0 - absCosThetaI * absCosThetaI);

	if (sin2ThetaT > 1.0) {
		return RefractionResult(vec3<f32>(0.0), 0.0);
	}

	let cosThetaT = sqrt(max(1.0 - sin2ThetaT, 0.0));
	let refraction = eta * -v + (eta * absCosThetaI - cosThetaT) * refractNormal;
	return RefractionResult(safeNormalize(refraction, -v), 1.0);
}

fn encodeOutput(color: vec3<f32>) -> vec3<f32> {
	return color;
}

fn encodeNormalForGBuffer(normal: vec3<f32>) -> vec2<f32> {
	let right = frame.skyboxBasisRight.xyz;
	let up = frame.skyboxBasisUp.xyz;
	let backward = frame.skyboxBasisBackward.xyz;
	let vn = vec3<f32>(dot(normal, right), dot(normal, up), dot(normal, backward));
	return vn.xy * 0.5 + vec2<f32>(0.5, 0.5);
}

fn buildSceneOutput(
	sceneLinear: vec3<f32>,
	alpha: f32,
	albedo: vec3<f32>,
	worldNormal: vec3<f32>,
	roughness: f32,
	metalness: f32,
	emissive: vec3<f32>,
	occlusion: f32,
	motion: vec2<f32>,
	linearDepth: f32
) -> SceneFragmentOutput {
	let fogMode = i32(floor(fog.fogParams0.x + 0.5));
	let fogFactor = ignisComputeFogFactor(
		fogMode,
		max(linearDepth, 0.0),
		fog.fogParams0.y,
		fog.fogParams0.z,
		fog.fogParams0.w,
		fog.fogParams1.w
	);
	let foggedSceneLinear = max(
		mix(sceneLinear, fog.fogParams1.rgb, fogFactor),
		vec3<f32>(0.0)
	);
	var output: SceneFragmentOutput;
	output.sceneColor = vec4<f32>(
		clamp(foggedSceneLinear, vec3<f32>(0.0), vec3<f32>(65504.0)),
		clamp(alpha, 0.0, 1.0)
	);
	output.gAlbedoAlpha = vec4<f32>(
		clamp(albedo, vec3<f32>(0.0), vec3<f32>(1.0)),
		clamp(alpha, 0.0, 1.0)
	);
	output.gNormalRoughMetal = vec4<f32>(
		encodeNormalForGBuffer(worldNormal),
		clamp(roughness, 0.0, 1.0),
		clamp(metalness, 0.0, 1.0)
	);
	output.gEmissiveOcclusion = vec4<f32>(
		clamp(emissive, vec3<f32>(0.0), vec3<f32>(65504.0)),
		clamp(occlusion, 0.0, 1.0)
	);
	output.gMotionDepth = vec4<f32>(
		clamp(motion, vec2<f32>(-1.0), vec2<f32>(1.0)),
		max(linearDepth, 0.0),
		0.0
	);
	return output;
}

fn useSHAmbient() -> bool {
	return frame.environmentOptionsA.x > 0.5 && frame.environmentOptionsA.y > 0.5;
}

fn hasEnvSpecular() -> bool {
	return frame.environmentOptionsA.w > 0.5;
}

fn hasBRDFLUT() -> bool {
	return frame.environmentOptionsB.x > 0.5;
}

fn envSpecularMaxMipLevel() -> f32 {
	return max(frame.environmentOptionsB.y, 0.0);
}

fn reflectionProbeCount() -> u32 {
	let count = u32(max(frame.lightCounts.w + 0.5, 0.0));
	return min(count, MAX_REFLECTION_PROBES);
}

fn evalSHBasis(direction: vec3<f32>) -> array<f32, 16> {
	let x = direction.x;
	let y = direction.y;
	let z = direction.z;

	let y00 = 0.282095;
	let y1_1 = 0.488603 * x;
	let y10 = 0.488603 * y;
	let y11 = 0.488603 * z;
	let y2_2 = 1.092548 * x * z;
	let y2_1 = 1.092548 * x * y;
	let y20 = 0.315392 * (3.0 * y * y - 1.0);
	let y21 = 1.092548 * y * z;
	let y22 = 0.546274 * (x * x - z * z);
	let y3_3 = 0.590835 * x * (x * x - 3.0 * z * z);
	let y3_2 = 2.893641 * x * y * z;
	let y3_1 = 0.457619 * x * (5.0 * y * y - 1.0);
	let y30 = 0.373176 * y * (5.0 * y * y - 3.0);
	let y31 = 0.457619 * z * (5.0 * y * y - 1.0);
	let y32 = 1.446821 * y * (x * x - z * z);
	let y33 = 0.590835 * z * (3.0 * x * x - z * z);

	return array<f32, 16>(
		y00,
		y1_1,
		y10,
		y11,
		y2_2,
		y2_1,
		y20,
		y21,
		y22,
		y3_3,
		y3_2,
		y3_1,
		y30,
		y31,
		y32,
		y33
	);
}

fn calculateIrradianceFromSH(normal: vec3<f32>) -> vec3<f32> {
	let basis = evalSHBasis(normal);
	let c1 = PI;
	let c2 = (2.0 * PI) / 3.0;
	let c3 = PI / 4.0;
	var result = vec3<f32>(0.0);
	for (var i: u32 = 0u; i < 16u; i = i + 1u) {
		var factor = select(0.0, c3, i >= 4u && i < 9u);
		factor = select(factor, c2, i >= 1u && i < 4u);
		factor = select(factor, c1, i == 0u);
		result += frame.shAmbientCoeffs[i].xyz * basis[i] * factor;
	}
	return max(result, vec3<f32>(0.0));
}

fn sampleSHRadiance(direction: vec3<f32>) -> vec3<f32> {
	let basis = evalSHBasis(direction);
	var result = vec3<f32>(0.0);
	for (var i: u32 = 0u; i < 16u; i = i + 1u) {
		result += frame.shAmbientCoeffs[i].xyz * basis[i];
	}
	return max(result, vec3<f32>(0.0));
}

fn directionToEquirectUV(direction: vec3<f32>) -> vec2<f32> {
	let phi = atan2(direction.x, direction.z);
	let theta = acos(clamp(direction.y, -1.0, 1.0));
	return vec2<f32>((phi + PI) / (2.0 * PI), theta / PI);
}

fn sampleEnvironmentSpecularFromDirection(
	direction: vec3<f32>,
	roughness: f32,
	layer: u32,
	layerCount: u32
) -> vec3<f32> {
	var uv = directionToEquirectUV(safeNormalize(direction, vec3<f32>(0.0, 1.0, 0.0)));
	if (layerCount > 1u) {
		uv.x = (uv.x + f32(layer)) / f32(layerCount);
	}
	let level = clamp(roughness, 0.0, 1.0) * envSpecularMaxMipLevel();
	return textureSampleLevel(envSpecularTexture, envSpecularSampler, uv, level).rgb;
}

fn worldToProbePoint(probe: ReflectionProbeData, worldPosition: vec3<f32>) -> vec3<f32> {
	return vec3<f32>(
		dot(probe.worldToProbeRow0.xyz, worldPosition) + probe.worldToProbeRow0.w,
		dot(probe.worldToProbeRow1.xyz, worldPosition) + probe.worldToProbeRow1.w,
		dot(probe.worldToProbeRow2.xyz, worldPosition) + probe.worldToProbeRow2.w
	);
}

fn worldToProbeDirection(
	probe: ReflectionProbeData,
	worldDirection: vec3<f32>
) -> vec3<f32> {
	return vec3<f32>(
		dot(probe.worldToProbeRow0.xyz, worldDirection),
		dot(probe.worldToProbeRow1.xyz, worldDirection),
		dot(probe.worldToProbeRow2.xyz, worldDirection)
	);
}

fn probeToWorldPoint(probe: ReflectionProbeData, probePosition: vec3<f32>) -> vec3<f32> {
	return vec3<f32>(
		dot(probe.probeToWorldRow0.xyz, probePosition) + probe.probeToWorldRow0.w,
		dot(probe.probeToWorldRow1.xyz, probePosition) + probe.probeToWorldRow1.w,
		dot(probe.probeToWorldRow2.xyz, probePosition) + probe.probeToWorldRow2.w
	);
}

fn computeReflectionProbeMetric(
	worldPosition: vec3<f32>,
	probe: ReflectionProbeData
) -> f32 {
	let localPosition = worldToProbePoint(probe, worldPosition);
	let shapeCode = probe.dataB.w;
	if (shapeCode > 0.5) {
		return max(
			max(
				abs(localPosition.x) * probe.dataA.x,
				abs(localPosition.y) * probe.dataA.y
			),
			abs(localPosition.z) * probe.dataA.z
		);
	}
	return length(localPosition) * probe.dataA.w;
}

fn computeReflectionProbeWeight(metric: f32, probe: ReflectionProbeData) -> f32 {
	let safeBlendDistance = max(probe.dataC.y, 1e-5);
	let x = clamp((metric - 1.0) / safeBlendDistance, 0.0, 1.0);
	var weight = 1.0 - smoothstep(0.0, 1.0, x);
	let blendExponent = max(probe.dataC.z, 0.01);
	if (abs(blendExponent - 1.0) > 1e-5) {
		weight = pow(max(weight, 0.0), blendExponent);
	}
	return weight;
}

fn isBetterReflectionProbeCandidate(
	candidateWeight: f32,
	candidateIndex: i32,
	currentWeight: f32,
	currentIndex: i32
) -> bool {
	if (candidateWeight > currentWeight + 1e-6) {
		return true;
	}
	if (abs(candidateWeight - currentWeight) <= 1e-6 && candidateIndex < currentIndex) {
		return true;
	}
	return false;
}

fn selectTopTwoReflectionProbes(worldPosition: vec3<f32>) -> vec4<f32> {
	let count = reflectionProbeCount();
	var firstIndex: i32 = -1;
	var secondIndex: i32 = -1;
	var firstWeight = 0.0;
	var secondWeight = 0.0;

	for (var i: u32 = 0u; i < count; i = i + 1u) {
		let probe = frame.reflectionProbes[i];
		let metric = computeReflectionProbeMetric(worldPosition, probe);
		let weight = computeReflectionProbeWeight(metric, probe);
		if (!isFiniteF32(weight) || weight <= 1e-6) {
			continue;
		}

		let index = i32(i);
		if (
			firstIndex < 0 ||
			isBetterReflectionProbeCandidate(weight, index, firstWeight, firstIndex)
		) {
			secondIndex = firstIndex;
			secondWeight = firstWeight;
			firstIndex = index;
			firstWeight = weight;
			continue;
		}

		if (
			secondIndex < 0 ||
			isBetterReflectionProbeCandidate(weight, index, secondWeight, secondIndex)
		) {
			secondIndex = index;
			secondWeight = weight;
		}
	}

	if (firstIndex < 0) {
		return vec4<f32>(-1.0, -1.0, 0.0, 0.0);
	}

	let sumWeight = firstWeight + max(secondWeight, 0.0);
	if (sumWeight <= 1e-6) {
		return vec4<f32>(f32(firstIndex), -1.0, 1.0, 0.0);
	}

	if (secondIndex < 0) {
		return vec4<f32>(f32(firstIndex), -1.0, 1.0, 0.0);
	}

	return vec4<f32>(
		f32(firstIndex),
		f32(secondIndex),
		firstWeight / sumWeight,
		secondWeight / sumWeight
	);
}

fn intersectReflectionProbeBox(
	localOrigin: vec3<f32>,
	localDirection: vec3<f32>,
	probe: ReflectionProbeData
) -> vec4<f32> {
	let halfExtents = vec3<f32>(
		1.0 / max(probe.dataA.x, 1e-5),
		1.0 / max(probe.dataA.y, 1e-5),
		1.0 / max(probe.dataA.z, 1e-5)
	);
	var tMin = -1e20;
	var tMax = 1e20;

	if (abs(localDirection.x) <= EPSILON) {
		if (localOrigin.x < -halfExtents.x || localOrigin.x > halfExtents.x) {
			return vec4<f32>(0.0, 0.0, 0.0, 0.0);
		}
	} else {
		let invDirection = 1.0 / localDirection.x;
		var t0 = (-halfExtents.x - localOrigin.x) * invDirection;
		var t1 = (halfExtents.x - localOrigin.x) * invDirection;
		if (t0 > t1) {
			let swap = t0;
			t0 = t1;
			t1 = swap;
		}
		tMin = max(tMin, t0);
		tMax = min(tMax, t1);
	}

	if (abs(localDirection.y) <= EPSILON) {
		if (localOrigin.y < -halfExtents.y || localOrigin.y > halfExtents.y) {
			return vec4<f32>(0.0, 0.0, 0.0, 0.0);
		}
	} else {
		let invDirection = 1.0 / localDirection.y;
		var t0 = (-halfExtents.y - localOrigin.y) * invDirection;
		var t1 = (halfExtents.y - localOrigin.y) * invDirection;
		if (t0 > t1) {
			let swap = t0;
			t0 = t1;
			t1 = swap;
		}
		tMin = max(tMin, t0);
		tMax = min(tMax, t1);
	}

	if (abs(localDirection.z) <= EPSILON) {
		if (localOrigin.z < -halfExtents.z || localOrigin.z > halfExtents.z) {
			return vec4<f32>(0.0, 0.0, 0.0, 0.0);
		}
	} else {
		let invDirection = 1.0 / localDirection.z;
		var t0 = (-halfExtents.z - localOrigin.z) * invDirection;
		var t1 = (halfExtents.z - localOrigin.z) * invDirection;
		if (t0 > t1) {
			let swap = t0;
			t0 = t1;
			t1 = swap;
		}
		tMin = max(tMin, t0);
		tMax = min(tMax, t1);
	}

	if (tMax < max(tMin, 0.0)) {
		return vec4<f32>(0.0, 0.0, 0.0, 0.0);
	}
	let t = select(tMax, tMin, tMin > EPSILON);
	if (!isFiniteF32(t) || t <= EPSILON) {
		return vec4<f32>(0.0, 0.0, 0.0, 0.0);
	}
	let hit = localOrigin + localDirection * t;
	return vec4<f32>(hit, 1.0);
}

fn intersectReflectionProbeSphere(
	localOrigin: vec3<f32>,
	localDirection: vec3<f32>,
	probe: ReflectionProbeData
) -> vec4<f32> {
	let radius = 1.0 / max(probe.dataA.w, 1e-5);
	let b = dot(localOrigin, localDirection);
	let c = dot(localOrigin, localOrigin) - radius * radius;
	let discriminant = b * b - c;
	if (discriminant < 0.0) {
		return vec4<f32>(0.0, 0.0, 0.0, 0.0);
	}

	let sqrtDiscriminant = sqrt(discriminant);
	let t0 = -b - sqrtDiscriminant;
	let t1 = -b + sqrtDiscriminant;
	var t = 1e20;
	if (t0 > EPSILON) {
		t = min(t, t0);
	}
	if (t1 > EPSILON) {
		t = min(t, t1);
	}
	if (!isFiniteF32(t) || t >= 1e19) {
		return vec4<f32>(0.0, 0.0, 0.0, 0.0);
	}
	let hit = localOrigin + localDirection * t;
	return vec4<f32>(hit, 1.0);
}

fn computeReflectionProbeParallaxDirection(
	worldPosition: vec3<f32>,
	reflectionDirection: vec3<f32>,
	probe: ReflectionProbeData
) -> vec3<f32> {
	let fallback = safeNormalize(reflectionDirection, vec3<f32>(0.0, 0.0, 1.0));
	let parallaxMode = i32(probe.dataC.x + 0.5);
	if (parallaxMode <= 0) {
		return fallback;
	}

	let localOrigin = worldToProbePoint(probe, worldPosition);
	let localDirection = safeNormalize(
		worldToProbeDirection(probe, fallback),
		fallback
	);

	let hitLocal =
		select(
			intersectReflectionProbeSphere(localOrigin, localDirection, probe),
			intersectReflectionProbeBox(localOrigin, localDirection, probe),
			parallaxMode == 1
		);
	if (hitLocal.w < 0.5) {
		return fallback;
	}

	let worldHit = probeToWorldPoint(probe, hitLocal.xyz);
	let corrected = worldHit - probe.dataB.xyz;
	return safeNormalize(corrected, fallback);
}

fn sampleEnvironmentSpecular(
	direction: vec3<f32>,
	roughness: f32,
	worldPosition: vec3<f32>
) -> vec3<f32> {
	if (!hasEnvSpecular()) {
		return vec3<f32>(0.0);
	}

	let normalizedDirection = safeNormalize(direction, vec3<f32>(0.0, 1.0, 0.0));
	let probeCount = reflectionProbeCount();
	if (probeCount == 0u) {
		return sampleEnvironmentSpecularFromDirection(
			normalizedDirection,
			roughness,
			0u,
			1u
		);
	}

	let selection = selectTopTwoReflectionProbes(worldPosition);
	if (selection.x < 0.0) {
		return vec3<f32>(0.0);
	}

	let firstIndex = u32(max(selection.x, 0.0));
	let firstProbe = frame.reflectionProbes[firstIndex];
	let firstDirection = computeReflectionProbeParallaxDirection(
		worldPosition,
		normalizedDirection,
		firstProbe
	);
	let firstLayer = u32(max(firstProbe.dataC.w + 0.5, 0.0));
	let firstSample = sampleEnvironmentSpecularFromDirection(
		firstDirection,
		roughness,
		firstLayer,
		probeCount
	);

	if (selection.y < 0.0 || selection.w <= 1e-6) {
		return firstSample;
	}

	let secondIndex = u32(max(selection.y, 0.0));
	let secondProbe = frame.reflectionProbes[secondIndex];
	let secondDirection = computeReflectionProbeParallaxDirection(
		worldPosition,
		normalizedDirection,
		secondProbe
	);
	let secondLayer = u32(max(secondProbe.dataC.w + 0.5, 0.0));
	let secondSample = sampleEnvironmentSpecularFromDirection(
		secondDirection,
		roughness,
		secondLayer,
		probeCount
	);

	return firstSample * selection.z + secondSample * selection.w;
}

fn sampleBRDFLUT(nDotV: f32, roughness: f32) -> vec2<f32> {
	let nv = clamp(nDotV, 0.0, 1.0);
	let r = clamp(roughness, 0.0, 1.0);
	let a = r * r;
	let k = a * 0.5;
	let visibility = nv / max(nv * (1.0 - k) + k, 0.0001);
	let grazingBias = pow(1.0 - nv, 5.0) * (0.04 + (1.0 - r) * 0.2);
	return vec2<f32>(visibility, grazingBias);
}

fn clampShadowTexelCoord(coord: vec2<i32>, size: i32) -> vec2<i32> {
	let maxCoord = max(size - 1, 0);
	return vec2<i32>(
		clamp(coord.x, 0, maxCoord),
		clamp(coord.y, 0, maxCoord)
	);
}

fn loadShadowDepthTexel(coord: vec2<i32>) -> f32 {
	return textureLoad(shadowAtlas, coord, 0);
}

fn sampleShadowVisibility(
	shadowType: u32,
	index: u32,
	shadowData: ShadowData,
	worldPosition: vec3<f32>,
	normal: vec3<f32>,
	lightDirection: vec3<f32>
) -> f32 {
	if (frame.options.z < 0.5 || shadowData.paramsA.x < 0.5) {
		return 1.0;
	}

	let shadowSize = max(i32(shadowData.paramsB.z + 0.5), 1);
	let atlasTileSize = max(i32(shadowData.paramsB.w + 0.5), shadowSize);
	let slopeBias = max(shadowData.paramsC.x, 0.0);
	let maxNormalBias = max(shadowData.paramsA.z, 0.0);
	let minNormalBias = max(shadowData.paramsA.w, 0.0);
	let cosTheta = max(dot(normal, lightDirection), 0.0);
	let bias = max(shadowData.paramsA.y + slopeBias * (1.0 - cosTheta), 0.0);
	let normalBias = minNormalBias + (maxNormalBias - minNormalBias) * (1.0 - cosTheta);
	let shadowWorldPosition = worldPosition + normal * normalBias;
	let shadowClip = shadowData.viewProjection * vec4<f32>(shadowWorldPosition, 1.0);
	if (shadowClip.w <= EPSILON) {
		return 1.0;
	}

	let invW = 1.0 / shadowClip.w;
	let shadowNdc = shadowClip.xyz * invW;
	let shadowUv = vec2<f32>(
		shadowNdc.x * 0.5 + 0.5,
		0.5 - shadowNdc.y * 0.5
	);
	let currentDepth = shadowNdc.z * 0.5 + 0.5;
	if (
		shadowUv.x < 0.0 ||
		shadowUv.x > 1.0 ||
		shadowUv.y < 0.0 ||
		shadowUv.y > 1.0 ||
		currentDepth < 0.0 ||
		currentDepth > 1.0
	) {
		return 1.0;
	}

	let pcfRadius = max(shadowData.paramsB.x, 1.0);
	let texelPosition = shadowUv * vec2<f32>(f32(shadowSize - 1), f32(shadowSize - 1));
	let atlasDimensions = textureDimensions(shadowAtlas);
	let atlasWidth = max(i32(atlasDimensions.x), 1);
	let atlasColumns = max(atlasWidth / max(atlasTileSize, 1), 1);
	let shadowGlobalIndex = select(
		index,
		MAX_DIRECTIONAL_LIGHTS + index,
		shadowType == 1u
	);
	let tileX = i32(shadowGlobalIndex % u32(atlasColumns));
	let tileY = i32(shadowGlobalIndex / u32(atlasColumns));
	let tileOffset = vec2<i32>(tileX * atlasTileSize, tileY * atlasTileSize);
	var visible = 0.0;
	var sampleCount = 0.0;

	for (var y: i32 = -1; y <= 1; y = y + 1) {
		for (var x: i32 = -1; x <= 1; x = x + 1) {
			let samplePosition =
				texelPosition + vec2<f32>(f32(x), f32(y)) * pcfRadius;
			if (
				samplePosition.x < 0.0 ||
				samplePosition.x > f32(shadowSize - 1) ||
				samplePosition.y < 0.0 ||
				samplePosition.y > f32(shadowSize - 1)
			) {
				continue;
			}
			let roundedSamplePosition = round(samplePosition);
			let sampleCoord = vec2<i32>(
				i32(roundedSamplePosition.x),
				i32(roundedSamplePosition.y)
			);
			let sampleDepth = loadShadowDepthTexel(tileOffset + sampleCoord);
			visible += select(0.0, 1.0, currentDepth - bias <= sampleDepth);
			sampleCount += 1.0;
		}
	}

	if (sampleCount < 1.0) {
		return 1.0;
	}

	let filteredVisibility = visible / max(sampleCount, 1.0);
	let strength = clamp(shadowData.paramsB.y, 0.0, 1.0);
	return 1.0 - strength + strength * filteredVisibility;
}

fn sampleDirectionalShadowVisibility(
	index: u32,
	worldPosition: vec3<f32>,
	normal: vec3<f32>,
	lightDirection: vec3<f32>
) -> f32 {
	return sampleShadowVisibility(
		0u,
		index,
		frame.directionalShadows[index],
		worldPosition,
		normal,
		lightDirection
	);
}

fn sampleSpotShadowVisibility(
	index: u32,
	worldPosition: vec3<f32>,
	normal: vec3<f32>,
	lightDirection: vec3<f32>
) -> f32 {
	return sampleShadowVisibility(
		1u,
		index,
		frame.spotShadows[index],
		worldPosition,
		normal,
		lightDirection
	);
}

struct ClusteredLightRef {
	lightIndex: u32,
	lightType: u32,
	shadowed: bool,
	volumetric: bool,
}

fn isClusteredLightingEnabled() -> bool {
	return frame.environmentOptionsB.w > 0.5 &&
		clusterGrid.zSlices > 0u &&
		clusterGrid.clusterCount > 0u &&
		clusterGrid.logScale > 0.0;
}

fn computeClusterSliceFromLinearDepth(linearDepth: f32) -> u32 {
	let nearPlane = max(clusterGrid.near, 0.05);
	let farPlane = max(clusterGrid.far, nearPlane + 1e-3);
	let z = max(linearDepth, nearPlane);
	let numerator = log(z) - log(nearPlane);
	let denominator = max(log(farPlane) - log(nearPlane), EPSILON);
	let scaled = numerator / denominator * f32(clusterGrid.zSlices);
	let slice = i32(floor(scaled));
	return u32(clamp(slice, 0, i32(clusterGrid.zSlices) - 1));
}

fn computeClusterIndex(worldPosition: vec3<f32>, linearDepth: f32) -> u32 {
	let clip = frame.viewProjection * vec4<f32>(worldPosition, 1.0);
	if (abs(clip.w) <= EPSILON) {
		return 0u;
	}
	let invW = 1.0 / clip.w;
	let ndc = clip.xy * invW;
	let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
	let tilesX = max(clusterGrid.tilesX, 1u);
	let tilesY = max(clusterGrid.tilesY, 1u);
	let tileX = u32(
		clamp(
			i32(floor(uv.x * f32(tilesX))),
			0,
			i32(tilesX) - 1
		)
	);
	let tileY = u32(
		clamp(
			i32(floor(uv.y * f32(tilesY))),
			0,
			i32(tilesY) - 1
		)
	);
	let slice = computeClusterSliceFromLinearDepth(linearDepth);
	return slice * (tilesX * tilesY) + tileY * tilesX + tileX;
}

fn getClusterHeaderForFragment(
	worldPosition: vec3<f32>,
	linearDepth: f32
) -> ClusterHeader {
	if (!isClusteredLightingEnabled()) {
		return ClusterHeader(0u, 0u, 0u, 0u);
	}
	let clusterIndex = computeClusterIndex(worldPosition, linearDepth);
	if (clusterIndex >= clusterGrid.clusterCount) {
		return ClusterHeader(0u, 0u, 0u, 0u);
	}
	return clusterHeaders.headers[clusterIndex];
}

fn getClusterEntryCount(header: ClusterHeader) -> u32 {
	let totalEntries = u32(arrayLength(&clusterIndices.indices));
	if (header.offset >= totalEntries) {
		return 0u;
	}
	return min(header.count, totalEntries - header.offset);
}

fn decodeClusteredLightRef(value: u32) -> ClusteredLightRef {
	return ClusteredLightRef(
		value & CLUSTER_INDEX_LIGHT_MASK,
		(value & CLUSTER_INDEX_TYPE_MASK) >> CLUSTER_INDEX_TYPE_SHIFT,
		(value & CLUSTER_INDEX_SHADOW_BIT) != 0u,
		(value & CLUSTER_INDEX_VOLUMETRIC_BIT) != 0u
	);
}
