#import <ignis/color/srgb>
#import <ignis/postprocess/fog>
#import <ignis/webgpu/constants>
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

fn transformUV(
	slotIndex: u32,
	uv0: vec2<f32>,
	uv1: vec2<f32>,
	uv2: vec2<f32>,
	uv3: vec2<f32>
) -> vec2<f32> {
	let transformA = model.textureTransformA[slotIndex];
	let transformB = model.textureTransformB[slotIndex];
	let uvSet = u32(clamp(floor(transformB.y + 0.5), 0.0, 3.0));
	var uv = uv0;
	if (uvSet == 1u) {
		uv = uv1;
	} else if (uvSet == 2u) {
		uv = uv2;
	} else if (uvSet >= 3u) {
		uv = uv3;
	}
	uv = uv * transformA.zw;

	let rotation = transformB.x;
	if (abs(rotation) > EPSILON) {
		let c = cos(rotation);
		let s = sin(rotation);
		uv = vec2<f32>(uv.x * c - uv.y * s, uv.x * s + uv.y * c);
	}

	return uv + transformA.xy;
}

fn transformAnisotropyUV(
	uv0: vec2<f32>,
	uv1: vec2<f32>,
	uv2: vec2<f32>,
	uv3: vec2<f32>
) -> vec2<f32> {
	let transformA = model.anisotropyTextureTransformA;
	let transformB = model.anisotropyTextureTransformB;
	let uvSet = u32(clamp(floor(transformB.y + 0.5), 0.0, 3.0));
	var uv = uv0;
	if (uvSet == 1u) {
		uv = uv1;
	} else if (uvSet == 2u) {
		uv = uv2;
	} else if (uvSet >= 3u) {
		uv = uv3;
	}
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
	uv1: vec2<f32>,
	uv2: vec2<f32>,
	uv3: vec2<f32>
) -> vec4<f32> {
	return textureSample(
		textureRef,
		samplerRef,
		transformUV(slotIndex, uv0, uv1, uv2, uv3)
	);
}

fn sampleColorTexture(
	textureRef: texture_2d<f32>,
	samplerRef: sampler,
	slotIndex: u32,
	uv0: vec2<f32>,
	uv1: vec2<f32>,
	uv2: vec2<f32>,
	uv3: vec2<f32>
) -> vec4<f32> {
	let sampled = sampleLinearTexture(
		textureRef,
		samplerRef,
		slotIndex,
		uv0,
		uv1,
		uv2,
		uv3
	);
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

fn fallbackTangentFromNormal(n: vec3<f32>) -> vec3<f32> {
	let axis = select(
		vec3<f32>(0.0, 1.0, 0.0),
		vec3<f32>(1.0, 0.0, 0.0),
		abs(n.y) > 0.999
	);
	return safeNormalize(cross(axis, n), vec3<f32>(1.0, 0.0, 0.0));
}

fn resolveAnisotropyDirection(
	uv0: vec2<f32>,
	uv1: vec2<f32>,
	uv2: vec2<f32>,
	uv3: vec2<f32>
) -> vec3<f32> {
	var strength = clamp(model.anisotropyParams.x, 0.0, 1.0);
	var direction = vec2<f32>(1.0, 0.0);
	if (model.anisotropyTextureTransformB.w > 0.5) {
		let texel = textureSample(
			anisotropyTexture,
			thicknessSampler,
			transformAnisotropyUV(uv0, uv1, uv2, uv3)
		);
		direction = texel.rg * 2.0 - vec2<f32>(1.0);
		let directionLen = length(direction);
		if (directionLen > EPSILON) {
			direction = direction / directionLen;
		} else {
			direction = vec2<f32>(1.0, 0.0);
		}
		strength = clamp(strength * texel.b, 0.0, 1.0);
	}

	let c = model.anisotropyParams.y;
	let s = model.anisotropyParams.z;
	let rotated = vec2<f32>(
		direction.x * c - direction.y * s,
		direction.x * s + direction.y * c
	);
	return vec3<f32>(normalize(rotated), strength);
}

fn resolveAnisotropyTangent(
	n: vec3<f32>,
	tangent: vec4<f32>,
	direction: vec2<f32>
) -> vec3<f32> {
	let tangentLenSq = dot(tangent.xyz, tangent.xyz);
	let hasValidTangent = tangentLenSq > 1e-12 && abs(tangent.w) > EPSILON;
	var t = fallbackTangentFromNormal(n);
	var handedness = 1.0;
	if (hasValidTangent) {
		let candidate = tangent.xyz - n * dot(n, tangent.xyz);
		let candidateLen = length(candidate);
		if (candidateLen > EPSILON) {
			t = candidate / candidateLen;
			handedness = select(1.0, -1.0, tangent.w < 0.0);
		}
	}
	let b = cross(n, t) * handedness;
	return safeNormalize(t * direction.x + b * direction.y, t);
}

fn distributionAnisotropicGGX(
	nDotH: f32,
	tDotH: f32,
	bDotH: f32,
	at: f32,
	ab: f32
) -> f32 {
	let a2 = max(at * ab, 1e-6);
	let f = vec3<f32>(ab * tDotH, at * bDotH, a2 * nDotH);
	let w2 = a2 / max(dot(f, f), 0.0001);
	return a2 * w2 * w2 / PI;
}

fn visibilityAnisotropicGGX(
	nDotL: f32,
	nDotV: f32,
	bDotV: f32,
	tDotV: f32,
	tDotL: f32,
	bDotL: f32,
	at: f32,
	ab: f32
) -> f32 {
	let ggxV = nDotL * length(vec3<f32>(at * tDotV, ab * bDotV, nDotV));
	let ggxL = nDotV * length(vec3<f32>(at * tDotL, ab * bDotL, nDotL));
	return clamp(0.5 / max(ggxV + ggxL, 0.0001), 0.0, 1.0);
}

fn resolveAnisotropicSpecular(
	fresnel: vec3<f32>,
	roughness: f32,
	anisotropy: f32,
	nDotL: f32,
	nDotV: f32,
	nDotH: f32,
	tDotV: f32,
	bDotV: f32,
	tDotL: f32,
	bDotL: f32,
	tDotH: f32,
	bDotH: f32
) -> vec3<f32> {
	let alphaRoughness = roughness * roughness;
	let at = mix(alphaRoughness, 1.0, anisotropy * anisotropy);
	let ab = alphaRoughness;
	let d = distributionAnisotropicGGX(nDotH, tDotH, bDotH, at, ab);
	let v = visibilityAnisotropicGGX(
		nDotL,
		nDotV,
		bDotV,
		tDotV,
		tDotL,
		bDotL,
		at,
		ab
	);
	return fresnel * d * v;
}

fn pointAttenuation(distanceSq: f32, range: f32) -> f32 {
	let rangeSq = max(range * range, EPSILON);
	let rangeFactor = distanceSq / rangeSq;
	let smoothFactor = max(0.0, 1.0 - rangeFactor * rangeFactor);
	return (smoothFactor * smoothFactor) / (distanceSq + 1.0);
}

struct AreaLightSample {
	direction: vec3<f32>,
	radiance: vec3<f32>,
	valid: bool,
}

const AREA_LIGHT_SAMPLE_GRID_SIZE: u32 = 3u;
const AREA_LIGHT_SAMPLE_COUNT: u32 =
	AREA_LIGHT_SAMPLE_GRID_SIZE * AREA_LIGHT_SAMPLE_GRID_SIZE;

fn areaLightCount() -> u32 {
	return min(
		u32(max(frame.areaLightCounts.x + 0.5, 0.0)),
		u32(__WEBGPU_MAX_AREA_LIGHTS__)
	);
}

fn areaLightRangeAttenuation(distanceSq: f32, range: f32) -> f32 {
	let rangeSq = max(range * range, EPSILON);
	let rangeFactor = distanceSq / rangeSq;
	let smoothFactor = max(0.0, 1.0 - rangeFactor * rangeFactor);
	return smoothFactor * smoothFactor;
}

fn sphericalTriangleSolidAngle(
	a: vec3<f32>,
	b: vec3<f32>,
	c: vec3<f32>
) -> f32 {
	let an = safeNormalize(a, vec3<f32>(0.0, 1.0, 0.0));
	let bn = safeNormalize(b, vec3<f32>(1.0, 0.0, 0.0));
	let cn = safeNormalize(c, vec3<f32>(0.0, 0.0, 1.0));
	let numerator = dot(an, cross(bn, cn));
	let denominator = 1.0 + dot(an, bn) + dot(bn, cn) + dot(cn, an);
	return 2.0 * atan2(numerator, denominator);
}

// Exact projected solid angle for one rectangular cell. The caller evaluates
// the BRDF at the cell center, so large emitters no longer collapse to one point.
fn rectangleProjectedSolidAngle(
	center: vec3<f32>,
	right: vec3<f32>,
	up: vec3<f32>,
	halfWidth: f32,
	halfHeight: f32,
	worldPosition: vec3<f32>
) -> f32 {
	let rightExtent = right * halfWidth;
	let upExtent = up * halfHeight;
	let p0 = center - rightExtent - upExtent - worldPosition;
	let p1 = center + rightExtent - upExtent - worldPosition;
	let p2 = center + rightExtent + upExtent - worldPosition;
	let p3 = center - rightExtent + upExtent - worldPosition;
	let solidAngle =
		sphericalTriangleSolidAngle(p0, p1, p2) +
		sphericalTriangleSolidAngle(p0, p2, p3);
	return abs(solidAngle);
}

fn areaLightSampleOffset(sampleIndex: u32) -> vec2<f32> {
	let gridSize = f32(AREA_LIGHT_SAMPLE_GRID_SIZE);
	let sampleX = f32(sampleIndex % AREA_LIGHT_SAMPLE_GRID_SIZE);
	let sampleY = f32(sampleIndex / AREA_LIGHT_SAMPLE_GRID_SIZE);
	return vec2<f32>(
		(sampleX + 0.5) / gridSize - 0.5,
		(sampleY + 0.5) / gridSize - 0.5
	);
}

fn evaluateAreaLight(
	light: AreaLightData,
	worldPosition: vec3<f32>,
	sampleIndex: u32
) -> AreaLightSample {
	let range = max(light.positionRange.w, 0.0);
	let width = max(light.rightWidth.w, 0.0);
	let height = max(light.upHeight.w, 0.0);
	let area = max(light.normalAreaScale.w, 0.0);
	if (range <= 0.0 || width <= 0.0 || height <= 0.0 || area <= 0.0) {
		return AreaLightSample(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0), false);
	}

	let center = light.positionRange.xyz;
	let right = safeNormalize(light.rightWidth.xyz, vec3<f32>(1.0, 0.0, 0.0));
	let up = safeNormalize(light.upHeight.xyz, vec3<f32>(0.0, 0.0, 1.0));
	let normal = safeNormalize(
		light.normalAreaScale.xyz,
		vec3<f32>(0.0, 1.0, 0.0)
	);
	let relPos = worldPosition - center;
	let distToPlane = dot(relPos, normal);
	if (distToPlane <= 0.0) {
		return AreaLightSample(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0), false);
	}

	let offset = areaLightSampleOffset(sampleIndex);
	let samplePoint =
		center +
		right * (offset.x * width) +
		up * (offset.y * height);
	let toLight = samplePoint - worldPosition;
	let distanceSq = dot(toLight, toLight);
	let distanceValue = sqrt(max(distanceSq, EPSILON));
	if (distanceValue > range) {
		return AreaLightSample(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0), false);
	}

	let direction = toLight / distanceValue;
	let cosLight = max(0.0, dot(-normal, direction));
	if (cosLight <= 0.0) {
		return AreaLightSample(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0), false);
	}

	let cellHalfWidth =
		(width / f32(AREA_LIGHT_SAMPLE_GRID_SIZE)) * 0.5;
	let cellHalfHeight =
		(height / f32(AREA_LIGHT_SAMPLE_GRID_SIZE)) * 0.5;
	let projectedSolidAngle = rectangleProjectedSolidAngle(
		samplePoint,
		right,
		up,
		cellHalfWidth,
		cellHalfHeight,
		worldPosition
	);
	let attenuation =
		projectedSolidAngle * areaLightRangeAttenuation(distanceSq, range);
	return AreaLightSample(direction, light.color.xyz * attenuation, true);
}

fn clusteredRecordToAreaLight(lightIndex: u32) -> AreaLightData {
	let payload = clusterAreaPayloads.values[lightIndex];
	return AreaLightData(
		clusterPositionRanges.values[lightIndex],
		payload.rightWidth,
		payload.upHeight,
		payload.normalAreaScale,
		vec4<f32>(clusterColorInners.values[lightIndex].xyz, 0.0)
	);
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

fn iorToFresnel0(transmittedIor: f32, incidentIor: f32) -> f32 {
	let value = (transmittedIor - incidentIor) / max(transmittedIor + incidentIor, EPSILON);
	return value * value;
}

fn fresnel0ToIor(f0: vec3<f32>) -> vec3<f32> {
	let sqrtF0 = sqrt(clamp(f0, vec3<f32>(0.0), vec3<f32>(0.9999)));
	return (vec3<f32>(1.0) + sqrtF0) /
		max(vec3<f32>(1.0) - sqrtF0, vec3<f32>(EPSILON));
}

fn evalIridescenceSensitivity(opd: f32, shift: vec3<f32>) -> vec3<f32> {
	let phase = 2.0 * PI * opd * 1.0e-9;
	let phaseSq = phase * phase;
	let val = vec3<f32>(5.4856e-13, 4.4201e-13, 5.2481e-13);
	let pos = vec3<f32>(1.6810e6, 1.7953e6, 2.2084e6);
	let variance = vec3<f32>(4.3278e9, 9.3046e9, 6.6121e9);
	var xyz =
		val *
		sqrt(vec3<f32>(2.0 * PI) * variance) *
		cos(pos * phase + shift) *
		exp(-variance * phaseSq);
	xyz.x =
		xyz.x +
		9.7470e-14 *
			sqrt(2.0 * PI * 4.5282e9) *
			cos(2.2399e6 * phase + shift.x) *
			exp(-4.5282e9 * phaseSq);
	xyz = xyz / 1.0685e-7;

	let xyzToRec709 = mat3x3<f32>(
		vec3<f32>(3.2404542, -0.9692660, 0.0556434),
		vec3<f32>(-1.5371385, 1.8760108, -0.2040259),
		vec3<f32>(-0.4985314, 0.0415560, 1.0572252)
	);
	return xyzToRec709 * xyz;
}

fn iridescentFresnel(
	outsideIor: f32,
	iridescenceIor: f32,
	baseF0: vec3<f32>,
	iridescenceThickness: f32,
	cosTheta1: f32
) -> vec3<f32> {
	let filmIor = max(iridescenceIor, EPSILON);
	let cos1 = clamp(cosTheta1, 0.0, 1.0);
	let eta = outsideIor / filmIor;
	let sinTheta2Sq = eta * eta * (1.0 - cos1 * cos1);
	if (sinTheta2Sq > 1.0) {
		return vec3<f32>(1.0);
	}

	let cosTheta2 = sqrt(max(1.0 - sinTheta2Sq, 0.0));
	let r0 = iorToFresnel0(filmIor, outsideIor);
	let r12 = fresnelSchlickScalar(cos1, r0);
	let t121 = 1.0 - r12;
	let baseIor = fresnel0ToIor(baseF0 + vec3<f32>(0.0001));
	let r1 = vec3<f32>(
		iorToFresnel0(baseIor.x, filmIor),
		iorToFresnel0(baseIor.y, filmIor),
		iorToFresnel0(baseIor.z, filmIor)
	);
	let r23 = fresnelSchlick(cosTheta2, r1);

	let phi12 = select(0.0, PI, filmIor < outsideIor);
	let phi21 = PI - phi12;
	let phi23 = select(
		vec3<f32>(0.0),
		vec3<f32>(PI),
		baseIor < vec3<f32>(filmIor)
	);
	let phi = vec3<f32>(phi21) + phi23;
	let opd = 2.0 * filmIor * iridescenceThickness * cosTheta2;
	let r123 = clamp(vec3<f32>(r12) * r23, vec3<f32>(1e-5), vec3<f32>(0.9999));
	let sqrtR123 = sqrt(r123);
	let rs = (t121 * t121) * r23 / (vec3<f32>(1.0) - r123);

	var interference = vec3<f32>(r12) + rs;
	var cm = rs - vec3<f32>(t121);
	for (var order = 1; order <= 2; order = order + 1) {
		cm = cm * sqrtR123;
		let orderValue = f32(order);
		let sensitivity = evalIridescenceSensitivity(orderValue * opd, orderValue * phi);
		interference = interference + cm * 2.0 * sensitivity;
	}

	return max(interference, vec3<f32>(0.0));
}

fn resolveIridescenceFresnel(
	cosTheta: f32,
	baseF0: vec3<f32>,
	iridescence: f32,
	iridescenceThickness: f32,
	iridescenceIor: f32
) -> vec3<f32> {
	let base = fresnelSchlick(cosTheta, baseF0);
	let strength = clamp(iridescence, 0.0, 1.0);
	if (strength <= EPSILON || iridescenceThickness <= 0.0) {
		return base;
	}

	let iridescent = iridescentFresnel(
		1.0,
		max(iridescenceIor, 1.0),
		clamp(baseF0, vec3<f32>(0.0), vec3<f32>(0.9999)),
		iridescenceThickness,
		cosTheta
	);
	return clamp(mix(base, iridescent, vec3<f32>(strength)), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn diffuseFresnelWeight(fresnel: vec3<f32>, iridescence: f32) -> vec3<f32> {
	if (iridescence > EPSILON) {
		let fresnelMax = max(max(fresnel.x, fresnel.y), fresnel.z);
		return vec3<f32>(1.0 - fresnelMax);
	}
	return vec3<f32>(1.0) - fresnel;
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

fn resolveAnisotropicReflectionDirection(
	n: vec3<f32>,
	v: vec3<f32>,
	anisotropicB: vec3<f32>,
	roughness: f32,
	anisotropy: f32
) -> vec3<f32> {
	var bentNormal = cross(anisotropicB, v);
	bentNormal = safeNormalize(
		cross(bentNormal, anisotropicB),
		n
	);
	let a = 1.0 - anisotropy * (1.0 - roughness);
	let blendToNormal = a * a * a * a;
	bentNormal = safeNormalize(mix(bentNormal, n, blendToNormal), n);
	var reflectionDir = reflectViewDirection(bentNormal, v);
	reflectionDir = safeNormalize(
		mix(reflectionDir, bentNormal, roughness * roughness),
		reflectionDir
	);
	return reflectionDir;
}

fn encodeOutput(color: vec3<f32>) -> vec3<f32> {
	return color;
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

fn encodeOctahedralNormal(normal: vec3<f32>) -> vec2<f32> {
	let n = safeNormalize(normal, vec3<f32>(0.0, 0.0, 1.0));
	let denom = max(abs(n.x) + abs(n.y) + abs(n.z), EPSILON);
	var oct = n.xy / denom;
	if (n.z < 0.0) {
		oct = octahedralWrap(oct);
	}
	return oct * 0.5 + vec2<f32>(0.5);
}

fn encodeNormalForGBuffer(normal: vec3<f32>) -> vec2<f32> {
	let right = frame.environmentBasisRight.xyz;
	let up = frame.environmentBasisUp.xyz;
	let backward = frame.environmentBasisBackward.xyz;
	let vn = vec3<f32>(dot(normal, right), dot(normal, up), dot(normal, backward));
	return encodeOctahedralNormal(vn);
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

fn hasEnvSpecularFallback() -> bool {
	return frame.environmentOptionsA.z > 0.5;
}

fn hasBRDFLUT() -> bool {
	return frame.environmentOptionsB.x > 0.5;
}

fn envSpecularMaxMipLevel() -> f32 {
	return max(frame.environmentOptionsB.y, 0.0);
}

fn envSpecularFallbackMaxMipLevel() -> f32 {
	return max(frame.environmentOptionsA.z - 1.0, 0.0);
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

fn localLightProbeCount() -> u32 {
	let count = u32(max(frame.localLightProbeCounts.x + 0.5, 0.0));
	return min(count, __WEBGPU_MAX_LOCAL_LIGHT_PROBES__u);
}

fn worldToLocalLightProbePoint(
	probeIndex: u32,
	worldPosition: vec3<f32>
) -> vec3<f32> {
	let row0 = frame.localLightProbeWorldToProbeRow0[probeIndex];
	let row1 = frame.localLightProbeWorldToProbeRow1[probeIndex];
	let row2 = frame.localLightProbeWorldToProbeRow2[probeIndex];
	return vec3<f32>(
		dot(row0.xyz, worldPosition) + row0.w,
		dot(row1.xyz, worldPosition) + row1.w,
		dot(row2.xyz, worldPosition) + row2.w
	);
}

fn computeLocalLightProbeMetric(
	worldPosition: vec3<f32>,
	probeIndex: u32
) -> f32 {
	let localPosition = worldToLocalLightProbePoint(probeIndex, worldPosition);
	let dataA = frame.localLightProbeDataA[probeIndex];
	let shape = frame.localLightProbeDataB[probeIndex].z;
	if (shape > 0.5) {
		return max(
			max(abs(localPosition.x) * dataA.x, abs(localPosition.y) * dataA.y),
			abs(localPosition.z) * dataA.z
		);
	}
	return length(localPosition) * dataA.w;
}

fn computeLocalLightProbeWeight(metric: f32, probeIndex: u32) -> f32 {
	let blendDistance = max(frame.localLightProbeDataB[probeIndex].x, 1e-5);
	let x = clamp((metric - 1.0) / blendDistance, 0.0, 1.0);
	return 1.0 - smoothstep(0.0, 1.0, x);
}

fn localLightProbePriority(probeIndex: u32) -> i32 {
	return i32(frame.localLightProbeDataB[probeIndex].y);
}

fn localLightProbeCoeffIndex(probeIndex: u32, coeffIndex: u32) -> u32 {
	return probeIndex * 16u + coeffIndex;
}

fn sampleLocalLightProbeSHCoeff(probeIndex: u32, coeffIndex: u32) -> vec3<f32> {
	return frame.localLightProbeSHAmbientCoeffs[
		localLightProbeCoeffIndex(probeIndex, coeffIndex)
	].xyz;
}

fn sampleLocalLightProbeIrradiance(
	probeIndex: u32,
	normal: vec3<f32>
) -> vec3<f32> {
	let basis = evalSHBasis(normal);
	let c1 = PI;
	let c2 = (2.0 * PI) / 3.0;
	let c3 = PI / 4.0;
	var result = vec3<f32>(0.0);
	for (var i: u32 = 0u; i < 16u; i = i + 1u) {
		var factor = select(0.0, c3, i >= 4u && i < 9u);
		factor = select(factor, c2, i >= 1u && i < 4u);
		factor = select(factor, c1, i == 0u);
		result += sampleLocalLightProbeSHCoeff(probeIndex, i) * basis[i] * factor;
	}
	return max(result, vec3<f32>(0.0));
}

fn sampleLocalLightProbeRadiance(
	probeIndex: u32,
	direction: vec3<f32>
) -> vec3<f32> {
	let basis = evalSHBasis(direction);
	var result = vec3<f32>(0.0);
	for (var i: u32 = 0u; i < 16u; i = i + 1u) {
		result += sampleLocalLightProbeSHCoeff(probeIndex, i) * basis[i];
	}
	return max(result, vec3<f32>(0.0));
}

fn isBetterLocalLightProbeCandidate(
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

fn selectTopTwoLocalLightProbes(worldPosition: vec3<f32>) -> vec4<f32> {
	let count = localLightProbeCount();
	var bestPriority = -2147483647;
	var firstIndex: i32 = -1;
	var secondIndex: i32 = -1;
	var firstWeight = 0.0;
	var secondWeight = 0.0;

	for (var i: u32 = 0u; i < count; i = i + 1u) {
		let metric = computeLocalLightProbeMetric(worldPosition, i);
		let weight = computeLocalLightProbeWeight(metric, i);
		if (!isFiniteF32(weight) || weight <= 1e-6) {
			continue;
		}

		let priority = localLightProbePriority(i);
		let index = i32(i);
		if (priority > bestPriority) {
			bestPriority = priority;
			firstIndex = index;
			secondIndex = -1;
			firstWeight = weight;
			secondWeight = 0.0;
			continue;
		}
		if (priority < bestPriority) {
			continue;
		}

		if (
			firstIndex < 0 ||
			isBetterLocalLightProbeCandidate(weight, index, firstWeight, firstIndex)
		) {
			secondIndex = firstIndex;
			secondWeight = firstWeight;
			firstIndex = index;
			firstWeight = weight;
			continue;
		}

		if (
			secondIndex < 0 ||
			isBetterLocalLightProbeCandidate(weight, index, secondWeight, secondIndex)
		) {
			secondIndex = index;
			secondWeight = weight;
		}
	}

	return vec4<f32>(
		f32(firstIndex),
		f32(secondIndex),
		firstWeight,
		secondWeight
	);
}

fn sampleBlendedLocalLightProbeIrradiance(
	selection: vec4<f32>,
	normal: vec3<f32>
) -> vec4<f32> {
	if (selection.x < 0.0 || selection.z <= 1e-6) {
		return vec4<f32>(0.0);
	}

	let rawSum = selection.z + max(selection.w, 0.0);
	let coverage = clamp(rawSum, 0.0, 1.0);
	if (coverage <= 1e-6) {
		return vec4<f32>(0.0);
	}

	let invWeight = 1.0 / max(rawSum, 1e-6);
	var result = sampleLocalLightProbeIrradiance(
		u32(max(selection.x, 0.0)),
		normal
	) * (selection.z * invWeight);
	if (selection.y >= 0.0 && selection.w > 1e-6) {
		result += sampleLocalLightProbeIrradiance(
			u32(max(selection.y, 0.0)),
			normal
		) * (selection.w * invWeight);
	}

	return vec4<f32>(result, coverage);
}

fn sampleBlendedLocalLightProbeRadiance(
	selection: vec4<f32>,
	direction: vec3<f32>
) -> vec4<f32> {
	if (selection.x < 0.0 || selection.z <= 1e-6) {
		return vec4<f32>(0.0);
	}

	let rawSum = selection.z + max(selection.w, 0.0);
	let coverage = clamp(rawSum, 0.0, 1.0);
	if (coverage <= 1e-6) {
		return vec4<f32>(0.0);
	}

	let invWeight = 1.0 / max(rawSum, 1e-6);
	var result = sampleLocalLightProbeRadiance(
		u32(max(selection.x, 0.0)),
		direction
	) * (selection.z * invWeight);
	if (selection.y >= 0.0 && selection.w > 1e-6) {
		result += sampleLocalLightProbeRadiance(
			u32(max(selection.y, 0.0)),
			direction
		) * (selection.w * invWeight);
	}

	return vec4<f32>(result, coverage);
}

fn irradianceProbeGridEnabled() -> bool {
	return frame.irradianceProbeGridDataC.x > 0.5 &&
		frame.irradianceProbeGridDataA.w > 0.5;
}

fn worldToIrradianceProbeGridPoint(worldPosition: vec3<f32>) -> vec3<f32> {
	let row0 = frame.irradianceProbeGridWorldToGridRow0;
	let row1 = frame.irradianceProbeGridWorldToGridRow1;
	let row2 = frame.irradianceProbeGridWorldToGridRow2;
	return vec3<f32>(
		dot(row0.xyz, worldPosition) + row0.w,
		dot(row1.xyz, worldPosition) + row1.w,
		dot(row2.xyz, worldPosition) + row2.w
	);
}

fn computeIrradianceProbeGridCoverage(localPosition: vec3<f32>) -> f32 {
	let invHalfExtents = frame.irradianceProbeGridDataB.xyz;
	let metric = max(
		max(abs(localPosition.x) * invHalfExtents.x, abs(localPosition.y) * invHalfExtents.y),
		abs(localPosition.z) * invHalfExtents.z
	);
	let blendDistance = max(frame.irradianceProbeGridDataB.w, 1e-5);
	let x = clamp((metric - 1.0) / blendDistance, 0.0, 1.0);
	return 1.0 - smoothstep(0.0, 1.0, x);
}

fn resolveIrradianceProbeGridAxis(
	localValue: f32,
	invHalfExtent: f32,
	dimension: u32
) -> f32 {
	if (dimension <= 1u) {
		return 0.0;
	}
	let normalized = clamp(localValue * invHalfExtent * 0.5 + 0.5, 0.0, 1.0);
	return normalized * f32(dimension - 1u);
}

fn irradianceProbeGridCellIndex(x: u32, y: u32, z: u32, dims: vec3<u32>) -> u32 {
	return x + y * dims.x + z * dims.x * dims.y;
}

fn sampleIrradianceProbeGridCoeff(cellIndex: u32, coeffIndex: u32) -> vec4<f32> {
	return textureLoad(
		irradianceProbeGridCoeffs,
		vec2<i32>(i32(coeffIndex), i32(cellIndex)),
		0
	);
}

fn sampleIrradianceProbeGridIrradiance(
	worldPosition: vec3<f32>,
	normal: vec3<f32>
) -> vec4<f32> {
	if (!irradianceProbeGridEnabled()) {
		return vec4<f32>(0.0);
	}
	let dims = vec3<u32>(
		max(u32(frame.irradianceProbeGridDataA.x + 0.5), 1u),
		max(u32(frame.irradianceProbeGridDataA.y + 0.5), 1u),
		max(u32(frame.irradianceProbeGridDataA.z + 0.5), 1u)
	);
	let cellCount = max(u32(frame.irradianceProbeGridDataA.w + 0.5), 1u);
	let localPosition = worldToIrradianceProbeGridPoint(worldPosition);
	let coverage = computeIrradianceProbeGridCoverage(localPosition);
	if (coverage <= 1e-6) {
		return vec4<f32>(0.0);
	}

	let gridX = resolveIrradianceProbeGridAxis(
		localPosition.x,
		frame.irradianceProbeGridDataB.x,
		dims.x
	);
	let gridY = resolveIrradianceProbeGridAxis(
		localPosition.y,
		frame.irradianceProbeGridDataB.y,
		dims.y
	);
	let gridZ = resolveIrradianceProbeGridAxis(
		localPosition.z,
		frame.irradianceProbeGridDataB.z,
		dims.z
	);
	let x0 = min(u32(floor(gridX)), dims.x - 1u);
	let y0 = min(u32(floor(gridY)), dims.y - 1u);
	let z0 = min(u32(floor(gridZ)), dims.z - 1u);
	let x1 = min(x0 + 1u, dims.x - 1u);
	let y1 = min(y0 + 1u, dims.y - 1u);
	let z1 = min(z0 + 1u, dims.z - 1u);
	let tx = gridX - f32(x0);
	let ty = gridY - f32(y0);
	let tz = gridZ - f32(z0);
	let basis = evalSHBasis(normal);
	let c1 = PI;
	let c2 = (2.0 * PI) / 3.0;
	let c3 = PI / 4.0;
	var result = vec3<f32>(0.0);
	var totalWeight = 0.0;

	for (var corner: u32 = 0u; corner < 8u; corner = corner + 1u) {
		let useX1 = (corner & 1u) != 0u;
		let useY1 = (corner & 2u) != 0u;
		let useZ1 = (corner & 4u) != 0u;
		let cellX = select(x0, x1, useX1);
		let cellY = select(y0, y1, useY1);
		let cellZ = select(z0, z1, useZ1);
		let weightX = select(1.0 - tx, tx, useX1);
		let weightY = select(1.0 - ty, ty, useY1);
		let weightZ = select(1.0 - tz, tz, useZ1);
		let weight = weightX * weightY * weightZ;
		if (weight <= 1e-6) {
			continue;
		}
		let cellIndex = irradianceProbeGridCellIndex(cellX, cellY, cellZ, dims);
		if (cellIndex >= cellCount) {
			continue;
		}
		let valid = sampleIrradianceProbeGridCoeff(cellIndex, 0u).w;
		if (valid <= 0.5) {
			continue;
		}
		for (var coeffIndex: u32 = 0u; coeffIndex < 16u; coeffIndex = coeffIndex + 1u) {
			var factor = select(0.0, c3, coeffIndex >= 4u && coeffIndex < 9u);
			factor = select(factor, c2, coeffIndex >= 1u && coeffIndex < 4u);
			factor = select(factor, c1, coeffIndex == 0u);
			let coeff = sampleIrradianceProbeGridCoeff(cellIndex, coeffIndex).xyz;
			result += coeff * basis[coeffIndex] * factor * weight;
		}
		totalWeight += weight;
	}

	if (totalWeight <= 1e-6) {
		return vec4<f32>(0.0);
	}
	return vec4<f32>(max(result / totalWeight, vec3<f32>(0.0)), clamp(coverage, 0.0, 1.0));
}

fn sampleDiffuseProbeIrradiance(
	worldPosition: vec3<f32>,
	normal: vec3<f32>
) -> vec3<f32> {
	let localSelection = selectTopTwoLocalLightProbes(worldPosition);
	let globalDiffuseAmbient = calculateIrradianceFromSH(normal);
	let localDiffuseAmbient = sampleBlendedLocalLightProbeIrradiance(
		localSelection,
		normal
	);
	let fallback = mix(
		globalDiffuseAmbient,
		localDiffuseAmbient.rgb,
		localDiffuseAmbient.w
	);
	let gridDiffuseAmbient = sampleIrradianceProbeGridIrradiance(
		worldPosition,
		normal
	);
	return mix(fallback, gridDiffuseAmbient.rgb, gridDiffuseAmbient.w);
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
	var sampled = textureSampleLevel(
		envSpecularTexture,
		envSpecularSampler,
		uv,
		level
	).rgb;
	if (layerCount <= 1u && frame.environmentOptionsB.z < 0.5) {
		sampled = srgbToLinear(sampled);
	}
	return sampled;
}

fn sampleFallbackEnvironmentSpecular(
	direction: vec3<f32>,
	roughness: f32
) -> vec3<f32> {
	if (!hasEnvSpecularFallback()) {
		return vec3<f32>(0.0);
	}

	let uv = directionToEquirectUV(
		safeNormalize(direction, vec3<f32>(0.0, 1.0, 0.0))
	);
	let level =
		clamp(roughness, 0.0, 1.0) * envSpecularFallbackMaxMipLevel();
	var sampled = textureSampleLevel(
		envSpecularFallbackTexture,
		envSpecularFallbackSampler,
		uv,
		level
	).rgb;
	if (frame.environmentOptionsB.z < 0.5) {
		sampled = srgbToLinear(sampled);
	}
	return sampled;
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

fn computeReflectionProbeDepthOcclusion(metric: f32, probe: ReflectionProbeData) -> f32 {
	let safeBlendDistance = max(probe.dataC.y, 1e-5);
	let normalizedDepth = clamp((1.0 - metric) / safeBlendDistance, 0.0, 1.0);
	return smoothstep(0.0, 1.0, normalizedDepth);
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

	let fallbackSample = sampleFallbackEnvironmentSpecular(
		normalizedDirection,
		roughness
	);
	let selection = selectTopTwoReflectionProbes(worldPosition);
	if (selection.x < 0.0) {
		return fallbackSample;
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
	let firstMetric = computeReflectionProbeMetric(worldPosition, firstProbe);
	let firstDepthOcclusion = computeReflectionProbeDepthOcclusion(
		firstMetric,
		firstProbe
	);
	let firstContribution = selection.z * firstDepthOcclusion;

	if (selection.y < 0.0 || selection.w <= 1e-6) {
		return firstSample * firstContribution +
			fallbackSample * (1.0 - clamp(firstContribution, 0.0, 1.0));
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
	let secondMetric = computeReflectionProbeMetric(worldPosition, secondProbe);
	let secondDepthOcclusion = computeReflectionProbeDepthOcclusion(
		secondMetric,
		secondProbe
	);
	let secondContribution = selection.w * secondDepthOcclusion;
	let combinedContribution = clamp(
		firstContribution + secondContribution,
		0.0,
		1.0
	);

	return
		firstSample * firstContribution +
		secondSample * secondContribution +
		fallbackSample * (1.0 - combinedContribution);
}

fn sampleBRDFLUT(nDotV: f32, roughness: f32) -> vec2<f32> {
	let nv = clamp(nDotV, 0.0, 1.0);
	let r = clamp(roughness, 0.0, 1.0);
	if (hasBRDFLUT()) {
		return textureSampleLevel(
			brdfLUTTexture,
			envSpecularSampler,
			vec2<f32>(min(nv, 0.999999), min(sqrt(r), 0.999999)),
			0.0
		).rg;
	}
	let a = r * r;
	let k = a * 0.5;
	let visibility = nv / max(nv * (1.0 - k) + k, 0.0001);
	let grazingBias = pow(1.0 - nv, 5.0) * (0.04 + (1.0 - r) * 0.2);
	return vec2<f32>(visibility, grazingBias);
}

fn resolveSpecularEnergyCompensation(
	nDotV: f32,
	roughness: f32,
	f0: vec3<f32>
) -> vec3<f32> {
	if (!hasBRDFLUT()) {
		return vec3<f32>(1.0);
	}

	let brdf = sampleBRDFLUT(nDotV, roughness);
	let singleScatterEnergy = brdf.x + brdf.y;
	if (singleScatterEnergy <= 0.0 || singleScatterEnergy >= 1.0) {
		return vec3<f32>(1.0);
	}

	let factor = 1.0 / singleScatterEnergy - 1.0;
	return vec3<f32>(1.0) + clamp(f0, vec3<f32>(0.0), vec3<f32>(1.0)) * factor;
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

fn loadShadowTransmittanceTexel(coord: vec2<i32>) -> vec3<f32> {
	return textureLoad(shadowTransmittanceAtlas, coord, 0).rgb;
}

const PAGED_SHADOW_NON_RESIDENT: u32 = 0xffffffffu;
const SHADOW_GOLDEN_ANGLE: f32 = 2.39996323;
const MAX_PCSS_FILTER_SAMPLES: i32 = 64;
const MAX_PCSS_SEARCH_SAMPLES: i32 = 64;

fn hashShadowRotation(position: vec3<f32>) -> f32 {
	return
		fract(sin(dot(position, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453123) *
		(2.0 * PI);
}

fn vogelDiskSample(sampleIndex: i32, sampleCount: i32, theta: f32) -> vec2<f32> {
	let indexF = f32(sampleIndex);
	let countF = max(f32(sampleCount), 1.0);
	let radius = sqrt((indexF + 0.5) / countF);
	let angle = indexF * SHADOW_GOLDEN_ANGLE + theta;
	return vec2<f32>(cos(angle), sin(angle)) * radius;
}

fn samplePagedShadowVisibilityForCascade(
	shadowData: ShadowData,
	shadowUv: vec2<f32>,
	currentDepth: f32,
	bias: f32,
	cascadeIndex: u32
) -> vec3<f32> {
	let pageGridSize = max(u32(floor(shadowData.paramsE.z + 0.5)), 1u);
	let pageSize = max(i32(floor(shadowData.paramsF.z + 0.5)), 1);
	let physicalAtlasSize = max(i32(floor(shadowData.paramsF.x + 0.5)), 1);
	let physicalGridSize = max(u32(floor(shadowData.paramsF.y + 0.5)), 1u);
	let pageTableBase = u32(max(floor(shadowData.paramsE.y + 0.5), 0.0));
	let pageTableCascadeStride = max(
		u32(floor(shadowData.paramsF.w + 0.5)),
		pageGridSize * pageGridSize
	);
	let pageCoord = vec2<u32>(
		u32(clamp(floor(shadowUv.x * f32(pageGridSize)), 0.0, f32(pageGridSize - 1u))),
		u32(clamp(floor(shadowUv.y * f32(pageGridSize)), 0.0, f32(pageGridSize - 1u)))
	);
	let tableIndex =
		pageTableBase +
		cascadeIndex * pageTableCascadeStride +
		pageCoord.y * pageGridSize +
		pageCoord.x;
	if (tableIndex >= arrayLength(&pagedShadowPageTable.entries)) {
		return vec3<f32>(1.0);
	}
	let physicalPageIndex = pagedShadowPageTable.entries[tableIndex];
	if (physicalPageIndex == PAGED_SHADOW_NON_RESIDENT) {
		return vec3<f32>(1.0);
	}
	let maxPhysicalPages = physicalGridSize * physicalGridSize;
	if (physicalPageIndex >= maxPhysicalPages) {
		return vec3<f32>(1.0);
	}

	let physicalPageCoord = vec2<i32>(
		i32(physicalPageIndex % physicalGridSize),
		i32(physicalPageIndex / physicalGridSize)
	);
	let localUv = fract(shadowUv * vec2<f32>(f32(pageGridSize)));
	let texelPosition = localUv * vec2<f32>(f32(pageSize - 1), f32(pageSize - 1));
	let pcfRadius = max(shadowData.paramsB.x, 1.0);
	var visible = vec3<f32>(0.0);
	var sampleCount = 0.0;
	for (var y: i32 = -1; y <= 1; y = y + 1) {
		for (var x: i32 = -1; x <= 1; x = x + 1) {
			let samplePosition = texelPosition + vec2<f32>(f32(x), f32(y)) * pcfRadius;
			if (
				samplePosition.x < 0.0 ||
				samplePosition.x > f32(pageSize - 1) ||
				samplePosition.y < 0.0 ||
				samplePosition.y > f32(pageSize - 1)
			) {
				continue;
			}
			let roundedSamplePosition = round(samplePosition);
			let sampleCoord = vec2<i32>(
				i32(roundedSamplePosition.x),
				i32(roundedSamplePosition.y)
			);
			let atlasCoord = physicalPageCoord * pageSize + sampleCoord;
			if (
				atlasCoord.x < 0 ||
				atlasCoord.x >= physicalAtlasSize ||
				atlasCoord.y < 0 ||
				atlasCoord.y >= physicalAtlasSize
			) {
				continue;
			}
			let sampleDepth = textureLoad(pagedShadowPhysicalDepth, atlasCoord, 0);
			if (currentDepth - bias <= sampleDepth) {
				visible += vec3<f32>(1.0);
			}
			sampleCount += 1.0;
		}
	}
	if (sampleCount < 1.0) {
		return vec3<f32>(1.0);
	}
	let filteredVisibility = visible / max(sampleCount, 1.0);
	let strength = clamp(shadowData.paramsB.y, 0.0, 1.0);
	return vec3<f32>(1.0 - strength) + strength * filteredVisibility;
}

fn sampleShadowVisibilityForCascade(
	shadowType: u32,
	index: u32,
	shadowData: ShadowData,
	worldPosition: vec3<f32>,
	normal: vec3<f32>,
	lightDirection: vec3<f32>,
	cascadeIndex: u32
) -> vec3<f32> {
	if (frame.options.z < 0.5 || shadowData.paramsA.x < 0.5) {
		return vec3<f32>(1.0);
	}

	let requestedShadowSize = max(i32(shadowData.paramsB.z + 0.5), 1);
	let atlasTileSize = max(i32(shadowData.paramsB.w + 0.5), requestedShadowSize);
	let isCSM = shadowData.paramsC.y > 0.5 && shadowData.paramsC.z > 1.5;
	let cascadeCount = u32(clamp(floor(shadowData.paramsC.z + 0.5), 1.0, 4.0));
	let clampedCascadeIndex = min(cascadeIndex, cascadeCount - 1u);
	let localTileSpan = select(1, 2, isCSM);
	let subTileSize = max(1, atlasTileSize / localTileSpan);
	let shadowSize = max(1, min(requestedShadowSize, subTileSize));
	let cascadeSplit = shadowData.cascadeSplits[clampedCascadeIndex];
	let localTileX = select(
		0,
		i32(clamp(floor(cascadeSplit.z + 0.5), 0.0, 1.0)),
		isCSM
	);
	let localTileY = select(
		0,
		i32(clamp(floor(cascadeSplit.w + 0.5), 0.0, 1.0)),
		isCSM
	);
	let slopeBias = max(shadowData.paramsC.x, 0.0);
	let maxNormalBias = max(shadowData.paramsA.z, 0.0);
	let minNormalBias = max(shadowData.paramsA.w, 0.0);
	let cosTheta = max(dot(normal, lightDirection), 0.0);
	let bias = max(shadowData.paramsA.y + slopeBias * (1.0 - cosTheta), 0.0);
	let normalBias = minNormalBias + (maxNormalBias - minNormalBias) * (1.0 - cosTheta);
	let shadowWorldPosition = worldPosition + normal * normalBias;
	var shadowMatrix = shadowData.viewProjection;
	if (isCSM) {
		shadowMatrix = shadowData.cascadeViewProjections[clampedCascadeIndex];
	}
	let shadowClip = shadowMatrix * vec4<f32>(shadowWorldPosition, 1.0);
	if (shadowClip.w <= EPSILON) {
		return vec3<f32>(1.0);
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
		return vec3<f32>(1.0);
	}

	if (shadowType == 0u && shadowData.paramsE.x > 0.5) {
		return samplePagedShadowVisibilityForCascade(
			shadowData,
			shadowUv,
			currentDepth,
			bias,
			clampedCascadeIndex
		) * sampleParticleShadowVolumeTransmittance(
			shadowType,
			index,
			clampedCascadeIndex,
			worldPosition
		);
	}

	let pcfRadius = max(shadowData.paramsB.x, 1.0);
	let pcssEnabled = shadowData.paramsD.x > 0.5 && shadowData.paramsD.y > 0.0;
	let pcssRadius = max(shadowData.paramsD.y, 0.0);
	let pcssFilterSamples = i32(
		clamp(floor(shadowData.paramsD.z + 0.5), 1.0, f32(MAX_PCSS_FILTER_SAMPLES))
	);
	let pcssSearchSamples = i32(
		clamp(floor(shadowData.paramsD.w + 0.5), 1.0, f32(MAX_PCSS_SEARCH_SAMPLES))
	);
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
	let tileOffset =
		vec2<i32>(tileX * atlasTileSize, tileY * atlasTileSize) +
		vec2<i32>(localTileX * subTileSize, localTileY * subTileSize);
	var visible = vec3<f32>(0.0);
	var sampleCount = 0.0;
	if (pcssEnabled) {
		let theta = hashShadowRotation(worldPosition);
		var blockerDepthSum = 0.0;
		var blockerCount = 0.0;
		for (var i: i32 = 0; i < MAX_PCSS_SEARCH_SAMPLES; i = i + 1) {
			if (i >= pcssSearchSamples) {
				break;
			}
			let samplePosition =
				texelPosition + vogelDiskSample(i, pcssSearchSamples, theta) * pcssRadius;
			if (
				samplePosition.x < 0.0 ||
				samplePosition.x > f32(shadowSize - 1) ||
				samplePosition.y < 0.0 ||
				samplePosition.y > f32(shadowSize - 1)
			) {
				continue;
			}
			let sampleCoord = vec2<i32>(
				i32(round(samplePosition.x)),
				i32(round(samplePosition.y))
			);
			let sampleDepth = loadShadowDepthTexel(tileOffset + sampleCoord);
			if (currentDepth - bias > sampleDepth) {
				blockerDepthSum += sampleDepth;
				blockerCount += 1.0;
			}
		}

		if (blockerCount < 1.0) {
			return vec3<f32>(1.0);
		}

		let avgBlockerDepth = blockerDepthSum / blockerCount;
		var penumbraRatio = 0.0;
		if (currentDepth > avgBlockerDepth) {
			penumbraRatio = clamp(
				(currentDepth - avgBlockerDepth) / max(avgBlockerDepth, 1e-4),
				0.0,
				1.0
			);
		}
		let filterRadius = pcssRadius * penumbraRatio;
		let effectiveRadius = select(filterRadius, pcfRadius, filterRadius < 0.1);

		for (var i: i32 = 0; i < MAX_PCSS_FILTER_SAMPLES; i = i + 1) {
			if (i >= pcssFilterSamples) {
				break;
			}
			let samplePosition =
				texelPosition + vogelDiskSample(i, pcssFilterSamples, theta) * effectiveRadius;
			if (
				samplePosition.x < 0.0 ||
				samplePosition.x > f32(shadowSize - 1) ||
				samplePosition.y < 0.0 ||
				samplePosition.y > f32(shadowSize - 1)
			) {
				continue;
			}
			let sampleCoord = vec2<i32>(
				i32(round(samplePosition.x)),
				i32(round(samplePosition.y))
			);
			let atlasCoord = tileOffset + sampleCoord;
			let sampleDepth = loadShadowDepthTexel(atlasCoord);
			if (currentDepth - bias <= sampleDepth) {
				visible += loadShadowTransmittanceTexel(atlasCoord);
			}
			sampleCount += 1.0;
		}
	} else {
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
				let atlasCoord = tileOffset + sampleCoord;
				let sampleDepth = loadShadowDepthTexel(atlasCoord);
				if (currentDepth - bias <= sampleDepth) {
					visible += loadShadowTransmittanceTexel(atlasCoord);
				}
				sampleCount += 1.0;
			}
		}
	}

	if (sampleCount < 1.0) {
		return vec3<f32>(1.0);
	}

	let filteredVisibility = visible / max(sampleCount, 1.0);
	let strength = clamp(shadowData.paramsB.y, 0.0, 1.0);
	return (vec3<f32>(1.0 - strength) + strength * filteredVisibility) *
		sampleParticleShadowVolumeTransmittance(shadowType, index, cascadeIndex, worldPosition);
}

fn sampleParticleShadowVolumeTransmittance(
	shadowType: u32,
	index: u32,
	cascadeIndex: u32,
	worldPosition: vec3<f32>
) -> f32 {
	if (shadowType != 0u || index != 0u || cascadeIndex >= 4u) {
		return 1.0;
	}

	let metaBase = cascadeIndex * 24u;
	let bufferLength = arrayLength(&particleShadowVolumes.data);
	if (metaBase + 20u >= bufferLength) {
		return 1.0;
	}
	if (particleShadowVolumes.data[metaBase + 16u] < 0.5) {
		return 1.0;
	}

	let width = u32(max(particleShadowVolumes.data[metaBase + 17u], 1.0));
	let height = u32(max(particleShadowVolumes.data[metaBase + 18u], 1.0));
	let depth = u32(max(particleShadowVolumes.data[metaBase + 19u], 1.0));
	let densityOffset = u32(max(particleShadowVolumes.data[metaBase + 20u], 0.0));
	let viewProjection = mat4x4<f32>(
		vec4<f32>(
			particleShadowVolumes.data[metaBase + 0u],
			particleShadowVolumes.data[metaBase + 1u],
			particleShadowVolumes.data[metaBase + 2u],
			particleShadowVolumes.data[metaBase + 3u]
		),
		vec4<f32>(
			particleShadowVolumes.data[metaBase + 4u],
			particleShadowVolumes.data[metaBase + 5u],
			particleShadowVolumes.data[metaBase + 6u],
			particleShadowVolumes.data[metaBase + 7u]
		),
		vec4<f32>(
			particleShadowVolumes.data[metaBase + 8u],
			particleShadowVolumes.data[metaBase + 9u],
			particleShadowVolumes.data[metaBase + 10u],
			particleShadowVolumes.data[metaBase + 11u]
		),
		vec4<f32>(
			particleShadowVolumes.data[metaBase + 12u],
			particleShadowVolumes.data[metaBase + 13u],
			particleShadowVolumes.data[metaBase + 14u],
			particleShadowVolumes.data[metaBase + 15u]
		)
	);
	let clip = viewProjection * vec4<f32>(worldPosition, 1.0);
	if (clip.w <= 1e-6) {
		return 1.0;
	}
	let ndc = clip.xyz / clip.w;
	if (
		ndc.x < -1.0 || ndc.x > 1.0 ||
		ndc.y < -1.0 || ndc.y > 1.0 ||
		ndc.z < -1.0 || ndc.z > 1.0
	) {
		return 1.0;
	}

	let x = u32(clamp(
		round((ndc.x * 0.5 + 0.5) * f32(width - 1u)),
		0.0,
		f32(width - 1u)
	));
	let y = u32(clamp(
		round((0.5 - ndc.y * 0.5) * f32(height - 1u)),
		0.0,
		f32(height - 1u)
	));
	let zMax = u32(clamp(
		round((ndc.z * 0.5 + 0.5) * f32(depth - 1u)),
		0.0,
		f32(depth - 1u)
	));

	var opticalDepth = 0.0;
	var z = 0u;
	loop {
		if (z > zMax) {
			break;
		}
		let densityIndex = densityOffset + z * width * height + y * width + x;
		if (densityIndex < bufferLength) {
			opticalDepth += particleShadowVolumes.data[densityIndex];
		}
		z = z + 1u;
	}

	return exp(-max(opticalDepth / max(f32(depth), 1.0), 0.0));
}

fn resolveDirectionalCascadeIndex(shadowData: ShadowData, linearDepth: f32) -> u32 {
	let cascadeCount = u32(clamp(floor(shadowData.paramsC.z + 0.5), 1.0, 4.0));
	if (cascadeCount <= 1u || shadowData.paramsC.y < 0.5) {
		return 0u;
	}

	var selected = cascadeCount - 1u;
	for (var i: u32 = 0u; i < 4u; i = i + 1u) {
		if (i >= cascadeCount) {
			break;
		}
		let splitFar = shadowData.cascadeSplits[i].y;
		if (linearDepth <= splitFar) {
			selected = i;
			break;
		}
	}
	return selected;
}

fn sampleDirectionalShadowVisibility(
	index: u32,
	worldPosition: vec3<f32>,
	normal: vec3<f32>,
	lightDirection: vec3<f32>,
	linearDepth: f32
) -> vec3<f32> {
	let shadowData = frame.directionalShadows[index];
	let cascadeIndex = resolveDirectionalCascadeIndex(shadowData, linearDepth);
	let baseVisibility = sampleShadowVisibilityForCascade(
		0u,
		index,
		shadowData,
		worldPosition,
		normal,
		lightDirection,
		cascadeIndex
	);
	let cascadeCount = u32(clamp(floor(shadowData.paramsC.z + 0.5), 1.0, 4.0));
	let blendRatio = clamp(shadowData.paramsC.w, 0.0, 1.0);
	let hasBlend =
		shadowData.paramsC.y > 0.5 &&
		blendRatio > EPSILON &&
		cascadeIndex + 1u < cascadeCount;
	if (!hasBlend) {
		return baseVisibility;
	}

	let split = shadowData.cascadeSplits[cascadeIndex];
	let cascadeRange = max(split.y - split.x, 1e-4);
	let blendStart = split.y - cascadeRange * blendRatio;
	if (linearDepth <= blendStart) {
		return baseVisibility;
	}

	let nextVisibility = sampleShadowVisibilityForCascade(
		0u,
		index,
		shadowData,
		worldPosition,
		normal,
		lightDirection,
		cascadeIndex + 1u
	);
	let blendFactor = clamp(
		(linearDepth - blendStart) / max(split.y - blendStart, 1e-4),
		0.0,
		1.0
	);
	return mix(baseVisibility, nextVisibility, blendFactor);
}

fn sampleSpotShadowVisibility(
	index: u32,
	worldPosition: vec3<f32>,
	normal: vec3<f32>,
	lightDirection: vec3<f32>
) -> vec3<f32> {
	return sampleShadowVisibilityForCascade(
		1u,
		index,
		frame.spotShadows[index],
		worldPosition,
		normal,
		lightDirection,
		0u
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

fn activeClusteredLightCount() -> u32 {
	return min(clusterGrid.lightCount, arrayLength(&clusterPositionRanges.values));
}

fn computeClusterSliceFromLinearDepth(linearDepth: f32) -> u32 {
	let nearPlane = max(clusterGrid.near, 0.05);
	let z = clamp(linearDepth, nearPlane, max(clusterGrid.far, nearPlane));
	let slice = i32(floor(log(z) * clusterGrid.logScale + clusterGrid.logBias));
	return u32(clamp(slice, 0, i32(clusterGrid.zSlices) - 1));
}

fn computeClusterIndex(pixelPosition: vec2<f32>, linearDepth: f32) -> u32 {
	let tilesX = max(clusterGrid.tilesX, 1u);
	let tilesY = max(clusterGrid.tilesY, 1u);
	let screenSize = vec2<f32>(
		f32(max(clusterGrid.screenWidth, 1u)),
		f32(max(clusterGrid.screenHeight, 1u))
	);
	let uv = clamp(pixelPosition / screenSize, vec2<f32>(0.0), vec2<f32>(0.999999));
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
	pixelPosition: vec2<f32>,
	linearDepth: f32
) -> ClusterHeader {
	if (!isClusteredLightingEnabled()) {
		return ClusterHeader(0u, 0u, 0u, 0u);
	}
	let clusterIndex = computeClusterIndex(pixelPosition, linearDepth);
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
	let clusterCapacity = max(clusterGrid.maxLightsPerCluster, 1u);
	return min(min(header.count, clusterCapacity), totalEntries - header.offset);
}

fn decodeClusteredLightRef(value: u32) -> ClusteredLightRef {
	return ClusteredLightRef(
		value & CLUSTER_INDEX_LIGHT_MASK,
		(value & CLUSTER_INDEX_TYPE_MASK) >> CLUSTER_INDEX_TYPE_SHIFT,
		(value & CLUSTER_INDEX_SHADOW_BIT) != 0u,
		(value & CLUSTER_INDEX_VOLUMETRIC_BIT) != 0u
	);
}

struct ClusteredDirectLightSample {
	direction: vec3<f32>,
	radiance: vec3<f32>,
	valid: bool,
}

fn evaluateClusteredDirectLightSample(
	lightRef: ClusteredLightRef,
	worldPosition: vec3<f32>,
	sampleIndex: u32
) -> ClusteredDirectLightSample {
	if (lightRef.lightType == CLUSTER_LIGHT_TYPE_AREA) {
		let areaLight = evaluateAreaLight(
			clusteredRecordToAreaLight(lightRef.lightIndex),
			worldPosition,
			sampleIndex
		);
		return ClusteredDirectLightSample(
			areaLight.direction,
			areaLight.radiance,
			areaLight.valid
		);
	}
	if (lightRef.lightType != CLUSTER_LIGHT_TYPE_POINT &&
		lightRef.lightType != CLUSTER_LIGHT_TYPE_SPOT) {
		return ClusteredDirectLightSample(
			vec3<f32>(0.0, 1.0, 0.0),
			vec3<f32>(0.0),
			false
		);
	}

	let positionRange = clusterPositionRanges.values[lightRef.lightIndex];
	let colorInner = clusterColorInners.values[lightRef.lightIndex];
	let toLight = positionRange.xyz - worldPosition;
	let distanceSq = dot(toLight, toLight);
	let distanceValue = sqrt(max(distanceSq, EPSILON));
	if (distanceValue > positionRange.w) {
		return ClusteredDirectLightSample(
			vec3<f32>(0.0, 1.0, 0.0),
			vec3<f32>(0.0),
			false
		);
	}
	let lightDirection = toLight / distanceValue;
	var attenuation = pointAttenuation(distanceSq, positionRange.w);
	if (lightRef.lightType == CLUSTER_LIGHT_TYPE_SPOT) {
		let directionOuter = clusterDirectionOuters.values[lightRef.lightIndex];
		let coneAttenuation = spotAttenuation(
			dot(
				-lightDirection,
				safeNormalize(directionOuter.xyz, vec3<f32>(0.0, -1.0, 0.0))
			),
			directionOuter.w,
			colorInner.w
		);
		if (coneAttenuation <= 0.0) {
			return ClusteredDirectLightSample(
				lightDirection,
				vec3<f32>(0.0),
				false
			);
		}
		attenuation *= coneAttenuation;
	}
	return ClusteredDirectLightSample(
		lightDirection,
		colorInner.xyz * attenuation,
		true
	);
}
