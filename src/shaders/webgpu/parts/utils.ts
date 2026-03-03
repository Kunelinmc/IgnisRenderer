export const WEBGPU_SCENE_SHADER_UTILS = /* wgsl */ `
fn saturate(value: f32) -> f32 {
	return clamp(value, 0.0, 1.0);
}

fn safeNormalize(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
	let len = length(value);
	return select(fallback, value / max(len, EPSILON), len > EPSILON);
}

fn srgbToLinear(color: vec3<f32>) -> vec3<f32> {
	return pow(max(color, vec3<f32>(0.0)), vec3<f32>(2.2));
}

fn linearToSrgb(color: vec3<f32>) -> vec3<f32> {
	return pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
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
	return select(color, linearToSrgb(color), frame.options.y > 0.5);
}

fn decodePackedShadowDepth(packed: vec4<f32>) -> f32 {
	let bytes = round(clamp(packed, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0);
	let value =
		bytes.x * 16777216.0 +
		bytes.y * 65536.0 +
		bytes.z * 256.0 +
		bytes.w;
	return clamp(value / 4294967295.0, 0.0, 1.0);
}

fn clampShadowTexelCoord(coord: vec2<i32>, size: i32) -> vec2<i32> {
	let maxCoord = max(size - 1, 0);
	return vec2<i32>(
		clamp(coord.x, 0, maxCoord),
		clamp(coord.y, 0, maxCoord)
	);
}

fn loadShadowTexel(shadowType: u32, coord: vec2<i32>) -> vec4<f32> {
	if (shadowType == 0u) {
		return textureLoad(directionalShadowAtlas, coord, 0);
	}

	return textureLoad(spotShadowAtlas, coord, 0);
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
	let bias = max(shadowData.paramsA.y, 0.0);
	let maxNormalBias = max(shadowData.paramsA.z, 0.0);
	let minNormalBias = max(shadowData.paramsA.w, 0.0);
	let cosTheta = max(dot(normal, lightDirection), 0.0);
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
	let tileOffset = vec2<i32>(
		i32(index % 2u) * atlasTileSize,
		i32(index / 2u) * atlasTileSize
	);
	var visible = 0.0;
	var sampleCount = 0.0;

	for (var y: i32 = -1; y <= 1; y = y + 1) {
		for (var x: i32 = -1; x <= 1; x = x + 1) {
			let samplePosition =
				texelPosition + vec2<f32>(f32(x), f32(y)) * pcfRadius;
			let roundedSamplePosition = round(samplePosition);
			let sampleCoord = clampShadowTexelCoord(
				vec2<i32>(
					i32(roundedSamplePosition.x),
					i32(roundedSamplePosition.y)
				),
				shadowSize
			);
			let sampleDepth = decodePackedShadowDepth(
				loadShadowTexel(shadowType, tileOffset + sampleCoord)
			);
			visible += select(0.0, 1.0, currentDepth - bias <= sampleDepth);
			sampleCount += 1.0;
		}
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
`
