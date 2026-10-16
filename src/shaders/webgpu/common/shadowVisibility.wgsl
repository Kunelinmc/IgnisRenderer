fn sampleDirectionalShadowVisibility(
	index: u32,
	worldPosition: vec3<f32>,
	normal: vec3<f32>,
	lightDirection: vec3<f32>,
	linearDepth: f32
) -> vec3<f32> {
	if (frameShadows.shadowsEnabled == 0u) {
		return vec3<f32>(1.0);
	}
	if (frameShadows.directionalShadows[index].paramsA.x < 0.5) {
		return vec3<f32>(1.0);
	}
	let cascadeIndex = resolveDirectionalCascadeIndex(index, linearDepth);
	let baseVisibility = sampleShadowVisibilityForCascade(
		0u,
		index,
		worldPosition,
		normal,
		lightDirection,
		cascadeIndex
	);
	let paramsC = frameShadows.directionalShadows[index].paramsC;
	let cascadeCount = u32(clamp(floor(paramsC.z + 0.5), 1.0, 4.0));
	let blendRatio = clamp(paramsC.w, 0.0, 1.0);
	let hasBlend =
		paramsC.y > 0.5 &&
		blendRatio > EPSILON &&
		cascadeIndex + 1u < cascadeCount;
	if (!hasBlend) {
		return baseVisibility;
	}

	let split = frameShadows.directionalShadows[index].cascadeSplits[cascadeIndex];
	let cascadeRange = max(split.y - split.x, 1e-4);
	let blendStart = split.y - cascadeRange * blendRatio;
	if (linearDepth <= blendStart) {
		return baseVisibility;
	}

	let nextVisibility = sampleShadowVisibilityForCascade(
		0u,
		index,
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
	if (frameShadows.shadowsEnabled == 0u) {
		return vec3<f32>(1.0);
	}
	if (frameShadows.spotShadows[index].paramsA.x < 0.5) {
		return vec3<f32>(1.0);
	}
	return sampleShadowVisibilityForCascade(
		1u,
		index,
		worldPosition,
		normal,
		lightDirection,
		0u
	);
}
