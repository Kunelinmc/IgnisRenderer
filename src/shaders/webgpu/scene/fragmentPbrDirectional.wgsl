let directionalCount = frameLights.lightCounts.x;
for (var i: u32 = 0u; i < directionalCount; i = i + 1u) {
	let lightDirection = safeNormalize(
		frameLights.directionalLights[i].direction.xyz,
		vec3<f32>(0.0, 1.0, 0.0)
	);
	let radiance = frameLights.directionalLights[i].color.xyz;
	let nDotLRaw = dot(pbrNormal, lightDirection);
	let nDotL = max(nDotLRaw, 0.0);
	let nDotLTransmission = max(-nDotLRaw, 0.0);
	if (nDotL <= 0.0 && nDotLTransmission <= 0.0) {
		continue;
	}

	var shadow = vec3<f32>(1.0);
	if (input.instanceMeta.y > 0.5) {
		shadow = sampleDirectionalShadowVisibility(
			i,
			input.worldPosition,
			pbrShadowNormal,
			lightDirection,
			linearDepth
		);
	}
	directLight += evaluateOpaquePBRLight(
		opaquePBRSurface,
		lightDirection,
		radiance,
		shadow
	);
	directLight += evaluatePBRTransmissionLight(
		opaquePBRSurface, lightDirection, radiance, shadow, ior, volumeAttenuation
	);
}
