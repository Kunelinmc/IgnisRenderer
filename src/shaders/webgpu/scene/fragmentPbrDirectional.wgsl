let directionalCount = u32(frame.lightCounts.x + 0.5);
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

	let shadow = sampleDirectionalShadowVisibility(
		i,
		input.worldPosition,
		pbrShadowNormal,
		lightDirection,
		linearDepth
	);
	let fView = resolveIridescenceFresnel(
		nDotV,
		realF0,
		iridescence,
		iridescenceThickness,
		iridescenceIor
	);
	let kT = (vec3<f32>(1.0) - fView) * (1.0 - metalness) * transmission;
	let transmittedDiffuse = (kT * volumeAttenuation * albedo) / PI;
	let ncDotV = max(dot(clearcoatNormal, viewDir), PBR_MIN_NDOTV);
	let clearcoatTransmissionFresnel = select(0.0, fresnelSchlickScalar(ncDotV, 0.04), clearcoat > 0.0);
	let transmissionAttenuation = vec3<f32>(1.0 - clearcoatTransmissionFresnel * clearcoat);

	directLight += evaluateOpaquePBRLight(
		opaquePBRSurface,
		lightDirection,
		radiance,
		shadow
	);
	directLight +=
		transmittedDiffuse *
		transmissionAttenuation *
		nDotLTransmission *
		radiance *
		shadow;
}
