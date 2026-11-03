if (!isClusteredLightingEnabled()) {
	let areaCount = areaLightCount();
	for (var i: u32 = 0u; i < areaCount; i = i + 1u) {
		for (
			var sampleIndex: u32 = 0u;
			sampleIndex < AREA_LIGHT_SAMPLE_COUNT;
			sampleIndex = sampleIndex + 1u
		) {
			let areaLight = evaluateAreaLight(
				frameLights.areaLights[i],
				input.worldPosition,
				sampleIndex
			);
			if (!areaLight.valid) {
				continue;
			}

			let lightDirection = areaLight.direction;
			let radiance = areaLight.radiance;
			let nDotLRaw = dot(pbrNormal, lightDirection);
			let nDotL = max(nDotLRaw, 0.0);
			let nDotLTransmission = max(-nDotLRaw, 0.0);
			if (nDotL <= 0.0 && nDotLTransmission <= 0.0) {
				continue;
			}

			directLight += evaluatePBRDirectLight(
				pbrSurface, lightDirection, radiance, vec3<f32>(1.0), ior, volumeAttenuation
			);
		}
	}
}
