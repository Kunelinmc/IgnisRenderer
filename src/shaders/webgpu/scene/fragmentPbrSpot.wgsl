if (!isClusteredLightingEnabled()) {
	let spotCount = frameLights.lightCounts.z;
	for (var i: u32 = 0u; i < spotCount; i = i + 1u) {
		let toLight = frameLights.spotLights[i].positionRange.xyz - input.worldPosition;
		let distanceSq = dot(toLight, toLight);
		let distanceValue = sqrt(max(distanceSq, EPSILON));
		let lightRange = frameLights.spotLights[i].positionRange.w;
		if (distanceValue > lightRange) {
			continue;
		}

		let lightDirection = toLight / distanceValue;
		let lightToPoint = -lightDirection;
		let coneDirection = safeNormalize(
			frameLights.spotLights[i].directionOuter.xyz,
			vec3<f32>(0.0, -1.0, 0.0)
		);
		let coneAttenuation = spotAttenuation(
			dot(lightToPoint, coneDirection),
			frameLights.spotLights[i].directionOuter.w,
			frameLights.spotLights[i].colorInner.w
		);
		if (coneAttenuation <= 0.0) {
			continue;
		}

		let radiance =
			frameLights.spotLights[i].colorInner.xyz *
			pointAttenuation(distanceSq, lightRange) *
			coneAttenuation;
		let nDotLRaw = dot(pbrNormal, lightDirection);
		let nDotL = max(nDotLRaw, 0.0);
		let nDotLTransmission = max(-nDotLRaw, 0.0);
		if (nDotL <= 0.0 && nDotLTransmission <= 0.0) {
			continue;
		}

		var shadow = vec3<f32>(1.0);
		if (input.instanceMeta.y > 0.5) {
			shadow = sampleSpotShadowVisibility(
				i,
				input.worldPosition,
				pbrShadowNormal,
				lightDirection
			);
		}
		directLight += evaluatePBRDirectLight(
			pbrSurface, lightDirection, radiance, shadow, ior, volumeAttenuation
		);
	}
}
