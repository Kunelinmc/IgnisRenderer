if (isClusteredLightingEnabled()) {
	let clusterHeader = getClusterHeaderForFragment(
		input.position.xy,
		linearDepth
	);
	let clusterEntryCount = getClusterEntryCount(clusterHeader);
	let clusterLightCount = activeClusteredLightCount();
	for (
		var entryIndex: u32 = 0u;
		entryIndex < clusterEntryCount;
		entryIndex = entryIndex + 1u
	) {
		let packedRef = clusterIndices.indices[clusterHeader.offset + entryIndex];
		let clusterRef = decodeClusteredLightRef(packedRef);
		if (clusterRef.lightIndex >= clusterLightCount) {
			continue;
		}
		let sampleCount = select(
			1u,
			AREA_LIGHT_SAMPLE_COUNT,
			clusterRef.lightType == CLUSTER_LIGHT_TYPE_AREA
		);
		for (
			var sampleIndex: u32 = 0u;
			sampleIndex < sampleCount;
			sampleIndex = sampleIndex + 1u
		) {
			let lightSample = evaluateClusteredDirectLightSample(
				clusterRef,
				input.worldPosition,
				sampleIndex
			);
			if (!lightSample.valid) {
				continue;
			}
			let lightDirection = lightSample.direction;
			let radiance = lightSample.radiance;
			var shadow = vec3<f32>(1.0);
			if (clusterRef.lightType == CLUSTER_LIGHT_TYPE_SPOT &&
				clusterRef.shadowed && input.instanceMeta.y > 0.5) {
				let shadowIndex =
					clusterMetadata.values[clusterRef.lightIndex].shadowIndex;
				if (shadowIndex < 8u) {
					shadow = sampleSpotShadowVisibility(
						shadowIndex,
						input.worldPosition,
						pbrShadowNormal,
						lightDirection
					);
				}
			}
			let nDotLRaw = dot(pbrNormal, lightDirection);
			let nDotL = max(nDotLRaw, 0.0);
			let nDotLTransmission = max(-nDotLRaw, 0.0);
			if (nDotL <= 0.0 && nDotLTransmission <= 0.0) {
				continue;
			}
	
			directLight += evaluatePBRDirectLight(
				pbrSurface, lightDirection, radiance, shadow, ior, volumeAttenuation
			);
		}
	}
} else {
	let pointCount = frameLights.lightCounts.y;
	for (var i: u32 = 0u; i < pointCount; i = i + 1u) {
		let toLight = frameLights.pointLights[i].positionRange.xyz - input.worldPosition;
		let distanceSq = dot(toLight, toLight);
		let distanceValue = sqrt(max(distanceSq, EPSILON));
		let lightRange = frameLights.pointLights[i].positionRange.w;
		if (distanceValue > lightRange) {
			continue;
		}

		let lightDirection = safeNormalize(
			toLight,
			vec3<f32>(0.0, 1.0, 0.0)
		);
		let radiance = frameLights.pointLights[i].color.xyz * pointAttenuation(distanceSq, lightRange);
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
