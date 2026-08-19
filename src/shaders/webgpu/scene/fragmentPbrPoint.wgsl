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
	
			var diffuse = vec3<f32>(0.0);
			var specular = vec3<f32>(0.0);
			var clearcoatSpecular = vec3<f32>(0.0);
			var sheenSpecular = vec3<f32>(0.0);
			var clearcoatAttenuation = vec3<f32>(1.0);
			var baseLayerAttenuation = vec3<f32>(1.0);
	
			if (nDotL > 0.0) {
				let halfVector = safeNormalize(viewDir + lightDirection, viewDir);
				let fresnel = resolveIridescenceFresnel(
					max(dot(halfVector, viewDir), 0.0),
					realF0,
					iridescence,
					iridescenceThickness,
					iridescenceIor
				);
				if (anisotropyStrength > EPSILON) {
					specular = resolveAnisotropicSpecular(
						fresnel,
						roughness,
						anisotropyStrength,
						nDotL,
						nDotV,
						max(dot(pbrNormal, halfVector), 0.0),
						dot(anisotropyTangent, viewDir),
						dot(anisotropyBitangent, viewDir),
						dot(anisotropyTangent, lightDirection),
						dot(anisotropyBitangent, lightDirection),
						dot(anisotropyTangent, halfVector),
						dot(anisotropyBitangent, halfVector)
					);
				} else {
					let ndf = distributionGGX(pbrNormal, halfVector, roughness);
					let geometry = geometrySmith(nDotV, nDotL, roughness);
					let denominator = max(4.0 * nDotV * nDotL, 0.0001);
					specular = (ndf * geometry * fresnel) / denominator;
				}
				specular = specular * energyCompensation;
	
				let kd =
					diffuseFresnelWeight(fresnel, iridescence) *
					(1.0 - metalness) *
					(1.0 - transmission);
				diffuse = (kd * albedo) / PI;
	
				var clearcoatFresnel = vec3<f32>(0.0);
				if (clearcoat > 0.0) {
					let ncDotL = max(dot(clearcoatNormal, lightDirection), 0.0);
					if (ncDotL > 0.0) {
						let ccHalfVector = safeNormalize(viewDir + lightDirection, viewDir);
						let hccDotV = max(dot(ccHalfVector, viewDir), 0.0);
						let ccNdf = distributionGGX(clearcoatNormal, ccHalfVector, clearcoatRoughness);
						let ccGeometry = geometrySmithClearcoat(ncDotV, ncDotL, clearcoatRoughness);
						let ccF = fresnelSchlickScalar(hccDotV, 0.04);
						let ccDenom = max(4.0 * ncDotV * ncDotL, 0.0001);
						let ccValue = (ccNdf * ccGeometry * ccF) / ccDenom;
						clearcoatSpecular = vec3<f32>(ccValue);
						clearcoatFresnel = vec3<f32>(ccF);
					}
				}
	
				var albedoSheenScaling = vec3<f32>(1.0);
				if (maxSheenColor > 0.0) {
					let nDotH = max(dot(pbrNormal, halfVector), 0.0);
					let sheenNdf = distributionCharlie(nDotH, max(sheenRoughness, 0.04));
					let sheenVisibility = visibilityAshikhmin(nDotL, nDotV);
					sheenSpecular = sheenColor * sheenNdf * sheenVisibility;
					let hDotV = max(dot(halfVector, viewDir), 0.0);
					let sheenFresnel = fresnelSchlick(hDotV, sheenColor);
					albedoSheenScaling = max(vec3<f32>(0.0), vec3<f32>(1.0) - sheenFresnel);
				}
	
				clearcoatAttenuation = vec3<f32>(1.0) - clearcoatFresnel * clearcoat;
				baseLayerAttenuation = clearcoatAttenuation * albedoSheenScaling;
			}
	
			directLight += evaluateOpaquePBRLight(
				opaquePBRSurface, lightDirection, radiance, shadow
			);
			directLight += transmittedDiffuse * transmissionAttenuation *
				nDotLTransmission * radiance * shadow;
		}
	}
} else {
	let pointCount = u32(frame.lightCounts.y + 0.5);
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

		var diffuse = vec3<f32>(0.0);
		var specular = vec3<f32>(0.0);
		var clearcoatSpecular = vec3<f32>(0.0);
		var sheenSpecular = vec3<f32>(0.0);
		var clearcoatAttenuation = vec3<f32>(1.0);
		var baseLayerAttenuation = vec3<f32>(1.0);

		if (nDotL > 0.0) {
			let halfVector = safeNormalize(viewDir + lightDirection, viewDir);
			let fresnel = resolveIridescenceFresnel(
				max(dot(halfVector, viewDir), 0.0),
				realF0,
				iridescence,
				iridescenceThickness,
				iridescenceIor
			);
			if (anisotropyStrength > EPSILON) {
				specular = resolveAnisotropicSpecular(
					fresnel,
					roughness,
					anisotropyStrength,
					nDotL,
					nDotV,
					max(dot(pbrNormal, halfVector), 0.0),
					dot(anisotropyTangent, viewDir),
					dot(anisotropyBitangent, viewDir),
					dot(anisotropyTangent, lightDirection),
					dot(anisotropyBitangent, lightDirection),
					dot(anisotropyTangent, halfVector),
					dot(anisotropyBitangent, halfVector)
				);
			} else {
				let ndf = distributionGGX(pbrNormal, halfVector, roughness);
				let geometry = geometrySmith(nDotV, nDotL, roughness);
				let denominator = max(4.0 * nDotV * nDotL, 0.0001);
				specular = (ndf * geometry * fresnel) / denominator;
			}
			specular = specular * energyCompensation;

			let kd =
				diffuseFresnelWeight(fresnel, iridescence) *
				(1.0 - metalness) *
				(1.0 - transmission);
			diffuse = (kd * albedo) / PI;

			var clearcoatFresnel = vec3<f32>(0.0);
			if (clearcoat > 0.0) {
				let ncDotL = max(dot(clearcoatNormal, lightDirection), 0.0);
				if (ncDotL > 0.0) {
					let ccHalfVector = safeNormalize(viewDir + lightDirection, viewDir);
					let hccDotV = max(dot(ccHalfVector, viewDir), 0.0);
					let ccNdf = distributionGGX(clearcoatNormal, ccHalfVector, clearcoatRoughness);
					let ccGeometry = geometrySmithClearcoat(ncDotV, ncDotL, clearcoatRoughness);
					let ccF = fresnelSchlickScalar(hccDotV, 0.04);
					let ccDenom = max(4.0 * ncDotV * ncDotL, 0.0001);
					let ccValue = (ccNdf * ccGeometry * ccF) / ccDenom;
					clearcoatSpecular = vec3<f32>(ccValue);
					clearcoatFresnel = vec3<f32>(ccF);
				}
			}

			var albedoSheenScaling = vec3<f32>(1.0);
			if (maxSheenColor > 0.0) {
				let nDotH = max(dot(pbrNormal, halfVector), 0.0);
				let sheenNdf = distributionCharlie(nDotH, max(sheenRoughness, 0.04));
				let sheenVisibility = visibilityAshikhmin(nDotL, nDotV);
				sheenSpecular = sheenColor * sheenNdf * sheenVisibility;
				let hDotV = max(dot(halfVector, viewDir), 0.0);
				let sheenFresnel = fresnelSchlick(hDotV, sheenColor);
				albedoSheenScaling = max(vec3<f32>(0.0), vec3<f32>(1.0) - sheenFresnel);
			}

			clearcoatAttenuation = vec3<f32>(1.0) - clearcoatFresnel * clearcoat;
			baseLayerAttenuation = clearcoatAttenuation * albedoSheenScaling;
		}

		directLight += evaluateOpaquePBRLight(
			opaquePBRSurface, lightDirection, radiance, vec3<f32>(1.0)
		);
		directLight += transmittedDiffuse * transmissionAttenuation *
			nDotLTransmission * radiance;
	}
}
