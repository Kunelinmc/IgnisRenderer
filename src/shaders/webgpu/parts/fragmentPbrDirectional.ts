export const WEBGPU_SCENE_FRAGMENT_PBR_DIRECTIONAL = /* wgsl */ `
	let directionalCount = u32(frame.lightCounts.x + 0.5);
	for (var i: u32 = 0u; i < directionalCount; i = i + 1u) {
		let lightDirection = safeNormalize(
			frame.directionalLights[i].direction.xyz,
			vec3<f32>(0.0, 1.0, 0.0)
		);
		let radiance = frame.directionalLights[i].color.xyz;
		let nDotLRaw = dot(pbrNormal, lightDirection);
		let nDotL = max(nDotLRaw, 0.0);
		let nDotLTransmission = max(-nDotLRaw, 0.0);
		if (nDotL <= 0.0 && nDotLTransmission <= 0.0) {
			continue;
		}

		let shadow = sampleDirectionalShadowVisibility(
			i,
			input.worldPosition,
			pbrNormal,
			lightDirection
		);
		let fView = fresnelSchlick(nDotV, realF0);
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
			let ndf = distributionGGX(pbrNormal, halfVector, roughness);
			let geometry = geometrySmith(nDotV, nDotL, roughness);
			let fresnel = fresnelSchlick(max(dot(halfVector, viewDir), 0.0), realF0);
			let denominator = max(4.0 * nDotV * nDotL, 0.0001);

			specular = (ndf * geometry * fresnel) / denominator;

			let kd = (vec3<f32>(1.0) - fresnel) * (1.0 - metalness) * (1.0 - transmission);
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

		directLight += (
			((diffuse + specular) * baseLayerAttenuation +
				clearcoatSpecular * clearcoat +
				sheenSpecular * clearcoatAttenuation) * nDotL +
			transmittedDiffuse * transmissionAttenuation * nDotLTransmission
		) * radiance * shadow;
	}
`;
