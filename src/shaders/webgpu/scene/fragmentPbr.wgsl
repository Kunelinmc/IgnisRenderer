struct PBRSurfaceInput {
	normal: vec3<f32>,
	viewDirection: vec3<f32>,
	albedo: vec3<f32>,
	realF0: vec3<f32>,
	roughness: f32,
	metalness: f32,
	transmission: f32,
	energyCompensation: vec3<f32>,
	iridescence: f32,
	iridescenceIor: f32,
	iridescenceThickness: f32,
	anisotropyTangent: vec3<f32>,
	anisotropyBitangent: vec3<f32>,
	anisotropyStrength: f32,
	clearcoat: f32,
	clearcoatRoughness: f32,
	clearcoatNormal: vec3<f32>,
	sheenColor: vec3<f32>,
	sheenRoughness: f32,
}

// Isotropic thin-surface BTDF used by the direct-light evaluator's back hemisphere.
fn evaluatePBRTransmissionLight(
	surface: PBRSurfaceInput,
	lightDirection: vec3<f32>,
	radiance: vec3<f32>,
	visibility: vec3<f32>,
	ior: f32,
	volumeAttenuation: vec3<f32>
) -> vec3<f32> {
	let nDotV = clamp(dot(surface.normal, surface.viewDirection), 0.0, 1.0);
	let nDotL = clamp(-dot(surface.normal, lightDirection), 0.0, 1.0);
	let transmission = surface.transmission * (1.0 - surface.metalness);
	if (transmission <= 0.0 || nDotV <= 0.0 || nDotL <= 0.0) {
		return vec3<f32>(0.0);
	}

	// Mirroring the incoming light puts the transmission lobe along the view ray.
	let mirroredLight = reflect(lightDirection, surface.normal);
	let halfVector = safeNormalize(surface.viewDirection + mirroredLight, surface.normal);
	let vDotH = clamp(dot(surface.viewDirection, halfVector), 0.0, 1.0);
	if (vDotH <= 0.0 || dot(halfVector, lightDirection) >= 0.0) {
		return vec3<f32>(0.0);
	}
	let nDotH = clamp(dot(surface.normal, halfVector), 0.0, 1.0);
	let alpha = max(
		surface.roughness * surface.roughness * clamp(2.0 * ior - 2.0, 0.0, 1.0),
		0.0016
	);
	let alphaSq = alpha * alpha;
	let nDotHSq = nDotH * nDotH;
	// This form preserves the narrow GGX peak without a fixed denominator clamp.
	let distributionRatio = alpha / ((1.0 - nDotHSq) + alphaSq * nDotHSq);
	let distribution = distributionRatio * distributionRatio / PI;
	let smithL = nDotL + sqrt(alphaSq + (1.0 - alphaSq) * nDotL * nDotL);
	let smithV = nDotV + sqrt(alphaSq + (1.0 - alphaSq) * nDotV * nDotV);
	let btdf = distribution / (smithL * smithV);
	let fresnel = resolveIridescenceFresnel(
		vDotH, surface.realF0, surface.iridescence,
		surface.iridescenceThickness, surface.iridescenceIor
	);
	let clearcoatNdotV = max(dot(surface.clearcoatNormal, surface.viewDirection), PBR_MIN_NDOTV);
	let clearcoatAttenuation =
		1.0 - fresnelSchlickScalar(clearcoatNdotV, 0.04) * surface.clearcoat;
	return (vec3<f32>(1.0) - fresnel) * surface.albedo * volumeAttenuation *
		(transmission * clearcoatAttenuation * btdf * nDotL) * radiance * visibility;
}

fn evaluatePBRDirectLight(
	surface: PBRSurfaceInput,
	lightDirection: vec3<f32>,
	radiance: vec3<f32>,
	visibility: vec3<f32>,
	ior: f32,
	volumeAttenuation: vec3<f32>
) -> vec3<f32> {
	let nDotL = max(dot(surface.normal, lightDirection), 0.0);
	// Reflection and transmission occupy opposite light hemispheres.
	if (nDotL <= 0.0) {
		return evaluatePBRTransmissionLight(
			surface, lightDirection, radiance, visibility, ior, volumeAttenuation
		);
	}
	let nDotV = max(dot(surface.normal, surface.viewDirection), PBR_MIN_NDOTV);
	let halfVector = safeNormalize(
		surface.viewDirection + lightDirection,
		surface.viewDirection
	);
	let fresnel = resolveIridescenceFresnel(
		max(dot(halfVector, surface.viewDirection), 0.0),
		surface.realF0,
		surface.iridescence,
		surface.iridescenceThickness,
		surface.iridescenceIor
	);
	var specular = vec3<f32>(0.0);
	if (surface.anisotropyStrength > 1e-6) {
		specular = resolveAnisotropicSpecular(
			fresnel,
			surface.roughness,
			surface.anisotropyStrength,
			nDotL,
			nDotV,
			max(dot(surface.normal, halfVector), 0.0),
			dot(surface.anisotropyTangent, surface.viewDirection),
			dot(surface.anisotropyBitangent, surface.viewDirection),
			dot(surface.anisotropyTangent, lightDirection),
			dot(surface.anisotropyBitangent, lightDirection),
			dot(surface.anisotropyTangent, halfVector),
			dot(surface.anisotropyBitangent, halfVector)
		);
	} else {
		let ndf = distributionGGX(surface.normal, halfVector, surface.roughness);
		let geometry = geometrySmith(nDotV, nDotL, surface.roughness);
		let denominator = max(4.0 * nDotV * nDotL, 0.0001);
		specular = (ndf * geometry * fresnel) / denominator;
	}
	specular *= surface.energyCompensation;
	let kd =
		diffuseFresnelWeight(fresnel, surface.iridescence) *
		(1.0 - surface.metalness) *
		(1.0 - surface.transmission);
	let diffuse = (kd * surface.albedo) / PI;

	var clearcoatSpecular = vec3<f32>(0.0);
	var clearcoatFresnel = vec3<f32>(0.0);
	if (surface.clearcoat > 0.0) {
		let ncDotL = max(dot(surface.clearcoatNormal, lightDirection), 0.0);
		if (ncDotL > 0.0) {
			let ncDotV = max(
				dot(surface.clearcoatNormal, surface.viewDirection),
				PBR_MIN_NDOTV
			);
			let ccHalfVector = safeNormalize(
				surface.viewDirection + lightDirection,
				surface.viewDirection
			);
			let hccDotV = max(dot(ccHalfVector, surface.viewDirection), 0.0);
			let ccNdf = distributionGGX(
				surface.clearcoatNormal,
				ccHalfVector,
				surface.clearcoatRoughness
			);
			let ccGeometry = geometrySmithClearcoat(
				ncDotV,
				ncDotL,
				surface.clearcoatRoughness
			);
			let ccF = fresnelSchlickScalar(hccDotV, 0.04);
			let ccDenominator = max(4.0 * ncDotV * ncDotL, 0.0001);
			clearcoatSpecular = vec3<f32>((ccNdf * ccGeometry * ccF) / ccDenominator);
			clearcoatFresnel = vec3<f32>(ccF);
		}
	}

	let clearcoatAttenuation =
		vec3<f32>(1.0) - clearcoatFresnel * surface.clearcoat;
	var sheenSpecular = vec3<f32>(0.0);
	var albedoSheenScaling = vec3<f32>(1.0);
	if (max(max(surface.sheenColor.x, surface.sheenColor.y), surface.sheenColor.z) > 0.0) {
		let nDotH = max(dot(surface.normal, halfVector), 0.0);
		let sheenNdf = distributionCharlie(nDotH, max(surface.sheenRoughness, 0.04));
		let sheenVisibility = visibilityAshikhmin(nDotL, nDotV);
		sheenSpecular = surface.sheenColor * sheenNdf * sheenVisibility;
		let hDotV = max(dot(halfVector, surface.viewDirection), 0.0);
		let sheenFresnel = fresnelSchlick(hDotV, surface.sheenColor);
		albedoSheenScaling = max(vec3<f32>(0.0), vec3<f32>(1.0) - sheenFresnel);
	}
	let baseLayerAttenuation = clearcoatAttenuation * albedoSheenScaling;
	return (
		(diffuse + specular) * baseLayerAttenuation +
		clearcoatSpecular * surface.clearcoat +
		sheenSpecular * clearcoatAttenuation
	) * nDotL * radiance * visibility;
}

fn shadePBR(
	input: VertexOutput,
	material: ResolvedLightingMaterial,
	context: SceneShadingContext,
	includeTransmissionBackground: bool
) -> SceneFragmentOutput {
	let baseColor = context.baseColor;
	let alpha = context.alpha;
	let normal = context.normal;
	let geometryNormal = normal;
	let viewDir = context.viewDir;
	let emissive = context.emissive;
	let motion = context.motion;
	let linearDepth = context.linearDepth;

	var mrSample = vec4<f32>(1.0);
	if (hasPBRTexture(material, PBR_TEXTURE_METALLIC_ROUGHNESS_MAP)) {
		mrSample = sampleLinearTexture(
			metallicRoughnessTexture,
			metallicRoughnessSampler,
			TEX_METALLIC_ROUGHNESS,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var occlusionSample = vec4<f32>(1.0);
	if (hasPBRTexture(material, PBR_TEXTURE_OCCLUSION_MAP)) {
		occlusionSample = sampleLinearTexture(
			occlusionTexture,
			occlusionSampler,
			TEX_OCCLUSION,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var specularSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(material, PBR_FEATURE_SPECULAR) &&
		material.pbr.specularColorFactor.a > EPSILON &&
		max(max(material.pbr.specularColorFactor.r, material.pbr.specularColorFactor.g),
			material.pbr.specularColorFactor.b) > EPSILON &&
		hasPBRTexture(material, PBR_TEXTURE_SPECULAR_MAP)
	) {
		specularSample = sampleLinearTexture(
			specularTexture,
			specularSampler,
			TEX_SPECULAR,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var specularColorSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(material, PBR_FEATURE_SPECULAR) &&
		material.pbr.specularColorFactor.a > EPSILON &&
		max(max(material.pbr.specularColorFactor.r, material.pbr.specularColorFactor.g),
			material.pbr.specularColorFactor.b) > EPSILON &&
		hasPBRTexture(material, PBR_TEXTURE_SPECULAR_COLOR_MAP)
	) {
		specularColorSample = sampleColorTexture(
			specularColorTexture,
			specularColorSampler,
			TEX_SPECULAR_COLOR,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var clearcoatSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(material, PBR_FEATURE_CLEARCOAT) &&
		hasPBRTexture(material, PBR_TEXTURE_CLEARCOAT_MAP)
	) {
		clearcoatSample = sampleLinearTexture(
			clearcoatTexture,
			clearcoatSampler,
			TEX_CLEARCOAT,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var clearcoatRoughnessSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(material, PBR_FEATURE_CLEARCOAT) &&
		hasPBRTexture(material, PBR_TEXTURE_CLEARCOAT_ROUGHNESS_MAP)
	) {
		clearcoatRoughnessSample = sampleLinearTexture(
			clearcoatRoughnessTexture,
			clearcoatRoughnessSampler,
			TEX_CLEARCOAT_ROUGHNESS,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var sheenColorSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(material, PBR_FEATURE_SHEEN) &&
		hasPBRTexture(material, PBR_TEXTURE_SHEEN_COLOR_MAP)
	) {
		sheenColorSample = sampleColorTexture(
			sheenColorTexture,
			sheenColorSampler,
			TEX_SHEEN_COLOR,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var sheenRoughnessSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(material, PBR_FEATURE_SHEEN) &&
		hasPBRTexture(material, PBR_TEXTURE_SHEEN_ROUGHNESS_MAP)
	) {
		sheenRoughnessSample = sampleLinearTexture(
			sheenRoughnessTexture,
			sheenRoughnessSampler,
			TEX_SHEEN_ROUGHNESS,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var transmissionSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(material, PBR_FEATURE_TRANSMISSION) &&
		hasPBRTexture(material, PBR_TEXTURE_TRANSMISSION_MAP)
	) {
		transmissionSample = sampleLinearTexture(
			transmissionTexture,
			transmissionSampler,
			TEX_TRANSMISSION,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var thicknessSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(material, PBR_FEATURE_TRANSMISSION) &&
		hasPBRTexture(material, PBR_TEXTURE_THICKNESS_MAP)
	) {
		thicknessSample = sampleLinearTexture(
			thicknessTexture,
			transmissionSampler,
			TEX_THICKNESS,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var iridescenceSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(material, PBR_FEATURE_IRIDESCENCE) &&
		hasPBRTexture(material, PBR_TEXTURE_IRIDESCENCE_MAP)
	) {
		iridescenceSample = sampleLinearTexture(
			iridescenceTexture,
			transmissionSampler,
			TEX_IRIDESCENCE,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	var iridescenceThicknessSample = vec4<f32>(1.0);
	if (
		hasPBRFeature(material, PBR_FEATURE_IRIDESCENCE) &&
		hasPBRTexture(material, PBR_TEXTURE_IRIDESCENCE_THICKNESS_MAP)
	) {
		iridescenceThicknessSample = sampleLinearTexture(
			iridescenceThicknessTexture,
			transmissionSampler,
			TEX_IRIDESCENCE_THICKNESS,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}

	var normalSample = vec3<f32>(0.5, 0.5, 1.0);
	if (hasPBRTexture(material, PBR_TEXTURE_NORMAL_MAP)) {
		normalSample = sampleLinearTexture(
			normalTexture,
			normalSampler,
			TEX_NORMAL,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		).rgb;
	}
	var clearcoatNormalSample = vec3<f32>(0.5, 0.5, 1.0);
	if (
		hasPBRFeature(material, PBR_FEATURE_CLEARCOAT) &&
		hasPBRTexture(material, PBR_TEXTURE_CLEARCOAT_NORMAL_MAP)
	) {
		clearcoatNormalSample = sampleLinearTexture(
			clearcoatNormalTexture,
			clearcoatNormalSampler,
			TEX_CLEARCOAT_NORMAL,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		).rgb;
	}

	let roughness = clamp(material.pbr.surfaceParams0.x * mrSample.g, 0.04, 1.0);
	let metalness = clamp(material.pbr.surfaceParams0.y * mrSample.b, 0.0, 1.0);
	let reflectance = clamp(material.pbr.surfaceParams0.z, 0.0, 1.0);
	let occlusion = clamp(
		1.0 + material.pbr.surfaceParams1.x * (occlusionSample.r - 1.0),
		0.0,
		1.0
	);
	let clearcoat = clamp(material.pbr.surfaceParams1.z * clearcoatSample.r, 0.0, 1.0);
	let clearcoatRoughness = clamp(
		material.pbr.surfaceParams1.w * clearcoatRoughnessSample.g,
		0.04,
		1.0
	);
	let sheenColor = material.pbr.sheenColorClearcoatNormalScale.rgb * sheenColorSample.rgb;
	let sheenRoughness = clamp(material.pbr.surfaceParams2.x * sheenRoughnessSample.a, 0.0, 1.0);
	let transmission = clamp(material.pbr.surfaceParams2.y * transmissionSample.r, 0.0, 1.0);
	let ior = max(material.pbr.surfaceParams2.z, 1.0);
	let thickness = max(material.pbr.surfaceParams2.w * thicknessSample.g, 0.0);
	let attenuationDistance = material.pbr.surfaceParams3.x;
	let iridescence = clamp(material.pbr.surfaceParams3.y * iridescenceSample.r, 0.0, 1.0);
	let iridescenceIor = max(material.pbr.surfaceParams3.z, 1.0);
	let iridescenceThickness = max(
		mix(
			material.pbr.surfaceParams3.w,
			material.pbr.attenuationColor.a,
			iridescenceThicknessSample.g
		),
		0.0
	);
	let attenuationColor = clamp(material.pbr.attenuationColor.rgb, vec3<f32>(0.0001), vec3<f32>(1.0));

	let specularFactor = clamp(material.pbr.specularColorFactor.a * specularSample.a, 0.0, 1.0);
	let specularColor = clamp(
		material.pbr.specularColorFactor.rgb * specularColorSample.rgb,
		vec3<f32>(0.0),
		vec3<f32>(1.0)
	);

	var pbrNormal = applyNormalMap(
		normal,
		input.worldTangent,
		normalSample,
		material.pbr.surfaceParams1.y
	);
	if (dot(pbrNormal, geometryNormal) < 0.0) {
		pbrNormal = -pbrNormal;
	}
	let pbrGeometricNormal = resolvePBRGeometricNormal(input.worldPosition, viewDir, geometryNormal);

	var clearcoatNormal = applyNormalMap(
		pbrNormal,
		input.worldTangent,
		clearcoatNormalSample,
		material.pbr.sheenColorClearcoatNormalScale.a
	);
	if (dot(clearcoatNormal, pbrNormal) < 0.0) {
		clearcoatNormal = -clearcoatNormal;
	}
	// Build both authored normal-map frames before independently correcting them.
	pbrNormal = resolvePBRShadingNormal(pbrNormal, pbrGeometricNormal, viewDir);
	clearcoatNormal = resolvePBRShadingNormal(clearcoatNormal, pbrGeometricNormal, viewDir);
	let pbrShadowNormal = pbrNormal;

	let albedo = clamp(baseColor, vec3<f32>(0.0), vec3<f32>(1.0));
	let baseF0 = 0.16 * reflectance * reflectance;
	let f0Norm = min(vec3<f32>(baseF0) * specularColor * specularFactor, vec3<f32>(1.0));
	let realF0 = mix(f0Norm, albedo, vec3<f32>(metalness));
	let nDotV = max(dot(pbrNormal, viewDir), PBR_MIN_NDOTV);
	let energyCompensation =
		resolveSpecularEnergyCompensation(nDotV, roughness, realF0);
	let anisotropyData = resolveAnisotropyDirection(
		material,
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	);
	let anisotropyStrength = anisotropyData.z;
	let anisotropyTangent = resolveAnisotropyTangent(
		pbrNormal,
		input.worldTangent,
		anisotropyData.xy
	);
	let anisotropyBitangent = safeNormalize(
		cross(pbrNormal, anisotropyTangent),
		fallbackTangentFromNormal(pbrNormal)
	);
	let pbrSurface = PBRSurfaceInput(
		pbrNormal,
		viewDir,
		albedo,
		realF0,
		roughness,
		metalness,
		transmission,
		energyCompensation,
		iridescence,
		iridescenceIor,
		iridescenceThickness,
		anisotropyTangent,
		anisotropyBitangent,
		anisotropyStrength,
		clearcoat,
		clearcoatRoughness,
		clearcoatNormal,
		sheenColor,
		sheenRoughness
	);
	let reflectionDir = resolvePBRReflectionDirection(select(
		reflectViewDirection(pbrNormal, viewDir),
		resolveAnisotropicReflectionDirection(
			pbrNormal,
			viewDir,
			anisotropyBitangent,
			roughness,
			anisotropyStrength
		),
		anisotropyStrength > EPSILON
	), pbrGeometricNormal);
	let maxSheenColor = max(max(sheenColor.x, sheenColor.y), sheenColor.z);

	var transmissionPathLength = thickness;
	if (thickness > 0.0) {
		let eta = 1.0 / ior;
		let sin2ThetaT = eta * eta * (1.0 - nDotV * nDotV);
		let cosThetaT = sqrt(max(1.0 - sin2ThetaT, 0.0));
		transmissionPathLength = thickness / max(cosThetaT, PBR_MIN_NDOTV);
	}
	var volumeAttenuation = vec3<f32>(1.0);
	if (thickness > 0.0 && attenuationDistance > 0.0) {
		let absorb = -log(attenuationColor) / attenuationDistance;
		volumeAttenuation = exp(-absorb * transmissionPathLength);
	}

	var directLight = vec3<f32>(0.0);

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
		directLight += evaluatePBRDirectLight(
			pbrSurface, lightDirection, radiance, shadow, ior, volumeAttenuation
		);
	}

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

	let shAmbientEnabled = useSHAmbient();
	var ambientColor = frameLights.ambientColor.rgb;

	var diffuseAmbient = ambientColor / PI;
	var specularAmbientRadiance = vec3<f32>(0.0);
	if (shAmbientEnabled) {
		diffuseAmbient =
			sampleDiffuseProbeIrradiance(input.worldPosition, pbrNormal) / (255.0 * PI);

		let localSelection = selectTopTwoLocalLightProbes(input.worldPosition);
		let globalSpecularAmbient = sampleSHRadiance(reflectionDir);
		let localSpecularAmbient = sampleBlendedLocalLightProbeRadiance(
			localSelection,
			reflectionDir
		);
		specularAmbientRadiance = mix(
			globalSpecularAmbient,
			localSpecularAmbient.rgb,
			localSpecularAmbient.w
		) / 255.0;
	}

	let fAmbient = resolveIridescenceFresnel(
		nDotV,
		realF0,
		iridescence,
		iridescenceThickness,
		iridescenceIor
	);
	let refractionResult = refractViewDirection(viewDir, pbrNormal, ior);
	let isTIR = transmission > 0.0 && refractionResult.valid < 0.5;
	let effectiveFAmbient = select(fAmbient, vec3<f32>(1.0), isTIR);
	var specularIBLWeight = effectiveFAmbient * max(PBR_SPEC_FALLBACK, (1.0 - roughness) * 0.5);
	if (hasEnvSpecular()) {
		let splitSumFresnel = select(realF0, effectiveFAmbient, iridescence > EPSILON || isTIR);
		specularIBLWeight = resolveIBLSpecularEnergy(
			sampleBRDFLUT(nDotV, roughness), splitSumFresnel, energyCompensation
		);
	}
	var kdAmbient = vec3<f32>((1.0 - metalness) * (1.0 - transmission));
	if (shAmbientEnabled) {
		kdAmbient =
			(vec3<f32>(1.0) - specularIBLWeight) *
			(1.0 - metalness) *
			(1.0 - transmission);
	}
	let ktAmbient =
		(vec3<f32>(1.0) - effectiveFAmbient) *
		(1.0 - metalness) *
		transmission;

	let ccAmbientFresnel = select(0.0, fresnelSchlickScalar(nDotV, 0.04), clearcoat > 0.0);
	let diffuseCcAmbientFresnel = select(
		select(0.0, 0.04, clearcoat > 0.0),
		ccAmbientFresnel,
		shAmbientEnabled
	);
	let clearcoatDiffuseAttenuation = 1.0 - diffuseCcAmbientFresnel * clearcoat;
	let clearcoatSpecularAttenuation = 1.0 - ccAmbientFresnel * clearcoat;
	let baseAmbientAttenuation =
		vec3<f32>(clearcoatDiffuseAttenuation) * (vec3<f32>(1.0) - sheenColor * 0.5);

	var ambientLight = diffuseAmbient * albedo * kdAmbient * baseAmbientAttenuation;
	if (
		includeTransmissionBackground &&
		transmission > 0.0 &&
		refractionResult.valid > 0.5
	) {
		var transmissionRadiance = vec3<f32>(0.0);
		if (hasEnvSpecular()) {
			transmissionRadiance = sampleEnvironmentSpecular(
				refractionResult.direction,
				roughness,
				input.worldPosition
			);
		} else if (shAmbientEnabled) {
			let localSelection = selectTopTwoLocalLightProbes(input.worldPosition);
			let globalTransmissionRadiance = sampleSHRadiance(refractionResult.direction);
			let localTransmissionRadiance = sampleBlendedLocalLightProbeRadiance(
				localSelection,
				refractionResult.direction
			);
			transmissionRadiance = mix(
				globalTransmissionRadiance,
				localTransmissionRadiance.rgb,
				localTransmissionRadiance.w
			) / 255.0;
		}
		ambientLight +=
			transmissionRadiance *
			albedo *
			ktAmbient *
			volumeAttenuation *
			clearcoatSpecularAttenuation;
	}

	let specularAmbientFactor = max(PBR_SPEC_FALLBACK, (1.0 - roughness) * 0.5);
	let clearcoatAmbientFactor = max(PBR_SPEC_FALLBACK, (1.0 - clearcoatRoughness) * 0.5);
	if (hasEnvSpecular()) {
		let prefiltered = sampleEnvironmentSpecular(
			reflectionDir,
			roughness,
			input.worldPosition
		);
		ambientLight +=
			prefiltered *
			specularIBLWeight *
			clearcoatSpecularAttenuation;

		let clearcoatNdotV = max(dot(clearcoatNormal, viewDir), PBR_MIN_NDOTV);
		let clearcoatReflectionDir = reflectViewDirection(clearcoatNormal, viewDir);
		let clearcoatPrefiltered = sampleEnvironmentSpecular(
			clearcoatReflectionDir,
			clearcoatRoughness,
			input.worldPosition
		);
		let clearcoatBrdf = sampleBRDFLUT(clearcoatNdotV, clearcoatRoughness);
		ambientLight +=
			clearcoatPrefiltered *
			(ccAmbientFresnel * clearcoatBrdf.x + clearcoatBrdf.y) *
			clearcoat;
	} else {
		ambientLight +=
			specularAmbientRadiance *
			effectiveFAmbient *
			specularAmbientFactor *
			clearcoatSpecularAttenuation;
		ambientLight +=
			specularAmbientRadiance *
			ccAmbientFresnel *
			clearcoatAmbientFactor *
			clearcoat;
	}

	if (maxSheenColor > 0.0) {
		let sheenAmbientFactor = max(PBR_SPEC_FALLBACK, (1.0 - max(sheenRoughness, 0.04)) * 0.5);
		ambientLight +=
			specularAmbientRadiance *
			sheenColor *
			sheenAmbientFactor *
			clearcoatSpecularAttenuation;
	}

	ambientLight *= occlusion;

	let finalLinear = max(directLight + ambientLight + emissive, vec3<f32>(0.0));
	let outputAlpha = resolveTransmissionAlpha(
		alpha,
		transmission,
		nDotV,
		realF0,
		iridescence,
		iridescenceThickness,
		iridescenceIor
	);
	return buildSceneOutput(
		finalLinear,
		outputAlpha,
		albedo,
		pbrNormal,
		roughness,
		metalness,
		emissive,
		occlusion,
		motion,
		linearDepth
	);
}

fn resolveTransmissionAlpha(
	baseAlpha: f32,
	transmission: f32,
	nDotV: f32,
	f0: vec3<f32>,
	iridescence: f32,
	iridescenceThickness: f32,
	iridescenceIor: f32
) -> f32 {
	let clampedTransmission = clamp(transmission, 0.0, 1.0);
	if (clampedTransmission <= EPSILON) {
		return clamp(baseAlpha, 0.0, 1.0);
	}

	let fresnel = resolveIridescenceFresnel(
		nDotV,
		f0,
		iridescence,
		iridescenceThickness,
		iridescenceIor
	);
	let fresnelAverage = clamp(
		(fresnel.x + fresnel.y + fresnel.z) * (1.0 / 3.0),
		0.0,
		1.0
	);
	let floorAlpha = max(0, fresnelAverage);
	let blended =
		baseAlpha * (1.0 - clampedTransmission) +
		floorAlpha * clampedTransmission;
	return clamp(max(floorAlpha, blended), 0.0, 1.0);
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
