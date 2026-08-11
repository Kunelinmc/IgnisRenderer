@group(3) @binding(0) var gAlbedoAlphaIn: texture_2d<f32>;
@group(3) @binding(1) var gNormalRoughMetalIn: texture_2d<f32>;
@group(3) @binding(2) var gEmissiveOcclusionIn: texture_2d<f32>;
@group(3) @binding(3) var gMotionDepthIn: texture_2d<f32>;
@group(3) @binding(4) var gSpecularIn: texture_2d<f32>;
@group(3) @binding(5) var gCoatSheenIn: texture_2d<f32>;
@group(3) @binding(6) var gSheenReflectanceIn: texture_2d<f32>;
@group(3) @binding(7) var gMaterialExt0In: texture_2d<f32>;
@group(3) @binding(8) var gMaterialExt3In: texture_2d<u32>;

struct DeferredVSOut {
	@builtin(position) position: vec4<f32>,
	@location(0) uv: vec2<f32>,
}

struct DeferredSurface {
	worldPosition: vec3<f32>,
	viewDir: vec3<f32>,
	normal: vec3<f32>,
	shadowNormal: vec3<f32>,
	albedo: vec3<f32>,
	alpha: f32,
	roughness: f32,
	metalness: f32,
	emissive: vec3<f32>,
	occlusion: f32,
	linearDepth: f32,
	shadingModel: u32,
	specularColor: vec3<f32>,
	specularFactor: f32,
	clearcoat: f32,
	clearcoatRoughness: f32,
	clearcoatNormal: vec3<f32>,
	sheenColor: vec3<f32>,
	sheenRoughness: f32,
	reflectance: f32,
	iridescence: f32,
	iridescenceIor: f32,
	iridescenceThickness: f32,
	anisotropyTangent: vec3<f32>,
	anisotropyBitangent: vec3<f32>,
	anisotropyStrength: f32,
	receiveShadows: bool,
	pixelPosition: vec2<f32>,
}

struct DeferredPBRContext {
	realF0: vec3<f32>,
	nDotV: f32,
	energyCompensation: vec3<f32>,
}

@vertex
fn vsMainDeferredLighting(
	@builtin(vertex_index) vertexIndex: u32
) -> DeferredVSOut {
	var positions = array<vec2<f32>, 3>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>(3.0, -1.0),
		vec2<f32>(-1.0, 3.0)
	);

	let pos = positions[vertexIndex];
	var output: DeferredVSOut;
	output.position = vec4<f32>(pos, 0.0, 1.0);
	output.uv = vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
	return output;
}

fn isOrthographicDeferredCamera() -> bool {
	return frame.environmentBasisBackward.w > 0.5;
}

fn decodeOctahedralNormal(encoded: vec2<f32>) -> vec3<f32> {
	let oct = encoded * 2.0 - vec2<f32>(1.0);
	var n = vec3<f32>(oct.x, oct.y, 1.0 - abs(oct.x) - abs(oct.y));
	if (n.z < 0.0) {
		n = vec3<f32>(octahedralWrap(n.xy), n.z);
	}
	return safeNormalize(n, vec3<f32>(0.0, 0.0, 1.0));
}

fn decodeDeferredNormal(encoded: vec2<f32>) -> vec3<f32> {
	let vn = decodeOctahedralNormal(encoded);
	return safeNormalize(
		frame.environmentBasisRight.xyz * vn.x +
			frame.environmentBasisUp.xyz * vn.y +
			frame.environmentBasisBackward.xyz * vn.z,
		frame.environmentBasisBackward.xyz
	);
}

fn reconstructDeferredWorldPosition(uv: vec2<f32>, depth: f32) -> vec3<f32> {
	let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
	let right = frame.environmentBasisRight.xyz;
	let up = frame.environmentBasisUp.xyz;
	let backward = frame.environmentBasisBackward.xyz;
	if (isOrthographicDeferredCamera()) {
		return frame.cameraPosition.xyz +
			right * ndc.x * frame.environmentBasisRight.w +
			up * ndc.y * frame.environmentBasisUp.w -
			backward * depth;
	}
	let cx = ndc.x * frame.environmentBasisUp.w * frame.environmentBasisRight.w * depth;
	let cy = ndc.y * frame.environmentBasisRight.w * depth;
	return frame.cameraPosition.xyz + right * cx + up * cy - backward * depth;
}

fn applyDeferredFog(color: vec3<f32>, linearDepth: f32) -> vec3<f32> {
	let fogMode = i32(floor(fog.fogParams0.x + 0.5));
	let fogFactor = ignisComputeFogFactor(
		fogMode,
		max(linearDepth, 0.0),
		fog.fogParams0.y,
		fog.fogParams0.z,
		fog.fogParams0.w,
		fog.fogParams1.w
	);
	return max(mix(color, fog.fogParams1.rgb, fogFactor), vec3<f32>(0.0));
}

fn loadDeferredSurface(input: DeferredVSOut) -> DeferredSurface {
	let coord = vec2<i32>(i32(input.position.x), i32(input.position.y));
	let albedoAlpha = textureLoad(gAlbedoAlphaIn, coord, 0);
	let normalRoughMetal = textureLoad(gNormalRoughMetalIn, coord, 0);
	let emissiveOcclusion = textureLoad(gEmissiveOcclusionIn, coord, 0);
	let motionDepth = textureLoad(gMotionDepthIn, coord, 0);
	let materialWord = decodeDeferredMaterialWord(motionDepth.w);
	let shadingModel = deferredShadingModel(materialWord);
	let isPhong = shadingModel == SHADING_PHONG || shadingModel == SHADING_FLAT;
	let hasClearcoat = deferredHasFeature(
		materialWord,
		DEFERRED_MATERIAL_CLEARCOAT_BIT
	);
	let hasSheen = deferredHasFeature(materialWord, DEFERRED_MATERIAL_SHEEN_BIT);
	let hasIridescence = deferredHasFeature(
		materialWord,
		DEFERRED_MATERIAL_IRIDESCENCE_BIT
	);
	let hasAnisotropy = deferredHasFeature(
		materialWord,
		DEFERRED_MATERIAL_ANISOTROPY_BIT
	);
	let hasSpecular = deferredHasFeature(
		materialWord,
		DEFERRED_MATERIAL_SPECULAR_BIT
	);
	let receiveShadows = deferredHasFeature(
		materialWord,
		DEFERRED_MATERIAL_RECEIVE_SHADOWS_BIT
	);
	var specular = vec4<f32>(1.0);
	var coatSheen = vec4<f32>(0.0, 0.04, 0.0, 1.3);
	var sheenReflectance = vec4<f32>(0.0, 0.0, 0.0, 0.5);
	var materialExt0 = vec4<f32>(0.5, 0.5, 0.0, 0.0);
	var materialExt3 = vec4<u32>(0u);
	if (isPhong || hasSpecular) {
		specular = textureLoad(gSpecularIn, coord, 0);
	}
	if (hasClearcoat || hasSheen || hasIridescence) {
		coatSheen = textureLoad(gCoatSheenIn, coord, 0);
	}
	if (isPhong || hasSheen || hasSpecular) {
		sheenReflectance = textureLoad(gSheenReflectanceIn, coord, 0);
	}
	if (hasClearcoat || hasIridescence) {
		materialExt0 = textureLoad(gMaterialExt0In, coord, 0);
	}
	if (hasAnisotropy) {
		materialExt3 = textureLoad(gMaterialExt3In, coord, 0);
	}
	let normal = decodeDeferredNormal(normalRoughMetal.xy);
	let clearcoatNormal = select(
		normal,
		decodeDeferredNormal(materialExt0.xy),
		hasClearcoat
	);
	let anisotropyTangentCandidate = decodeDeferredNormal(
		decodeDeferredExt3Normal(materialExt3.xy)
	);
	let anisotropyTangent = safeNormalize(
		anisotropyTangentCandidate -
			normal * dot(normal, anisotropyTangentCandidate),
		fallbackTangentFromNormal(normal)
	);
	let anisotropyBitangent = safeNormalize(
		cross(normal, anisotropyTangent),
		fallbackTangentFromNormal(normal)
	);
	let worldPosition = reconstructDeferredWorldPosition(input.uv, motionDepth.z);
	let viewDir = safeNormalize(
		frame.cameraPosition.xyz - worldPosition,
		-frame.environmentBasisBackward.xyz
	);
	return DeferredSurface(
		worldPosition,
		viewDir,
		normal,
		normal,
		clamp(albedoAlpha.rgb, vec3<f32>(0.0), vec3<f32>(1.0)),
		clamp(albedoAlpha.a, 0.0, 1.0),
		clamp(normalRoughMetal.z, 0.04, 1.0),
		clamp(normalRoughMetal.w, 0.0, 1.0),
		max(emissiveOcclusion.rgb, vec3<f32>(0.0)),
		clamp(emissiveOcclusion.a, 0.0, 1.0),
		max(motionDepth.z, 0.0),
		shadingModel,
		clamp(specular.rgb, vec3<f32>(0.0), vec3<f32>(1.0)),
		select(clamp(specular.a, 0.0, 1.0), max(specular.a, 0.0), isPhong),
		clamp(coatSheen.x, 0.0, 1.0),
		clamp(coatSheen.y, 0.04, 1.0),
		clearcoatNormal,
		clamp(sheenReflectance.rgb, vec3<f32>(0.0), vec3<f32>(1.0)),
		clamp(coatSheen.z, 0.0, 1.0),
		clamp(sheenReflectance.a, 0.0, 1.0),
		clamp(materialExt0.z, 0.0, 1.0),
		max(coatSheen.w, 1.0),
		max(materialExt0.w, 0.0),
		anisotropyTangent,
		anisotropyBitangent,
		decodeDeferredExt3Strength(materialExt3.z),
		receiveShadows,
		input.position.xy
	);
}

fn sampleDeferredDirectionalShadow(
	surface: DeferredSurface,
	lightIndex: u32,
	lightDirection: vec3<f32>
) -> vec3<f32> {
	if (!surface.receiveShadows) {
		return vec3<f32>(1.0);
	}
	return sampleDirectionalShadowVisibility(
		lightIndex,
		surface.worldPosition,
		surface.shadowNormal,
		lightDirection,
		surface.linearDepth
	);
}

fn sampleDeferredSpotShadow(
	surface: DeferredSurface,
	shadowIndex: u32,
	lightDirection: vec3<f32>
) -> vec3<f32> {
	if (!surface.receiveShadows) {
		return vec3<f32>(1.0);
	}
	return sampleSpotShadowVisibility(
		shadowIndex,
		surface.worldPosition,
		surface.shadowNormal,
		lightDirection
	);
}

fn buildDeferredPBRContext(surface: DeferredSurface) -> DeferredPBRContext {
	let baseF0 = 0.16 * surface.reflectance * surface.reflectance;
	let f0Norm = min(
		vec3<f32>(baseF0) * surface.specularColor * surface.specularFactor,
		vec3<f32>(1.0)
	);
	let realF0 = mix(f0Norm, surface.albedo, vec3<f32>(surface.metalness));
	let nDotV = max(dot(surface.normal, surface.viewDir), PBR_MIN_NDOTV);
	let energyCompensation = resolveSpecularEnergyCompensation(
		nDotV,
		surface.roughness,
		realF0
	);
	return DeferredPBRContext(realF0, nDotV, energyCompensation);
}

fn evaluateDeferredPBRLight(
	surface: DeferredSurface,
	pbr: DeferredPBRContext,
	lightDirection: vec3<f32>,
	radiance: vec3<f32>,
	shadow: vec3<f32>
) -> vec3<f32> {
	return evaluateOpaquePBRLight(
		OpaquePBRSurfaceInput(
			surface.normal,
			surface.viewDir,
			surface.albedo,
			pbr.realF0,
			surface.roughness,
			surface.metalness,
			0.0,
			pbr.energyCompensation,
			surface.iridescence,
			surface.iridescenceIor,
			surface.iridescenceThickness,
			surface.anisotropyTangent,
			surface.anisotropyBitangent,
			surface.anisotropyStrength,
			surface.clearcoat,
			surface.clearcoatRoughness,
			surface.clearcoatNormal,
			surface.sheenColor,
			surface.sheenRoughness
		),
		lightDirection,
		radiance,
		shadow
	);
}

fn evaluateDeferredPBR(surface: DeferredSurface) -> vec3<f32> {
	var directLight = vec3<f32>(0.0);
	let pbr = buildDeferredPBRContext(surface);
	let directionalCount = u32(frame.lightCounts.x + 0.5);
	for (var i: u32 = 0u; i < directionalCount; i = i + 1u) {
		let lightDirection = safeNormalize(
			frameLights.directionalLights[i].direction.xyz,
			vec3<f32>(0.0, 1.0, 0.0)
		);
		let shadow = sampleDeferredDirectionalShadow(surface, i, lightDirection);
		directLight += evaluateDeferredPBRLight(
			surface,
			pbr,
			lightDirection,
			frameLights.directionalLights[i].color.xyz,
			shadow
		);
	}

	if (isClusteredLightingEnabled()) {
		let clusterHeader = getClusterHeaderForFragment(
			surface.pixelPosition,
			surface.linearDepth
		);
		let clusterEntryCount = getClusterEntryCount(clusterHeader);
		let clusterLightCount = activeClusteredLightCount();
		for (var entryIndex: u32 = 0u; entryIndex < clusterEntryCount; entryIndex = entryIndex + 1u) {
			let packedRef = clusterIndices.indices[clusterHeader.offset + entryIndex];
			let clusterRef = decodeClusteredLightRef(packedRef);
			if (clusterRef.lightIndex >= clusterLightCount) {
				continue;
			}
			if (clusterRef.lightType == CLUSTER_LIGHT_TYPE_AREA) {
				let areaRecord = clusteredRecordToAreaLight(clusterRef.lightIndex);
				for (
					var sampleIndex: u32 = 0u;
					sampleIndex < AREA_LIGHT_SAMPLE_COUNT;
					sampleIndex = sampleIndex + 1u
				) {
					let areaLight = evaluateAreaLight(
						areaRecord,
						surface.worldPosition,
						sampleIndex
					);
					if (!areaLight.valid) {
						continue;
					}
					directLight += evaluateDeferredPBRLight(
						surface,
						pbr,
						areaLight.direction,
						areaLight.radiance,
						vec3<f32>(1.0)
					);
				}
				continue;
			}
			let positionRange = clusterPositionRanges.values[clusterRef.lightIndex];
			let colorInner = clusterColorInners.values[clusterRef.lightIndex];
			let toLight = positionRange.xyz - surface.worldPosition;
			let distanceSq = dot(toLight, toLight);
			let distanceValue = sqrt(max(distanceSq, EPSILON));
			let lightRange = positionRange.w;
			if (distanceValue > lightRange) {
				continue;
			}

			let lightDirection = toLight / distanceValue;
			var attenuation = pointAttenuation(distanceSq, lightRange);
			var shadow = vec3<f32>(1.0);
			if (clusterRef.lightType == CLUSTER_LIGHT_TYPE_SPOT) {
				let directionOuter =
					clusterDirectionOuters.values[clusterRef.lightIndex];
				let lightToPoint = -lightDirection;
				let coneDirection = safeNormalize(
					directionOuter.xyz,
					vec3<f32>(0.0, -1.0, 0.0)
				);
				let coneAttenuation = spotAttenuation(
					dot(lightToPoint, coneDirection),
					directionOuter.w,
					colorInner.w
				);
				if (coneAttenuation <= 0.0) {
					continue;
				}
				attenuation *= coneAttenuation;
				let shadowIndex =
					clusterMetadata.values[clusterRef.lightIndex].shadowIndex;
				if (clusterRef.shadowed && shadowIndex < 8u) {
					shadow = sampleDeferredSpotShadow(
						surface,
						shadowIndex,
						lightDirection
					);
				}
			} else if (clusterRef.lightType != CLUSTER_LIGHT_TYPE_POINT) {
				continue;
			}

			directLight += evaluateDeferredPBRLight(
				surface,
				pbr,
				lightDirection,
				colorInner.xyz * attenuation,
				shadow
			);
		}
	} else {
		let pointCount = u32(frame.lightCounts.y + 0.5);
		for (var i: u32 = 0u; i < pointCount; i = i + 1u) {
			let toLight = frameLights.pointLights[i].positionRange.xyz - surface.worldPosition;
			let distanceSq = dot(toLight, toLight);
			let distanceValue = sqrt(max(distanceSq, EPSILON));
			let lightRange = frameLights.pointLights[i].positionRange.w;
			if (distanceValue > lightRange) {
				continue;
			}
			let lightDirection = toLight / distanceValue;
			let radiance =
				frameLights.pointLights[i].color.xyz *
				pointAttenuation(distanceSq, lightRange);
			directLight += evaluateDeferredPBRLight(
				surface,
				pbr,
				lightDirection,
				radiance,
				vec3<f32>(1.0)
			);
		}

		let spotCount = u32(frame.lightCounts.z + 0.5);
		for (var i: u32 = 0u; i < spotCount; i = i + 1u) {
			let toLight = frameLights.spotLights[i].positionRange.xyz - surface.worldPosition;
			let distanceSq = dot(toLight, toLight);
			let distanceValue = sqrt(max(distanceSq, EPSILON));
			let lightRange = frameLights.spotLights[i].positionRange.w;
			if (distanceValue > lightRange) {
				continue;
			}
			let lightDirection = toLight / distanceValue;
			let coneDirection = safeNormalize(
				frameLights.spotLights[i].directionOuter.xyz,
				vec3<f32>(0.0, -1.0, 0.0)
			);
			let coneAttenuation = spotAttenuation(
				dot(-lightDirection, coneDirection),
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
			let shadow = sampleDeferredSpotShadow(surface, i, lightDirection);
			directLight += evaluateDeferredPBRLight(
				surface,
				pbr,
				lightDirection,
				radiance,
				shadow
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
					surface.worldPosition,
					sampleIndex
				);
				if (!areaLight.valid) {
					continue;
				}
				directLight += evaluateDeferredPBRLight(
					surface,
					pbr,
					areaLight.direction,
					areaLight.radiance,
					vec3<f32>(1.0)
				);
			}
		}
	}

	var ambientColor = frame.ambientColor.rgb;

	var diffuseAmbient = ambientColor / PI;
	var specularAmbientRadiance = ambientColor / PI;
	if (useSHAmbient()) {
		diffuseAmbient =
			sampleDiffuseProbeIrradiance(surface.worldPosition, surface.normal) /
			(255.0 * PI);

		let reflectionDir = select(
			reflectViewDirection(surface.normal, surface.viewDir),
			resolveAnisotropicReflectionDirection(
				surface.normal,
				surface.viewDir,
				surface.anisotropyBitangent,
				surface.roughness,
				surface.anisotropyStrength
			),
			surface.anisotropyStrength > EPSILON
		);
		let localSelection = selectTopTwoLocalLightProbes(surface.worldPosition);
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
		pbr.nDotV,
		pbr.realF0,
		surface.iridescence,
		surface.iridescenceThickness,
		surface.iridescenceIor
	);
	let kdAmbient =
		diffuseFresnelWeight(fAmbient, surface.iridescence) *
		(1.0 - surface.metalness);
	let ccAmbientFresnel = select(
		0.0,
		fresnelSchlickScalar(pbr.nDotV, 0.04),
		surface.clearcoat > 0.0
	);
	let clearcoatAmbientAttenuation =
		1.0 - ccAmbientFresnel * surface.clearcoat;
	let baseAmbientAttenuation =
		vec3<f32>(clearcoatAmbientAttenuation) *
		(vec3<f32>(1.0) - surface.sheenColor * 0.5);
	var ambientLight =
		diffuseAmbient * surface.albedo * kdAmbient * baseAmbientAttenuation;

	let reflectionDir = select(
		reflectViewDirection(surface.normal, surface.viewDir),
		resolveAnisotropicReflectionDirection(
			surface.normal,
			surface.viewDir,
			surface.anisotropyBitangent,
			surface.roughness,
			surface.anisotropyStrength
		),
		surface.anisotropyStrength > EPSILON
	);
	let clearcoatNdotV =
		max(dot(surface.clearcoatNormal, surface.viewDir), PBR_MIN_NDOTV);
	let clearcoatReflectionDir = reflectViewDirection(
		surface.clearcoatNormal,
		surface.viewDir
	);
	if (hasEnvSpecular()) {
		let prefiltered = sampleEnvironmentSpecular(
			reflectionDir,
			surface.roughness,
			surface.worldPosition
		);
		let brdf = sampleBRDFLUT(pbr.nDotV, surface.roughness);
		let splitSumFresnel = select(
			pbr.realF0,
			fAmbient,
			surface.iridescence > EPSILON
		);
		ambientLight +=
			prefiltered *
			(splitSumFresnel * brdf.x + vec3<f32>(brdf.y)) *
			pbr.energyCompensation *
			clearcoatAmbientAttenuation;

		let clearcoatPrefiltered = sampleEnvironmentSpecular(
			clearcoatReflectionDir,
			surface.clearcoatRoughness,
			surface.worldPosition
		);
		let clearcoatBrdf = sampleBRDFLUT(
			clearcoatNdotV,
			surface.clearcoatRoughness
		);
		ambientLight +=
			clearcoatPrefiltered *
			(ccAmbientFresnel * clearcoatBrdf.x + clearcoatBrdf.y) *
			surface.clearcoat;
	} else {
		let specularAmbientFactor =
			max(PBR_SPEC_FALLBACK, (1.0 - surface.roughness) * 0.5);
		let clearcoatAmbientFactor =
			max(PBR_SPEC_FALLBACK, (1.0 - surface.clearcoatRoughness) * 0.5);
		ambientLight +=
			specularAmbientRadiance *
			fAmbient *
			specularAmbientFactor *
			clearcoatAmbientAttenuation;
		ambientLight +=
			specularAmbientRadiance *
			ccAmbientFresnel *
			clearcoatAmbientFactor *
			surface.clearcoat;
	}

	let maxSheenColor =
		max(max(surface.sheenColor.x, surface.sheenColor.y), surface.sheenColor.z);
	if (maxSheenColor > 0.0) {
		let sheenAmbientFactor = max(
			PBR_SPEC_FALLBACK,
			(1.0 - max(surface.sheenRoughness, 0.04)) * 0.5
		);
		ambientLight +=
			specularAmbientRadiance *
			surface.sheenColor *
			sheenAmbientFactor *
			clearcoatAmbientAttenuation;
	}

	return max(directLight + ambientLight * surface.occlusion + surface.emissive, vec3<f32>(0.0));
}

fn evaluateDeferredPhong(surface: DeferredSurface) -> vec3<f32> {
	var ambientBase = frame.ambientColor.rgb;
	if (useSHAmbient()) {
		ambientBase =
			sampleDiffuseProbeIrradiance(surface.worldPosition, surface.normal) / 255.0;
	}
	var direct = vec3<f32>(0.0);
	let shininess = max(surface.specularFactor, 0.0);

	let directionalCount = u32(frame.lightCounts.x + 0.5);
	for (var i: u32 = 0u; i < directionalCount; i = i + 1u) {
		let lightDirection = safeNormalize(
			frameLights.directionalLights[i].direction.xyz,
			vec3<f32>(0.0, 1.0, 0.0)
		);
		let nDotL = max(dot(surface.normal, lightDirection), 0.0);
		if (nDotL <= 0.0) {
			continue;
		}
		let shadow = sampleDeferredDirectionalShadow(surface, i, lightDirection);
		direct += evaluateOpaquePhongLight(
			surface.normal, surface.viewDir, surface.albedo,
			surface.specularColor, shininess, lightDirection,
			frameLights.directionalLights[i].color.xyz, shadow
		);
	}

	if (isClusteredLightingEnabled()) {
		let clusterHeader = getClusterHeaderForFragment(
			surface.pixelPosition,
			surface.linearDepth
		);
		let clusterEntryCount = getClusterEntryCount(clusterHeader);
		let clusterLightCount = activeClusteredLightCount();
		for (var entryIndex: u32 = 0u; entryIndex < clusterEntryCount; entryIndex = entryIndex + 1u) {
			let packedRef = clusterIndices.indices[clusterHeader.offset + entryIndex];
			let clusterRef = decodeClusteredLightRef(packedRef);
			if (clusterRef.lightIndex >= clusterLightCount) {
				continue;
			}
			if (clusterRef.lightType == CLUSTER_LIGHT_TYPE_AREA) {
				let areaRecord = clusteredRecordToAreaLight(clusterRef.lightIndex);
				for (
					var sampleIndex: u32 = 0u;
					sampleIndex < AREA_LIGHT_SAMPLE_COUNT;
					sampleIndex = sampleIndex + 1u
				) {
					let areaLight = evaluateAreaLight(
						areaRecord,
						surface.worldPosition,
						sampleIndex
					);
					if (!areaLight.valid) {
						continue;
					}
					let lightDirection = areaLight.direction;
					let nDotL = max(dot(surface.normal, lightDirection), 0.0);
					if (nDotL <= 0.0) {
						continue;
					}
					direct += evaluateOpaquePhongLight(
						surface.normal, surface.viewDir, surface.albedo,
						surface.specularColor, shininess, lightDirection,
						areaLight.radiance, vec3<f32>(1.0)
					);
				}
				continue;
			}
			let positionRange = clusterPositionRanges.values[clusterRef.lightIndex];
			let colorInner = clusterColorInners.values[clusterRef.lightIndex];
			let toLight = positionRange.xyz - surface.worldPosition;
			let distanceSq = dot(toLight, toLight);
			let distanceValue = sqrt(max(distanceSq, EPSILON));
			let lightRange = positionRange.w;
			if (distanceValue > lightRange) {
				continue;
			}
			let lightDirection = toLight / distanceValue;
			var attenuation = pointAttenuation(distanceSq, lightRange);
			var shadow = vec3<f32>(1.0);
			if (clusterRef.lightType == CLUSTER_LIGHT_TYPE_SPOT) {
				let directionOuter =
					clusterDirectionOuters.values[clusterRef.lightIndex];
				let coneDirection = safeNormalize(
					directionOuter.xyz,
					vec3<f32>(0.0, -1.0, 0.0)
				);
				let coneAttenuation = spotAttenuation(
					dot(-lightDirection, coneDirection),
					directionOuter.w,
					colorInner.w
				);
				if (coneAttenuation <= 0.0) {
					continue;
				}
				attenuation *= coneAttenuation;
				let shadowIndex =
					clusterMetadata.values[clusterRef.lightIndex].shadowIndex;
				if (clusterRef.shadowed && shadowIndex < 8u) {
					shadow = sampleDeferredSpotShadow(
						surface,
						shadowIndex,
						lightDirection
					);
				}
			} else if (clusterRef.lightType != CLUSTER_LIGHT_TYPE_POINT) {
				continue;
			}
			let nDotL = max(dot(surface.normal, lightDirection), 0.0);
			if (nDotL <= 0.0) {
				continue;
			}
			direct += evaluateOpaquePhongLight(
				surface.normal, surface.viewDir, surface.albedo,
				surface.specularColor, shininess, lightDirection,
				colorInner.xyz * attenuation, shadow
			);
		}
	} else {
		let pointCount = u32(frame.lightCounts.y + 0.5);
		for (var i: u32 = 0u; i < pointCount; i = i + 1u) {
			let toLight = frameLights.pointLights[i].positionRange.xyz - surface.worldPosition;
			let distanceSq = dot(toLight, toLight);
			let distanceValue = sqrt(max(distanceSq, EPSILON));
			let lightRange = frameLights.pointLights[i].positionRange.w;
			if (distanceValue > lightRange) {
				continue;
			}
			let lightDirection = toLight / distanceValue;
			let nDotL = max(dot(surface.normal, lightDirection), 0.0);
			if (nDotL <= 0.0) {
				continue;
			}
			direct += evaluateOpaquePhongLight(
				surface.normal, surface.viewDir, surface.albedo,
				surface.specularColor, shininess, lightDirection,
				frameLights.pointLights[i].color.xyz *
					pointAttenuation(distanceSq, lightRange),
				vec3<f32>(1.0)
			);
		}

		let spotCount = u32(frame.lightCounts.z + 0.5);
		for (var i: u32 = 0u; i < spotCount; i = i + 1u) {
			let toLight = frameLights.spotLights[i].positionRange.xyz - surface.worldPosition;
			let distanceSq = dot(toLight, toLight);
			let distanceValue = sqrt(max(distanceSq, EPSILON));
			let lightRange = frameLights.spotLights[i].positionRange.w;
			if (distanceValue > lightRange) {
				continue;
			}
			let lightDirection = toLight / distanceValue;
			let coneDirection = safeNormalize(
				frameLights.spotLights[i].directionOuter.xyz,
				vec3<f32>(0.0, -1.0, 0.0)
			);
			let coneAttenuation = spotAttenuation(
				dot(-lightDirection, coneDirection),
				frameLights.spotLights[i].directionOuter.w,
				frameLights.spotLights[i].colorInner.w
			);
			if (coneAttenuation <= 0.0) {
				continue;
			}
			let nDotL = max(dot(surface.normal, lightDirection), 0.0);
			if (nDotL <= 0.0) {
				continue;
			}
			let shadow = sampleDeferredSpotShadow(surface, i, lightDirection);
			direct += evaluateOpaquePhongLight(
				surface.normal, surface.viewDir, surface.albedo,
				surface.specularColor, shininess, lightDirection,
				frameLights.spotLights[i].colorInner.xyz *
					pointAttenuation(distanceSq, lightRange) * coneAttenuation,
				shadow
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
					surface.worldPosition,
					sampleIndex
				);
				if (!areaLight.valid) {
					continue;
				}
				let lightDirection = areaLight.direction;
				let nDotL = max(dot(surface.normal, lightDirection), 0.0);
				if (nDotL <= 0.0) {
					continue;
				}
				direct += evaluateOpaquePhongLight(
					surface.normal, surface.viewDir, surface.albedo,
					surface.specularColor, shininess, lightDirection,
					areaLight.radiance, vec3<f32>(1.0)
				);
			}
		}
	}

	return max(
		ambientBase * surface.sheenColor / PI + direct + surface.emissive,
		vec3<f32>(0.0)
	);
}

@fragment
fn fsMainDeferredLighting(input: DeferredVSOut) -> @location(0) vec4<f32> {
	let surface = loadDeferredSurface(input);
	if (surface.linearDepth <= 0.0) {
		discard;
	}

	var linearColor = surface.albedo;
	if (frame.options.x > 0.5 && surface.shadingModel == SHADING_PBR) {
		linearColor = evaluateDeferredPBR(surface);
	} else if (
		frame.options.x > 0.5 &&
		(surface.shadingModel == SHADING_PHONG ||
			surface.shadingModel == SHADING_FLAT)
	) {
		linearColor = evaluateDeferredPhong(surface);
	}
	linearColor = applyDeferredFog(linearColor, surface.linearDepth);
	return vec4<f32>(
		clamp(linearColor, vec3<f32>(0.0), vec3<f32>(65504.0)),
		surface.alpha
	);
}
