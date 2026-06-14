@group(3) @binding(0) var gAlbedoAlphaIn: texture_2d<f32>;
@group(3) @binding(1) var gNormalRoughMetalIn: texture_2d<f32>;
@group(3) @binding(2) var gEmissiveOcclusionIn: texture_2d<f32>;
@group(3) @binding(3) var gMotionDepthIn: texture_2d<f32>;
@group(3) @binding(4) var gSpecularIn: texture_2d<f32>;
@group(3) @binding(5) var gCoatSheenIn: texture_2d<f32>;
@group(3) @binding(6) var gSheenReflectanceIn: texture_2d<f32>;
@group(3) @binding(7) var gMaterialExt0In: texture_2d<f32>;
@group(3) @binding(8) var gMaterialExt1In: texture_2d<f32>;
@group(3) @binding(9) var gMaterialExt2In: texture_2d<f32>;
@group(3) @binding(10) var gMaterialExt3In: texture_2d<f32>;

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
	ior: f32,
	thickness: f32,
	attenuationColor: vec3<f32>,
	attenuationDistance: f32,
	iridescence: f32,
	iridescenceIor: f32,
	iridescenceThickness: f32,
	anisotropyTangent: vec3<f32>,
	anisotropyBitangent: vec3<f32>,
	anisotropyStrength: f32,
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
	let specular = textureLoad(gSpecularIn, coord, 0);
	let coatSheen = textureLoad(gCoatSheenIn, coord, 0);
	let sheenReflectance = textureLoad(gSheenReflectanceIn, coord, 0);
	let materialExt0 = textureLoad(gMaterialExt0In, coord, 0);
	let materialExt1 = textureLoad(gMaterialExt1In, coord, 0);
	let materialExt2 = textureLoad(gMaterialExt2In, coord, 0);
	let materialExt3 = textureLoad(gMaterialExt3In, coord, 0);
	let normal = decodeDeferredNormal(normalRoughMetal.xy);
	let clearcoatNormal = decodeDeferredNormal(materialExt0.xy);
	let anisotropyTangentCandidate = decodeDeferredNormal(materialExt3.xy);
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
		u32(clamp(floor(motionDepth.w + 0.5), 0.0, 3.0)),
		clamp(specular.rgb, vec3<f32>(0.0), vec3<f32>(1.0)),
		clamp(specular.a, 0.0, 1.0),
		clamp(coatSheen.x, 0.0, 1.0),
		clamp(coatSheen.y, 0.04, 1.0),
		clearcoatNormal,
		clamp(sheenReflectance.rgb, vec3<f32>(0.0), vec3<f32>(1.0)),
		clamp(coatSheen.z, 0.0, 1.0),
		clamp(sheenReflectance.a, 0.0, 1.0),
		max(materialExt0.z, 1.0),
		max(materialExt0.w, 0.0),
		clamp(materialExt1.rgb, vec3<f32>(0.0001), vec3<f32>(1.0)),
		materialExt1.a,
		clamp(materialExt2.x, 0.0, 1.0),
		max(materialExt2.y, 1.0),
		max(materialExt2.z, 0.0),
		anisotropyTangent,
		anisotropyBitangent,
		clamp(materialExt3.z, 0.0, 1.0),
		input.position.xy
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
	let nDotL = max(dot(surface.normal, lightDirection), 0.0);
	if (nDotL <= 0.0) {
		return vec3<f32>(0.0);
	}

	let halfVector = safeNormalize(
		surface.viewDir + lightDirection,
		surface.viewDir
	);
	let fresnel = resolveIridescenceFresnel(
		max(dot(halfVector, surface.viewDir), 0.0),
		pbr.realF0,
		surface.iridescence,
		surface.iridescenceThickness,
		surface.iridescenceIor
	);
	var specular = vec3<f32>(0.0);
	if (surface.anisotropyStrength > EPSILON) {
		specular = resolveAnisotropicSpecular(
			fresnel,
			surface.roughness,
			surface.anisotropyStrength,
			nDotL,
			pbr.nDotV,
			max(dot(surface.normal, halfVector), 0.0),
			dot(surface.anisotropyTangent, surface.viewDir),
			dot(surface.anisotropyBitangent, surface.viewDir),
			dot(surface.anisotropyTangent, lightDirection),
			dot(surface.anisotropyBitangent, lightDirection),
			dot(surface.anisotropyTangent, halfVector),
			dot(surface.anisotropyBitangent, halfVector)
		);
	} else {
		let ndf = distributionGGX(surface.normal, halfVector, surface.roughness);
		let geometry = geometrySmith(pbr.nDotV, nDotL, surface.roughness);
		let denominator = max(4.0 * pbr.nDotV * nDotL, 0.0001);
		specular = (ndf * geometry * fresnel) / denominator;
	}
	specular = specular * pbr.energyCompensation;
	let kd =
		diffuseFresnelWeight(fresnel, surface.iridescence) *
		(1.0 - surface.metalness);
	let diffuse = (kd * surface.albedo) / PI;

	var clearcoatSpecular = vec3<f32>(0.0);
	var clearcoatFresnel = vec3<f32>(0.0);
	let ncDotV = max(dot(surface.clearcoatNormal, surface.viewDir), PBR_MIN_NDOTV);
	if (surface.clearcoat > 0.0) {
		let ncDotL = max(dot(surface.clearcoatNormal, lightDirection), 0.0);
		if (ncDotL > 0.0) {
			let ccHalfVector = safeNormalize(
				surface.viewDir + lightDirection,
				surface.viewDir
			);
			let hccDotV = max(dot(ccHalfVector, surface.viewDir), 0.0);
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
			let ccDenom = max(4.0 * ncDotV * ncDotL, 0.0001);
			clearcoatSpecular = vec3<f32>((ccNdf * ccGeometry * ccF) / ccDenom);
			clearcoatFresnel = vec3<f32>(ccF);
		}
	}

	var sheenSpecular = vec3<f32>(0.0);
	var albedoSheenScaling = vec3<f32>(1.0);
	let maxSheenColor = max(max(surface.sheenColor.x, surface.sheenColor.y), surface.sheenColor.z);
	if (maxSheenColor > 0.0) {
		let nDotH = max(dot(surface.normal, halfVector), 0.0);
		let sheenNdf = distributionCharlie(
			nDotH,
			max(surface.sheenRoughness, 0.04)
		);
		let sheenVisibility = visibilityAshikhmin(nDotL, pbr.nDotV);
		sheenSpecular = surface.sheenColor * sheenNdf * sheenVisibility;
		let hDotV = max(dot(halfVector, surface.viewDir), 0.0);
		let sheenFresnel = fresnelSchlick(hDotV, surface.sheenColor);
		albedoSheenScaling = max(vec3<f32>(0.0), vec3<f32>(1.0) - sheenFresnel);
	}

	let clearcoatAttenuation = vec3<f32>(1.0) - clearcoatFresnel * surface.clearcoat;
	let baseLayerAttenuation = clearcoatAttenuation * albedoSheenScaling;
	return (
		(diffuse + specular) * baseLayerAttenuation +
		clearcoatSpecular * surface.clearcoat +
		sheenSpecular * clearcoatAttenuation
	) * nDotL * radiance * shadow;
}

fn evaluateDeferredPBR(surface: DeferredSurface) -> vec3<f32> {
	var directLight = vec3<f32>(0.0);
	let pbr = buildDeferredPBRContext(surface);
	let directionalCount = u32(frame.lightCounts.x + 0.5);
	for (var i: u32 = 0u; i < directionalCount; i = i + 1u) {
		let lightDirection = safeNormalize(
			frame.directionalLights[i].direction.xyz,
			vec3<f32>(0.0, 1.0, 0.0)
		);
		let shadow = sampleDirectionalShadowVisibility(
			i,
			surface.worldPosition,
			surface.shadowNormal,
			lightDirection,
			surface.linearDepth
		);
		directLight += evaluateDeferredPBRLight(
			surface,
			pbr,
			lightDirection,
			frame.directionalLights[i].color.xyz,
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
					shadow = sampleSpotShadowVisibility(
						shadowIndex,
						surface.worldPosition,
						surface.shadowNormal,
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
			let toLight = frame.pointLights[i].positionRange.xyz - surface.worldPosition;
			let distanceSq = dot(toLight, toLight);
			let distanceValue = sqrt(max(distanceSq, EPSILON));
			let lightRange = frame.pointLights[i].positionRange.w;
			if (distanceValue > lightRange) {
				continue;
			}
			let lightDirection = toLight / distanceValue;
			let radiance =
				frame.pointLights[i].color.xyz *
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
			let toLight = frame.spotLights[i].positionRange.xyz - surface.worldPosition;
			let distanceSq = dot(toLight, toLight);
			let distanceValue = sqrt(max(distanceSq, EPSILON));
			let lightRange = frame.spotLights[i].positionRange.w;
			if (distanceValue > lightRange) {
				continue;
			}
			let lightDirection = toLight / distanceValue;
			let coneDirection = safeNormalize(
				frame.spotLights[i].directionOuter.xyz,
				vec3<f32>(0.0, -1.0, 0.0)
			);
			let coneAttenuation = spotAttenuation(
				dot(-lightDirection, coneDirection),
				frame.spotLights[i].directionOuter.w,
				frame.spotLights[i].colorInner.w
			);
			if (coneAttenuation <= 0.0) {
				continue;
			}
			let radiance =
				frame.spotLights[i].colorInner.xyz *
				pointAttenuation(distanceSq, lightRange) *
				coneAttenuation;
			let shadow = sampleSpotShadowVisibility(
				i,
				surface.worldPosition,
				surface.shadowNormal,
				lightDirection
			);
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
					frame.areaLights[i],
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
	if (ambientColor.x + ambientColor.y + ambientColor.z == 0.0) {
		ambientColor = vec3<f32>(PBR_AMBIENT_FALLBACK_LINEAR);
	}

	var diffuseAmbient = ambientColor;
	var specularAmbientRadiance = ambientColor / PI;
	if (useSHAmbient()) {
		diffuseAmbient =
			sampleDiffuseProbeIrradiance(surface.worldPosition, surface.normal) / 255.0;

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
		ambientLight +=
			prefiltered *
			(fAmbient * brdf.x + vec3<f32>(brdf.y)) *
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
			frame.directionalLights[i].direction.xyz,
			vec3<f32>(0.0, 1.0, 0.0)
		);
		let nDotL = max(dot(surface.normal, lightDirection), 0.0);
		if (nDotL <= 0.0) {
			continue;
		}
		let shadow = sampleDirectionalShadowVisibility(
			i,
			surface.worldPosition,
			surface.shadowNormal,
			lightDirection,
			surface.linearDepth
		);
		let halfVector = safeNormalize(
			surface.viewDir + lightDirection,
			surface.viewDir
		);
		let specFactor = pow(max(dot(surface.normal, halfVector), 0.0), shininess);
		let radiance = frame.directionalLights[i].color.xyz * shadow;
		direct += radiance * nDotL * surface.albedo;
		direct += radiance * specFactor * surface.specularColor;
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
					let halfVector = safeNormalize(
						surface.viewDir + lightDirection,
						surface.viewDir
					);
					let specFactor = pow(
						max(dot(surface.normal, halfVector), 0.0),
						shininess
					);
					direct += areaLight.radiance * nDotL * surface.albedo;
					direct +=
						areaLight.radiance * specFactor * surface.specularColor;
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
					shadow = sampleSpotShadowVisibility(
						shadowIndex,
						surface.worldPosition,
						surface.shadowNormal,
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
			let halfVector = safeNormalize(
				surface.viewDir + lightDirection,
				surface.viewDir
			);
			let specFactor = pow(max(dot(surface.normal, halfVector), 0.0), shininess);
			let radiance = colorInner.xyz * attenuation * shadow;
			direct += radiance * nDotL * surface.albedo;
			direct += radiance * specFactor * surface.specularColor;
		}
	} else {
		let pointCount = u32(frame.lightCounts.y + 0.5);
		for (var i: u32 = 0u; i < pointCount; i = i + 1u) {
			let toLight = frame.pointLights[i].positionRange.xyz - surface.worldPosition;
			let distanceSq = dot(toLight, toLight);
			let distanceValue = sqrt(max(distanceSq, EPSILON));
			let lightRange = frame.pointLights[i].positionRange.w;
			if (distanceValue > lightRange) {
				continue;
			}
			let lightDirection = toLight / distanceValue;
			let nDotL = max(dot(surface.normal, lightDirection), 0.0);
			if (nDotL <= 0.0) {
				continue;
			}
			let halfVector = safeNormalize(
				surface.viewDir + lightDirection,
				surface.viewDir
			);
			let specFactor = pow(max(dot(surface.normal, halfVector), 0.0), shininess);
			let radiance =
				frame.pointLights[i].color.xyz *
				pointAttenuation(distanceSq, lightRange);
			direct += radiance * nDotL * surface.albedo;
			direct += radiance * specFactor * surface.specularColor;
		}

		let spotCount = u32(frame.lightCounts.z + 0.5);
		for (var i: u32 = 0u; i < spotCount; i = i + 1u) {
			let toLight = frame.spotLights[i].positionRange.xyz - surface.worldPosition;
			let distanceSq = dot(toLight, toLight);
			let distanceValue = sqrt(max(distanceSq, EPSILON));
			let lightRange = frame.spotLights[i].positionRange.w;
			if (distanceValue > lightRange) {
				continue;
			}
			let lightDirection = toLight / distanceValue;
			let coneDirection = safeNormalize(
				frame.spotLights[i].directionOuter.xyz,
				vec3<f32>(0.0, -1.0, 0.0)
			);
			let coneAttenuation = spotAttenuation(
				dot(-lightDirection, coneDirection),
				frame.spotLights[i].directionOuter.w,
				frame.spotLights[i].colorInner.w
			);
			if (coneAttenuation <= 0.0) {
				continue;
			}
			let nDotL = max(dot(surface.normal, lightDirection), 0.0);
			if (nDotL <= 0.0) {
				continue;
			}
			let shadow = sampleSpotShadowVisibility(
				i,
				surface.worldPosition,
				surface.shadowNormal,
				lightDirection
			);
			let halfVector = safeNormalize(
				surface.viewDir + lightDirection,
				surface.viewDir
			);
			let specFactor = pow(max(dot(surface.normal, halfVector), 0.0), shininess);
			let radiance =
				frame.spotLights[i].colorInner.xyz *
				pointAttenuation(distanceSq, lightRange) *
				coneAttenuation *
				shadow;
			direct += radiance * nDotL * surface.albedo;
			direct += radiance * specFactor * surface.specularColor;
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
					frame.areaLights[i],
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
				let halfVector = safeNormalize(
					surface.viewDir + lightDirection,
					surface.viewDir
				);
				let specFactor = pow(
					max(dot(surface.normal, halfVector), 0.0),
					shininess
				);
				direct += areaLight.radiance * nDotL * surface.albedo;
				direct +=
					areaLight.radiance * specFactor * surface.specularColor;
			}
		}
	}

	return max(ambientBase * surface.sheenColor + direct + surface.emissive, vec3<f32>(0.0));
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
