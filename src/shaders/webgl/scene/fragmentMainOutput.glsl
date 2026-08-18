vec2 encodeNormalForGBuffer(vec3 normal) {
	vec3 n = normalize(normal);
	float invL1 = 1.0 / max(abs(n.x) + abs(n.y) + abs(n.z), EPSILON);
	vec2 encoded = n.xy * invL1;
	if (n.z < 0.0) {
		encoded = (vec2(1.0) - abs(encoded.yx)) * sign(encoded.xy);
	}
	return encoded * 0.5 + 0.5;
}

#if WEBGL_MATERIAL_TRANSMISSION || WEBGL_MATERIAL_MODEL_FULL
vec2 traceTransmissionBackground(
	vec2 startUv,
	vec2 rayUvOffset,
	float startDepth,
	float rayDepthOffset,
	out bool hit
) {
	hit = false;
	float lowerT = 0.0;
	float upperT = 1.0;
	for (int stepIndex = 1; stepIndex <= 16; stepIndex++) {
		float t = float(stepIndex) / 16.0;
		vec2 candidateUv = startUv + rayUvOffset * t;
		if (any(lessThan(candidateUv, vec2(0.0))) ||
			any(greaterThan(candidateUv, vec2(1.0)))) {
			return startUv;
		}
		float sceneDepth = texture(uTransmissionDepthMap, candidateUv).b;
		float rayDepth = startDepth + rayDepthOffset * t;
		if (sceneDepth > 0.0 && rayDepth >= sceneDepth) {
			hit = true;
			upperT = t;
			lowerT = float(stepIndex - 1) / 16.0;
			break;
		}
	}
	if (!hit) {
		return startUv;
	}
	for (int refinementIndex = 0; refinementIndex < 4; refinementIndex++) {
		float middleT = (lowerT + upperT) * 0.5;
		vec2 candidateUv = startUv + rayUvOffset * middleT;
		float sceneDepth = texture(uTransmissionDepthMap, candidateUv).b;
		float rayDepth = startDepth + rayDepthOffset * middleT;
		if (sceneDepth > 0.0 && rayDepth >= sceneDepth) {
			upperT = middleT;
		} else {
			lowerT = middleT;
		}
	}
	return startUv + rayUvOffset * upperT;
}
#endif

void main() {
	vec3 albedo = uBaseColor.rgb;
	float alpha = clamp(uBaseColor.a, 0.0, 1.0);
#if WEBGL_MATERIAL_BASE_MAP
	vec2 baseUv = resolveMappedUV(
		uBaseMapUV,
		uBaseMapTransformA,
		uBaseMapTransformB
	);
	if (uHasBaseMap == 1) {
		vec4 texel = texture(uBaseMap, baseUv);
		vec3 texColor = uBaseMapIsLinear == 1 ? texel.rgb : srgbToLinear(texel.rgb);
		albedo *= texColor;
		alpha *= texel.a;
	}
#endif

#if WEBGL_MATERIAL_ALPHA_MASK
	if (uAlpha.y > 0.5 && alpha < uAlpha.x) {
		discard;
	}
#endif

	vec3 geometryNormal = normalize(vNormal);
	vec3 viewDir = safeNormalize(uCameraPosition - vWorldPos, vec3(0.0, 0.0, 1.0));
	if (uDoubleSided == 1 && !gl_FrontFacing) {
		geometryNormal = -geometryNormal;
	}
	vec3 normal = geometryNormal;

#if WEBGL_MATERIAL_MODEL_PBR || WEBGL_MATERIAL_MODEL_FULL
	float roughness = clamp(uPBR.x, 0.04, 1.0);
	float metalness = clamp(uPBR.y, 0.0, 1.0);
	float reflectance = clamp(uPBR.z, 0.0, 1.0);
	float specularFactor = clamp(uSpecular.a, 0.0, 1.0);
	vec3 specularColor = clamp(uSpecular.rgb, vec3(0.0), vec3(1.0));
#if WEBGL_MATERIAL_SPECULAR_MAP
	if (uHasSpecularMap == 1) {
		vec2 specularUv = resolveMappedUV(
			uSpecularMapUV, uSpecularMapTransformA, uSpecularMapTransformB
		);
		specularFactor *= texture(uSpecularMap, specularUv).a;
	}
#endif
#if WEBGL_MATERIAL_SPECULAR_COLOR_MAP
	if (uHasSpecularColorMap == 1) {
		vec2 specularColorUv = resolveMappedUV(
			uSpecularColorMapUV,
			uSpecularColorMapTransformA,
			uSpecularColorMapTransformB
		);
		specularColor *= srgbToLinear(texture(uSpecularColorMap, specularColorUv).rgb);
	}
#endif
#if WEBGL_MATERIAL_TRANSMISSION || WEBGL_MATERIAL_MODEL_FULL
	float transmission = clamp(uPBR.w, 0.0, 1.0);
#if WEBGL_MATERIAL_TRANSMISSION_MAP
	if (uHasTransmissionMap == 1) {
		vec2 transmissionUv = resolveMappedUV(
			uTransmissionMapUV,
			uTransmissionMapTransformA,
			uTransmissionMapTransformB
		);
		transmission *= texture(uTransmissionMap, transmissionUv).r;
	}
#endif
#else
	float transmission = 0.0;
#endif
#if WEBGL_MATERIAL_METALLIC_ROUGHNESS_MAP
	if (uHasMetallicRoughnessMap == 1) {
		vec2 metallicRoughnessUv = resolveMappedUV(
			uMetallicRoughnessMapUV,
			uMetallicRoughnessMapTransformA,
			uMetallicRoughnessMapTransformB
		);
		vec4 metallicRoughnessTexel = texture(uMetallicRoughnessMap, metallicRoughnessUv);
		roughness = clamp(roughness * metallicRoughnessTexel.g, 0.04, 1.0);
		metalness = clamp(metalness * metallicRoughnessTexel.b, 0.0, 1.0);
	}
#endif

	float occlusion = 1.0;
#if WEBGL_MATERIAL_OCCLUSION_MAP
	if (uHasOcclusionMap == 1) {
		vec2 occlusionUv = resolveMappedUV(
			uOcclusionMapUV,
			uOcclusionMapTransformA,
			uOcclusionMapTransformB
		);
		float occlusionTexel = texture(uOcclusionMap, occlusionUv).r;
		occlusion = clamp(
			1.0 + clamp(uOcclusionStrength, 0.0, 1.0) * (occlusionTexel - 1.0),
			0.0,
			1.0
		);
	}
#endif

	float iridescence = 0.0;
	float iridescenceIor = 1.3;
	float iridescenceThickness = 0.0;
#if WEBGL_MATERIAL_IRIDESCENCE || WEBGL_MATERIAL_MODEL_FULL
	iridescence = clamp(uIridescence.x, 0.0, 1.0);
	iridescenceIor = max(uIridescence.y, 1.0);
	iridescenceThickness = max(uIridescence.w, 0.0);
#if WEBGL_MATERIAL_IRIDESCENCE_MAP
	if (uHasIridescenceMap == 1) {
		vec2 iridescenceUv = resolveMappedUV(
			uIridescenceMapUV,
			uIridescenceMapTransformA,
			uIridescenceMapTransformB
		);
		iridescence *= texture(uIridescenceMap, iridescenceUv).r;
	}
#endif
#if WEBGL_MATERIAL_IRIDESCENCE_THICKNESS_MAP
	if (uHasIridescenceThicknessMap == 1) {
		vec2 iridescenceThicknessUv = resolveMappedUV(
			uIridescenceThicknessMapUV,
			uIridescenceThicknessMapTransformA,
			uIridescenceThicknessMapTransformB
		);
		iridescenceThickness = max(
			mix(
				uIridescence.z,
				uIridescence.w,
				texture(uIridescenceThicknessMap, iridescenceThicknessUv).g
			),
			0.0
		);
	}
#endif
#endif

#if WEBGL_MATERIAL_NORMAL_MAP
	vec2 normalUv = resolveMappedUV(
		uNormalMapUV,
		uNormalMapTransformA,
		uNormalMapTransformB
	);
	if (uHasNormalMap == 1) {
		normal = applyNormalMap(
			normal,
			vTangent,
			texture(uNormalMap, normalUv).xyz,
			max(uNormalScale, 0.0)
		);
	}
	if (dot(normal, geometryNormal) < 0.0) {
		normal = -normal;
	}
#endif
	float clearcoat = 0.0;
	float clearcoatRoughness = 0.01;
	vec3 clearcoatNormal = normal;
#if WEBGL_MATERIAL_CLEARCOAT || WEBGL_MATERIAL_MODEL_FULL
	clearcoat = clamp(uClearcoat.x, 0.0, 1.0);
	clearcoatRoughness = clamp(uClearcoat.y, 0.01, 1.0);
#if WEBGL_MATERIAL_CLEARCOAT_MAP
	if (uHasClearcoatMap == 1) {
		vec2 clearcoatUv = resolveMappedUV(
			uClearcoatMapUV, uClearcoatMapTransformA, uClearcoatMapTransformB
		);
		clearcoat *= texture(uClearcoatMap, clearcoatUv).r;
	}
#endif
#if WEBGL_MATERIAL_CLEARCOAT_ROUGHNESS_MAP
	if (uHasClearcoatRoughnessMap == 1) {
		vec2 clearcoatRoughnessUv = resolveMappedUV(
			uClearcoatRoughnessMapUV,
			uClearcoatRoughnessMapTransformA,
			uClearcoatRoughnessMapTransformB
		);
		clearcoatRoughness = clamp(
			clearcoatRoughness * texture(uClearcoatRoughnessMap, clearcoatRoughnessUv).g,
			0.01,
			1.0
		);
	}
#endif
#if WEBGL_MATERIAL_CLEARCOAT_NORMAL_MAP
	if (uHasClearcoatNormalMap == 1) {
		vec2 clearcoatNormalUv = resolveMappedUV(
			uClearcoatNormalMapUV,
			uClearcoatNormalMapTransformA,
			uClearcoatNormalMapTransformB
		);
		clearcoatNormal = applyNormalMap(
			normal,
			vTangent,
			texture(uClearcoatNormalMap, clearcoatNormalUv).xyz,
			max(uClearcoat.z, 0.0)
		);
	}
#endif
#endif
	if (dot(clearcoatNormal, normal) < 0.0) {
		clearcoatNormal = -clearcoatNormal;
	}
	vec3 sheenColor = vec3(0.0);
	float sheenRoughness = 0.0;
#if WEBGL_MATERIAL_SHEEN || WEBGL_MATERIAL_MODEL_FULL
	sheenColor = clamp(uSheen.rgb, vec3(0.0), vec3(1.0));
	sheenRoughness = clamp(uSheen.a, 0.0, 1.0);
#if WEBGL_MATERIAL_SHEEN_COLOR_MAP
	if (uHasSheenColorMap == 1) {
		vec2 sheenColorUv = resolveMappedUV(
			uSheenColorMapUV,
			uSheenColorMapTransformA,
			uSheenColorMapTransformB
		);
		sheenColor *= srgbToLinear(texture(uSheenColorMap, sheenColorUv).rgb);
	}
#endif
#if WEBGL_MATERIAL_SHEEN_ROUGHNESS_MAP
	if (uHasSheenRoughnessMap == 1) {
		vec2 sheenRoughnessUv = resolveMappedUV(
			uSheenRoughnessMapUV,
			uSheenRoughnessMapTransformA,
			uSheenRoughnessMapTransformB
		);
		sheenRoughness *= texture(uSheenRoughnessMap, sheenRoughnessUv).a;
	}
#endif
#endif
	float resolvedThickness = 0.0;
#if WEBGL_MATERIAL_TRANSMISSION || WEBGL_MATERIAL_MODEL_FULL
	resolvedThickness = max(uTransmissionVolume.y, 0.0);
#if WEBGL_MATERIAL_THICKNESS_MAP
	if (uHasThicknessMap == 1) {
		vec2 thicknessUv = resolveMappedUV(
			uThicknessMapUV, uThicknessMapTransformA, uThicknessMapTransformB
		);
		resolvedThickness *= texture(uThicknessMap, thicknessUv).g;
	}
#endif
	resolvedThickness *= max(uTransmissionModelScale, 0.0001);
#endif

	float anisotropyStrength = 0.0;
	vec3 anisotropyTangent = fallbackTangentFromNormal(normal);
	vec3 anisotropyBitangent = safeNormalize(
		cross(normal, anisotropyTangent),
		fallbackTangentFromNormal(normal)
	);
#if WEBGL_MATERIAL_ANISOTROPY || WEBGL_MATERIAL_MODEL_FULL
	anisotropyStrength = clamp(uAnisotropy.x, 0.0, 1.0);
	vec2 anisotropyDirection = vec2(1.0, 0.0);
#if WEBGL_MATERIAL_ANISOTROPY_MAP
	if (uHasAnisotropyMap == 1) {
		vec2 anisotropyUv = resolveMappedUV(
			uAnisotropyMapUV,
			uAnisotropyMapTransformA,
			uAnisotropyMapTransformB
		);
		vec3 anisotropyTexel = texture(uAnisotropyMap, anisotropyUv).rgb;
		anisotropyDirection = anisotropyTexel.rg * 2.0 - vec2(1.0);
		float anisotropyDirectionLen = length(anisotropyDirection);
		anisotropyDirection =
			anisotropyDirectionLen > EPSILON ?
				anisotropyDirection / anisotropyDirectionLen
			:	vec2(1.0, 0.0);
		anisotropyStrength = clamp(
			anisotropyStrength * anisotropyTexel.b,
			0.0,
			1.0
		);
	}
#endif
	anisotropyDirection = rotateAnisotropyDirection(anisotropyDirection);
	vec3 anisotropyBaseTangent;
	vec3 anisotropyBaseBitangent;
	resolveTangentFrame(
		normal,
		vTangent,
		anisotropyBaseTangent,
		anisotropyBaseBitangent
	);
	anisotropyTangent = safeNormalize(
		anisotropyBaseTangent * anisotropyDirection.x +
			anisotropyBaseBitangent * anisotropyDirection.y,
		anisotropyBaseTangent
	);
	anisotropyBitangent = safeNormalize(
		cross(normal, anisotropyTangent),
		anisotropyBaseBitangent
	);
#endif
#endif
	vec3 shadowNormal = normal;

	vec3 emissive = uEmissive.rgb;
#if WEBGL_MATERIAL_EMISSIVE_MAP
	if (uHasEmissiveMap == 1) {
		vec2 emissiveUv = resolveMappedUV(
			uEmissiveMapUV,
			uEmissiveMapTransformA,
			uEmissiveMapTransformB
		);
		vec3 emissiveTexel = texture(uEmissiveMap, emissiveUv).rgb;
		emissive *=
			uEmissiveMapIsLinear == 1 ? emissiveTexel : srgbToLinear(emissiveTexel);
	}
#endif

	vec3 color;
#if WEBGL_MATERIAL_MODEL_UNLIT
	color = albedo;
#elif WEBGL_MATERIAL_MODEL_PBR
	if (uEnableLighting == 0) {
		color = albedo;
	} else {
		color = shadePBR(
			albedo,
			normal,
			shadowNormal,
			viewDir,
			roughness,
			metalness,
			reflectance,
			specularFactor,
			specularColor,
			transmission,
			resolvedThickness,
			clearcoat,
			clearcoatRoughness,
			clearcoatNormal,
			sheenColor,
			sheenRoughness,
			anisotropyStrength,
			anisotropyTangent,
			anisotropyBitangent,
			iridescence,
			iridescenceThickness,
			iridescenceIor,
			occlusion
		);
	}
#if WEBGL_MATERIAL_TRANSMISSION
	if (transmission > EPSILON) {
		vec3 dielectricF0 = vec3(0.16 * reflectance * reflectance) *
			specularColor * specularFactor;
		vec3 f0 = mix(dielectricF0, albedo, metalness);
		float nDotV = max(dot(normal, viewDir), PBR_MIN_NDOTV);
		vec3 fresnel = resolveIridescenceFresnel(
			nDotV,
			f0,
			iridescence,
			iridescenceThickness,
			iridescenceIor
		);
		float fresnelAverage = clamp(dot(fresnel, vec3(1.0 / 3.0)), 0.0, 1.0);
		alpha = resolveTransmissionAlpha(alpha, transmission, fresnelAverage);
	}
#endif
#elif WEBGL_MATERIAL_MODEL_LEGACY
	if (uEnableLighting == 0) {
		color = albedo;
	} else {
		color = shadePhong(albedo, normal, shadowNormal, viewDir);
	}
#else
	if (uEnableLighting == 0 || uShadingModel == 2) {
		color = albedo;
	} else if (uShadingModel == 1) {
		color = shadePBR(
			albedo,
			normal,
			shadowNormal,
			viewDir,
			roughness,
			metalness,
			reflectance,
			specularFactor,
			specularColor,
			transmission,
			resolvedThickness,
			clearcoat,
			clearcoatRoughness,
			clearcoatNormal,
			sheenColor,
			sheenRoughness,
			anisotropyStrength,
			anisotropyTangent,
			anisotropyBitangent,
			iridescence,
			iridescenceThickness,
			iridescenceIor,
			occlusion
		);
		if (transmission > EPSILON) {
			vec3 dielectricF0 = vec3(0.16 * reflectance * reflectance) *
				specularColor * specularFactor;
			vec3 f0 = mix(dielectricF0, albedo, metalness);
			float nDotV = max(dot(normal, viewDir), PBR_MIN_NDOTV);
			vec3 fresnel = resolveIridescenceFresnel(
				nDotV,
				f0,
				iridescence,
				iridescenceThickness,
				iridescenceIor
			);
			float fresnelAverage = clamp(
				dot(fresnel, vec3(1.0 / 3.0)),
				0.0,
				1.0
			);
			alpha = resolveTransmissionAlpha(alpha, transmission, fresnelAverage);
		}
	} else {
		color = shadePhong(albedo, normal, shadowNormal, viewDir);
	}
#endif

	color += emissive;
#if WEBGL_MATERIAL_TRANSMISSION || WEBGL_MATERIAL_MODEL_FULL
	if (transmission > EPSILON && uHasTransmissionBackgroundMap == 1) {
		vec2 screenUv = gl_FragCoord.xy * uTransmissionBackgroundInvSize;
		float ior = max(uTransmissionVolume.x, 1.0);
		vec3 refractedDirection = refract(-viewDir, normal, 1.0 / ior);
		bool totalInternalReflection =
			dot(refractedDirection, refractedDirection) <= EPSILON;
		bool rayHit = resolvedThickness <= EPSILON;
		vec2 refractedUv = screenUv;
		if (!totalInternalReflection && resolvedThickness > EPSILON &&
			uHasTransmissionDepthMap == 1) {
			float projectedZ = max(abs(refractedDirection.z), 0.1);
			vec2 rayUvOffset = refractedDirection.xy / projectedZ *
				resolvedThickness * 0.025;
			float rayDepthOffset = projectedZ * resolvedThickness;
			refractedUv = traceTransmissionBackground(
				screenUv,
				rayUvOffset,
				max(vViewDepth, 0.0),
				rayDepthOffset,
				rayHit
			);
		}
		float mipLevel = roughness * 8.0;
		if (!totalInternalReflection && rayHit) {
			vec3 currentBackground = textureLod(
				uTransmissionBackgroundMap, refractedUv, mipLevel
			).rgb;
			float attenuationDistance = uTransmissionVolume.z;
			vec3 volumeAttenuation = vec3(1.0);
			if (attenuationDistance > EPSILON && resolvedThickness > EPSILON) {
				volumeAttenuation = pow(
					clamp(uAttenuationColor.rgb, vec3(0.0001), vec3(1.0)),
					vec3(resolvedThickness / attenuationDistance)
				);
			}
			currentBackground *= volumeAttenuation;
			float coverage = clamp(alpha, 0.0, 1.0);
			color = mix(currentBackground, color, coverage);
		}
		alpha = 1.0;
	}
#endif
	int fogMode = int(floor(uFogParams0.x + 0.5));
	float fogFactor = ignisComputeFogFactor(
		fogMode,
		max(vViewDepth, 0.0),
		uFogParams0.y,
		uFogParams0.z,
		uFogParams0.w,
		uFogParams1.w
	);
	color = max(mix(color, uFogParams1.rgb, fogFactor), vec3(0.0));
	vec3 finalColor = max(color, vec3(0.0));
	float finalAlpha = clamp(alpha, 0.0, 1.0);
#if WEBGL_SCENE_OIT
	if (uOITPassMode == 1) {
		float weight = resolveOITWeight(finalAlpha, max(vViewDepth, 0.0));
		fragColor = vec4(finalColor * finalAlpha, finalAlpha) * weight;
		fragMotion = vec4(0.0);
		fragNormal = vec4(0.0);
		return;
	}
	if (uOITPassMode == 2) {
		fragColor = vec4(finalAlpha);
		fragMotion = vec4(0.0);
		fragNormal = vec4(0.0);
		return;
	}
#endif
	fragColor = vec4(finalColor, finalAlpha);
#if WEBGL_SCENE_OUTPUT_MRT
	#if WEBGL_SCENE_OUTPUT_MATERIAL_GBUFFER && (WEBGL_MATERIAL_MODEL_PBR || WEBGL_MATERIAL_MODEL_FULL)
	fragNormal = vec4(encodeNormalForGBuffer(normal), roughness, metalness);
	#elif WEBGL_SCENE_OUTPUT_MATERIAL_GBUFFER
	fragNormal = vec4(encodeNormalForGBuffer(normal), 1.0, 0.0);
	#else
	fragNormal = vec4(normal * 0.5 + 0.5, 1.0);
	#endif
	vec2 curUV = (vCurrentClip.xy / vCurrentClip.w) * 0.5 + 0.5;
	vec2 prevUV = (vPrevClip.xy / vPrevClip.w) * 0.5 + 0.5;
	fragMotion = vec4(curUV - prevUV, vViewDepth, 1.0);
	#if WEBGL_SCENE_OUTPUT_MATERIAL_GBUFFER
		fragAlbedo = vec4(albedo, alpha);
		#if WEBGL_MATERIAL_MODEL_PBR || WEBGL_MATERIAL_MODEL_FULL
			fragSpecular = vec4(
				mix(
					vec3(0.16 * reflectance * reflectance) *
						specularColor * specularFactor,
					albedo,
					metalness
				),
				specularFactor
			);
		#else
			fragSpecular = vec4(0.0);
		#endif
	#endif
#endif
}
