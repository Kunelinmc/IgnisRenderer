vec2 encodeNormalForGBuffer(vec3 normal) {
	vec3 n = normalize(normal);
	float invL1 = 1.0 / max(abs(n.x) + abs(n.y) + abs(n.z), EPSILON);
	vec2 encoded = n.xy * invL1;
	if (n.z < 0.0) {
		encoded = (vec2(1.0) - abs(encoded.yx)) * sign(encoded.xy);
	}
	return encoded * 0.5 + 0.5;
}

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

	vec3 normal = normalize(vNormal);
	vec3 viewDir = safeNormalize(uCameraPosition - vWorldPos, vec3(0.0, 0.0, 1.0));
	if (uDoubleSided == 1 && dot(normal, viewDir) < 0.0) {
		normal = -normal;
	}

#if WEBGL_MATERIAL_MODEL_PBR || WEBGL_MATERIAL_MODEL_FULL
	float roughness = clamp(uPBR.x, 0.04, 1.0);
	float metalness = clamp(uPBR.y, 0.0, 1.0);
	float reflectance = clamp(uPBR.z, 0.0, 1.0);
#if WEBGL_MATERIAL_TRANSMISSION || WEBGL_MATERIAL_MODEL_FULL
	float transmission = clamp(uPBR.w, 0.0, 1.0);
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
	if (uDoubleSided == 1 && dot(normal, viewDir) < 0.0) {
		normal = -normal;
	}
#endif
	vec3 shadowNormal = normal;

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
		vec3 anisotropyTexel = texture(uIridescenceThicknessMap, anisotropyUv).rgb;
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
			transmission,
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
		float dielectricF0 = 0.16 * reflectance * reflectance;
		vec3 f0 = mix(vec3(dielectricF0), albedo, metalness);
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
			transmission,
			anisotropyStrength,
			anisotropyTangent,
			anisotropyBitangent,
			iridescence,
			iridescenceThickness,
			iridescenceIor,
			occlusion
		);
		if (transmission > EPSILON) {
			float dielectricF0 = 0.16 * reflectance * reflectance;
			vec3 f0 = mix(vec3(dielectricF0), albedo, metalness);
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
			float dielectricF0 = 0.16 * reflectance * reflectance;
			float specularFactor = uSpecular.a;
			vec3 specularColor = uSpecular.rgb;
			fragSpecular = vec4(
				mix(vec3(dielectricF0), albedo, metalness) * specularColor,
				specularFactor
			);
		#else
			fragSpecular = vec4(0.0);
		#endif
	#endif
#endif
}
