layout(std140) uniform IgnisMaterialCommon {
	vec4 ignisBaseColor;
	vec4 ignisEmissive;
	vec4 ignisAlpha;
	vec4 ignisMaterialRenderParams;
#if WEBGL_MATERIAL_BASE_MAP
	vec4 ignisBaseMapTransformA;
	vec4 ignisBaseMapTransformB;
#endif
#if WEBGL_MATERIAL_EMISSIVE_MAP
	vec4 ignisEmissiveMapTransformA;
	vec4 ignisEmissiveMapTransformB;
#endif
};

#if WEBGL_MATERIAL_MODEL_PBR
layout(std140) uniform IgnisPBRMaterial {
	vec4 ignisPBR;
	vec4 ignisSpecular;
	vec4 ignisTransmissionVolume;
	vec4 ignisClearcoat;
	vec4 ignisSheen;
	vec4 ignisIridescence;
	vec4 ignisAttenuationColor;
	vec4 ignisAnisotropy;
	vec4 ignisPBRScales;
#if WEBGL_MATERIAL_METALLIC_ROUGHNESS_MAP
	vec4 ignisMetallicRoughnessMapTransformA;
	vec4 ignisMetallicRoughnessMapTransformB;
#endif
#if WEBGL_MATERIAL_SPECULAR_MAP
	vec4 ignisSpecularMapTransformA;
	vec4 ignisSpecularMapTransformB;
#endif
#if WEBGL_MATERIAL_SPECULAR_COLOR_MAP
	vec4 ignisSpecularColorMapTransformA;
	vec4 ignisSpecularColorMapTransformB;
#endif
#if WEBGL_MATERIAL_CLEARCOAT_MAP
	vec4 ignisClearcoatMapTransformA;
	vec4 ignisClearcoatMapTransformB;
#endif
#if WEBGL_MATERIAL_CLEARCOAT_ROUGHNESS_MAP
	vec4 ignisClearcoatRoughnessMapTransformA;
	vec4 ignisClearcoatRoughnessMapTransformB;
#endif
#if WEBGL_MATERIAL_CLEARCOAT_NORMAL_MAP
	vec4 ignisClearcoatNormalMapTransformA;
	vec4 ignisClearcoatNormalMapTransformB;
#endif
#if WEBGL_MATERIAL_SHEEN_COLOR_MAP
	vec4 ignisSheenColorMapTransformA;
	vec4 ignisSheenColorMapTransformB;
#endif
#if WEBGL_MATERIAL_SHEEN_ROUGHNESS_MAP
	vec4 ignisSheenRoughnessMapTransformA;
	vec4 ignisSheenRoughnessMapTransformB;
#endif
#if WEBGL_MATERIAL_TRANSMISSION_MAP
	vec4 ignisTransmissionMapTransformA;
	vec4 ignisTransmissionMapTransformB;
#endif
#if WEBGL_MATERIAL_THICKNESS_MAP
	vec4 ignisThicknessMapTransformA;
	vec4 ignisThicknessMapTransformB;
#endif
#if WEBGL_MATERIAL_NORMAL_MAP
	vec4 ignisNormalMapTransformA;
	vec4 ignisNormalMapTransformB;
#endif
#if WEBGL_MATERIAL_OCCLUSION_MAP
	vec4 ignisOcclusionMapTransformA;
	vec4 ignisOcclusionMapTransformB;
#endif
#if WEBGL_MATERIAL_IRIDESCENCE_MAP
	vec4 ignisIridescenceMapTransformA;
	vec4 ignisIridescenceMapTransformB;
#endif
#if WEBGL_MATERIAL_IRIDESCENCE_THICKNESS_MAP
	vec4 ignisIridescenceThicknessMapTransformA;
	vec4 ignisIridescenceThicknessMapTransformB;
#endif
#if WEBGL_MATERIAL_ANISOTROPY_MAP
	vec4 ignisAnisotropyMapTransformA;
	vec4 ignisAnisotropyMapTransformB;
#endif
};
#elif WEBGL_MATERIAL_MODEL_PHONG || WEBGL_MATERIAL_MODEL_FLAT
layout(std140) uniform IgnisPhongMaterial {
	vec4 ignisSpecular;
	vec4 ignisPhong;
	vec4 ignisPhongAmbient;
};
#endif

#define uBaseColor ignisBaseColor
#define uEmissive ignisEmissive
#define uAlpha ignisAlpha
#define uDoubleSided int(ignisMaterialRenderParams.x + 0.5)

#if WEBGL_MATERIAL_BASE_MAP
#define uHasBaseMap 1
#define uBaseMapUV int(ignisBaseMapTransformB.z + 0.5)
#define uBaseMapTransformA ignisBaseMapTransformA
#define uBaseMapTransformB ignisBaseMapTransformB.xy
#endif
#if WEBGL_MATERIAL_EMISSIVE_MAP
#define uHasEmissiveMap 1
#define uEmissiveMapUV int(ignisEmissiveMapTransformB.z + 0.5)
#define uEmissiveMapTransformA ignisEmissiveMapTransformA
#define uEmissiveMapTransformB ignisEmissiveMapTransformB.xy
#endif

#if WEBGL_MATERIAL_MODEL_PBR
#define uPBR ignisPBR
#define uSpecular ignisSpecular
#define uTransmissionVolume ignisTransmissionVolume
#define uClearcoat ignisClearcoat
#define uSheen ignisSheen
#define uIridescence ignisIridescence
#define uAttenuationColor ignisAttenuationColor
#define uAnisotropy ignisAnisotropy
#define uNormalScale ignisPBRScales.x
#define uOcclusionStrength ignisPBRScales.y
#elif WEBGL_MATERIAL_MODEL_PHONG || WEBGL_MATERIAL_MODEL_FLAT
#define uSpecular ignisSpecular
#define uPhong ignisPhong
#define uPhongAmbient ignisPhongAmbient
#endif

#if WEBGL_MATERIAL_METALLIC_ROUGHNESS_MAP
#define uHasMetallicRoughnessMap 1
#define uMetallicRoughnessMapUV int(ignisMetallicRoughnessMapTransformB.z + 0.5)
#define uMetallicRoughnessMapTransformA ignisMetallicRoughnessMapTransformA
#define uMetallicRoughnessMapTransformB ignisMetallicRoughnessMapTransformB.xy
#endif
#if WEBGL_MATERIAL_SPECULAR_MAP
#define uHasSpecularMap 1
#define uSpecularMapUV int(ignisSpecularMapTransformB.z + 0.5)
#define uSpecularMapTransformA ignisSpecularMapTransformA
#define uSpecularMapTransformB ignisSpecularMapTransformB.xy
#endif
#if WEBGL_MATERIAL_SPECULAR_COLOR_MAP
#define uHasSpecularColorMap 1
#define uSpecularColorMapUV int(ignisSpecularColorMapTransformB.z + 0.5)
#define uSpecularColorMapTransformA ignisSpecularColorMapTransformA
#define uSpecularColorMapTransformB ignisSpecularColorMapTransformB.xy
#endif
#if WEBGL_MATERIAL_CLEARCOAT_MAP
#define uHasClearcoatMap 1
#define uClearcoatMapUV int(ignisClearcoatMapTransformB.z + 0.5)
#define uClearcoatMapTransformA ignisClearcoatMapTransformA
#define uClearcoatMapTransformB ignisClearcoatMapTransformB.xy
#endif
#if WEBGL_MATERIAL_CLEARCOAT_ROUGHNESS_MAP
#define uHasClearcoatRoughnessMap 1
#define uClearcoatRoughnessMapUV int(ignisClearcoatRoughnessMapTransformB.z + 0.5)
#define uClearcoatRoughnessMapTransformA ignisClearcoatRoughnessMapTransformA
#define uClearcoatRoughnessMapTransformB ignisClearcoatRoughnessMapTransformB.xy
#endif
#if WEBGL_MATERIAL_CLEARCOAT_NORMAL_MAP
#define uHasClearcoatNormalMap 1
#define uClearcoatNormalMapUV int(ignisClearcoatNormalMapTransformB.z + 0.5)
#define uClearcoatNormalMapTransformA ignisClearcoatNormalMapTransformA
#define uClearcoatNormalMapTransformB ignisClearcoatNormalMapTransformB.xy
#endif
#if WEBGL_MATERIAL_SHEEN_COLOR_MAP
#define uHasSheenColorMap 1
#define uSheenColorMapUV int(ignisSheenColorMapTransformB.z + 0.5)
#define uSheenColorMapTransformA ignisSheenColorMapTransformA
#define uSheenColorMapTransformB ignisSheenColorMapTransformB.xy
#endif
#if WEBGL_MATERIAL_SHEEN_ROUGHNESS_MAP
#define uHasSheenRoughnessMap 1
#define uSheenRoughnessMapUV int(ignisSheenRoughnessMapTransformB.z + 0.5)
#define uSheenRoughnessMapTransformA ignisSheenRoughnessMapTransformA
#define uSheenRoughnessMapTransformB ignisSheenRoughnessMapTransformB.xy
#endif
#if WEBGL_MATERIAL_TRANSMISSION_MAP
#define uHasTransmissionMap 1
#define uTransmissionMapUV int(ignisTransmissionMapTransformB.z + 0.5)
#define uTransmissionMapTransformA ignisTransmissionMapTransformA
#define uTransmissionMapTransformB ignisTransmissionMapTransformB.xy
#endif
#if WEBGL_MATERIAL_THICKNESS_MAP
#define uHasThicknessMap 1
#define uThicknessMapUV int(ignisThicknessMapTransformB.z + 0.5)
#define uThicknessMapTransformA ignisThicknessMapTransformA
#define uThicknessMapTransformB ignisThicknessMapTransformB.xy
#endif
#if WEBGL_MATERIAL_NORMAL_MAP
#define uHasNormalMap 1
#define uNormalMapUV int(ignisNormalMapTransformB.z + 0.5)
#define uNormalMapTransformA ignisNormalMapTransformA
#define uNormalMapTransformB ignisNormalMapTransformB.xy
#endif
#if WEBGL_MATERIAL_OCCLUSION_MAP
#define uHasOcclusionMap 1
#define uOcclusionMapUV int(ignisOcclusionMapTransformB.z + 0.5)
#define uOcclusionMapTransformA ignisOcclusionMapTransformA
#define uOcclusionMapTransformB ignisOcclusionMapTransformB.xy
#endif
#if WEBGL_MATERIAL_IRIDESCENCE_MAP
#define uHasIridescenceMap 1
#define uIridescenceMapUV int(ignisIridescenceMapTransformB.z + 0.5)
#define uIridescenceMapTransformA ignisIridescenceMapTransformA
#define uIridescenceMapTransformB ignisIridescenceMapTransformB.xy
#endif
#if WEBGL_MATERIAL_IRIDESCENCE_THICKNESS_MAP
#define uHasIridescenceThicknessMap 1
#define uIridescenceThicknessMapUV int(ignisIridescenceThicknessMapTransformB.z + 0.5)
#define uIridescenceThicknessMapTransformA ignisIridescenceThicknessMapTransformA
#define uIridescenceThicknessMapTransformB ignisIridescenceThicknessMapTransformB.xy
#endif
#if WEBGL_MATERIAL_ANISOTROPY_MAP
#define uHasAnisotropyMap 1
#define uAnisotropyMapUV int(ignisAnisotropyMapTransformB.z + 0.5)
#define uAnisotropyMapTransformA ignisAnisotropyMapTransformA
#define uAnisotropyMapTransformB ignisAnisotropyMapTransformB.xy
#endif
