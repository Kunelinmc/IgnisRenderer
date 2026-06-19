in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUv;
in vec2 vUv1;
in vec2 vUv2;
in vec2 vUv3;
in vec4 vTangent;
in vec4 vCurrentClip;
in vec4 vPrevClip;
in float vViewDepth;

uniform vec3 uCameraPosition;
uniform vec3 uAmbientColor;
__WEBGL_SCENE_LIGHTING_UNIFORMS__
uniform int uEnableLighting;
__WEBGL_SCENE_SH_UNIFORMS__
uniform int uEnableSH;
uniform vec3 uSHAmbientCoeffs[SH_COEFFICIENT_COUNT];
__WEBGL_MATERIAL_SHADING_MODEL_UNIFORMS__
uniform int uShadingModel;
uniform int uDoubleSided;
uniform vec4 uBaseColor;
uniform vec4 uEmissive;
__WEBGL_MATERIAL_PBR_UNIFORMS__
uniform vec4 uPBR;
__WEBGL_MATERIAL_TRANSMISSION_UNIFORMS__
uniform vec4 uTransmissionVolume;
uniform vec4 uAttenuationColor;
__WEBGL_MATERIAL_IRIDESCENCE_UNIFORMS__
uniform vec4 uIridescence;
__WEBGL_MATERIAL_ANISOTROPY_UNIFORMS__
uniform vec4 uAnisotropy;
__WEBGL_MATERIAL_PHONG_UNIFORMS__
uniform vec4 uPhong;
__WEBGL_MATERIAL_ALPHA_MASK_UNIFORMS__
uniform vec4 uAlpha;
__WEBGL_MATERIAL_BASE_MAP_UNIFORMS__
uniform sampler2D uBaseMap;
uniform int uHasBaseMap;
uniform int uBaseMapIsLinear;
uniform int uBaseMapUV;
uniform vec4 uBaseMapTransformA;
uniform vec2 uBaseMapTransformB;
__WEBGL_MATERIAL_METALLIC_ROUGHNESS_MAP_UNIFORMS__
uniform sampler2D uMetallicRoughnessMap;
uniform int uHasMetallicRoughnessMap;
uniform int uMetallicRoughnessMapUV;
uniform vec4 uMetallicRoughnessMapTransformA;
uniform vec2 uMetallicRoughnessMapTransformB;
__WEBGL_MATERIAL_NORMAL_MAP_UNIFORMS__
uniform sampler2D uNormalMap;
uniform int uHasNormalMap;
uniform int uNormalMapUV;
uniform vec4 uNormalMapTransformA;
uniform vec2 uNormalMapTransformB;
uniform float uNormalScale;
__WEBGL_MATERIAL_EMISSIVE_MAP_UNIFORMS__
uniform sampler2D uEmissiveMap;
uniform int uHasEmissiveMap;
uniform int uEmissiveMapIsLinear;
uniform int uEmissiveMapUV;
uniform vec4 uEmissiveMapTransformA;
uniform vec2 uEmissiveMapTransformB;
__WEBGL_MATERIAL_OCCLUSION_MAP_UNIFORMS__
uniform sampler2D uOcclusionMap;
uniform int uHasOcclusionMap;
uniform int uOcclusionMapUV;
uniform vec4 uOcclusionMapTransformA;
uniform vec2 uOcclusionMapTransformB;
uniform float uOcclusionStrength;
__WEBGL_MATERIAL_IRIDESCENCE_MAP_UNIFORMS__
uniform sampler2D uIridescenceMap;
uniform int uHasIridescenceMap;
uniform int uIridescenceMapUV;
uniform vec4 uIridescenceMapTransformA;
uniform vec2 uIridescenceMapTransformB;
__WEBGL_MATERIAL_IRIDESCENCE_THICKNESS_MAP_UNIFORMS__
uniform sampler2D uIridescenceThicknessMap;
uniform int uHasIridescenceThicknessMap;
uniform int uIridescenceThicknessMapUV;
uniform vec4 uIridescenceThicknessMapTransformA;
uniform vec2 uIridescenceThicknessMapTransformB;
__WEBGL_MATERIAL_ANISOTROPY_MAP_UNIFORMS__
uniform int uHasAnisotropyMap;
uniform int uAnisotropyMapUV;
uniform vec4 uAnisotropyMapTransformA;
uniform vec2 uAnisotropyMapTransformB;
__WEBGL_SCENE_ENVIRONMENT_SPECULAR_UNIFORMS__
uniform sampler2D uEnvSpecularMap;
uniform int uHasEnvSpecularMap;
uniform int uEnvSpecularMapIsLinear;
uniform float uEnvSpecularMaxMipLevel;
uniform sampler2D uEnvSpecularFallbackMap;
uniform int uHasEnvSpecularFallbackMap;
uniform int uEnvSpecularFallbackMapIsLinear;
uniform float uEnvSpecularFallbackMaxMipLevel;
uniform sampler2D uBrdfLUT;
__WEBGL_SCENE_LOCAL_LIGHT_PROBE_UNIFORMS__
uniform int uLocalLightProbeCount;
uniform vec4 uLocalLightProbeWorldToProbeRow0[MAX_LOCAL_LIGHT_PROBES];
uniform vec4 uLocalLightProbeWorldToProbeRow1[MAX_LOCAL_LIGHT_PROBES];
uniform vec4 uLocalLightProbeWorldToProbeRow2[MAX_LOCAL_LIGHT_PROBES];
uniform vec4 uLocalLightProbeDataA[MAX_LOCAL_LIGHT_PROBES];
uniform vec4 uLocalLightProbeDataB[MAX_LOCAL_LIGHT_PROBES];
uniform sampler2D uLocalLightProbeCoeffs;
uniform vec2 uLocalLightProbeCoeffsSize;
__WEBGL_IRRADIANCE_PROBE_GRID_UNIFORMS__
__WEBGL_SCENE_REFLECTION_PROBE_UNIFORMS__
uniform int uReflectionProbeCount;
uniform vec4 uReflectionProbeWorldToProbeRow0[MAX_REFLECTION_PROBES];
uniform vec4 uReflectionProbeWorldToProbeRow1[MAX_REFLECTION_PROBES];
uniform vec4 uReflectionProbeWorldToProbeRow2[MAX_REFLECTION_PROBES];
uniform vec4 uReflectionProbeProbeToWorldRow0[MAX_REFLECTION_PROBES];
uniform vec4 uReflectionProbeProbeToWorldRow1[MAX_REFLECTION_PROBES];
uniform vec4 uReflectionProbeProbeToWorldRow2[MAX_REFLECTION_PROBES];
uniform vec4 uReflectionProbeDataA[MAX_REFLECTION_PROBES];
uniform vec4 uReflectionProbeDataB[MAX_REFLECTION_PROBES];
uniform vec4 uReflectionProbeDataC[MAX_REFLECTION_PROBES];

__WEBGL_SCENE_FORWARD_LIGHT_UNIFORMS__
uniform int uDirLightCount;
uniform vec4 uDirLightDirection[MAX_DIRECTIONAL_LIGHTS];
uniform vec4 uDirLightColor[MAX_DIRECTIONAL_LIGHTS];

uniform int uPointLightCount;
uniform vec4 uPointLightPositionRange[MAX_POINT_LIGHTS];
uniform vec4 uPointLightColor[MAX_POINT_LIGHTS];

uniform int uSpotLightCount;
uniform vec4 uSpotLightPositionRange[MAX_SPOT_LIGHTS];
uniform vec4 uSpotLightDirectionOuter[MAX_SPOT_LIGHTS];
uniform vec4 uSpotLightColorInner[MAX_SPOT_LIGHTS];
__WEBGL_SCENE_SHADOW_UNIFORMS__
uniform int uEnableShadows;
uniform sampler2D uShadowAtlas;
__WEBGL_SHADOW_TRANSMITTANCE_UNIFORMS__
uniform sampler2D uParticleShadowVolumeAtlas;
uniform vec2 uParticleShadowVolumeAtlasSize;
uniform vec4 uParticleShadowVolumeGridSize;
uniform vec4 uParticleShadowVolumeSliceParams[4];
uniform mat4 uDirShadowViewProjection[MAX_DIRECTIONAL_LIGHTS];
uniform mat4 uDirShadowCascadeViewProjection[MAX_DIRECTIONAL_LIGHTS * 4];
uniform vec4 uDirShadowCascadeSplits[MAX_DIRECTIONAL_LIGHTS * 4];
uniform vec4 uDirShadowParamsA[MAX_DIRECTIONAL_LIGHTS];
uniform vec4 uDirShadowParamsB[MAX_DIRECTIONAL_LIGHTS];
uniform vec4 uDirShadowParamsC[MAX_DIRECTIONAL_LIGHTS];
uniform vec4 uDirShadowParamsD[MAX_DIRECTIONAL_LIGHTS];
uniform mat4 uSpotShadowViewProjection[MAX_SPOT_LIGHTS];
uniform vec4 uSpotShadowParamsA[MAX_SPOT_LIGHTS];
uniform vec4 uSpotShadowParamsB[MAX_SPOT_LIGHTS];
uniform vec4 uSpotShadowParamsC[MAX_SPOT_LIGHTS];
uniform vec4 uSpotShadowParamsD[MAX_SPOT_LIGHTS];
__WEBGL_SCENE_CLUSTERED_LIGHT_UNIFORMS__
uniform int uEnableClusteredLighting;
uniform vec4 uClusterParams0;
uniform vec4 uClusterParams1;
uniform sampler2D uClusterHeaderTexture;
uniform sampler2D uClusterIndexTexture;
uniform sampler2D uClusterLightTexture;
uniform vec2 uClusterHeaderTexSize;
uniform vec2 uClusterIndexTexSize;
uniform vec2 uClusterLightTexSize;
uniform vec4 uFogParams0;
uniform vec4 uFogParams1;
__WEBGL_SCENE_OIT_UNIFORMS__
uniform int uOITPassMode;

layout(location = 0) out vec4 fragColor;
__WEBGL_SCENE_EXTRA_OUTPUTS__
layout(location = 1) out vec4 fragMotion;
layout(location = 2) out vec4 fragNormal;
