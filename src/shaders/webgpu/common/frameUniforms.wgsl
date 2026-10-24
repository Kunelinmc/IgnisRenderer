#import <ignis/webgpu/constants>

struct FrameLightUniforms {
	ambientColor: vec4<f32>,
	lightCounts: vec4<u32>,
	lightingEnabled: u32,
	clusteredLightingEnabled: u32,
	_pad0: u32,
	_pad1: u32,
	directionalLights: array<DirectionalLightData, __WEBGPU_MAX_DIRECTIONAL_LIGHTS__>,
	pointLights: array<PointLightData, __WEBGPU_MAX_POINT_LIGHTS__>,
	spotLights: array<SpotLightData, __WEBGPU_MAX_SPOT_LIGHTS__>,
	areaLights: array<AreaLightData, __WEBGPU_MAX_AREA_LIGHTS__>,
}

struct FrameShadowUniforms {
	shadowsEnabled: u32,
	_pad0: u32,
	_pad1: u32,
	_pad2: u32,
	directionalShadows: array<ShadowData, __WEBGPU_MAX_DIRECTIONAL_LIGHTS__>,
	spotShadows: array<ShadowData, __WEBGPU_MAX_SPOT_LIGHTS__>,
}

struct FrameEnvironmentUniforms {
	shAmbientEnabled: u32,
	hasEnvSpecular: u32,
	hasBRDFLUT: u32,
	envSpecularMaxMipLevel: f32,
	shAmbientCoeffs: array<vec4<f32>, __WEBGPU_SH_COEFFICIENT_COUNT__>,
	reflectionProbes: array<ReflectionProbeData, __WEBGPU_MAX_REFLECTION_PROBES__>,
	localLightProbeCounts: vec4<f32>,
	localLightProbeWorldToProbeRow0: array<vec4<f32>, __WEBGPU_MAX_LOCAL_LIGHT_PROBES__>,
	localLightProbeWorldToProbeRow1: array<vec4<f32>, __WEBGPU_MAX_LOCAL_LIGHT_PROBES__>,
	localLightProbeWorldToProbeRow2: array<vec4<f32>, __WEBGPU_MAX_LOCAL_LIGHT_PROBES__>,
	localLightProbeDataA: array<vec4<f32>, __WEBGPU_MAX_LOCAL_LIGHT_PROBES__>,
	localLightProbeDataB: array<vec4<f32>, __WEBGPU_MAX_LOCAL_LIGHT_PROBES__>,
	localLightProbeSHAmbientCoeffs:
		array<vec4<f32>, __WEBGPU_LOCAL_LIGHT_PROBE_COEFFICIENT_COUNT__>,
	irradianceProbeGridWorldToGridRow0: vec4<f32>,
	irradianceProbeGridWorldToGridRow1: vec4<f32>,
	irradianceProbeGridWorldToGridRow2: vec4<f32>,
	irradianceProbeGridDataA: vec4<f32>,
	irradianceProbeGridDataB: vec4<f32>,
	irradianceProbeGridDataC: vec4<f32>,
}
