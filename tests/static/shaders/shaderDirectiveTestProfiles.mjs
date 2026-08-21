import {
	MAX_AREA_LIGHTS,
	MAX_CLUSTER_LIGHTS_PER_FRAGMENT,
	MAX_DIRECTIONAL_LIGHTS,
	MAX_LOCAL_LIGHT_PROBES,
	MAX_POINT_LIGHTS,
	MAX_REFLECTION_PROBES,
	MAX_SPOT_LIGHTS,
} from "../../../src/backends/constants.ts";
import {
	WEBGPU_SH_COEFFICIENT_COUNT,
	WEBGPU_TEXTURE_SLOT_COUNT,
} from "../../../src/backends/webgpu/constants.ts";
import {
	createWebGLShaderDirectiveProfile,
	prepareWebGLShaderDirectiveProfileBase,
} from "../../../src/shaders/webgl/webGLProfile.ts";
import {
	createWebGPUShaderDirectiveProfile,
	prepareWebGPUShaderDirectiveProfileBase,
} from "../../../src/shaders/webgpu/webGPUProfile.ts";

export const WEBGL_TEST_PROFILE = createWebGLShaderDirectiveProfile(
	await prepareWebGLShaderDirectiveProfileBase(),
	{
		maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
		maxPointLights: MAX_POINT_LIGHTS,
		maxSpotLights: MAX_SPOT_LIGHTS,
		maxClusterLightsPerFragment: MAX_CLUSTER_LIGHTS_PER_FRAGMENT,
		maxLocalLightProbes: MAX_LOCAL_LIGHT_PROBES,
		maxReflectionProbes: MAX_REFLECTION_PROBES,
	},
);

export const WEBGPU_TEST_PROFILE = createWebGPUShaderDirectiveProfile(
	await prepareWebGPUShaderDirectiveProfileBase(),
	{
		maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
		maxPointLights: MAX_POINT_LIGHTS,
		maxSpotLights: MAX_SPOT_LIGHTS,
		maxAreaLights: MAX_AREA_LIGHTS,
		maxLocalLightProbes: MAX_LOCAL_LIGHT_PROBES,
		maxReflectionProbes: MAX_REFLECTION_PROBES,
		shCoefficientCount: WEBGPU_SH_COEFFICIENT_COUNT,
		textureSlotCount: WEBGPU_TEXTURE_SLOT_COUNT,
	},
);
