import { createBuiltinInjectionFeaturePacks } from "../features/builtinInjectionScripts";
import { ShaderSource } from "../ShaderSource";
import { composeShaderDirectiveProfile } from "../runtime/DirectiveProfile";
import type {
	CompositeShaderSource,
	ShaderDirectiveFeaturePack,
	ShaderDirectiveProfile,
	ShaderDirectiveProfileBase,
	ShaderIncludeModule,
} from "../runtime/types";

export interface WebGPUShaderDirectiveProfileLimits {
	maxDirectionalLights: number;
	maxPointLights: number;
	maxSpotLights: number;
	maxAreaLights: number;
	maxLocalLightProbes: number;
	maxReflectionProbes: number;
	shCoefficientCount: number;
	textureSlotCount: number;
}

const WEBGPU_DIRECTIVE_MODULES = [
	["constants", "ignis/webgpu/constants-base.wgsl"],
	["srgb", "ignis/color/srgb.wgsl"],
	["fog", "ignis/postprocess/fog.wgsl"],
	["lumaWeights", "ignis/postprocess/luma-weights.wgsl"],
	["lumaCommon", "ignis/postprocess/luma-common.wgsl"],
] as const;

function toIncludeModule(
	id: string,
	composite: CompositeShaderSource,
): ShaderIncludeModule {
	const code = composite.code;
	return {
		language: "wgsl",
		id,
		code,
		sourcePath:
			composite.sourceMap.segments[0]?.sourcePath ??
			`runtime://ignis/includes/wgsl/${id}`,
	};
}

/** @internal WebGPU backend directive-profile preparation. */
export async function prepareWebGPUShaderDirectiveProfileBase():
	Promise<ShaderDirectiveProfileBase> {
	const composites = await Promise.all(
		WEBGPU_DIRECTIVE_MODULES.map(([part]) =>
			ShaderSource.load(`webgpu.directive.${part}.composite`),
		),
	);
	const includeModules = composites.map((composite, index) =>
		toIncludeModule(WEBGPU_DIRECTIVE_MODULES[index][1], composite),
	);
	const assetPack: ShaderDirectiveFeaturePack = {
		id: "ignis/webgpu-directive-assets",
		backend: "webgpu",
		revision: 1,
		includeModules,
		injectionScripts: [],
	};
	return {
		id: "ignis/webgpu-profile-base",
		backend: "webgpu",
		packs: [assetPack, ...createBuiltinInjectionFeaturePacks("webgpu")],
	};
}

/** @internal WebGPU backend directive-profile composition. */
export function createWebGPUShaderDirectiveProfile(
	base: ShaderDirectiveProfileBase,
	limits: WebGPUShaderDirectiveProfileLimits,
): ShaderDirectiveProfile {
	const localProbeCoefficientCount =
		limits.maxLocalLightProbes * limits.shCoefficientCount;
	const code = [
		"#include <ignis/webgpu/constants-base>",
		`#define __WEBGPU_MAX_DIRECTIONAL_LIGHTS__ ${limits.maxDirectionalLights}`,
		`#define __WEBGPU_MAX_POINT_LIGHTS__ ${limits.maxPointLights}`,
		`#define __WEBGPU_MAX_SPOT_LIGHTS__ ${limits.maxSpotLights}`,
		`#define __WEBGPU_MAX_AREA_LIGHTS__ ${limits.maxAreaLights}`,
		`#define __WEBGPU_MAX_LOCAL_LIGHT_PROBES__ ${limits.maxLocalLightProbes}`,
		`#define __WEBGPU_MAX_LOCAL_LIGHT_PROBES__u ${limits.maxLocalLightProbes}u`,
		`#define __WEBGPU_MAX_REFLECTION_PROBES__ ${limits.maxReflectionProbes}`,
		`#define __WEBGPU_SH_COEFFICIENT_COUNT__ ${limits.shCoefficientCount}`,
		`#define __WEBGPU_LOCAL_LIGHT_PROBE_COEFFICIENT_COUNT__ ${localProbeCoefficientCount}`,
		`#define __WEBGPU_TEXTURE_SLOT_COUNT__ ${limits.textureSlotCount}`,
		`#define __WEBGPU_FRAME_DIRECTIONAL_LIGHT_VEC4_COUNT__ ${limits.maxDirectionalLights * 2}`,
		`#define __WEBGPU_FRAME_POINT_LIGHT_VEC4_COUNT__ ${limits.maxPointLights * 2}`,
		`#define __WEBGPU_FRAME_SPOT_LIGHT_VEC4_COUNT__ ${limits.maxSpotLights * 3}`,
	].join("\n");
	return composeShaderDirectiveProfile(base, {
		id: "ignis/webgpu-instance-overlay",
		backend: "webgpu",
		includeModules: [
			{
				language: "wgsl",
				id: "ignis/webgpu/constants.wgsl",
				code,
				sourcePath: "runtime://ignis/includes/wgsl/webgpu/constants.wgsl",
			},
		],
	});
}
