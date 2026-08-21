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

export interface WebGLShaderDirectiveProfileLimits {
	maxDirectionalLights: number;
	maxPointLights: number;
	maxSpotLights: number;
	maxClusterLightsPerFragment: number;
	maxLocalLightProbes: number;
	maxReflectionProbes: number;
}

const WEBGL_DIRECTIVE_MODULES = [
	["animation", "ignis/webgl/animation.glsl"],
	["constants", "ignis/webgl/constants-base.glsl"],
	["srgb", "ignis/color/srgb.glsl"],
	["fog", "ignis/postprocess/fog.glsl"],
	["lumaWeights", "ignis/postprocess/luma-weights.glsl"],
	["lumaCommon", "ignis/postprocess/luma-common.glsl"],
] as const;

function toIncludeModule(
	id: string,
	composite: CompositeShaderSource,
): ShaderIncludeModule {
	const code = composite.code;
	return {
		language: "glsl",
		id,
		code,
		sourcePath:
			composite.sourceMap.segments[0]?.sourcePath ??
			`runtime://ignis/includes/glsl/${id}`,
	};
}

/** @internal WebGL backend directive-profile preparation. */
export async function prepareWebGLShaderDirectiveProfileBase():
	Promise<ShaderDirectiveProfileBase> {
	const composites = await Promise.all(
		WEBGL_DIRECTIVE_MODULES.map(([part]) =>
			ShaderSource.load(`webgl.directive.${part}.composite`),
		),
	);
	const includeModules = composites.map((composite, index) =>
		toIncludeModule(WEBGL_DIRECTIVE_MODULES[index][1], composite),
	);
	const assetPack: ShaderDirectiveFeaturePack = {
		id: "ignis/webgl-directive-assets",
		backend: "webgl",
		revision: 1,
		includeModules,
		injectionScripts: [],
	};
	return {
		id: "ignis/webgl-profile-base",
		backend: "webgl",
		packs: [assetPack, ...createBuiltinInjectionFeaturePacks("webgl")],
	};
}

/** @internal WebGL backend directive-profile composition. */
export function createWebGLShaderDirectiveProfile(
	base: ShaderDirectiveProfileBase,
	limits: WebGLShaderDirectiveProfileLimits,
): ShaderDirectiveProfile {
	const code = [
		"#include <ignis/webgl/constants-base>",
		`#define __WEBGL_MAX_DIRECTIONAL_LIGHTS__ ${limits.maxDirectionalLights}`,
		`#define __WEBGL_MAX_POINT_LIGHTS__ ${limits.maxPointLights}`,
		`#define __WEBGL_MAX_SPOT_LIGHTS__ ${limits.maxSpotLights}`,
		`#define __WEBGL_MAX_CLUSTER_LIGHTS_PER_FRAGMENT__ ${limits.maxClusterLightsPerFragment}`,
		`#define __WEBGL_MAX_LOCAL_LIGHT_PROBES__ ${limits.maxLocalLightProbes}`,
		`#define __WEBGL_MAX_REFLECTION_PROBES__ ${limits.maxReflectionProbes}`,
	].join("\n");
	return composeShaderDirectiveProfile(base, {
		id: "ignis/webgl-instance-overlay",
		backend: "webgl",
		includeModules: [
			{
				language: "glsl",
				id: "ignis/webgl/constants.glsl",
				code,
				sourcePath: "runtime://ignis/includes/glsl/webgl/constants.glsl",
			},
		],
	});
}
