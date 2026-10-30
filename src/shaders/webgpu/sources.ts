import {
	createShaderModuleGroup,
	mapShaderManifestEntries,
	type ShaderBackendManifest,
	type ShaderSourceNode,
} from "../ShaderManifest";

export type WebGPUSceneShaderPart = keyof typeof WEBGPU_SCENE_SHADER_FILES;
export type WebGPUPostProcessShaderPart = keyof typeof WEBGPU_POST_PROCESS_SHADER_FILES;
export type WebGPUShadowShaderPart = keyof typeof WEBGPU_SHADOW_SHADER_FILES;
export type WebGPUUtilityShaderPart = keyof typeof WEBGPU_UTILITY_SHADER_FILES;
export type WebGPUDirectiveShaderPart = keyof typeof WEBGPU_DIRECTIVE_SHADER_FILES;

const WEBGPU_POST_PROCESS_PARTS_USING_SHARED_LIGHT_DATA =
	new Set<WebGPUPostProcessShaderPart>(["ssr", "volumetric"]);

const WEBGPU_SCENE_SHADER_FILES = {
	lightData: "./webgpu/common/lightData.wgsl",
	constants: "./webgpu/common/constants.wgsl",
	definitions: "./webgpu/common/definitions.wgsl",
	utils: "./webgpu/common/utils.wgsl",
	deferredGBufferCodec: "./webgpu/common/deferredGBufferCodec.wgsl",
	surfaceLighting: "./webgpu/common/surfaceLighting.wgsl",
	vertexStage: "./webgpu/scene/vertexStage.wgsl",
	fragmentPrelude: "./webgpu/scene/fragmentPrelude.wgsl",
	fragmentPhong: "./webgpu/scene/fragmentPhong.wgsl",
	fragmentPbrSetup: "./webgpu/scene/fragmentPbrSetup.wgsl",
	fragmentPbrDirectional: "./webgpu/scene/fragmentPbrDirectional.wgsl",
	fragmentPbrPoint: "./webgpu/scene/fragmentPbrPoint.wgsl",
	fragmentPbrSpot: "./webgpu/scene/fragmentPbrSpot.wgsl",
	fragmentPbrArea: "./webgpu/scene/fragmentPbrArea.wgsl",
	fragmentPbrAmbient: "./webgpu/scene/fragmentPbrAmbient.wgsl",
	fragmentGBuffer: "./webgpu/scene/fragmentGBuffer.wgsl",
	fragmentSingleTarget: "./webgpu/scene/fragmentSingleTarget.wgsl",
} as const satisfies Record<string, string>;

const WEBGPU_POST_PROCESS_SHADER_FILES = {
	ssao: "./webgpu/postprocess/ssao.wgsl",
	ssgi: "./webgpu/postprocess/ssgi.wgsl",
	denoise: "./webgpu/postprocess/denoise.wgsl",
	taa: "./webgpu/postprocess/taa.wgsl",
	hiz: "./webgpu/postprocess/hiz.wgsl",
	ssr: "./webgpu/postprocess/ssr.wgsl",
	screenSpaceRefractions: "./webgpu/postprocess/ssrf.wgsl",
	volumetric: "./webgpu/postprocess/volumetric.wgsl",
	fog: "./webgpu/postprocess/fog.wgsl",
	motionBlur: "./webgpu/postprocess/motionBlur.wgsl",
	dof: "./webgpu/postprocess/dof.wgsl",
	bloom: "./webgpu/postprocess/bloom.wgsl",
	gamma: "./webgpu/postprocess/gamma.wgsl",
	toneMapping: "./webgpu/postprocess/toneMapping.wgsl",
	colorFilter: "./webgpu/postprocess/colorFilter.wgsl",
	fxaa: "./webgpu/postprocess/fxaa.wgsl",
	sobelNormal: "./webgpu/postprocess/sobelNormal.wgsl",
} as const satisfies Record<string, string>;

const WEBGPU_SHADOW_SHADER_FILES = {
	depth: "./webgpu/shadow/depth.wgsl",
} as const satisfies Record<string, string>;

const WEBGPU_UTILITY_SHADER_FILES = {
	planarReflectionComposite: "./webgpu/utility/planarReflectionComposite.wgsl",
	present: "./webgpu/utility/present.wgsl",
	colorDirtyClear: "./webgpu/utility/colorDirtyClear.wgsl",
	depthDirtyClear: "./webgpu/utility/depthDirtyClear.wgsl",
	decal: "./webgpu/scene/decal.wgsl",
	oitResolve: "./webgpu/utility/oitResolve.wgsl",
	occlusionCulling: "./webgpu/utility/occlusionCulling.wgsl",
	mipmapBlit: { path: "./webgpu/utility/mipmapBlit.wgsl", sync: true },
} as const;

const WEBGPU_FIXED_SHADER_FILES = {
	environment: "./webgpu/environment/background.wgsl",
	deferredLighting: "./webgpu/lighting/deferredLighting.wgsl",
	particle: "./webgpu/particles/render.wgsl",
	particleSimulation: "./webgpu/particles/simulation.wgsl",
	clusteredLightingCull: "./webgpu/lighting/clusteredLightingCull.wgsl",
	iblPrefilter: "./webgpu/environment/iblPrefilter.wgsl",
} as const satisfies Record<string, string>;

const WEBGPU_DIRECTIVE_SHADER_FILES = {
	animation: "./webgpu/common/animation.wgsl",
	frameCamera: "./webgpu/common/frameCamera.wgsl",
	frameUniforms: "./webgpu/common/frameUniforms.wgsl",
	constants: "./webgpu/directives/constants.wgsl",
	srgb: "./webgpu/directives/srgb.wgsl",
	lumaWeights: "./webgpu/directives/lumaWeights.wgsl",
	lumaCommon: "./webgpu/directives/lumaCommon.wgsl",
} as const satisfies Record<string, string>;

const asset = <const T extends string>(id: T) => ({ asset: id });
const concat = (
	ids: readonly string[],
	fallbackSourcePath: string,
): ShaderSourceNode => ({
	concat: ids.map(asset),
	fallbackSourcePath,
});

const sceneModules = createShaderModuleGroup(
	WEBGPU_SCENE_SHADER_FILES, "scene.", "webgpu.scene.part.", "builtin-scene",
);
const shadowModules = createShaderModuleGroup(
	WEBGPU_SHADOW_SHADER_FILES, "shadow.", "webgpu.shadow.", "shadow",
);
const utilityModules = createShaderModuleGroup(
	WEBGPU_UTILITY_SHADER_FILES, "utility.", "webgpu.utility.", "unknown",
);
const directiveModules = createShaderModuleGroup(
	WEBGPU_DIRECTIVE_SHADER_FILES, "directive.", "webgpu.directive.", "unknown",
);

const assets = {
	...sceneModules.assets,
	...mapShaderManifestEntries(WEBGPU_POST_PROCESS_SHADER_FILES, "postprocess.",
		(path) => ({ path })),
	...shadowModules.assets,
	...utilityModules.assets,
	...directiveModules.assets,
	...mapShaderManifestEntries(WEBGPU_FIXED_SHADER_FILES, "fixed.", (path) => ({ path })),
	"scene.depthPrepass": { path: "./webgpu/scene/depthPrepass.wgsl" },
	"scene.shadowVisibility": { path: "./webgpu/common/shadowVisibility.wgsl" },
} as const;

const sources = {
	...sceneModules.sources,
	...shadowModules.sources,
	...utilityModules.sources,
	...directiveModules.sources,
	...mapShaderManifestEntries(WEBGPU_POST_PROCESS_SHADER_FILES, "webgpu.postprocess.",
		(_path, part) => ({
			kind: "module",
			sourceKind: "postprocess",
			source: concat(
				WEBGPU_POST_PROCESS_PARTS_USING_SHARED_LIGHT_DATA.has(part) ?
					["scene.lightData", `postprocess.${part}`] : [`postprocess.${part}`],
				"<webgpu-postprocess-part>",
			),
		} as const)),
	"webgpu.scene.depth": {
		kind: "module",
		sourceKind: "builtin-scene",
		source: asset("scene.depthPrepass"),
	},
	"webgpu.utility.decal": {
		kind: "module",
		sourceKind: "decal",
		source: concat(["scene.deferredGBufferCodec", "utility.decal"], "<webgpu-decal-part>"),
	},
	"webgpu.scene": {
		kind: "module",
		sourceKind: "builtin-scene",
		source: concat(
			[...Object.keys(sceneModules.assets), "scene.shadowVisibility"],
			"<webgpu-scene-part>",
		),
	},
	"webgpu.environment": {
		kind: "module",
		sourceKind: "builtin-environment",
		source: concat(["scene.lightData", "fixed.environment"], "<webgpu-shared-light-data-part>"),
	},
	"webgpu.deferredLighting": {
		kind: "module",
		sourceKind: "builtin-scene",
		source: concat(
			[
				"scene.lightData",
				"scene.constants",
				"scene.definitions",
				"scene.utils",
				"scene.shadowVisibility",
				"scene.deferredGBufferCodec",
				"scene.surfaceLighting",
				"fixed.deferredLighting",
			],
			"<webgpu-deferred-lighting-part>",
		),
	},
	"webgpu.particle": {
		kind: "module",
		sourceKind: "particle",
		source: asset("fixed.particle"),
	},
	"webgpu.particleSimulation": {
		kind: "module",
		sourceKind: "particle",
		source: asset("fixed.particleSimulation"),
	},
	"webgpu.clusteredLightingCull": {
		kind: "module",
		sourceKind: "clustered",
		source: concat(
			["scene.lightData", "fixed.clusteredLightingCull"],
			"<webgpu-shared-light-data-part>",
		),
	},
	"webgpu.iblPrefilter": {
		kind: "module",
		sourceKind: "builtin-environment",
		source: asset("fixed.iblPrefilter"),
	},
} as const;

export const WEBGPU_SHADER_MANIFEST = {
	backend: "webgpu",
	language: "wgsl",
	assets,
	sources,
	preloadGroups: {
		backendInit: ["webgpu.utility.mipmapBlit"],
	},
	profile: {
		baseId: "ignis/webgpu-profile-base",
		revision: 1,
		includes: [
			{ id: "ignis/webgpu/animation.wgsl", source: "webgpu.directive.animation" },
			{ id: "ignis/webgpu/frame-camera.wgsl", source: "webgpu.directive.frameCamera" },
			{ id: "ignis/webgpu/frame-uniforms.wgsl", source: "webgpu.directive.frameUniforms" },
			{ id: "ignis/webgpu/constants-base.wgsl", source: "webgpu.directive.constants" },
			{ id: "ignis/color/srgb.wgsl", source: "webgpu.directive.srgb" },
			{ id: "ignis/postprocess/luma-weights.wgsl", source: "webgpu.directive.lumaWeights" },
			{ id: "ignis/postprocess/luma-common.wgsl", source: "webgpu.directive.lumaCommon" },
		],
		overlay: {
			id: "ignis/webgpu-instance-overlay",
			includeId: "ignis/webgpu/constants.wgsl",
			sourcePath: "runtime://ignis/includes/wgsl/webgpu/constants.wgsl",
			baseInclude: "ignis/webgpu/constants-base",
			parameters: {
				type: "record",
				fields: Object.fromEntries(
					[
						"maxDirectionalLights", "maxPointLights", "maxSpotLights",
						"maxAreaLights", "maxLocalLightProbes", "maxReflectionProbes",
						"shCoefficientCount", "textureSlotCount",
					].map((name) => [name, { type: "integer", required: true, min: 0 }]),
				),
			},
			defines: {
				__WEBGPU_MAX_DIRECTIONAL_LIGHTS__: { parameter: "maxDirectionalLights" },
				__WEBGPU_MAX_POINT_LIGHTS__: { parameter: "maxPointLights" },
				__WEBGPU_MAX_SPOT_LIGHTS__: { parameter: "maxSpotLights" },
				__WEBGPU_MAX_AREA_LIGHTS__: { parameter: "maxAreaLights" },
				__WEBGPU_MAX_LOCAL_LIGHT_PROBES__: { parameter: "maxLocalLightProbes" },
				__WEBGPU_MAX_LOCAL_LIGHT_PROBES__u: { parameter: "maxLocalLightProbes" },
				__WEBGPU_MAX_REFLECTION_PROBES__: { parameter: "maxReflectionProbes" },
				__WEBGPU_SH_COEFFICIENT_COUNT__: { parameter: "shCoefficientCount" },
				__WEBGPU_LOCAL_LIGHT_PROBE_COEFFICIENT_COUNT__: {
					multiply: [{ parameter: "maxLocalLightProbes" }, { parameter: "shCoefficientCount" }],
				},
				__WEBGPU_TEXTURE_SLOT_COUNT__: { parameter: "textureSlotCount" },
				__WEBGPU_FRAME_DIRECTIONAL_LIGHT_VEC4_COUNT__: {
					multiply: [{ parameter: "maxDirectionalLights" }, { literal: 2 }],
				},
				__WEBGPU_FRAME_POINT_LIGHT_VEC4_COUNT__: {
					multiply: [{ parameter: "maxPointLights" }, { literal: 2 }],
				},
				__WEBGPU_FRAME_SPOT_LIGHT_VEC4_COUNT__: {
					multiply: [{ parameter: "maxSpotLights" }, { literal: 3 }],
				},
			},
		},
	},
} as const satisfies ShaderBackendManifest<typeof sources>;
