import type { ShaderSourceSegmentKind } from "../runtime";
import type { ShaderBackendManifest, ShaderSourceNode } from "../ShaderManifest";

type ImportMetaGlobLoaderMap = Record<string, () => Promise<string>>;

export interface WebGPUShaderFileDescriptor {
	scope: "webgpu";
	key: string;
	path: string;
	segmentKind?: ShaderSourceSegmentKind;
}

export type WebGPUSceneShaderPart =
	| "lightData"
	| "constants"
	| "definitions"
	| "utils"
	| "deferredGBufferCodec"
	| "surfaceLighting"
	| "vertexStage"
	| "fragmentPrelude"
	| "fragmentPhong"
	| "fragmentPbrSetup"
	| "fragmentPbrDirectional"
	| "fragmentPbrPoint"
	| "fragmentPbrSpot"
	| "fragmentPbrArea"
	| "fragmentPbrAmbient"
	| "fragmentGBuffer"
	| "fragmentSingleTarget";

export type WebGPUPostProcessShaderPart =
	| "ssao"
	| "ssgi"
	| "denoise"
	| "taa"
	| "hiz"
	| "ssr"
	| "screenSpaceRefractions"
	| "volumetric"
	| "fog"
	| "motionBlur"
	| "dof"
	| "bloomDownsample"
	| "bloomBlurH"
	| "bloomBlurV"
	| "bloomUpsample"
	| "bloomComposite"
	| "gamma"
	| "toneMapping"
	| "colorFilter"
	| "fxaa"
	| "sobelNormal";

export type WebGPUShadowShaderPart =
	| "depth"
	| "pagedShadowRequestMark"
	| "pagedShadowRequestCompact"
	| "pagedShadowResidencyAllocate"
	| "pagedShadowDirtyCompact"
	| "pagedShadowDirtyGridBuild"
	| "pagedShadowDrawBuild"
	| "pagedShadowFeedback"
	| "pagedShadowPageTableCopy"
	| "pagedShadowClear";

export type WebGPUUtilityShaderPart =
	| "planarReflectionComposite"
	| "present"
	| "colorDirtyClear"
	| "depthDirtyClear"
	| "decal"
	| "oitResolve"
	| "occlusionCulling"
	| "mipmapBlit";

export type WebGPUDirectiveShaderPart =
	| "constants"
	| "srgb"
	| "fog"
	| "lumaWeights"
	| "lumaCommon";

const WEBGPU_SCENE_SHADER_PARTS: readonly WebGPUSceneShaderPart[] = [
	"lightData",
	"constants",
	"definitions",
	"utils",
	"deferredGBufferCodec",
	"surfaceLighting",
	"vertexStage",
	"fragmentPrelude",
	"fragmentPhong",
	"fragmentPbrSetup",
	"fragmentPbrDirectional",
	"fragmentPbrPoint",
	"fragmentPbrSpot",
	"fragmentPbrArea",
	"fragmentPbrAmbient",
	"fragmentGBuffer",
	"fragmentSingleTarget",
];

const WEBGPU_POST_PROCESS_PARTS_USING_SHARED_LIGHT_DATA =
	new Set<WebGPUPostProcessShaderPart>(["ssr", "volumetric"]);

const WEBGPU_SCENE_SHADER_FILES: Record<WebGPUSceneShaderPart, string> = {
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
};

const WEBGPU_POST_PROCESS_SHADER_FILES: Record<
	WebGPUPostProcessShaderPart,
	string
> = {
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
	bloomDownsample: "./webgpu/postprocess/bloomDownsample.wgsl",
	bloomBlurH: "./webgpu/postprocess/bloomBlurH.wgsl",
	bloomBlurV: "./webgpu/postprocess/bloomBlurV.wgsl",
	bloomUpsample: "./webgpu/postprocess/bloomUpsample.wgsl",
	bloomComposite: "./webgpu/postprocess/bloomComposite.wgsl",
	gamma: "./webgpu/postprocess/gamma.wgsl",
	toneMapping: "./webgpu/postprocess/toneMapping.wgsl",
	colorFilter: "./webgpu/postprocess/colorFilter.wgsl",
	fxaa: "./webgpu/postprocess/fxaa.wgsl",
	sobelNormal: "./webgpu/postprocess/sobelNormal.wgsl",
};

const WEBGPU_SHADOW_SHADER_FILES: Record<WebGPUShadowShaderPart, string> = {
	depth: "./webgpu/shadow/depth.wgsl",
	pagedShadowRequestMark: "./webgpu/shadow/pagedShadowRequestMark.wgsl",
	pagedShadowRequestCompact: "./webgpu/shadow/pagedShadowRequestCompact.wgsl",
	pagedShadowResidencyAllocate:
		"./webgpu/shadow/pagedShadowResidencyAllocate.wgsl",
	pagedShadowDirtyCompact: "./webgpu/shadow/pagedShadowDirtyCompact.wgsl",
	pagedShadowDirtyGridBuild: "./webgpu/shadow/pagedShadowDirtyGridBuild.wgsl",
	pagedShadowDrawBuild: "./webgpu/shadow/pagedShadowDrawBuild.wgsl",
	pagedShadowFeedback: "./webgpu/shadow/pagedShadowFeedback.wgsl",
	pagedShadowPageTableCopy: "./webgpu/shadow/pagedShadowPageTableCopy.wgsl",
	pagedShadowClear: "./webgpu/shadow/pagedShadowClear.wgsl",
};

const WEBGPU_UTILITY_SHADER_FILES: Record<WebGPUUtilityShaderPart, string> = {
	planarReflectionComposite: "./webgpu/utility/planarReflectionComposite.wgsl",
	present: "./webgpu/utility/present.wgsl",
	colorDirtyClear: "./webgpu/utility/colorDirtyClear.wgsl",
	depthDirtyClear: "./webgpu/utility/depthDirtyClear.wgsl",
	decal: "./webgpu/scene/decal.wgsl",
	oitResolve: "./webgpu/utility/oitResolve.wgsl",
	occlusionCulling: "./webgpu/utility/occlusionCulling.wgsl",
	mipmapBlit: "./webgpu/utility/mipmapBlit.wgsl",
};

const WEBGPU_FIXED_SHADER_FILES = {
	environment: {
		scope: "webgpu",
		key: "webgpu.environment",
		path: "./webgpu/environment/background.wgsl",
	},
	deferredLighting: {
		scope: "webgpu",
		key: "webgpu.deferredLighting",
		path: "./webgpu/lighting/deferredLighting.wgsl",
	},
	particle: {
		scope: "webgpu",
		key: "webgpu.particle",
		path: "./webgpu/particles/render.wgsl",
	},
	particleSimulation: {
		scope: "webgpu",
		key: "webgpu.particleSimulation",
		path: "./webgpu/particles/simulation.wgsl",
	},
	clusteredLightingCull: {
		scope: "webgpu",
		key: "webgpu.clusteredLightingCull",
		path: "./webgpu/lighting/clusteredLightingCull.wgsl",
	},
	iblPrefilter: {
		scope: "webgpu",
		key: "webgpu.iblPrefilter",
		path: "./webgpu/environment/iblPrefilter.wgsl",
	},
} as const satisfies Record<string, WebGPUShaderFileDescriptor>;

const WEBGPU_DIRECTIVE_SHADER_FILES: Record<
	WebGPUDirectiveShaderPart,
	string
> = {
	constants: "./webgpu/directives/constants.wgsl",
	srgb: "./webgpu/directives/srgb.wgsl",
	fog: "./webgpu/directives/fog.wgsl",
	lumaWeights: "./webgpu/directives/lumaWeights.wgsl",
	lumaCommon: "./webgpu/directives/lumaCommon.wgsl",
};

const WEBGPU_SHADER_MATERIAL_FILES = {
	textureHelpers: "./webgpu/material/shaderMaterialTextureHelpers.wgsl",
} as const;

const asset = (id: string): ShaderSourceNode => ({ asset: id });
const concat = (
	ids: readonly string[],
	fallbackSourcePath: string,
): ShaderSourceNode => ({
	concat: ids.map(asset),
	fallbackSourcePath,
});

const assets: Record<string, { path: string; sync?: boolean }> = {};
const sources: Record<string, any> = {};

for (const [part, path] of Object.entries(WEBGPU_SCENE_SHADER_FILES)) {
	const id = `scene.${part}`;
	assets[id] = { path };
	sources[`webgpu.scene.part.${part}`] = {
		kind: "module",
		sourceKind: "builtin-scene",
		source: asset(id),
	};
}
for (const [part, path] of Object.entries(WEBGPU_POST_PROCESS_SHADER_FILES)) {
	const id = `postprocess.${part}`;
	assets[id] = { path };
	const nodes = WEBGPU_POST_PROCESS_PARTS_USING_SHARED_LIGHT_DATA.has(
		part as WebGPUPostProcessShaderPart,
	) ? ["scene.lightData", id] : [id];
	sources[`webgpu.postprocess.${part}`] = {
		kind: "module",
		sourceKind: "postprocess",
		source: concat(nodes, "<webgpu-postprocess-part>"),
	};
}
for (const [part, path] of Object.entries(WEBGPU_SHADOW_SHADER_FILES)) {
	const id = `shadow.${part}`;
	assets[id] = { path };
	sources[`webgpu.shadow.${part}`] = {
		kind: "module",
		sourceKind: "shadow",
		source: asset(id),
	};
}
for (const [part, path] of Object.entries(WEBGPU_UTILITY_SHADER_FILES)) {
	const id = `utility.${part}`;
	assets[id] = { path, sync: part === "mipmapBlit" };
	sources[`webgpu.utility.${part}`] = {
		kind: "module",
		sourceKind: part === "decal" ? "decal" : "unknown",
		source:
			part === "decal" ?
				concat(["scene.deferredGBufferCodec", id], "<webgpu-decal-part>")
			: asset(id),
	};
}
for (const [part, path] of Object.entries(WEBGPU_DIRECTIVE_SHADER_FILES)) {
	const id = `directive.${part}`;
	assets[id] = { path };
	sources[`webgpu.directive.${part}`] = {
		kind: "module",
		sourceKind: "unknown",
		source: asset(id),
	};
}
for (const [part, path] of Object.entries(WEBGPU_SHADER_MATERIAL_FILES)) {
	const id = `material.${part}`;
	assets[id] = { path, sync: true };
	sources[`webgpu.material.${part}`] = {
		kind: "module",
		sourceKind: "custom-material",
		source: asset(id),
	};
}
for (const [part, descriptor] of Object.entries(WEBGPU_FIXED_SHADER_FILES)) {
	const id = `fixed.${part}`;
	assets[id] = { path: descriptor.path };
}

sources["webgpu.scene"] = {
	kind: "module",
	sourceKind: "builtin-scene",
	source: concat(
		WEBGPU_SCENE_SHADER_PARTS.map((part) => `scene.${part}`),
		"<webgpu-scene-part>",
	),
};
sources["webgpu.environment"] = {
	kind: "module",
	sourceKind: "builtin-environment",
	source: concat(["scene.lightData", "fixed.environment"], "<webgpu-shared-light-data-part>"),
};
sources["webgpu.deferredLighting"] = {
	kind: "module",
	sourceKind: "builtin-scene",
	source: concat(
		[
			"scene.lightData",
			"scene.constants",
			"scene.definitions",
			"scene.utils",
			"scene.deferredGBufferCodec",
			"scene.surfaceLighting",
			"fixed.deferredLighting",
		],
		"<webgpu-deferred-lighting-part>",
	),
};
sources["webgpu.particle"] = {
	kind: "module",
	sourceKind: "particle",
	source: concat(["scene.lightData", "fixed.particle"], "<webgpu-shared-light-data-part>"),
};
sources["webgpu.particleSimulation"] = {
	kind: "module",
	sourceKind: "particle",
	source: asset("fixed.particleSimulation"),
};
sources["webgpu.clusteredLightingCull"] = {
	kind: "module",
	sourceKind: "clustered",
	source: concat(
		["scene.lightData", "fixed.clusteredLightingCull"],
		"<webgpu-shared-light-data-part>",
	),
};
sources["webgpu.iblPrefilter"] = {
	kind: "module",
	sourceKind: "builtin-environment",
	source: asset("fixed.iblPrefilter"),
};

export const WEBGPU_SHADER_MANIFEST: ShaderBackendManifest = {
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
			{ id: "ignis/webgpu/constants-base.wgsl", source: "webgpu.directive.constants" },
			{ id: "ignis/color/srgb.wgsl", source: "webgpu.directive.srgb" },
			{ id: "ignis/postprocess/fog.wgsl", source: "webgpu.directive.fog" },
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
};
