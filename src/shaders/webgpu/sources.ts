import { Platform } from "../../foundation/Platform";
import type { ShaderSourceSegmentKind } from "../runtime";

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
	| "depthDirtyClear"
	| "decal"
	| "oitResolve"
	| "occlusionCulling"
	| "mipmapBlit";

export type WebGPUShaderSourceSyncKey = "webgpu.utility.mipmapBlit.raw";

export const WEBGPU_SCENE_SHADER_PARTS: readonly WebGPUSceneShaderPart[] = [
	"lightData",
	"constants",
	"definitions",
	"utils",
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

export const WEBGPU_POST_PROCESS_PARTS_USING_SHARED_LIGHT_DATA =
	new Set<WebGPUPostProcessShaderPart>(["ssr", "volumetric"]);

export const WEBGPU_SCENE_SHADER_FILES: Record<WebGPUSceneShaderPart, string> = {
	lightData: "./webgpu/common/lightData.wgsl",
	constants: "./webgpu/common/constants.wgsl",
	definitions: "./webgpu/common/definitions.wgsl",
	utils: "./webgpu/common/utils.wgsl",
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

export const WEBGPU_POST_PROCESS_SHADER_FILES: Record<
	WebGPUPostProcessShaderPart,
	string
> = {
	ssao: "./webgpu/postprocess/ssao.wgsl",
	ssgi: "./webgpu/postprocess/ssgi.wgsl",
	taa: "./webgpu/postprocess/taa.wgsl",
	hiz: "./webgpu/postprocess/hiz.wgsl",
	ssr: "./webgpu/postprocess/ssr.wgsl",
	screenSpaceRefractions: "./webgpu/postprocess/screenSpaceRefractions.wgsl",
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

export const WEBGPU_SHADOW_SHADER_FILES: Record<WebGPUShadowShaderPart, string> = {
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

export const WEBGPU_UTILITY_SHADER_FILES: Record<WebGPUUtilityShaderPart, string> = {
	planarReflectionComposite: "./webgpu/utility/planarReflectionComposite.wgsl",
	present: "./webgpu/utility/present.wgsl",
	depthDirtyClear: "./webgpu/utility/depthDirtyClear.wgsl",
	decal: "./webgpu/scene/decal.wgsl",
	oitResolve: "./webgpu/utility/oitResolve.wgsl",
	occlusionCulling: "./webgpu/utility/occlusionCulling.wgsl",
	mipmapBlit: "./webgpu/utility/mipmapBlit.wgsl",
};

export const WEBGPU_FIXED_SHADER_FILES = {
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

export const WEBGPU_SYNC_SHADER_FILES: Record<
	WebGPUShaderSourceSyncKey,
	WebGPUShaderFileDescriptor
> = {
	"webgpu.utility.mipmapBlit.raw": {
		scope: "webgpu",
		key: "webgpu.utility.mipmapBlit",
		path: "./webgpu/utility/mipmapBlit.wgsl",
	},
};

export function createWebGPUBrowserShaderSources(): ImportMetaGlobLoaderMap {
	if (Platform.isNodeRuntime()) {
		return {};
	}
	try {
		return prefixWebGPUShaderPaths(
			import.meta.glob<string>(["./**/*.wgsl", "!./utility/mipmapBlit.wgsl"], {
				query: "?raw",
				import: "default",
			})
		);
	} catch {
		return {};
	}
}

export function createWebGPUBrowserSyncShaderSources(): Record<string, string> {
	if (Platform.isNodeRuntime()) {
		return {};
	}
	try {
		return prefixWebGPUShaderPaths(
			import.meta.glob<string>("./utility/mipmapBlit.wgsl", {
				query: "?raw",
				import: "default",
				eager: true,
			})
		);
	} catch {
		return {};
	}
}

function prefixWebGPUShaderPaths<T>(sources: Record<string, T>): Record<string, T> {
	const prefixed: Record<string, T> = {};
	for (const [path, source] of Object.entries(sources)) {
		prefixed[`./webgpu/${path.slice("./".length)}`] = source;
	}
	return prefixed;
}
