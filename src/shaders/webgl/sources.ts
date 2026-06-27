import { Platform } from "../../foundation/Platform";

type ImportMetaGlobLoaderMap = Record<string, () => Promise<string>>;

export type WebGLShaderPart =
	| "sceneVertex"
	| "sceneFragment"
	| "sceneDepthPrepassVertex"
	| "sceneDepthPrepassFragment"
	| "environmentVertex"
	| "environmentFragment"
	| "presentVertex"
	| "presentFragment"
	| "particleVertex"
	| "particleFragment"
	| "shadowDepthVertex"
	| "shadowDepthFragment"
	| "shadowTransmittanceFragment"
	| "copyFragment"
	| "oitResolveFragment"
	| "postProcessStubFragment"
	| "toneMappingFragment"
	| "colorFilterFragment"
	| "fxaaFragment"
	| "bloomFragment"
	| "motionBlurFragment"
	| "fogFragment"
	| "dofFragment"
	| "taaFragment"
	| "ssaoRawFragment"
	| "ssaoBlurFragment"
	| "ssaoCombineFragment";

export type WebGLSceneFragmentPart =
	| "fragmentPrelude"
	| "fragmentUniforms"
	| "fragmentUvTextureNormal"
	| "fragmentSh"
	| "fragmentLocalProbes"
	| "fragmentClusteredLighting"
	| "fragmentReflectionEnvironment"
	| "fragmentEnvironmentSpecular"
	| "fragmentReflectionProbes"
	| "fragmentLightAttenuation"
	| "fragmentShadows"
	| "fragmentBrdfPbr"
	| "fragmentPhong"
	| "fragmentPbrLighting"
	| "fragmentMainOutput";

export const WEBGL_SHADER_PARTS: readonly WebGLShaderPart[] = [
	"sceneVertex",
	"sceneFragment",
	"sceneDepthPrepassVertex",
	"sceneDepthPrepassFragment",
	"environmentVertex",
	"environmentFragment",
	"presentVertex",
	"presentFragment",
	"particleVertex",
	"particleFragment",
	"shadowDepthVertex",
	"shadowDepthFragment",
	"shadowTransmittanceFragment",
	"copyFragment",
	"oitResolveFragment",
	"postProcessStubFragment",
	"toneMappingFragment",
	"colorFilterFragment",
	"fxaaFragment",
	"bloomFragment",
	"motionBlurFragment",
	"fogFragment",
	"dofFragment",
	"taaFragment",
	"ssaoRawFragment",
	"ssaoBlurFragment",
	"ssaoCombineFragment",
];

export const WEBGL_SCENE_FRAGMENT_PARTS: readonly WebGLSceneFragmentPart[] = [
	"fragmentPrelude",
	"fragmentUniforms",
	"fragmentUvTextureNormal",
	"fragmentSh",
	"fragmentLocalProbes",
	"fragmentClusteredLighting",
	"fragmentReflectionEnvironment",
	"fragmentEnvironmentSpecular",
	"fragmentReflectionProbes",
	"fragmentLightAttenuation",
	"fragmentShadows",
	"fragmentBrdfPbr",
	"fragmentPhong",
	"fragmentPbrLighting",
	"fragmentMainOutput",
];

export const WEBGL_PIPELINE_SHADER_PARTS: readonly WebGLShaderPart[] =
	WEBGL_SHADER_PARTS.filter(
		(part) => part !== "sceneVertex" && part !== "sceneFragment"
	);

export const WEBGL_SHADER_FILES: Record<WebGLShaderPart, string> = {
	sceneVertex: "./webgl/scene/sceneVertex.glsl",
	sceneFragment: "./webgl/scene/sceneFragment.glsl",
	sceneDepthPrepassVertex: "./webgl/scene/sceneDepthPrepassVertex.glsl",
	sceneDepthPrepassFragment: "./webgl/scene/sceneDepthPrepassFragment.glsl",
	environmentVertex: "./webgl/environment/environmentVertex.glsl",
	environmentFragment: "./webgl/environment/environmentFragment.glsl",
	presentVertex: "./webgl/utility/presentVertex.glsl",
	presentFragment: "./webgl/utility/presentFragment.glsl",
	particleVertex: "./webgl/particles/particleVertex.glsl",
	particleFragment: "./webgl/particles/particleFragment.glsl",
	shadowDepthVertex: "./webgl/shadow/shadowDepthVertex.glsl",
	shadowDepthFragment: "./webgl/shadow/shadowDepthFragment.glsl",
	shadowTransmittanceFragment: "./webgl/shadow/shadowTransmittanceFragment.glsl",
	copyFragment: "./webgl/postprocess/copyFragment.glsl",
	oitResolveFragment: "./webgl/utility/oitResolveFragment.glsl",
	postProcessStubFragment: "./webgl/postprocess/postProcessStubFragment.glsl",
	toneMappingFragment: "./webgl/postprocess/toneMappingFragment.glsl",
	colorFilterFragment: "./webgl/postprocess/colorFilterFragment.glsl",
	fxaaFragment: "./webgl/postprocess/fxaaFragment.glsl",
	bloomFragment: "./webgl/postprocess/bloomFragment.glsl",
	motionBlurFragment: "./webgl/postprocess/motionBlurFragment.glsl",
	fogFragment: "./webgl/postprocess/fogFragment.glsl",
	dofFragment: "./webgl/postprocess/dofFragment.glsl",
	taaFragment: "./webgl/postprocess/taaFragment.glsl",
	ssaoRawFragment: "./webgl/postprocess/ssaoRawFragment.glsl",
	ssaoBlurFragment: "./webgl/postprocess/ssaoBlurFragment.glsl",
	ssaoCombineFragment: "./webgl/postprocess/ssaoCombineFragment.glsl",
};

export const WEBGL_SCENE_FRAGMENT_SHADER_FILES: Record<
	WebGLSceneFragmentPart,
	string
> = {
	fragmentPrelude: "./webgl/scene/fragmentPrelude.glsl",
	fragmentUniforms: "./webgl/scene/fragmentUniforms.glsl",
	fragmentUvTextureNormal: "./webgl/scene/fragmentUvTextureNormal.glsl",
	fragmentSh: "./webgl/scene/fragmentSh.glsl",
	fragmentLocalProbes: "./webgl/scene/fragmentLocalProbes.glsl",
	fragmentClusteredLighting:
		"./webgl/scene/fragmentClusteredLighting.glsl",
	fragmentReflectionEnvironment:
		"./webgl/scene/fragmentReflectionEnvironment.glsl",
	fragmentEnvironmentSpecular:
		"./webgl/scene/fragmentEnvironmentSpecular.glsl",
	fragmentReflectionProbes: "./webgl/scene/fragmentReflectionProbes.glsl",
	fragmentLightAttenuation: "./webgl/scene/fragmentLightAttenuation.glsl",
	fragmentShadows: "./webgl/scene/fragmentShadows.glsl",
	fragmentBrdfPbr: "./webgl/scene/fragmentBrdfPbr.glsl",
	fragmentPhong: "./webgl/scene/fragmentPhong.glsl",
	fragmentPbrLighting: "./webgl/scene/fragmentPbrLighting.glsl",
	fragmentMainOutput: "./webgl/scene/fragmentMainOutput.glsl",
};

export const WEBGL_INTERNAL_SHADER_FILES = {
	diffuseProbeFallbackFragment: "./webgl/environment/diffuseProbeFallbackFragment.glsl",
	irradianceProbeGridFragment: "./webgl/environment/irradianceProbeGridFragment.glsl",
} as const;

export function createWebGLBrowserShaderSources(): ImportMetaGlobLoaderMap {
	if (Platform.isNodeRuntime()) {
		return {};
	}
	try {
		return prefixWebGLShaderPaths(
			import.meta.glob<string>("./**/*.glsl", {
				query: "?raw",
				import: "default",
			})
		);
	} catch {
		return {};
	}
}

function prefixWebGLShaderPaths<T>(sources: Record<string, T>): Record<string, T> {
	const prefixed: Record<string, T> = {};
	for (const [path, source] of Object.entries(sources)) {
		prefixed[`./webgl/${path.slice("./".length)}`] = source;
	}
	return prefixed;
}
