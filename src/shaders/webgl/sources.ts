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

export const WEBGL_PIPELINE_SHADER_PARTS: readonly WebGLShaderPart[] =
	WEBGL_SHADER_PARTS.filter(
		(part) => part !== "sceneVertex" && part !== "sceneFragment"
	);

export const WEBGL_SHADER_FILES: Record<WebGLShaderPart, string> = {
	sceneVertex: "./webgl/parts/sceneVertex.glsl",
	sceneFragment: "./webgl/parts/sceneFragment.glsl",
	sceneDepthPrepassVertex: "./webgl/parts/sceneDepthPrepassVertex.glsl",
	sceneDepthPrepassFragment: "./webgl/parts/sceneDepthPrepassFragment.glsl",
	environmentVertex: "./webgl/parts/environmentVertex.glsl",
	environmentFragment: "./webgl/parts/environmentFragment.glsl",
	presentVertex: "./webgl/parts/presentVertex.glsl",
	presentFragment: "./webgl/parts/presentFragment.glsl",
	particleVertex: "./webgl/parts/particleVertex.glsl",
	particleFragment: "./webgl/parts/particleFragment.glsl",
	shadowDepthVertex: "./webgl/parts/shadowDepthVertex.glsl",
	shadowDepthFragment: "./webgl/parts/shadowDepthFragment.glsl",
	shadowTransmittanceFragment: "./webgl/parts/shadowTransmittanceFragment.glsl",
	copyFragment: "./webgl/parts/copyFragment.glsl",
	oitResolveFragment: "./webgl/parts/oitResolveFragment.glsl",
	postProcessStubFragment: "./webgl/parts/postProcessStubFragment.glsl",
	toneMappingFragment: "./webgl/parts/toneMappingFragment.glsl",
	colorFilterFragment: "./webgl/parts/colorFilterFragment.glsl",
	fxaaFragment: "./webgl/parts/fxaaFragment.glsl",
	bloomFragment: "./webgl/parts/bloomFragment.glsl",
	motionBlurFragment: "./webgl/parts/motionBlurFragment.glsl",
	fogFragment: "./webgl/parts/fogFragment.glsl",
	dofFragment: "./webgl/parts/dofFragment.glsl",
	taaFragment: "./webgl/parts/taaFragment.glsl",
	ssaoRawFragment: "./webgl/parts/ssaoRawFragment.glsl",
	ssaoBlurFragment: "./webgl/parts/ssaoBlurFragment.glsl",
	ssaoCombineFragment: "./webgl/parts/ssaoCombineFragment.glsl",
};

export const WEBGL_INTERNAL_SHADER_FILES = {
	diffuseProbeFallbackFragment: "./webgl/parts/diffuseProbeFallbackFragment.glsl",
	irradianceProbeGridFragment: "./webgl/parts/irradianceProbeGridFragment.glsl",
} as const;

export function createWebGLBrowserShaderSources(): ImportMetaGlobLoaderMap {
	if (Platform.isNodeRuntime()) {
		return {};
	}
	try {
		return prefixWebGLShaderPaths(
			import.meta.glob<string>("./parts/*.glsl", {
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
