import { Platform } from "../../foundation/Platform";
import { ShaderLoader } from "../../loaders/ShaderLoader";
import type { CompositeShaderSource } from "../runtime";

export type WebGLShaderPart =
	| "sceneVertex"
	| "sceneFragment"
	| "skyboxVertex"
	| "skyboxFragment"
	| "presentVertex"
	| "presentFragment"
	| "particleVertex"
	| "particleFragment"
	| "shadowDepthVertex"
	| "shadowDepthFragment"
	| "copyFragment"
	| "postProcessStubFragment"
	| "fxaaFragment"
	| "bloomFragment"
	| "interactionOutlineFragment"
	| "motionBlurFragment"
	| "dofFragment"
	| "taaFragment"
	| "ssaoRawFragment"
	| "ssaoBlurFragment"
	| "ssaoCombineFragment";

export const WEBGL_SHADER_PARTS: readonly WebGLShaderPart[] = [
	"sceneVertex",
	"sceneFragment",
	"skyboxVertex",
	"skyboxFragment",
	"presentVertex",
	"presentFragment",
	"particleVertex",
	"particleFragment",
	"shadowDepthVertex",
	"shadowDepthFragment",
	"copyFragment",
	"postProcessStubFragment",
	"fxaaFragment",
	"bloomFragment",
	"interactionOutlineFragment",
	"motionBlurFragment",
	"dofFragment",
	"taaFragment",
	"ssaoRawFragment",
	"ssaoBlurFragment",
	"ssaoCombineFragment",
];

type ImportMetaGlobLoaderMap = Record<string, () => Promise<string>>;

const webglParts: ImportMetaGlobLoaderMap = Platform.isNodeRuntime()
	? {}
	: import.meta.glob<string>("./parts/*.glsl", {
			query: "?raw",
			import: "default",
		});

const _shaderLoader = new ShaderLoader();
const _compositeCache = new Map<string, Promise<CompositeShaderSource>>();

function loadShaderCompositeFromFile(
	key: string,
	nodeRelativePath: string,
	browserLoader: () => Promise<string>
): Promise<CompositeShaderSource> {
	return _shaderLoader.loadComposite({
		key,
		nodeRelativePath,
		nodeBaseUrl: import.meta.url,
		browserLoader,
	});
}

function loadShader(
	key: string,
	nodeRelativePath: string,
	browserLoader: () => Promise<string>
): Promise<string> {
	return _shaderLoader.loadSource({
		key,
		nodeRelativePath,
		nodeBaseUrl: import.meta.url,
		browserLoader,
	});
}

const shaderFiles: Record<WebGLShaderPart, string> = {
	sceneVertex: "./parts/sceneVertex.glsl",
	sceneFragment: "./parts/sceneFragment.glsl",
	skyboxVertex: "./parts/skyboxVertex.glsl",
	skyboxFragment: "./parts/skyboxFragment.glsl",
	presentVertex: "./parts/presentVertex.glsl",
	presentFragment: "./parts/presentFragment.glsl",
	particleVertex: "./parts/particleVertex.glsl",
	particleFragment: "./parts/particleFragment.glsl",
	shadowDepthVertex: "./parts/shadowDepthVertex.glsl",
	shadowDepthFragment: "./parts/shadowDepthFragment.glsl",
	copyFragment: "./parts/copyFragment.glsl",
	postProcessStubFragment: "./parts/postProcessStubFragment.glsl",
	fxaaFragment: "./parts/fxaaFragment.glsl",
	bloomFragment: "./parts/bloomFragment.glsl",
	interactionOutlineFragment: "./parts/interactionOutlineFragment.glsl",
	motionBlurFragment: "./parts/motionBlurFragment.glsl",
	dofFragment: "./parts/dofFragment.glsl",
	taaFragment: "./parts/taaFragment.glsl",
	ssaoRawFragment: "./parts/ssaoRawFragment.glsl",
	ssaoBlurFragment: "./parts/ssaoBlurFragment.glsl",
	ssaoCombineFragment: "./parts/ssaoCombineFragment.glsl",
};

export function loadWebGLShaderPart(part: WebGLShaderPart): Promise<string> {
	return loadShader(`webgl:${part}`, shaderFiles[part], () => {
		const loader = webglParts[`./parts/${part}.glsl`];
		if (!loader) {
			return Promise.reject(
				new Error(`WebGL shader part not found: ${part}`)
			);
		}
		return loader();
	});
}

export function loadWebGLShaderPartComposite(
	part: WebGLShaderPart
): Promise<CompositeShaderSource> {
	const key = `webgl-composite:${part}`;
	let cached = _compositeCache.get(key);
	if (!cached) {
		cached = loadShaderCompositeFromFile(`webgl:${part}`, shaderFiles[part], () => {
			const loader = webglParts[`./parts/${part}.glsl`];
			if (!loader) {
				return Promise.reject(
					new Error(`WebGL shader part not found: ${part}`)
				);
			}
			return loader();
		});
		_compositeCache.set(key, cached);
	}
	return cached;
}
