import { Platform } from "../../foundation/Platform";
import {
	createInlineCompositeShaderSource,
	type CompositeShaderSource,
} from "../runtime";

type WebGLShaderPart =
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
	| "motionBlurFragment"
	| "dofFragment"
	| "taaFragment"
	| "ssaoRawFragment"
	| "ssaoBlurFragment"
	| "ssaoCombineFragment";

type RawShaderModule = {
	default: string;
};

type ImportMetaGlobLoaderMap = Record<string, () => Promise<string>>;

const webglParts: ImportMetaGlobLoaderMap = Platform.isNodeRuntime()
	? {}
	: import.meta.glob<string>("./parts/*.glsl", {
			query: "?raw",
			import: "default",
		});

const _cache = new Map<string, Promise<string>>();
const _compositeCache = new Map<string, Promise<CompositeShaderSource>>();

async function loadShader(
	key: string,
	nodeRelativePath: string,
	browserLoader: () => Promise<RawShaderModule>
): Promise<string> {
	let cached = _cache.get(key);
	if (!cached) {
		cached = (async () => {
			if (Platform.isNodeRuntime()) {
				const fsSpecifier = ["node", "fs/promises"].join(":");
				const fsModule = (await import(/* @vite-ignore */ fsSpecifier)) as {
					readFile: (
						path: string | URL,
						options?: string | { encoding?: string }
					) => Promise<string>;
				};
				return fsModule.readFile(
					new URL(nodeRelativePath, import.meta.url),
					"utf8"
				);
			}

			const module = await browserLoader();
			return module.default;
		})();
		_cache.set(key, cached);
	}
	return cached;
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
		return loader().then((content) => ({ default: content }));
	});
}

export function loadWebGLShaderPartComposite(
	part: WebGLShaderPart
): Promise<CompositeShaderSource> {
	const key = `webgl-composite:${part}`;
	let cached = _compositeCache.get(key);
	if (!cached) {
		cached = loadWebGLShaderPart(part).then((code) =>
			createInlineCompositeShaderSource(code, shaderFiles[part], "template")
		);
		_compositeCache.set(key, cached);
	}
	return cached;
}
