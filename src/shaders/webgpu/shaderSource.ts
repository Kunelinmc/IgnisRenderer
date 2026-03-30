import { Platform } from "../../foundation/Platform";
import { type CompositeShaderSource } from "../runtime";
import { preprocessEngineShaderDirectives } from "../runtime/engineDirectives";

type SceneShaderPart =
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
	| "fragmentPbrAmbient"
	| "fragmentSingleTarget";

type PostProcessShaderPart =
	| "ssao"
	| "ssgi"
	| "taa"
	| "hiz"
	| "ssr"
	| "volumetric"
	| "motionBlur"
	| "dof"
	| "bloomDownsample"
	| "bloomBlurH"
	| "bloomBlurV"
	| "bloomUpsample"
	| "bloomComposite"
	| "interactionOutline"
	| "fxaa"
	| "copy"
	| "sobelNormal";

type RawShaderModule = {
	default: string;
};

type ImportMetaGlobLoaderMap = Record<string, () => Promise<string>>;

const sceneParts: ImportMetaGlobLoaderMap = Platform.isNodeRuntime()
	? {}
	: import.meta.glob<string>("./parts/*.wgsl", {
			query: "?raw",
			import: "default",
		});

const postProcessParts: ImportMetaGlobLoaderMap = Platform.isNodeRuntime()
	? {}
	: import.meta.glob<string>("./postprocess/*.wgsl", {
			query: "?raw",
			import: "default",
		});

const _cache = new Map<string, Promise<string>>();
const _preprocessedCache = new Map<string, Promise<CompositeShaderSource>>();
const _compositeCache = new Map<string, Promise<CompositeShaderSource>>();

async function loadShaderCompositeFromFile(
	key: string,
	nodeRelativePath: string,
	browserLoader: () => Promise<RawShaderModule>
): Promise<CompositeShaderSource> {
	let cached = _preprocessedCache.get(key);
	if (!cached) {
		cached = (async () => {
			let code: string;
			if (Platform.isNodeRuntime()) {
				const fsSpecifier = ["node", "fs/promises"].join(":");
				const fsModule = (await import(/* @vite-ignore */ fsSpecifier)) as {
					readFile: (
						path: string | URL,
						options?: string | { encoding?: string }
					) => Promise<string>;
				};
				code = await fsModule.readFile(
					new URL(nodeRelativePath, import.meta.url),
					"utf8"
				);
			} else {
				const module = await browserLoader();
				code = module.default;
			}
			return preprocessEngineShaderDirectives({
				code,
				language: "wgsl",
				sourcePath: nodeRelativePath,
			});
		})();
		_preprocessedCache.set(key, cached);
	}
	return cached;
}

async function loadShader(
	key: string,
	nodeRelativePath: string,
	browserLoader: () => Promise<RawShaderModule>
): Promise<string> {
	let cached = _cache.get(key);
	if (!cached) {
		cached = loadShaderCompositeFromFile(
			key,
			nodeRelativePath,
			browserLoader
		).then((composite) => composite.code);
		_cache.set(key, cached);
	}
	return cached;
}

const sceneShaderFiles: Record<SceneShaderPart, string> = {
	constants: "./parts/constants.wgsl",
	definitions: "./parts/definitions.wgsl",
	utils: "./parts/utils.wgsl",
	vertexStage: "./parts/vertexStage.wgsl",
	fragmentPrelude: "./parts/fragmentPrelude.wgsl",
	fragmentPhong: "./parts/fragmentPhong.wgsl",
	fragmentPbrSetup: "./parts/fragmentPbrSetup.wgsl",
	fragmentPbrDirectional: "./parts/fragmentPbrDirectional.wgsl",
	fragmentPbrPoint: "./parts/fragmentPbrPoint.wgsl",
	fragmentPbrSpot: "./parts/fragmentPbrSpot.wgsl",
	fragmentPbrAmbient: "./parts/fragmentPbrAmbient.wgsl",
	fragmentSingleTarget: "./parts/fragmentSingleTarget.wgsl",
};

export function loadSceneShaderPart(part: SceneShaderPart): Promise<string> {
	const path = sceneShaderFiles[part];

	return loadShader(`scene:${part}`, path, () => {
		const loader = sceneParts[`./parts/${part}.wgsl`];
		if (!loader) {
			return Promise.reject(
				new Error(`Scene shader part not found: ${part}`)
			);
		}
		return loader().then((content) => ({ default: content }));
	});
}

export function loadSceneShaderPartComposite(
	part: SceneShaderPart
): Promise<CompositeShaderSource> {
	const key = `scene-composite:${part}`;
	let cached = _compositeCache.get(key);
	if (!cached) {
		cached = loadShaderCompositeFromFile(`scene:${part}`, sceneShaderFiles[part], () => {
			const loader = sceneParts[`./parts/${part}.wgsl`];
			if (!loader) {
				return Promise.reject(
					new Error(`Scene shader part not found: ${part}`)
				);
			}
			return loader().then((content) => ({ default: content }));
		});
		_compositeCache.set(key, cached);
	}
	return cached;
}

export function loadSkyboxShaderSource(): Promise<string> {
	return loadShader(
		"skybox",
		"./skyboxShader.wgsl",
		() => import("./skyboxShader.wgsl?raw")
	);
}

export function loadSkyboxShaderSourceComposite(): Promise<CompositeShaderSource> {
	const key = "skybox-composite";
	let cached = _compositeCache.get(key);
	if (!cached) {
		cached = loadShaderCompositeFromFile("skybox", "./skyboxShader.wgsl", () =>
			import("./skyboxShader.wgsl?raw")
		);
		_compositeCache.set(key, cached);
	}
	return cached;
}

export function loadParticleShaderSource(): Promise<string> {
	return loadShader(
		"particle",
		"./particleShader.wgsl",
		() => import("./particleShader.wgsl?raw")
	);
}

export function loadParticleShaderSourceComposite(): Promise<CompositeShaderSource> {
	const key = "particle-composite";
	let cached = _compositeCache.get(key);
	if (!cached) {
		cached = loadShaderCompositeFromFile("particle", "./particleShader.wgsl", () =>
			import("./particleShader.wgsl?raw")
		);
		_compositeCache.set(key, cached);
	}
	return cached;
}

const postProcessShaderFiles: Record<PostProcessShaderPart, string> = {
	ssao: "./postprocess/ssao.wgsl",
	ssgi: "./postprocess/ssgi.wgsl",
	taa: "./postprocess/taa.wgsl",
	hiz: "./postprocess/hiz.wgsl",
	ssr: "./postprocess/ssr.wgsl",
	volumetric: "./postprocess/volumetric.wgsl",
	motionBlur: "./postprocess/motionBlur.wgsl",
	dof: "./postprocess/dof.wgsl",
	bloomDownsample: "./postprocess/bloomDownsample.wgsl",
	bloomBlurH: "./postprocess/bloomBlurH.wgsl",
	bloomBlurV: "./postprocess/bloomBlurV.wgsl",
	bloomUpsample: "./postprocess/bloomUpsample.wgsl",
	bloomComposite: "./postprocess/bloomComposite.wgsl",
	interactionOutline: "./postprocess/interactionOutline.wgsl",
	fxaa: "./postprocess/fxaa.wgsl",
	copy: "./postprocess/copy.wgsl",
	sobelNormal: "./postprocess/sobelNormal.wgsl",
};

export function loadPostProcessShaderPart(
	part: PostProcessShaderPart
): Promise<string> {
	const path = postProcessShaderFiles[part];

	return loadShader(`post:${part}`, path, () => {
		const loader = postProcessParts[`./postprocess/${part}.wgsl`];
		if (!loader) {
			return Promise.reject(
				new Error(`Post-process shader part not found: ${part}`)
			);
		}
		return loader().then((content) => ({ default: content }));
	});
}

export function loadPostProcessShaderPartComposite(
	part: PostProcessShaderPart
): Promise<CompositeShaderSource> {
	const key = `post-composite:${part}`;
	let cached = _compositeCache.get(key);
	if (!cached) {
		cached = loadShaderCompositeFromFile(`post:${part}`, postProcessShaderFiles[part], () => {
			const loader = postProcessParts[`./postprocess/${part}.wgsl`];
			if (!loader) {
				return Promise.reject(
					new Error(`Post-process shader part not found: ${part}`)
				);
			}
			return loader().then((content) => ({ default: content }));
		});
		_compositeCache.set(key, cached);
	}
	return cached;
}
