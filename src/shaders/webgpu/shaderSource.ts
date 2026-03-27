import { Platform } from "../../foundation/Platform";
import {
	createInlineCompositeShaderSource,
	type CompositeShaderSource,
} from "../runtime";

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
	| "taa"
	| "hiz"
	| "ssr"
	| "volumetric"
	| "motionBlur"
	| "dof"
	| "bloom"
	| "fxaa"
	| "copy";

type RawShaderModule = {
	default: string;
};

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

	return loadShader(
		`scene:${part}`,
		path,
		() => import(`./parts/${part}.wgsl?raw`)
	);
}

export function loadSceneShaderPartComposite(
	part: SceneShaderPart
): Promise<CompositeShaderSource> {
	const key = `scene-composite:${part}`;
	let cached = _compositeCache.get(key);
	if (!cached) {
		cached = loadSceneShaderPart(part).then((code) =>
			createInlineCompositeShaderSource(code, sceneShaderFiles[part], "template")
		);
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
		cached = loadSkyboxShaderSource().then((code) =>
			createInlineCompositeShaderSource(
				code,
				"./skyboxShader.wgsl",
				"source"
			)
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
		cached = loadParticleShaderSource().then((code) =>
			createInlineCompositeShaderSource(
				code,
				"./particleShader.wgsl",
				"source"
			)
		);
		_compositeCache.set(key, cached);
	}
	return cached;
}

const postProcessShaderFiles: Record<PostProcessShaderPart, string> = {
	ssao: "./postprocess/ssao.wgsl",
	taa: "./postprocess/taa.wgsl",
	hiz: "./postprocess/hiz.wgsl",
	ssr: "./postprocess/ssr.wgsl",
	volumetric: "./postprocess/volumetric.wgsl",
	motionBlur: "./postprocess/motionBlur.wgsl",
	dof: "./postprocess/dof.wgsl",
	bloom: "./postprocess/bloom.wgsl",
	fxaa: "./postprocess/fxaa.wgsl",
	copy: "./postprocess/copy.wgsl",
};

export function loadPostProcessShaderPart(
	part: PostProcessShaderPart
): Promise<string> {
	const path = postProcessShaderFiles[part];

	return loadShader(
		`post:${part}`,
		path,
		() => import(`./postprocess/${part}.wgsl?raw`)
	);
}

export function loadPostProcessShaderPartComposite(
	part: PostProcessShaderPart
): Promise<CompositeShaderSource> {
	const key = `post-composite:${part}`;
	let cached = _compositeCache.get(key);
	if (!cached) {
		cached = loadPostProcessShaderPart(part).then((code) =>
			createInlineCompositeShaderSource(
				code,
				postProcessShaderFiles[part],
				"template"
			)
		);
		_compositeCache.set(key, cached);
	}
	return cached;
}
