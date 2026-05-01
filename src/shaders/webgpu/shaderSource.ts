import { Platform } from "../../foundation/Platform";
import { ShaderLoader, fromRawShaderModuleLoader } from "../../loaders/ShaderLoader";
import {
	composeCompositeShaderSources,
	type CompositeShaderSource,
} from "../runtime";

type SceneShaderPart =
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
	| "fragmentPbrAmbient"
	| "fragmentSingleTarget";

type PostProcessShaderPart =
	| "ssao"
	| "ssgi"
	| "taa"
	| "hiz"
	| "ssr"
	| "volumetric"
	| "fog"
	| "motionBlur"
	| "dof"
	| "bloomDownsample"
	| "bloomBlurH"
	| "bloomBlurV"
	| "bloomUpsample"
	| "bloomComposite"
	| "toneMapping"
	| "colorFilter"
	| "interactionOutline"
	| "fxaa"
	| "copy"
	| "sobelNormal";

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

const sceneShaderFiles: Record<SceneShaderPart, string> = {
	lightData: "./parts/lightData.wgsl",
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

const POST_PROCESS_PARTS_USING_SHARED_LIGHT_DATA = new Set<
	PostProcessShaderPart
>(["ssr", "volumetric"]);

function loadSharedLightDataComposite(): Promise<CompositeShaderSource> {
	const key = "webgpu-shared-light-data-composite";
	let cached = _compositeCache.get(key);
	if (!cached) {
		cached = loadSceneShaderPartComposite("lightData");
		_compositeCache.set(key, cached);
	}
	return cached;
}

function composeWithSharedLightData(
	key: string,
	shaderPromise: Promise<CompositeShaderSource>
): Promise<CompositeShaderSource> {
	let cached = _compositeCache.get(key);
	if (!cached) {
		cached = Promise.all([
			loadSharedLightDataComposite(),
			shaderPromise,
		]).then(([lightData, shader]) =>
			composeCompositeShaderSources(
				[
					{
						code: lightData.code,
						sourceMap: lightData.sourceMap,
						sourcePath:
							lightData.sourceMap.segments[0]?.sourcePath ??
							"<webgpu-light-data>",
						kind: "template",
					},
					{
						code: shader.code,
						sourceMap: shader.sourceMap,
						sourcePath:
							shader.sourceMap.segments[0]?.sourcePath ??
							"<webgpu-shader-part>",
						kind: "template",
					},
				],
				"\n\n",
				"template"
			)
		);
		_compositeCache.set(key, cached);
	}
	return cached;
}

export function loadSceneShaderPart(part: SceneShaderPart): Promise<string> {
	const path = sceneShaderFiles[part];

	return loadShader(`scene:${part}`, path, () => {
		const loader = sceneParts[`./parts/${part}.wgsl`];
		if (!loader) {
			return Promise.reject(
				new Error(`Scene shader part not found: ${part}`)
			);
		}
		return loader();
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
			return loader();
		});
		_compositeCache.set(key, cached);
	}
	return cached;
}

export function loadSkyboxShaderSource(): Promise<string> {
	return loadSkyboxShaderSourceComposite().then((composite) => composite.code);
}

export function loadSkyboxShaderSourceComposite(): Promise<CompositeShaderSource> {
	return composeWithSharedLightData(
		"skybox-composite",
		loadShaderCompositeFromFile(
			"skybox",
			"./skyboxShader.wgsl",
			fromRawShaderModuleLoader(() => import("./skyboxShader.wgsl?raw"))
		)
	);
}

export function loadParticleShaderSource(): Promise<string> {
	return loadParticleShaderSourceComposite().then((composite) => composite.code);
}

export function loadParticleShaderSourceComposite(): Promise<CompositeShaderSource> {
	return composeWithSharedLightData(
		"particle-composite",
		loadShaderCompositeFromFile(
			"particle",
			"./particleShader.wgsl",
			fromRawShaderModuleLoader(() => import("./particleShader.wgsl?raw"))
		)
	);
}

const postProcessShaderFiles: Record<PostProcessShaderPart, string> = {
	ssao: "./postprocess/ssao.wgsl",
	ssgi: "./postprocess/ssgi.wgsl",
	taa: "./postprocess/taa.wgsl",
	hiz: "./postprocess/hiz.wgsl",
	ssr: "./postprocess/ssr.wgsl",
	volumetric: "./postprocess/volumetric.wgsl",
	fog: "./postprocess/fog.wgsl",
	motionBlur: "./postprocess/motionBlur.wgsl",
	dof: "./postprocess/dof.wgsl",
	bloomDownsample: "./postprocess/bloomDownsample.wgsl",
	bloomBlurH: "./postprocess/bloomBlurH.wgsl",
	bloomBlurV: "./postprocess/bloomBlurV.wgsl",
	bloomUpsample: "./postprocess/bloomUpsample.wgsl",
	bloomComposite: "./postprocess/bloomComposite.wgsl",
	toneMapping: "./postprocess/toneMapping.wgsl",
	colorFilter: "./postprocess/colorFilter.wgsl",
	interactionOutline: "./postprocess/interactionOutline.wgsl",
	fxaa: "./postprocess/fxaa.wgsl",
	copy: "./postprocess/copy.wgsl",
	sobelNormal: "./postprocess/sobelNormal.wgsl",
};

export function loadPostProcessShaderPart(
	part: PostProcessShaderPart
): Promise<string> {
	if (POST_PROCESS_PARTS_USING_SHARED_LIGHT_DATA.has(part)) {
		return loadPostProcessShaderPartComposite(part).then(
			(composite) => composite.code
		);
	}

	const path = postProcessShaderFiles[part];

	return loadShader(`post:${part}`, path, () => {
		const loader = postProcessParts[`./postprocess/${part}.wgsl`];
		if (!loader) {
			return Promise.reject(
				new Error(`Post-process shader part not found: ${part}`)
			);
		}
		return loader();
	});
}

export function loadPostProcessShaderPartComposite(
	part: PostProcessShaderPart
): Promise<CompositeShaderSource> {
	if (POST_PROCESS_PARTS_USING_SHARED_LIGHT_DATA.has(part)) {
		return composeWithSharedLightData(
			`post-composite:${part}`,
			loadShaderCompositeFromFile(`post:${part}`, postProcessShaderFiles[part], () => {
				const loader = postProcessParts[`./postprocess/${part}.wgsl`];
				if (!loader) {
					return Promise.reject(
						new Error(`Post-process shader part not found: ${part}`)
					);
				}
				return loader();
			})
		);
	}

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
			return loader();
		});
		_compositeCache.set(key, cached);
	}
	return cached;
}

export function loadClusteredLightingCullShaderComposite():
	Promise<CompositeShaderSource> {
	return composeWithSharedLightData(
		"clustered-lighting-cull-composite",
		loadShaderCompositeFromFile(
			"clustered-lighting-cull",
			"./clusteredLightingCull.wgsl",
			fromRawShaderModuleLoader(() =>
				import("./clusteredLightingCull.wgsl?raw")
			)
		)
	);
}
