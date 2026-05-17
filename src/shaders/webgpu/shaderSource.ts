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
	| "fragmentPbrArea"
	| "fragmentPbrAmbient"
	| "fragmentGBuffer"
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

type UtilityShaderPart = "planarReflectionComposite";

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

const utilityParts: ImportMetaGlobLoaderMap = Platform.isNodeRuntime()
	? {}
	: import.meta.glob<string>("./*.wgsl", {
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
	fragmentPbrArea: "./parts/fragmentPbrArea.wgsl",
	fragmentPbrAmbient: "./parts/fragmentPbrAmbient.wgsl",
	fragmentGBuffer: "./parts/fragmentGBuffer.wgsl",
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

export function loadEnvironmentShaderSource(): Promise<string> {
	return loadEnvironmentShaderSourceComposite().then((composite) => composite.code);
}

export function loadEnvironmentShaderSourceComposite(): Promise<CompositeShaderSource> {
	return composeWithSharedLightData(
		"environment-composite",
		loadShaderCompositeFromFile(
			"environment",
			"./environmentShader.wgsl",
			fromRawShaderModuleLoader(() => import("./environmentShader.wgsl?raw"))
		)
	);
}

export function loadDeferredLightingShaderComposite():
	Promise<CompositeShaderSource> {
	const key = "deferred-lighting-composite";
	let cached = _compositeCache.get(key);
	if (!cached) {
		cached = Promise.all([
			loadSceneShaderPartComposite("lightData"),
			loadSceneShaderPartComposite("constants"),
			loadSceneShaderPartComposite("definitions"),
			loadSceneShaderPartComposite("utils"),
			loadShaderCompositeFromFile(
				"deferred-lighting",
				"./deferredLightingShader.wgsl",
				fromRawShaderModuleLoader(() =>
					import("./deferredLightingShader.wgsl?raw")
				)
			),
		]).then((parts) =>
			composeCompositeShaderSources(
				parts.map((part) => ({
					code: part.code,
					sourceMap: part.sourceMap,
					sourcePath:
						part.sourceMap.segments[0]?.sourcePath ??
						"<webgpu-deferred-lighting-part>",
					kind: "template",
				})),
				"\n\n",
				"template"
			)
		);
		_compositeCache.set(key, cached);
	}
	return cached;
}

export function loadDeferredLightingShaderSource(): Promise<string> {
	return loadDeferredLightingShaderComposite().then((composite) => composite.code);
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
			fromRawShaderModuleLoader(() =>
				import("./particleShader.wgsl?raw")
			)
		)
	);
}

export function loadParticleSimulationShaderSource(): Promise<string> {
	return loadShader(
		"particle-simulation",
		"./particleSimulation.wgsl",
		fromRawShaderModuleLoader(() => import("./particleSimulation.wgsl?raw"))
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

const utilityShaderFiles: Record<UtilityShaderPart, string> = {
	planarReflectionComposite: "./planarReflectionComposite.wgsl",
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

export function loadPlanarReflectionCompositeShaderComposite():
	Promise<CompositeShaderSource> {
	const key = "planar-reflection-composite";
	let cached = _compositeCache.get(key);
	if (!cached) {
		cached = loadShaderCompositeFromFile(
			key,
			utilityShaderFiles.planarReflectionComposite,
			() => {
				const loader =
					utilityParts["./planarReflectionComposite.wgsl"];
				if (!loader) {
					return Promise.reject(
						new Error("Utility shader part not found: planarReflectionComposite")
					);
				}
				return loader();
			}
		);
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
