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
	| "fxaa"
	| "copy";

type RawShaderModule = {
	default: string;
};

const _cache = new Map<string, Promise<string>>();

function isNodeRuntime(): boolean {
	const processObject = (
		globalThis as {
			process?: {
				versions?: {
					node?: string;
				};
			};
		}
	).process;
	const nodeVersion = processObject?.versions?.node;
	return typeof nodeVersion === "string" && nodeVersion.length > 0;
}

async function loadShader(
	key: string,
	nodeRelativePath: string,
	browserLoader: () => Promise<RawShaderModule>
): Promise<string> {
	let cached = _cache.get(key);
	if (!cached) {
		cached = (async () => {
			if (isNodeRuntime()) {
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

export function loadSkyboxShaderSource(): Promise<string> {
	return loadShader(
		"skybox",
		"./skyboxShader.wgsl",
		() => import("./skyboxShader.wgsl?raw")
	);
}

export function loadParticleShaderSource(): Promise<string> {
	return loadShader(
		"particle",
		"./particleShader.wgsl",
		() => import("./particleShader.wgsl?raw")
	);
}

const postProcessShaderFiles: Record<PostProcessShaderPart, string> = {
	ssao: "./postprocess/ssao.wgsl",
	taa: "./postprocess/taa.wgsl",
	hiz: "./postprocess/hiz.wgsl",
	ssr: "./postprocess/ssr.wgsl",
	volumetric: "./postprocess/volumetric.wgsl",
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
