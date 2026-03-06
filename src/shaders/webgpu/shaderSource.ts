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

type PostProcessShaderPart = "ssao" | "taa" | "hiz" | "ssr" | "fxaa" | "copy";

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

export function loadSceneShaderPart(part: SceneShaderPart): Promise<string> {
	switch (part) {
		case "constants":
			return loadShader(
				"scene:constants",
				"./parts/constants.wgsl",
				() => import("./parts/constants.wgsl?raw")
			);
		case "definitions":
			return loadShader(
				"scene:definitions",
				"./parts/definitions.wgsl",
				() => import("./parts/definitions.wgsl?raw")
			);
		case "utils":
			return loadShader(
				"scene:utils",
				"./parts/utils.wgsl",
				() => import("./parts/utils.wgsl?raw")
			);
		case "vertexStage":
			return loadShader(
				"scene:vertexStage",
				"./parts/vertexStage.wgsl",
				() => import("./parts/vertexStage.wgsl?raw")
			);
		case "fragmentPrelude":
			return loadShader(
				"scene:fragmentPrelude",
				"./parts/fragmentPrelude.wgsl",
				() => import("./parts/fragmentPrelude.wgsl?raw")
			);
		case "fragmentPhong":
			return loadShader(
				"scene:fragmentPhong",
				"./parts/fragmentPhong.wgsl",
				() => import("./parts/fragmentPhong.wgsl?raw")
			);
		case "fragmentPbrSetup":
			return loadShader(
				"scene:fragmentPbrSetup",
				"./parts/fragmentPbrSetup.wgsl",
				() => import("./parts/fragmentPbrSetup.wgsl?raw")
			);
		case "fragmentPbrDirectional":
			return loadShader(
				"scene:fragmentPbrDirectional",
				"./parts/fragmentPbrDirectional.wgsl",
				() => import("./parts/fragmentPbrDirectional.wgsl?raw")
			);
		case "fragmentPbrPoint":
			return loadShader(
				"scene:fragmentPbrPoint",
				"./parts/fragmentPbrPoint.wgsl",
				() => import("./parts/fragmentPbrPoint.wgsl?raw")
			);
		case "fragmentPbrSpot":
			return loadShader(
				"scene:fragmentPbrSpot",
				"./parts/fragmentPbrSpot.wgsl",
				() => import("./parts/fragmentPbrSpot.wgsl?raw")
			);
		case "fragmentPbrAmbient":
			return loadShader(
				"scene:fragmentPbrAmbient",
				"./parts/fragmentPbrAmbient.wgsl",
				() => import("./parts/fragmentPbrAmbient.wgsl?raw")
			);
		case "fragmentSingleTarget":
			return loadShader(
				"scene:fragmentSingleTarget",
				"./parts/fragmentSingleTarget.wgsl",
				() => import("./parts/fragmentSingleTarget.wgsl?raw")
			);
	}
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

export function loadPostProcessShaderPart(
	part: PostProcessShaderPart
): Promise<string> {
	switch (part) {
		case "ssao":
			return loadShader(
				"post:ssao",
				"./postprocess/ssao.wgsl",
				() => import("./postprocess/ssao.wgsl?raw")
			);
		case "taa":
			return loadShader(
				"post:taa",
				"./postprocess/taa.wgsl",
				() => import("./postprocess/taa.wgsl?raw")
			);
		case "hiz":
			return loadShader(
				"post:hiz",
				"./postprocess/hiz.wgsl",
				() => import("./postprocess/hiz.wgsl?raw")
			);
		case "ssr":
			return loadShader(
				"post:ssr",
				"./postprocess/ssr.wgsl",
				() => import("./postprocess/ssr.wgsl?raw")
			);
		case "fxaa":
			return loadShader(
				"post:fxaa",
				"./postprocess/fxaa.wgsl",
				() => import("./postprocess/fxaa.wgsl?raw")
			);
		case "copy":
			return loadShader(
				"post:copy",
				"./postprocess/copy.wgsl",
				() => import("./postprocess/copy.wgsl?raw")
			);
	}
}
