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
	| "fxaaFragment";

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
};

export function loadWebGLShaderPart(part: WebGLShaderPart): Promise<string> {
	return loadShader(`webgl:${part}`, shaderFiles[part], () =>
		import(`./parts/${part}.glsl?raw`)
	);
}
