import { Platform } from "../../foundation/Platform";

type RawShaderModule = {
	default: string;
};

const shaderParts: Record<string, () => Promise<string>> = Platform.isNodeRuntime()
	? {}
	: import.meta.glob<string>("./lightProbePrefilter.wgsl", {
			query: "?raw",
			import: "default",
		});

let _prefilterShaderCache: Promise<string> | null = null;

export function loadLightProbePrefilterShaderSource(): Promise<string> {
	if (_prefilterShaderCache) {
		return _prefilterShaderCache;
	}

	_prefilterShaderCache = (async () => {
		if (Platform.isNodeRuntime()) {
			const fsSpecifier = ["node", "fs/promises"].join(":");
			const fsModule = (await import(/* @vite-ignore */ fsSpecifier)) as {
				readFile: (
					path: string | URL,
					options?: string | { encoding?: string }
				) => Promise<string>;
			};
			return fsModule.readFile(
				new URL("./lightProbePrefilter.wgsl", import.meta.url),
				"utf8"
			);
		}

		const loader = shaderParts["./lightProbePrefilter.wgsl"];
		if (!loader) {
			throw new Error("Light probe prefilter shader source not found.");
		}
		const raw = (await loader()) as unknown as RawShaderModule["default"];
		return raw;
	})();

	return _prefilterShaderCache;
}
