import { Platform } from "../../foundation/Platform";
import { ShaderLoader } from "../../loaders/ShaderLoader";

const shaderParts: Record<string, () => Promise<string>> = Platform.isNodeRuntime()
	? {}
	: import.meta.glob<string>("./environmentIblPrefilter.wgsl", {
			query: "?raw",
			import: "default",
		});

const _shaderLoader = new ShaderLoader();

export function loadEnvironmentIBLPrefilterShaderSource(): Promise<string> {
	return _shaderLoader.loadSource({
		key: "environment-ibl-prefilter",
		nodeRelativePath: "./environmentIblPrefilter.wgsl",
		nodeBaseUrl: import.meta.url,
		browserLoader: () => {
			const loader = shaderParts["./environmentIblPrefilter.wgsl"];
			if (!loader) {
				return Promise.reject(
					new Error("Environment IBL prefilter shader source not found.")
				);
			}
			return loader();
		},
	});
}

/**
 * @deprecated Use loadEnvironmentIBLPrefilterShaderSource.
 */
export const loadLightProbePrefilterShaderSource =
	loadEnvironmentIBLPrefilterShaderSource;
