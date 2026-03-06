import { loadSkyboxShaderSource } from "./shaderSource";

let _skyboxShaderPromise: Promise<string> | null = null;

export function getWebGPUSkyboxShader(): Promise<string> {
	if (!_skyboxShaderPromise) {
		_skyboxShaderPromise = loadSkyboxShaderSource();
	}
	return _skyboxShaderPromise;
}
