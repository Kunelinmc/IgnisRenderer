import type { CompositeShaderSource } from "../runtime";
import {
	loadSkyboxShaderSource,
	loadSkyboxShaderSourceComposite,
} from "./shaderSource";

let _skyboxShaderPromise: Promise<string> | null = null;
let _skyboxShaderCompositePromise: Promise<CompositeShaderSource> | null = null;

export function getWebGPUSkyboxShader(): Promise<string> {
	if (!_skyboxShaderPromise) {
		_skyboxShaderPromise = loadSkyboxShaderSource();
	}
	return _skyboxShaderPromise;
}

export function getWebGPUSkyboxShaderComposite(): Promise<CompositeShaderSource> {
	if (!_skyboxShaderCompositePromise) {
		_skyboxShaderCompositePromise = loadSkyboxShaderSourceComposite();
	}
	return _skyboxShaderCompositePromise;
}
