import type { CompositeShaderSource } from "../runtime";
import {
	loadEnvironmentShaderSource,
	loadEnvironmentShaderSourceComposite,
} from "./shaderSource";

let _environmentShaderPromise: Promise<string> | null = null;
let _environmentShaderCompositePromise: Promise<CompositeShaderSource> | null = null;

export function getWebGPUEnvironmentShader(): Promise<string> {
	if (!_environmentShaderPromise) {
		_environmentShaderPromise = loadEnvironmentShaderSource();
	}
	return _environmentShaderPromise;
}

export function getWebGPUEnvironmentShaderComposite(): Promise<CompositeShaderSource> {
	if (!_environmentShaderCompositePromise) {
		_environmentShaderCompositePromise = loadEnvironmentShaderSourceComposite();
	}
	return _environmentShaderCompositePromise;
}
