import type { CompositeShaderSource } from "../runtime";
import {
	loadDeferredLightingShaderComposite,
	loadDeferredLightingShaderSource,
} from "./shaderSource";

let _deferredLightingShaderPromise: Promise<string> | null = null;
let _deferredLightingShaderCompositePromise:
	Promise<CompositeShaderSource> | null = null;

export function getWebGPUDeferredLightingShader(): Promise<string> {
	if (!_deferredLightingShaderPromise) {
		_deferredLightingShaderPromise = loadDeferredLightingShaderSource();
	}
	return _deferredLightingShaderPromise;
}

export function getWebGPUDeferredLightingShaderComposite():
	Promise<CompositeShaderSource> {
	if (!_deferredLightingShaderCompositePromise) {
		_deferredLightingShaderCompositePromise =
			loadDeferredLightingShaderComposite();
	}
	return _deferredLightingShaderCompositePromise;
}
