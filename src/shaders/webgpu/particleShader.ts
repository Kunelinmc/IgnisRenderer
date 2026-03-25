import type { CompositeShaderSource } from "../runtime";
import {
	loadParticleShaderSource,
	loadParticleShaderSourceComposite,
} from "./shaderSource";

let _particleShaderPromise: Promise<string> | null = null;
let _particleShaderCompositePromise: Promise<CompositeShaderSource> | null =
	null;

export function getWebGPUParticleShader(): Promise<string> {
	if (!_particleShaderPromise) {
		_particleShaderPromise = loadParticleShaderSource();
	}
	return _particleShaderPromise;
}

export function getWebGPUParticleShaderComposite():
	Promise<CompositeShaderSource> {
	if (!_particleShaderCompositePromise) {
		_particleShaderCompositePromise = loadParticleShaderSourceComposite();
	}
	return _particleShaderCompositePromise;
}
