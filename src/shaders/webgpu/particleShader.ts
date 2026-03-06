import { loadParticleShaderSource } from "./shaderSource";

let _particleShaderPromise: Promise<string> | null = null;

export function getWebGPUParticleShader(): Promise<string> {
	if (!_particleShaderPromise) {
		_particleShaderPromise = loadParticleShaderSource();
	}
	return _particleShaderPromise;
}
