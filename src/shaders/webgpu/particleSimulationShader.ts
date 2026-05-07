import { loadParticleSimulationShaderSource } from "./shaderSource";

let _particleSimulationShaderPromise: Promise<string> | null = null;

export function getWebGPUParticleSimulationShader(): Promise<string> {
	if (!_particleSimulationShaderPromise) {
		_particleSimulationShaderPromise = loadParticleSimulationShaderSource();
	}
	return _particleSimulationShaderPromise;
}
