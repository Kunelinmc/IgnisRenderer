import type { WebGPUPostProcessRuntimePassRegistry } from "./types";

/**
 * Reserved spatial delegate kept for backend-internal delegate symmetry.
 *
 * SSAO and SSGI are owned by logical pass implementations under
 * `src/postprocess/passes/`.
 */
export class SpatialPostProcessDelegate {
	constructor(_shared?: unknown) {}

	public registerPasses(_registry: WebGPUPostProcessRuntimePassRegistry): void {}

	public destroy(): void {}
}
