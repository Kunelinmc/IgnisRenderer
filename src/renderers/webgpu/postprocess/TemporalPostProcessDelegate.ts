import type { WebGPUPostProcessRuntimePassRegistry } from "./types";

/**
 * Temporal runtime delegate retained for future runtime-owned temporal passes.
 *
 * Pass-owned temporal built-ins such as SSR and volumetric lighting register
 * their implementation lifecycle through `src/postprocess/passes/` instead of
 * this backend runtime registry.
 */
export class TemporalPostProcessDelegate {
	/**
	 * Creates a temporal runtime delegate.
	 *
	 * @param _shared Shared post-process runtime context retained for constructor
	 * compatibility with `WebGPUPostProcessRuntime`.
	 * @returns New delegate.
	 * @sideEffects None.
	 */
	public constructor(_shared?: unknown) {}

	/**
	 * Registers temporal runtime passes with the owning runtime.
	 *
	 * @param _registry Runtime registry supplied by `WebGPUPostProcessRuntime`.
	 * @returns Nothing.
	 * @sideEffects None.
	 */
	public registerPasses(_registry: WebGPUPostProcessRuntimePassRegistry): void {}

	/**
	 * Invalidates cached temporal runtime bindings.
	 *
	 * @returns Nothing.
	 * @sideEffects None while all temporal built-ins are pass-owned.
	 */
	public invalidateBindings(): void {}

	/**
	 * Handles shader runtime replacement.
	 *
	 * @returns Nothing.
	 * @sideEffects None while all temporal built-ins are pass-owned.
	 */
	public onShaderRuntimeChanged(): void {}

	/**
	 * Releases runtime-owned temporal resources.
	 *
	 * @returns Nothing.
	 * @sideEffects None while all temporal built-ins are pass-owned.
	 */
	public destroy(): void {}
}
