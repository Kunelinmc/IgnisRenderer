/**
 * Gets the backend-private handle carried by a renderer resource.
 *
 * @internal WebGPU backend implementation detail. Shared renderer contracts
 * intentionally keep this handle opaque.
 */
export function getWebGPUResourceHandle(resource: unknown): unknown {
	if (
		resource === null ||
		(typeof resource !== "object" && typeof resource !== "function")
	) {
		return undefined;
	}
	return (resource as { _gpuResource?: unknown })._gpuResource;
}

/**
 * Gets a backend-private handle when it is an object suitable for identity
 * tracking.
 *
 * @internal WebGPU backend implementation detail.
 */
export function getWebGPUObjectResourceHandle(resource: unknown): object | null {
	const handle = getWebGPUResourceHandle(resource);
	return handle !== null && typeof handle === "object" ? handle : null;
}
