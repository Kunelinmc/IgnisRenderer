import { Logger } from "../../foundation/Logger";

/** @internal Destroys aliased managed handles exactly once. */
export function destroyUniqueWebGPUHandles(
	handles: readonly (object | null)[],
	kind: string,
	scope: string,
): void {
	const destroyed = new Set<object>();
	for (const handle of handles) {
		if (!handle || destroyed.has(handle)) continue;
		destroyed.add(handle);
		const destroy = (handle as { destroy?: () => void }).destroy;
		if (typeof destroy !== "function") continue;
		try {
			destroy.call(handle);
		} catch (error) {
			Logger.warn(
				`[webgpu-cache-resource-destroy-failed] Failed to destroy cached ${kind}: ${String(error)}`,
				{ scope },
			);
		}
	}
}
