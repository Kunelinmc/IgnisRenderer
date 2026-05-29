import type { PostProcessBackendAdapter } from "./types";

const POST_PROCESS_BACKEND_ADAPTERS = new WeakMap<
	object,
	PostProcessBackendAdapter
>();

/**
 * Registers the post-process backend adapter associated with an owner object.
 *
 * @param owner Backend or host object that owns the adapter lifetime.
 * @param adapter Adapter used by `Renderer` to execute logical post-process work.
 * @returns Nothing.
 * @constraints `owner` must remain stable for the backend lifetime.
 * @sideEffects Replaces any previously registered adapter for `owner`.
 */
export function registerPostProcessBackendAdapter(
	owner: object,
	adapter: PostProcessBackendAdapter
): void {
	POST_PROCESS_BACKEND_ADAPTERS.set(owner, adapter);
}

/**
 * Removes the post-process backend adapter associated with an owner object.
 *
 * @param owner Backend or host object whose adapter should be removed.
 * @returns Nothing.
 * @sideEffects Deletes the registry entry for `owner`.
 */
export function unregisterPostProcessBackendAdapter(owner: object): void {
	POST_PROCESS_BACKEND_ADAPTERS.delete(owner);
}

/**
 * Resolves the post-process backend adapter associated with an owner object.
 *
 * @param owner Backend or host object used during registration.
 * @returns Registered adapter, or `null` when the owner has no adapter.
 * @sideEffects None.
 */
export function resolvePostProcessBackendAdapter(
	owner: object
): PostProcessBackendAdapter | null {
	return POST_PROCESS_BACKEND_ADAPTERS.get(owner) ?? null;
}
