import {
	normalizeOcclusionCullingOptions,
	type OcclusionVisibilityProvider,
} from "../pipeline/OcclusionCulling";
import type { ResolvedFeatureState } from "../pipeline/types";
import type { IRenderBackend } from "./IRenderBackend";
import { resolveOcclusionCullingBackendExtension } from "./BackendExtensions";

/**
 * Coordinates renderer-owned occlusion culling decisions with backend snapshots.
 */
export class RendererOcclusionCullingController {
	private readonly _backend: IRenderBackend;

	public constructor(backend: IRenderBackend) {
		this._backend = backend;
	}

	/**
	 * Resolves the synchronous visibility provider for prepared-scene building.
	 *
	 * @param features Per-frame resolved renderer feature state.
	 * @returns Visibility provider, or `null` when occlusion culling is disabled
	 * or unsupported by the backend extension registry.
	 * @sideEffects May collect completed backend readbacks before returning the
	 * provider.
	 */
	public getVisibilityProvider(
		features: ResolvedFeatureState
	): OcclusionVisibilityProvider | null {
		if (features.enableOcclusionCulling !== true) {
			return null;
		}
		const adapter = resolveOcclusionCullingBackendExtension(this._backend)?.api;
		if (!adapter) {
			return null;
		}
		return adapter.getVisibilityProvider(
			normalizeOcclusionCullingOptions(features.occlusionCullingOptions)
		);
	}

	/**
	 * Resets backend occlusion visibility snapshots after renderer state changes.
	 *
	 * @returns Nothing.
	 * @sideEffects Clears backend visibility history when the backend exposes an
	 * occlusion culling extension.
	 */
	public reset(): void {
		resolveOcclusionCullingBackendExtension(this._backend)
			?.api.resetOcclusionCulling?.();
	}
}
