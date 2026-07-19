import {
	normalizeOcclusionCullingOptions,
	type OcclusionVisibilityProvider,
} from "../pipeline/OcclusionCulling";
import type { ResolvedFeatureState } from "../pipeline/types";
import {
	OCCLUSION_CULLING_EXTENSION,
	type RenderBackendExtensionReader,
} from "../backends/BackendExtensions";

/**
 * Coordinates renderer-owned occlusion culling decisions with backend snapshots.
 */
export class RendererOcclusionCullingController {
	private readonly _extensions: RenderBackendExtensionReader;

	public constructor(extensions: RenderBackendExtensionReader | any) {
		if (extensions && typeof extensions.getBackendExtension === "function") {
			this._extensions = extensions;
		} else if (extensions && extensions.extensions && typeof extensions.extensions.getBackendExtension === "function") {
			this._extensions = extensions.extensions;
		} else {
			this._extensions = extensions;
		}
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
		const adapter = this._extensions.getBackendExtension(
			OCCLUSION_CULLING_EXTENSION
		);
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
		this._extensions
			.getBackendExtension(OCCLUSION_CULLING_EXTENSION)
			?.resetOcclusionCulling?.();
	}
}
