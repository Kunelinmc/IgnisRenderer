import type { OcclusionCullingBackendAdapter } from "../pipeline/OcclusionCulling";
import type { PostProcessBackendAdapter } from "../postprocess/types";
import type { IRenderBackend } from "./IRenderBackend";

export const RENDERER_POST_PROCESS_EXTENSION_ID = "renderer.postprocess";
export const RENDERER_OCCLUSION_CULLING_EXTENSION_ID =
	"renderer.occlusion-culling";

export const RENDERER_POST_PROCESS_INSERTION_POINT = "renderer:postprocess";
export const RENDERER_OCCLUSION_VISIBILITY_INSERTION_POINT =
	"renderer:prepared-scene:occlusion-visibility";
export const WEBGPU_OCCLUSION_AFTER_DEPTH_INSERTION_POINT =
	"backend:webgpu:frame-graph:after-depth";

export type RenderBackendExtensionId =
	| typeof RENDERER_POST_PROCESS_EXTENSION_ID
	| typeof RENDERER_OCCLUSION_CULLING_EXTENSION_ID
	| (string & {});

export type RenderBackendExtensionInsertionPoint =
	| typeof RENDERER_POST_PROCESS_INSERTION_POINT
	| typeof RENDERER_OCCLUSION_VISIBILITY_INSERTION_POINT
	| typeof WEBGPU_OCCLUSION_AFTER_DEPTH_INSERTION_POINT
	| (string & {});

export interface RenderBackendExtension<TApi = unknown> {
	readonly id: RenderBackendExtensionId;
	readonly insertionPoints: readonly RenderBackendExtensionInsertionPoint[];
	readonly api: TApi;
}

export interface RenderBackendExtensionRegistry {
	getExtension<TApi = unknown>(
		id: RenderBackendExtensionId
	): RenderBackendExtension<TApi> | null;
	listExtensions(): readonly RenderBackendExtension[];
}

/**
 * Creates a stable backend extension registry.
 *
 * @param extensions Extension descriptors exposed by one backend.
 * @returns Registry used by `Renderer` to resolve backend-owned integration APIs.
 * @sideEffects None.
 */
export function createRenderBackendExtensionRegistry(
	extensions: readonly RenderBackendExtension[]
): RenderBackendExtensionRegistry {
	const extensionById = new Map<RenderBackendExtensionId, RenderBackendExtension>();
	for (const extension of extensions) {
		if (extensionById.has(extension.id)) {
			throw new Error(
				`Duplicate render backend extension id "${extension.id}".`
			);
		}
		extensionById.set(extension.id, extension);
	}
	const snapshot = Array.from(extensionById.values());
	return {
		getExtension<TApi = unknown>(
			id: RenderBackendExtensionId
		): RenderBackendExtension<TApi> | null {
			return (
				(extensionById.get(id) as RenderBackendExtension<TApi> | undefined) ??
				null
			);
		},
		listExtensions(): readonly RenderBackendExtension[] {
			return snapshot;
		},
	};
}

/**
 * Resolves the backend post-process extension.
 *
 * @param backend Backend that may expose renderer-owned post-process support.
 * @returns Typed post-process extension descriptor, or `null` when absent.
 * @sideEffects None.
 */
export function resolvePostProcessBackendExtension(
	backend: IRenderBackend
): RenderBackendExtension<PostProcessBackendAdapter> | null {
	return (
		backend.extensions?.getExtension<PostProcessBackendAdapter>(
			RENDERER_POST_PROCESS_EXTENSION_ID
		) ?? null
	);
}

/**
 * Resolves the backend occlusion culling extension.
 *
 * @param backend Backend that may expose previous-frame visibility snapshots.
 * @returns Typed occlusion culling extension descriptor, or `null` when absent.
 * @sideEffects None.
 */
export function resolveOcclusionCullingBackendExtension(
	backend: IRenderBackend
): RenderBackendExtension<OcclusionCullingBackendAdapter> | null {
	return (
		backend.extensions?.getExtension<OcclusionCullingBackendAdapter>(
			RENDERER_OCCLUSION_CULLING_EXTENSION_ID
		) ?? null
	);
}
