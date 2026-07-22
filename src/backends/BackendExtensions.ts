import type { OcclusionCullingBackendAdapter } from "../pipeline/OcclusionCulling";
import type { ProbeCaptureSource } from "../lights/runtime/ProbeCaptureRuntime";
import type { IWebGPUComputeFacade } from "./webgpu/ComputeFacade";

export const RENDERER_OCCLUSION_CULLING_EXTENSION_ID =
	"renderer.occlusion-culling";
export const RENDERER_PROBE_CAPTURE_EXTENSION_ID = "renderer.probe-capture";
export const WEBGPU_COMPUTE_EXTENSION_ID = "webgpu.compute";

export const RENDERER_OCCLUSION_VISIBILITY_INSERTION_POINT =
	"renderer:prepared-scene:occlusion-visibility";
export const WEBGPU_OCCLUSION_AFTER_DEPTH_INSERTION_POINT =
	"backend:webgpu:frame-graph:after-depth";

export type RenderBackendExtensionId =
	| typeof RENDERER_OCCLUSION_CULLING_EXTENSION_ID
	| typeof RENDERER_PROBE_CAPTURE_EXTENSION_ID
	| typeof WEBGPU_COMPUTE_EXTENSION_ID
	| (string & {});

export type RenderBackendExtensionInsertionPoint =
	| typeof RENDERER_OCCLUSION_VISIBILITY_INSERTION_POINT
	| typeof WEBGPU_OCCLUSION_AFTER_DEPTH_INSERTION_POINT
	| (string & {});

export interface RenderBackendExtensionKey<TApi> {
	readonly id: RenderBackendExtensionId;
}

export interface RenderBackendExtension<TApi = unknown> {
	readonly id: RenderBackendExtensionId;
	readonly insertionPoints: readonly RenderBackendExtensionInsertionPoint[];
	readonly api: TApi;
}

export interface RenderBackendExtensionReader {
	getBackendExtension<TApi>(key: RenderBackendExtensionKey<TApi>): TApi | null;
	requireBackendExtension<TApi>(key: RenderBackendExtensionKey<TApi>): TApi;
}

export interface RenderBackendExtensionRegistry
	extends RenderBackendExtensionReader {
	getExtension<TApi = unknown>(id: RenderBackendExtensionId):
		| RenderBackendExtension<TApi>
		| null;
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
		getBackendExtension<TApi>(key: RenderBackendExtensionKey<TApi>): TApi | null {
			return (
				(extensionById.get(key.id) as RenderBackendExtension<TApi> | undefined)
					?.api ?? null
			);
		},
		requireBackendExtension<TApi>(key: RenderBackendExtensionKey<TApi>): TApi {
			const extension = extensionById.get(key.id) as
				| RenderBackendExtension<TApi>
				| undefined;
			if (!extension) {
				throw new Error(`Render backend extension "${key.id}" is unavailable.`);
			}
			return extension.api;
		},
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
 * Resolves the backend occlusion culling extension.
 *
 * @param backend Backend that may expose previous-frame visibility snapshots.
 * @returns Typed occlusion culling extension descriptor, or `null` when absent.
 * @sideEffects None.
 */
export function resolveOcclusionCullingBackendExtension(
	reader: RenderBackendExtensionReader
): RenderBackendExtension<OcclusionCullingBackendAdapter> | null {
	const registry = (reader as any)?.extensions || reader;
	if (registry && typeof registry.getExtension === "function") {
		const ext = registry.getExtension(RENDERER_OCCLUSION_CULLING_EXTENSION_ID);
		if (ext) return ext;
	}

	let api: OcclusionCullingBackendAdapter | null = null;
	if (reader && typeof reader.getBackendExtension === "function") {
		api = reader.getBackendExtension(OCCLUSION_CULLING_EXTENSION);
	} else if (reader && (reader as any).extensions && typeof (reader as any).extensions.getBackendExtension === "function") {
		api = (reader as any).extensions.getBackendExtension(OCCLUSION_CULLING_EXTENSION);
	}
	if (!api) return null;
	return {
		id: RENDERER_OCCLUSION_CULLING_EXTENSION_ID,
		insertionPoints: [RENDERER_OCCLUSION_VISIBILITY_INSERTION_POINT],
		api,
	};
}

export const OCCLUSION_CULLING_EXTENSION: RenderBackendExtensionKey<
	OcclusionCullingBackendAdapter
> = {
	id: RENDERER_OCCLUSION_CULLING_EXTENSION_ID,
};

export const PROBE_CAPTURE_EXTENSION: RenderBackendExtensionKey<
	ProbeCaptureSource
> = {
	id: RENDERER_PROBE_CAPTURE_EXTENSION_ID,
};

export const WEBGPU_COMPUTE_EXTENSION: RenderBackendExtensionKey<
	IWebGPUComputeFacade
> = {
	id: WEBGPU_COMPUTE_EXTENSION_ID,
};
