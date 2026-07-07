/** @internal WebGL pass-owned implementation context metadata. */
export type WebGLPostProcessContextKind = "screen";

/** @internal WebGL pass-owned implementation history binding side. */
export type WebGLPostProcessHistorySide = "read" | "write";

/** @internal WebGL pass-owned implementation history binding metadata. */
export interface WebGLPostProcessHistoryBindingMetadata {
	readonly property: string;
	readonly historyId: string;
	readonly side: WebGLPostProcessHistorySide;
}

/** @internal WebGL pass-owned implementation context metadata. */
export interface WebGLPostProcessContextMetadata {
	readonly backend: "webgl";
	readonly kind: WebGLPostProcessContextKind;
	readonly sceneMotionTexture?: boolean;
	readonly sceneNormalTexture?: boolean;
	readonly ssaoTargets?: boolean;
	readonly frameJitter?: boolean;
	readonly warn?: boolean;
	readonly histories?: readonly WebGLPostProcessHistoryBindingMetadata[];
	readonly syncPipelineHistories?: boolean;
	readonly markTAAHistoryValidOnPublish?: boolean;
}

/** @internal WebGL screen-pass implementation context metadata. */
export const WEBGL_SCREEN_POST_PROCESS_CONTEXT_METADATA = {
	backend: "webgl",
	kind: "screen",
} as const satisfies WebGLPostProcessContextMetadata;

/**
 * Returns whether a context metadata value is owned by the WebGL backend.
 *
 * @internal Used by WebGL pass-owned implementation context packing.
 *
 * @param value Candidate implementation metadata context.
 * @returns `true` when WebGL can pack the declared context.
 * @sideEffects None.
 */
export function isWebGLPostProcessContextMetadata(
	value: unknown
): value is WebGLPostProcessContextMetadata {
	if (!value || typeof value !== "object") {
		return false;
	}
	const candidate = value as Partial<WebGLPostProcessContextMetadata>;
	return candidate.backend === "webgl" && candidate.kind === "screen";
}
