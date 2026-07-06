import type { IRenderTexture } from "../types";
import type { WebGPUFrameFeatureKey } from "./FrameFeatures";
import type { WebGPUPostProcessPassId } from "./postprocess/types";

/** @internal WebGPU pass-owned implementation context metadata. */
export type WebGPUPostProcessContextKind = "screen" | "present";

/** @internal WebGPU pass-owned implementation history binding side. */
export type WebGPUPostProcessHistorySide = "read" | "write";

/** @internal WebGPU pass-owned implementation history binding metadata. */
export interface WebGPUPostProcessHistoryBindingMetadata {
	/**
	 * Context property that receives the resolved history texture.
	 *
	 * @remarks The value is read from `PostProcessPassRequest.histories`.
	 * @sideEffects None.
	 */
	readonly property: string;
	/**
	 * Pipeline-owned history id to resolve.
	 *
	 * @remarks Must match a descriptor returned by the logical pass.
	 * @sideEffects None.
	 */
	readonly historyId: string;
	/**
	 * History side to bind into the context.
	 *
	 * @remarks `read` samples the previous valid resource and `write` targets
	 * the current frame's writable resource.
	 * @sideEffects None.
	 */
	readonly side: WebGPUPostProcessHistorySide;
}

/** @internal WebGPU pass-owned implementation motion-history copy metadata. */
export interface WebGPUPostProcessMotionHistoryCopyMetadata {
	/**
	 * Context property containing the writable motion-history texture.
	 *
	 * @remarks The executor copies current frame motion data into this target
	 * after command submission when the generated callback is invoked.
	 * @sideEffects None.
	 */
	readonly writeProperty: string;
	/**
	 * Context method installed by the executor.
	 *
	 * @remarks Omit to use `writeMotionHistoryFromCurrent`.
	 * @sideEffects None.
	 */
	readonly method?: string;
}

/** @internal WebGPU pass-owned implementation transient binding metadata. */
export interface WebGPUPostProcessTransientBindingMetadata {
	/**
	 * Context property that receives the resolved transient texture.
	 *
	 * @remarks The value is read from `PostProcessPassRequest.transients`.
	 * @sideEffects None.
	 */
	readonly property: string;
	/**
	 * Pipeline-owned transient id to resolve.
	 *
	 * @remarks Must match a descriptor returned by the logical pass.
	 * @sideEffects None.
	 */
	readonly transientId: string;
}

/** @internal WebGPU pass-owned frame feature data binding metadata. */
export interface WebGPUPostProcessFrameDataBindingMetadata<TValue = unknown> {
	/**
	 * Context property that receives the requested frame feature data.
	 *
	 * @remarks The value is read from `WebGPUPreparedFrameResources.featureData`.
	 * @sideEffects None.
	 */
	readonly property: string;
	/**
	 * Typed internal frame feature data key.
	 *
	 * @remarks Missing data resolves to `undefined`; passes must handle absence.
	 * @sideEffects None.
	 */
	readonly key: WebGPUFrameFeatureKey<TValue>;
}

/** @internal WebGPU pass-owned implementation context metadata. */
export interface WebGPUPostProcessContextMetadata {
	/**
	 * Backend that owns this context metadata.
	 *
	 * @remarks WebGPU executors ignore metadata whose backend is not `webgpu`.
	 * @sideEffects None.
	 */
	readonly backend: "webgpu";
	/**
	 * Base context shape requested by the implementation.
	 *
	 * @remarks `screen` provides compute/screen-target helpers. `present`
	 * provides final presentation helpers.
	 * @sideEffects None.
	 */
	readonly kind: WebGPUPostProcessContextKind;
	/**
	 * Whether the context must expose `publishColorTarget(texture)`.
	 *
	 * @remarks The callback records the pass color output. The WebGPU executor
	 * validates and applies the target after the pass completes successfully.
	 * @sideEffects None.
	 */
	readonly publishColorTarget?: boolean;
	/**
	 * Whether the context must expose the current frame binding group.
	 *
	 * @remarks Used by passes that sample camera or frame-wide uniforms.
	 * @sideEffects None.
	 */
	readonly frameBinding?: boolean;
	/**
	 * Whether the context must expose the current lighting state.
	 *
	 * @remarks Used by passes that evaluate scene lights.
	 * @sideEffects None.
	 */
	readonly lightingState?: boolean;
	/**
	 * Frame feature data bindings to pack into the implementation context.
	 *
	 * @remarks Used by optional WebGPU features, such as volumetric lighting,
	 * that consume pass-specific frame data without depending on the full
	 * lighting state.
	 * @sideEffects None.
	 */
	readonly frameData?: readonly WebGPUPostProcessFrameDataBindingMetadata[];
	/**
	 * History texture bindings to pack into the context.
	 *
	 * @remarks Each entry resolves one `request.histories` slot by id and side.
	 * @sideEffects None.
	 */
	readonly histories?: readonly WebGPUPostProcessHistoryBindingMetadata[];
	/**
	 * Transient texture bindings to pack into the context.
	 *
	 * @remarks Each entry resolves one `request.transients` slot by id.
	 * @sideEffects None.
	 */
	readonly transients?: readonly WebGPUPostProcessTransientBindingMetadata[];
	/**
	 * Motion-history copy callback to pack into the context.
	 *
	 * @remarks The callback records which history texture receives the current
	 * frame motion buffer after command submission.
	 * @sideEffects None.
	 */
	readonly motionHistoryCopy?: WebGPUPostProcessMotionHistoryCopyMetadata;
}

/** @internal WebGPU screen-pass implementation context metadata. */
export const WEBGPU_SCREEN_POST_PROCESS_CONTEXT_METADATA = {
	backend: "webgpu",
	kind: "screen",
	publishColorTarget: true,
} as const satisfies WebGPUPostProcessContextMetadata;

/** @internal WebGPU present-pass implementation context metadata. */
export const WEBGPU_PRESENT_POST_PROCESS_CONTEXT_METADATA = {
	backend: "webgpu",
	kind: "present",
} as const satisfies WebGPUPostProcessContextMetadata;

/**
 * Returns whether a context metadata value is owned by the WebGPU backend.
 *
 * @internal Used by WebGPU pass-owned implementation context packing.
 *
 * @param value Candidate implementation metadata context.
 * @returns `true` when WebGPU can pack the declared context.
 * @sideEffects None.
 */
export function isWebGPUPostProcessContextMetadata(
	value: unknown
): value is WebGPUPostProcessContextMetadata {
	if (!value || typeof value !== "object") {
		return false;
	}
	const candidate = value as Partial<WebGPUPostProcessContextMetadata>;
	return (
		candidate.backend === "webgpu" &&
		(candidate.kind === "screen" || candidate.kind === "present")
	);
}

export type WebGPUReservedPostProcessPassId =
	| WebGPUPostProcessPassId
	| "gamma";

const WEBGPU_RESERVED_POST_PROCESS_PASS_ID_SET = new Set<string>([
	"ssao",
	"ssgi",
	"taa",
	"ssr",
	"ssrefraction",
	"volumetric",
	"fog",
	"motion-blur",
	"dof",
	"bloom",
	"color-filter",
	"fxaa",
	"tonemap",
	"gamma",
]);

/**
 * Returns whether `id` is reserved by an engine-provided WebGPU
 * post-process pass.
 *
 * @param id Candidate runtime pass id.
 * @returns `true` when the id cannot be used by custom post-process passes.
 */
export function isWebGPUReservedPostProcessPassId(id: string): boolean {
	return WEBGPU_RESERVED_POST_PROCESS_PASS_ID_SET.has(id);
}

/** @internal WebGPU frame target set exposed to pass-owned implementations. */
export interface WebGPUFrameTargets {
	sceneColor: IRenderTexture;
	sceneColorMain: IRenderTexture;
	postPing?: IRenderTexture | null;
	postPong?: IRenderTexture | null;
	gAlbedoAlpha?: IRenderTexture | null;
	gNormalRoughMetal?: IRenderTexture | null;
	gEmissiveOcclusion?: IRenderTexture | null;
	gMotionDepth?: IRenderTexture | null;
	gSpecular?: IRenderTexture | null;
	gCoatSheen?: IRenderTexture | null;
	gSheenReflectance?: IRenderTexture | null;
	gMaterialExt0?: IRenderTexture | null;
	gMaterialExt1?: IRenderTexture | null;
	gMaterialExt2?: IRenderTexture | null;
	gMaterialExt3?: IRenderTexture | null;
	depth: IRenderTexture;
	oitAccum?: IRenderTexture | null;
	oitReveal?: IRenderTexture | null;
	oitSceneColorCopy?: IRenderTexture | null;
	transmissionSceneColorCopy?: IRenderTexture | null;
	transmissionLighting?: IRenderTexture | null;
	gTransmissionSurface0?: IRenderTexture | null;
	gTransmissionSurface1?: IRenderTexture | null;
	gTransmissionSurface2?: IRenderTexture | null;
	transmissionDepth?: IRenderTexture | null;
	planarReflectionMask?: IRenderTexture | null;
}

/** @internal Read-only WebGPU frame target view for pass-owned implementations. */
export type WebGPUPostProcessFrameTargets = Readonly<WebGPUFrameTargets>;
