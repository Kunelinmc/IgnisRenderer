import type {
	FrameAttachments,
	FrameContext,
	FramePass,
} from "../pipeline/types";
import type {
	RenderTargetReadbackOptions,
	RenderTargetReadbackResult,
} from "../rendering/CustomRenderTargets";
import type { RenderDirtyReason } from "../pipeline/incremental";
import type { ShaderCompileError } from "../shaders/runtime";
import type { RenderBackendExtensionRegistry } from "./BackendExtensions";
import type {
	DisplayOutputOptions,
	DisplayOutputState,
	ResolvedDisplayOutputOptions,
} from "../rendering/DisplayOutput";

/** Built-in renderer backend identifiers. */
export type KnownBackendType = "software" | "webgpu" | "webgl";

/** Backend identifier, including custom backend implementations. */
export type RenderBackendType = KnownBackendType | (string & {});

/** Backend frame scheduling policy reported to renderer coordination. */
export type FrameSchedulingMode = "always" | "on-demand";

/** Diagnostic details reported when a backend device or context is lost. */
export interface RenderBackendDeviceLostInfo {
	/** Backend-specific reason for the loss, when available. */
	reason?: string;
	/** Human-readable diagnostic message, when available. */
	message?: string;
}

/** Progress snapshot for one backend warmup operation. */
export interface WarmupProgress {
	/** Name of the warmup phase currently being processed. */
	phase: string;
	/** Number of work items completed in the current phase or report. */
	completed: number;
	/** Total number of work items in the current phase or report. */
	total: number;
	/** Optional backend-specific detail about the current work item. */
	detail?: string;
}

/** Scheduling policy used to start backend warmup work. */
export type WarmupSchedulingMode = "immediate" | "next-frame" | "idle";

/** Options controlling backend shader and pass warmup. */
export interface WarmupOptions {
	/** Include the backend's core rendering passes. */
	includeCorePasses?: boolean;
	/** Include shadow rendering passes. */
	includeShadowPass?: boolean;
	/** Include post-process passes. */
	includePostProcess?: boolean;
	/** Include particle rendering passes. */
	includeParticles?: boolean;
	/** Include shader compilation diagnostics in the report. */
	logCompilationInfo?: boolean;
	/**
	 * Controls when warmup begins.
	 *
	 * `immediate` preserves the legacy eager behavior, `next-frame` allows the
	 * browser to paint once before warmup, and `idle` schedules warmup work for
	 * browser idle time when supported.
	 */
	scheduling?: WarmupSchedulingMode;
	/**
	 * Approximate main-thread budget for cooperative warmup chunks.
	 *
	 * A value of `0` disables cooperative yielding. When `scheduling` is `idle`
	 * and this option is omitted, backends use a small default time slice.
	 */
	yieldIntervalMs?: number;
	/** Receives progress updates as warmup phases complete. */
	onProgress?: (progress: WarmupProgress) => void;
}

/** Summary of warmup work performed for one backend phase. */
export interface WarmupPhaseReport {
	/** Name of the completed warmup phase. */
	phase: string;
	/** Number of work items considered by the phase. */
	total: number;
	/** Number of work items compiled successfully. */
	compiled: number;
	/** Number of work items skipped by the backend. */
	skipped: number;
	/** Number of work items that failed to compile. */
	failed: number;
}

/** Complete result of a backend warmup operation. */
export interface WarmupReport {
	/** Backend that produced the report. */
	backend: RenderBackendType;
	/** Start timestamp in milliseconds. */
	startedAt: number;
	/** Completion timestamp in milliseconds. */
	finishedAt: number;
	/** Elapsed warmup duration in milliseconds. */
	durationMs: number;
	/** Total number of work items considered. */
	total: number;
	/** Total number of work items compiled successfully. */
	compiled: number;
	/** Total number of work items skipped by the backend. */
	skipped: number;
	/** Total number of work items that failed to compile. */
	failed: number;
	/** Per-phase warmup summaries in execution order. */
	phases: WarmupPhaseReport[];
	/** Shader compilation failures collected during warmup. */
	errors: ShaderCompileError[];
}

/** Optional device and adapter identifiers collected for diagnostics. */
export interface RenderBackendDeviceDebugInfo {
	/** Adapter vendor name, when reported by the runtime. */
	readonly vendor?: string;
	/** Adapter renderer name, when reported by the runtime. */
	readonly renderer?: string;
	/** Adapter architecture name, when reported by the runtime. */
	readonly architecture?: string;
	/** Adapter device name, when reported by the runtime. */
	readonly device?: string;
	/** Human-readable device description, when reported by the runtime. */
	readonly description?: string;
	/** Whether the runtime selected a fallback adapter. */
	readonly isFallbackAdapter?: boolean;
	/** Driver version, when reported by the runtime. */
	readonly driverVersion?: string;
	/** Additional backend-specific diagnostic values. */
	readonly raw?: Record<string, string | number | boolean>;
}

/** Best-effort runtime diagnostics for an attached backend. */
export interface RenderBackendDebugInfo {
	/** Backend identifier that produced the diagnostic snapshot. */
	readonly backend: RenderBackendType;
	/** Graphics API surface used by the backend. */
	readonly api: "software" | "webgpu" | "webgl2";
	/** Whether runtime diagnostics were collected successfully. */
	readonly available: boolean;
	/** Reason diagnostics are unavailable, when `available` is `false`. */
	readonly unavailableReason?: string;
	/** Best-effort adapter and device details. */
	readonly device?: RenderBackendDeviceDebugInfo;
	/** Selected numeric device limits, when available. */
	readonly limits?: Record<string, number>;
	/** Backend feature or extension names, when available. */
	readonly features?: readonly string[];
}

/** Feature support reported by a backend implementation. */
export interface BackendCapabilities {
	/** Whether HDR display output is supported. */
	displayHDR: boolean;
	/** Whether spherical-harmonics lighting is supported. */
	sh: boolean;
	/** Whether shadow rendering is supported. */
	shadows: boolean;
	/** Whether reflection rendering is supported. */
	reflection: boolean;
	/** Whether environment lighting is supported. */
	environment: boolean;
	/** Whether post-processing is supported. */
	postProcess: boolean;
	/** Whether mesh particle templates render through regular mesh packets. */
	meshParticles: boolean;
	/** Whether clustered lighting is supported. */
	clusteredLighting: boolean;
	/** Whether weighted or order-independent transparency is supported. */
	oit: boolean;
	/** Whether occlusion culling is supported. */
	occlusionCulling: boolean;
	/** Whether custom render targets are supported. */
	customRenderTargets: boolean;
	/** Whether custom render passes are supported. */
	customRenderPasses: boolean;
	/** Whether custom render-target readback is supported. */
	renderTargetReadback: boolean;
}

/** Runtime capabilities and scheduling metadata for one backend. */
export interface RenderBackendProfile {
	/** Backend identifier. */
	readonly id: RenderBackendType;
	/** Feature support advertised by the backend. */
	readonly capabilities: BackendCapabilities;
	/** Default frame scheduling policy for this backend. */
	readonly frameScheduling: FrameSchedulingMode;
	/** Backend-specific lighting integration metadata. */
	readonly lighting: {
		/** How localized light probes are accumulated by the backend. */
		readonly localizedProbeMode: "accumulate-globally" | "backend-local";
	};
}

/**
 * The final-output preservation guarantee reported after a successful frame.
 *
 * @internal Owned by renderer frame coordination. Applications receive the
 * normalized `IncrementalFrameStatus` from `Renderer.renderFrame()`.
 */
export type RenderBackendCompletedFrameCoverage =
	| "dirty-tiles"
	| "full-frame";

/** Presentation surface and resolved output configuration owned by a backend. */
export interface RenderSurface {
	/** Canvas used for presentation. */
	readonly canvas: HTMLCanvasElement;
	/** Effective display-output configuration for the attached surface. */
	readonly displayOutput: ResolvedDisplayOutputOptions;
}

/** Pixel dimensions used to configure backend presentation attachments. */
export interface RenderSurfaceSize {
	/** Width in physical pixels. */
	readonly width: number;
	/** Height in physical pixels. */
	readonly height: number;
}

/** Backend resource lifecycle action forwarded to renderer coordination. */
export type RendererBackendResourceEventAction = "invalidate" | "destroy";

/** Identifier of a backend-owned resource in a lifecycle event. */
export type RendererBackendResourceEventResource = string & {};

/** Notification that a backend-owned resource requires renderer action. */
export interface RendererBackendResourceEvent {
	/** Resource identifier or resource category. */
	readonly resource: RendererBackendResourceEventResource;
	/** Action the renderer must apply to the resource. */
	readonly action: RendererBackendResourceEventAction;
	/** Backend associated with the resource, when known. */
	readonly backend?: RenderBackendType;
	/** Optional reason for the lifecycle transition. */
	readonly reason?: string;
}

/** Discriminated union of backend lifecycle and invalidation notifications. */
export type RenderBackendEvent =
	| {
			type: "device-lost";
			info?: RenderBackendDeviceLostInfo;
	  }
	| { type: "device-restored" }
	| {
			type: "display-output-change";
			previous: DisplayOutputState;
			current: DisplayOutputState;
	  }
	| { type: "render-invalidated"; reason: RenderDirtyReason }
	| { type: "resource-lifecycle"; event: RendererBackendResourceEvent };

/** Event sink used by an attached backend to notify the renderer. */
export interface RenderBackendEventSink {
	/** Emits one backend state transition or invalidation event. */
	emit(event: RenderBackendEvent): void;
}

/** Renderer-owned context supplied when a backend is attached. */
export interface RenderBackendAttachContext {
	/** Presentation surface and resolved display-output options. */
	readonly surface: RenderSurface;
	/** Sink for backend lifecycle and resource events. */
	readonly events: RenderBackendEventSink;
}

/**
 * Backend runtime contract consumed by `Renderer` frame coordination.
 *
 * @internal Renderer and backend implementations own this contract. Applications
 * should use `Renderer` instead of invoking backend methods directly.
 */
export interface IRenderBackend {
	/**
	 * Runtime capabilities and scheduling metadata for this attached backend.
	 */
	readonly profile: RenderBackendProfile;
	/**
	 * Optional registry of backend-owned integration APIs.
	 *
	 * @remarks Renderer-facing optional capabilities must be exposed through this
	 * registry instead of adding feature-specific properties to `IRenderBackend`.
	 * @sideEffects None.
	 */
	readonly extensions: RenderBackendExtensionRegistry;
	/**
	 * Attaches this backend instance to exactly one renderer surface.
	 *
	 * @param context Presentation surface and backend event sink.
	 * @sideEffects Stores renderer-owned surface and event context. Implementations
	 * must throw if called more than once, including after `destroy()`.
	 */
	attach(context: RenderBackendAttachContext): void;
	/**
	 * Initializes the backend device or graphics context and its resources.
	 *
	 * @returns A promise that resolves when initialization is complete.
	 * @constraints `attach()` must have completed. Implementations must reject
	 * calls made after destruction or after initialization has completed.
	 * @sideEffects Acquires backend device resources and transitions the backend
	 * into its ready state.
	 */
	initialize(): Promise<void>;
	/**
	 * Returns a best-effort diagnostic snapshot for the backend runtime.
	 *
	 * @returns Backend debug information. The snapshot may report unavailable
	 * or redacted fields when the backend is not initialized or browser privacy
	 * policy hides GPU identifiers.
	 * @sideEffects None. Implementations must not initialize backend resources.
	 */
	getDebugInfo(): RenderBackendDebugInfo;
	/**
	 * Returns the resolved display-output state.
	 *
	 * @internal Owned by the Renderer display-output facade.
	 */
	getDisplayOutputState(): DisplayOutputState;
	/**
	 * Reconfigures the presentation output without rebuilding the backend.
	 *
	 * @internal Owned by the Renderer display-output facade.
	 */
	setDisplayOutput(options: DisplayOutputOptions): Promise<DisplayOutputState>;
	/**
	 * Rebuilds backend device or graphics context resources after loss.
	 *
	 * @returns A promise when restoration is asynchronous, otherwise `void`.
	 * @constraints The backend must be attached and in a restorable lost state.
	 * @sideEffects Recreates device resources, recovers backend-owned state, and
	 * emits a `device-restored` event when restoration completes.
	 */
	restore(): void | Promise<void>;
	/**
	 * Resizes backend presentation and frame attachments.
	 *
	 * @param size Target surface dimensions in physical pixels.
	 * @constraints Implementations must defer resource replacement until an active
	 * frame reaches a terminal state.
	 * @sideEffects Reconfigures size-dependent backend resources.
	 */
	resize(size: RenderSurfaceSize): void;
	/**
	 * Releases all backend device, surface, and cached rendering resources.
	 *
	 * @sideEffects Invalidates the backend for further rendering. Implementations
	 * must make repeated calls safe.
	 */
	destroy(): void | Promise<void>;
	/**
	 * Creates or returns attachments for a frame of the requested size.
	 *
	 * @param size Target attachment dimensions in physical pixels.
	 * @returns Backend-owned frame attachments suitable for `FrameContext`.
	 * @constraints The backend must be attached before attachments are requested.
	 * @sideEffects May allocate or resize backend-owned attachment resources.
	 */
	getAttachments(size: RenderSurfaceSize): FrameAttachments;
	/**
	 * Begins recording and execution for one frame.
	 *
	 * @param context Active frame context prepared by renderer coordination.
	 * @constraints The backend must be initialized and must not already have an
	 * active frame.
	 * @sideEffects Acquires per-frame resources and starts backend command or pass
	 * recording.
	 */
	beginFrame(context: FrameContext): void | Promise<void>;
	/**
	 * Aborts the active frame after a failed `beginFrame` or pass execution.
	 *
	 * @param error Optional original frame error used only for diagnostics.
	 * @returns Nothing.
	 * @constraints Implementations must tolerate repeated calls and calls when
	 * no frame is active.
	 * @sideEffects Releases per-frame state without presenting, submitting new
	 * frame work, or committing temporal history.
	 */
	abortFrame(error?: unknown): void | Promise<void>;
	/**
	 * Executes one planned frame pass.
	 *
	 * @param pass Pass descriptor to execute.
	 * @param context Active frame context associated with the current frame.
	 * @constraints A frame must be active and `context` must be the context passed
	 * to `beginFrame()`.
	 * @sideEffects Records and/or submits backend commands for the pass.
	 */
	executePass(pass: FramePass, context: FrameContext): void | Promise<void>;
	/**
	 * Notifies the backend that a planned pass was disabled.
	 *
	 * @param pass Disabled pass descriptor.
	 * @constraints Implementations may release or transition pass dependencies;
	 * they must not execute the disabled pass.
	 * @sideEffects May release or transition pass-owned backend state.
	 */
	skipPass?(pass: FramePass): void;
	/**
	 * Reads color data from a completed custom render target attachment.
	 *
	 * @param id Registered custom render-target identifier.
	 * @param attachmentIndex Zero-based color attachment index.
	 * @param options Optional readback dimensions.
	 * @returns Readback pixels in the attachment's storage format and origin.
	 * @constraints The target must be supported by the backend and have completed
	 * at least one successful frame before readback.
	 * @sideEffects May submit or await backend readback work; does not alter the
	 * target's published contents.
	 */
	readRenderTargetColor?(
		id: string,
		attachmentIndex?: number,
		options?: RenderTargetReadbackOptions
	): Promise<RenderTargetReadbackResult>;
	/**
	 * Warms backend shader and pass resources before regular frame execution.
	 *
	 * @param context Frame context used to resolve warmup resources.
	 * @param options Optional phase, scheduling, and progress settings.
	 * @returns A report containing phase totals and shader compilation failures.
	 * @sideEffects May compile shaders and allocate backend-owned warmup resources.
	 */
	warmup?(
		context: FrameContext,
		options?: WarmupOptions
	): Promise<WarmupReport>;
	/**
	 * Completes the active frame and presents its recorded output.
	 *
	 * @returns A promise when command submission or presentation is asynchronous,
	 * otherwise `void`.
	 * @constraints A frame must be active. State publication must occur only after
	 * all frame work and post-submit hooks succeed.
	 * @sideEffects Submits commands, presents the frame, and commits temporal and
	 * render-target state.
	 */
	endFrame(): void | Promise<void>;
	/**
	 * Returns the completed frame's final-output preservation guarantee.
	 *
	 * @internal Owned by renderer frame coordination. Implementations must return
	 * `"full-frame"` unless every final-output path preserved non-dirty tiles.
	 */
	getCompletedFrameCoverage(): RenderBackendCompletedFrameCoverage;
}
