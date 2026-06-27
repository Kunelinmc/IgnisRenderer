import type {
	FrameAttachments,
	FrameContext,
	FramePass,
} from "../pipeline/types";
import type {
	RenderTargetReadbackOptions,
} from "./CustomRenderTargets";
import type { TextureReadbackResult } from "./IComputeRuntime";
import type { RenderDirtyReason } from "../pipeline/incremental";
import type { IShadowBackendCapabilities } from "../lights/shadows";
import type { ShaderCompileError } from "../shaders/runtime";
import type { RenderBackendExtensionRegistry } from "./BackendExtensions";

export type KnownBackendType = "software" | "webgpu" | "webgl";
export type RenderBackendType = KnownBackendType | (string & {});
export type FrameSchedulingMode = "always" | "on-demand";

export interface RenderBackendDeviceLostInfo {
	reason?: string;
	message?: string;
}

export interface WarmupProgress {
	phase: string;
	completed: number;
	total: number;
	detail?: string;
}

export type WarmupSchedulingMode = "immediate" | "next-frame" | "idle";

export interface WarmupOptions {
	includeCorePasses?: boolean;
	includeShadowPass?: boolean;
	includePostProcess?: boolean;
	includeParticles?: boolean;
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
	onProgress?: (progress: WarmupProgress) => void;
}

export interface WarmupPhaseReport {
	phase: string;
	total: number;
	compiled: number;
	skipped: number;
	failed: number;
}

export interface WarmupReport {
	backend: RenderBackendType;
	startedAt: number;
	finishedAt: number;
	durationMs: number;
	total: number;
	compiled: number;
	skipped: number;
	failed: number;
	phases: WarmupPhaseReport[];
	errors: ShaderCompileError[];
}

export interface BackendCapabilities {
	sh: boolean;
	shadows: boolean;
	reflection: boolean;
	environment: boolean;
	postProcess: boolean;
	clusteredLighting: boolean;
	oit: boolean;
	occlusionCulling: boolean;
	customRenderTargets: boolean;
	customRenderPasses: boolean;
	renderTargetReadback: boolean;
}

export interface RenderBackendProfile {
	readonly id: RenderBackendType;
	readonly capabilities: BackendCapabilities;
	readonly frameScheduling: FrameSchedulingMode;
	readonly shadow: IShadowBackendCapabilities;
	readonly lighting: {
		readonly localizedProbeMode: "accumulate-globally" | "backend-local";
	};
}

export interface RenderSurface {
	readonly canvas: HTMLCanvasElement;
}

export interface RenderSurfaceSize {
	readonly width: number;
	readonly height: number;
}

export type RendererBackendResourceEventAction = "invalidate" | "destroy";
export type RendererBackendResourceEventResource = string & {};

export interface RendererBackendResourceEvent {
	readonly resource: RendererBackendResourceEventResource;
	readonly action: RendererBackendResourceEventAction;
	readonly backend?: RenderBackendType;
	readonly reason?: string;
}

export type RenderBackendEvent =
	| {
			type: "device-lost";
			info?: RenderBackendDeviceLostInfo;
	  }
	| { type: "device-restored" }
	| { type: "render-invalidated"; reason: RenderDirtyReason }
	| { type: "resource-lifecycle"; event: RendererBackendResourceEvent };

export interface RenderBackendEventSink {
	emit(event: RenderBackendEvent): void;
}

export interface RenderBackendAttachContext {
	readonly surface: RenderSurface;
	readonly events: RenderBackendEventSink;
}

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
	initialize(): Promise<void>;
	/**
	 * Rebuilds backend device or graphics context resources after loss.
	 */
	restore(): void | Promise<void>;
	resize(size: RenderSurfaceSize): void;
	destroy(): void | Promise<void>;
	getAttachments(size: RenderSurfaceSize): FrameAttachments;
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
	executePass(pass: FramePass, context: FrameContext): void | Promise<void>;
	skipPass?(pass: FramePass): void;
	readRenderTargetColor?(
		id: string,
		attachmentIndex?: number,
		options?: RenderTargetReadbackOptions
	): Promise<TextureReadbackResult>;
	warmup?(
		context: FrameContext,
		options?: WarmupOptions
	): Promise<WarmupReport>;
	endFrame(): void | Promise<void>;
}
