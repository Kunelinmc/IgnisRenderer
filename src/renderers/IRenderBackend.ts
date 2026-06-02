import type {
	FrameAttachments,
	FrameContext,
	FramePass,
} from "../pipeline/types";
import type { EnvironmentIBLBakeOptions } from "../pipeline/EnvironmentIBLBaker";
import type { ShaderCompileError } from "../shaders/runtime";

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

export interface WarmupOptions {
	includeCorePasses?: boolean;
	includeShadowPass?: boolean;
	includePostProcess?: boolean;
	includeParticles?: boolean;
	includeEnvironmentIBLBake?: boolean;
	environmentIBLBake?: Omit<EnvironmentIBLBakeOptions, "onProgress">;
	logCompilationInfo?: boolean;
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
	clusteredLighting: boolean;
	oit: boolean;
}

export type RendererBackendResourceEventAction = "invalidate" | "destroy";
export type RendererBackendResourceEventResource = "postprocess" | (string & {});

export interface RendererBackendResourceEvent {
	readonly resource: RendererBackendResourceEventResource;
	readonly action: RendererBackendResourceEventAction;
	readonly backend?: RenderBackendType;
	readonly reason?: string;
}

/**
 * Presentation-only renderer host state exposed to backends.
 * Frame data must flow through `FrameContext`.
 */
export interface RendererBackendBridge {
	readonly canvas: Pick<HTMLCanvasElement, "width" | "height">;
	pixels?: Uint8ClampedArray | null;
	/**
	 * Notifies the renderer that the backend observed device/context loss.
	 */
	onDeviceLost?(info?: RenderBackendDeviceLostInfo): void | Promise<void>;
	/**
	 * Notifies the renderer that backend-owned resources changed lifetime.
	 *
	 * @param event Backend resource event to handle.
	 * @returns Nothing.
	 * @sideEffects May invalidate or destroy renderer-owned resources that
	 * reference backend handles.
	 */
	onBackendResourceEvent?(event: RendererBackendResourceEvent): void;
}

export interface IRenderBackend {
	readonly type: RenderBackendType;
	readonly capabilities: BackendCapabilities;
	readonly frameScheduling: FrameSchedulingMode;
	setRenderer?(renderer: RendererBackendBridge): void;
	init(canvas: HTMLCanvasElement): Promise<void>;
	/**
	 * Marks backend device or graphics context resources as lost.
	 */
	onDeviceLost?(info?: RenderBackendDeviceLostInfo): void | Promise<void>;
	/**
	 * Rebuilds backend device or graphics context resources after loss.
	 */
	restore?(canvas?: HTMLCanvasElement): void | Promise<void>;
	resize(width: number, height: number): void;
	destroy?(): void;
	getAttachments(width: number, height: number): FrameAttachments;
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
	abortFrame?(error?: unknown): void | Promise<void>;
	executeSharedPass?(
		pass: FramePass,
		context: FrameContext
	): void | Promise<void>;
	executePass(pass: FramePass, context: FrameContext): void | Promise<void>;
	skipPass?(pass: FramePass): void;
	warmup?(
		context: FrameContext,
		options?: WarmupOptions
	): Promise<WarmupReport>;
	endFrame(): void | Promise<void>;
}
