import type {
	FrameAttachments,
	FrameContext,
	FramePass,
} from "../pipeline/types";
import type {
	IPostProcessExecutor,
	PostProcessBackendKind,
	PostProcessBackendSupport,
} from "../postprocess";
import type { PostProcessPassRegistry } from "../postprocess/PostProcessPass";
import type { EnvironmentIBLBakeOptions } from "../pipeline/EnvironmentIBLBaker";
import type { ShaderCompileError } from "../shaders/runtime";

export type KnownBackendType = "software" | "webgpu" | "webgl";
export type RenderBackendType = KnownBackendType | (string & {});
export type FrameSchedulingMode = "always" | "on-demand";

export interface RenderBackendDeviceLostInfo {
	reason?: string;
	message?: string;
}

export type PassExecutorMap = Partial<
	Record<FramePass["stage"], FramePass["executor"]>
>;

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

/**
 * Presentation-only renderer host state exposed to backends.
 * Frame data must flow through `FrameContext`.
 */
export interface RendererBackendBridge {
	readonly canvas: Pick<HTMLCanvasElement, "width" | "height">;
	readonly postProcess?: PostProcessPassRegistry;
	pixels?: Uint8ClampedArray | null;
	/**
	 * Releases renderer-owned post-process resources for a backend reset.
	 *
	 * @param backend Backend kind whose pass implementations must be destroyed.
	 * @param executor Executor that owns pipeline-created resource handles.
	 * @returns Nothing.
	 * @sideEffects Destroys post-process history, transient, and pass resources.
	 */
	destroyPostProcessResources?(
		backend: PostProcessBackendKind,
		executor: IPostProcessExecutor
	): void;
}

export interface IRenderBackend {
	readonly type: RenderBackendType;
	readonly capabilities: BackendCapabilities;
	readonly frameScheduling: FrameSchedulingMode;
	readonly passExecutors?: PassExecutorMap;
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

/**
 * Render backend contract required by `Renderer`.
 *
 * Core backend implementations may satisfy `IRenderBackend` without exposing
 * post-process execution support, but `Renderer` requires this combined
 * contract to run its renderer-owned logical `postprocess` stage.
 */
export type PostProcessCapableRenderBackend =
	IRenderBackend & PostProcessBackendSupport;
