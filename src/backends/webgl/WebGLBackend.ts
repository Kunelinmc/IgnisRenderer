import {
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
	type FrameContext,
	type FramePass,
	type FrameAttachments,
} from "../../pipeline/types";
import { DefaultParticleSimulator } from "../../simulation/particles/DefaultParticleSimulator";
import type {
	IRenderBackend,
	RenderBackendDebugInfo,
	RenderBackendDeviceLostInfo,
	RenderBackendAttachContext,
	RenderBackendCompletedFrameCoverage,
	RenderBackendProfile,
	RenderSurfaceSize,
	WarmupOptions,
	WarmupReport,
} from "../IRenderBackend";
import type {
	RenderTargetReadbackOptions,
	RenderTargetReadbackResult,
} from "../../rendering/CustomRenderTargets";
import type { WebGLFrameServiceOwner } from "./WebGLFrameServiceOwner";
import { WebGLContextServiceOwner } from "./WebGLContextServiceOwner";
import { WebGLBackendExtensionOwner } from "./WebGLBackendExtensions";
import { WebGLFrameGraphRuntime } from "./rendergraph/WebGLFrameGraphRuntime";
import { WebGLPostProcessExecutor } from "./WebGLPostProcessExecutor";
import { BackendPostProcessRuntime } from "../../postprocess/BackendPostProcessRuntime";
import type { PostProcessPlan } from "../../postprocess/PostProcessPlanner";
import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_POINT_LIGHTS,
	MAX_SPOT_LIGHTS,
} from "../constants";
import {
	ShaderBackendCompileStage,
	DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
	ShaderRuntime,
} from "../../shaders/runtime";
import type {
	ShaderDirectiveCompileHook,
	ShaderRuntimeMode,
} from "../../shaders/runtime";
import {
	ShaderSource,
	WEBGL_SHADER_PARTS,
} from "../../shaders/ShaderSource";
import {
	addWarmupPhase,
	buildWarmupPlan,
	createWarmupReport,
	finalizeWarmupReport,
	toShaderCompileError,
	type WarmupPhaseCounters,
	type WarmupPostProcessPlan,
} from "../../pipeline/WarmupPlanner";
import { Logger } from "../../foundation/Logger";
import { WebGLContextWorkError } from "../../foundation/Error";
import { assertWebGLHDRCapabilities } from "./WebGLHDRCapabilities";
import {
	FramePassPlanValidator,
	type FramePassPlanValidatorState,
} from "../../pipeline/FramePassPlanValidator";
import type { RenderBackendExtensionRegistry } from "../BackendExtensions";
import { WebGLContextWorkQueue } from "./WebGLContextWorkQueue";
import {
	DEFAULT_DISPLAY_OUTPUT_OPTIONS,
	displayOutputStatesEqual,
	resolveSDROnlyDisplayOutput,
	type DisplayOutputOptions,
	type DisplayOutputState,
} from "../../rendering/DisplayOutput";

const MAX_PARTICLE_SIM_DELTA_TIME_SECONDS = 0.5;
const WEBGL_DEBUG_INFO_UNINITIALIZED: RenderBackendDebugInfo = {
	backend: "webgl",
	api: "webgl2",
	available: false,
	unavailableReason: "WebGL backend has not been initialized.",
};
const WEBGL_DEBUG_LIMIT_KEYS = [
	"MAX_TEXTURE_SIZE",
	"MAX_RENDERBUFFER_SIZE",
	"MAX_TEXTURE_IMAGE_UNITS",
	"MAX_VERTEX_TEXTURE_IMAGE_UNITS",
	"MAX_COMBINED_TEXTURE_IMAGE_UNITS",
	"MAX_DRAW_BUFFERS",
	"MAX_COLOR_ATTACHMENTS",
] as const;

export interface WebGLBackendOptions {
	shaderMode?: ShaderRuntimeMode;
	directiveHook?: ShaderDirectiveCompileHook | null;
	validatePrograms?: boolean;
	enableEarlyZPrepass?: boolean;
}

export class WebGLBackend implements IRenderBackend {
	private readonly _postProcessRuntime = new BackendPostProcessRuntime({
		executor: new WebGLPostProcessExecutor({
			getDeviceServices: () => this._frameServices,
		}),
		backend: this,
		warn: (key, message) =>
			Logger.warn(`[${key}] ${message}`, {
				scope: "WebGLBackend",
				onceKey: key,
			}),
	});
	public readonly extensions: RenderBackendExtensionRegistry;
	public readonly profile: RenderBackendProfile = {
		id: "webgl",
		capabilities: {
			displayHDR: false,
			sh: true,
			shadows: true,
			reflection: false,
			environment: true,
			postProcess: true,
			meshParticles: false,
			clusteredLighting: true,
			oit: true,
			occlusionCulling: false,
			customRenderTargets: true,
			customRenderPasses: true,
			renderTargetReadback: true,
		},
		frameScheduling: "on-demand",
		lighting: { localizedProbeMode: "backend-local" },
	};

	private _attachContext: RenderBackendAttachContext | null = null;
	private _attached = false;
	private readonly _options: WebGLBackendOptions;
	private _canvas: HTMLCanvasElement | null = null;
	private _gl: WebGL2RenderingContext | null = null;
	private _contextServices: WebGLContextServiceOwner | null = null;
	private _frameGraphRuntime: WebGLFrameGraphRuntime | null = null;
	private _particleSimulator: DefaultParticleSimulator | null = null;
	private _activeContext: FrameContext | null = null;
	private _completedFrameCoverage: RenderBackendCompletedFrameCoverage = "full-frame";
	private _contextLost = false;
	private _contextLossHandler: ((event: Event) => void) | null = null;
	private _contextRestoreHandler: ((event: Event) => void) | null = null;
	private _width = 1;
	private _height = 1;
	public readonly shaderRuntime: ShaderRuntime;
	private _shaderCompileStage: ShaderBackendCompileStage;
	private _executedPasses = new Set<FramePass["stage"]>();
	private _plannedPasses = new Set<FramePass["stage"]>();
	private _plannedPassOrder = new Map<FramePass["stage"], number>();
	private readonly _framePlanner = new FramePassPlanValidator("WebGL");
	private _debugInfo: RenderBackendDebugInfo = WEBGL_DEBUG_INFO_UNINITIALIZED;
	private _displayOutputState = resolveSDROnlyDisplayOutput(
		DEFAULT_DISPLAY_OUTPUT_OPTIONS,
	);
	private readonly _contextWorkQueue =
		new WebGLContextWorkQueue<WebGLContextServiceOwner>({
		resolveServices: () => this._contextServices,
		restoreBaseline: (scope) => scope.services.restoreContextWorkBaseline(),
	});
	private readonly _extensionOwner: WebGLBackendExtensionOwner;

	private get _frameServices(): WebGLFrameServiceOwner | null {
		return this._contextServices?.frame ?? null;
	}

	constructor(options: WebGLBackendOptions = {}) {
		this._options = options;
		this._extensionOwner = new WebGLBackendExtensionOwner({
			contextWorkQueue: this._contextWorkQueue,
			resolveContextServices: () => this._contextServices,
		});
		this.extensions = this._extensionOwner.registry;
		const shaderMode = options.shaderMode ?? "warn";
		this.shaderRuntime = new ShaderRuntime({
			mode: shaderMode,
		});
		this._shaderCompileStage = new ShaderBackendCompileStage({
			backend: "webgl",
			runtime: this.shaderRuntime,
			profiles: DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
			hook: options.directiveHook ?? null,
			mode: shaderMode,
		});
		this._ensureParticleSimulator();
	}

	public attach(context: RenderBackendAttachContext): void {
		if (this._attached) {
			throw new Error("WebGLBackend is already attached to a renderer.");
		}
		this._attachContext = context;
		this._displayOutputState = resolveSDROnlyDisplayOutput(
			context.surface.displayOutput,
		);
		this._attached = true;
	}

	/**
	 * Reports whether the WebGL backend will run its internal Early Z pre-pass.
	 *
	 * @returns `true` when opaque WebGL frames use depth pre-pass optimization.
	 * @sideEffects None.
	 */
	public isEarlyZPrepassEnabled(): boolean {
		return this._options.enableEarlyZPrepass !== false;
	}

	/**
	 * Returns the current WebGL diagnostic snapshot.
	 *
	 * @returns WebGL adapter strings, limits, and extensions when initialized,
	 * otherwise an unavailable snapshot.
	 * @sideEffects None.
	 */
	public getDebugInfo(): RenderBackendDebugInfo {
		return this._debugInfo;
	}

	public getDisplayOutputState(): DisplayOutputState {
		return this._displayOutputState;
	}

	public async setDisplayOutput(
		options: DisplayOutputOptions,
	): Promise<DisplayOutputState> {
		const previous = this._displayOutputState;
		const current = resolveSDROnlyDisplayOutput(options, previous.requested);
		this._displayOutputState = current;
		if (current.fallbackReason === "backend-unsupported") {
			Logger.warn(
				"[display-hdr-unavailable] WebGLBackend supports SDR presentation only.",
				{ scope: "WebGLBackend", onceKey: "display-hdr-unavailable" },
			);
		}
		if (!displayOutputStatesEqual(previous, current)) {
			this._requireAttachContext().events.emit({
				type: "display-output-change",
				previous,
				current,
			});
		}
		return current;
	}

	public async initialize(): Promise<void> {
		const canvas = this._requireAttachContext().surface.canvas;
		if (this._displayOutputState.fallbackReason === "backend-unsupported") {
			Logger.warn(
				"[display-hdr-unavailable] WebGLBackend supports SDR presentation only.",
				{ scope: "WebGLBackend", onceKey: "display-hdr-unavailable" },
			);
		}
		this._ensureParticleSimulator();
		this._canvas = canvas;
		this._installContextLifecycleListeners(canvas);
		await ShaderSource.prepareMany([
			...WEBGL_SHADER_PARTS.flatMap((part) => [
				{ key: `webgl.part.${part}.raw` as const },
				{ key: `webgl.part.${part}.composite` as const },
			]),
			{
				key: "webgl.scene.raw" as const,
				params: {
					limits: {
						maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
						maxPointLights: MAX_POINT_LIGHTS,
						maxSpotLights: MAX_SPOT_LIGHTS,
						enableShadowTransmittance: false,
					},
				},
			},
			{
				key: "webgl.scene.composite" as const,
				params: {
					limits: {
						maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
						maxPointLights: MAX_POINT_LIGHTS,
						maxSpotLights: MAX_SPOT_LIGHTS,
						enableShadowTransmittance: false,
					},
				},
			},
			{
				key: "webgl.scene.raw" as const,
				params: {
					limits: {
						maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
						maxPointLights: MAX_POINT_LIGHTS,
						maxSpotLights: MAX_SPOT_LIGHTS,
						enableShadowTransmittance: true,
					},
				},
			},
			{
				key: "webgl.scene.composite" as const,
				params: {
					limits: {
						maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
						maxPointLights: MAX_POINT_LIGHTS,
						maxSpotLights: MAX_SPOT_LIGHTS,
						enableShadowTransmittance: true,
					},
				},
			},
			{
				key: "webgl.scene.raw" as const,
				params: {
					limits: {
						maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
						maxPointLights: MAX_POINT_LIGHTS,
						maxSpotLights: MAX_SPOT_LIGHTS,
						enableShadowTransmittance: false,
						enableIrradianceProbeGrid: true,
					},
				},
			},
			{
				key: "webgl.scene.composite" as const,
				params: {
					limits: {
						maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
						maxPointLights: MAX_POINT_LIGHTS,
						maxSpotLights: MAX_SPOT_LIGHTS,
						enableShadowTransmittance: false,
						enableIrradianceProbeGrid: true,
					},
				},
			},
			{
				key: "webgl.scene.raw" as const,
				params: {
					limits: {
						maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
						maxPointLights: MAX_POINT_LIGHTS,
						maxSpotLights: MAX_SPOT_LIGHTS,
						enableShadowTransmittance: true,
						enableIrradianceProbeGrid: true,
					},
				},
			},
			{
				key: "webgl.scene.composite" as const,
				params: {
					limits: {
						maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
						maxPointLights: MAX_POINT_LIGHTS,
						maxSpotLights: MAX_SPOT_LIGHTS,
						enableShadowTransmittance: true,
						enableIrradianceProbeGrid: true,
					},
				},
			},
		]);
		this._initializeGLContext(canvas);
	}

	/**
	 * Marks WebGL context resources as lost.
	 *
	 * @internal Backend lifecycle hook used by `webglcontextlost` handling and
	 * renderer recovery paths.
	 */
	public onDeviceLost(info?: RenderBackendDeviceLostInfo): void {
		if (this._contextLost) {
			return;
		}
		this._contextLost = true;
		this._contextWorkQueue.suspend();
		this._frameGraphRuntime?.abortGraphAnalysis?.(info);
		const detail =
			typeof info?.message === "string" && info.message.length > 0 ? `: ${info.message}` : "";
		Logger.warn(
			`WebGL context was lost${detail}. Rendering is paused until context restoration.`,
			{ scope: "WebGLBackend" },
		);
	}

	public restore(): void {
		const targetCanvas = this._requireAttachContext().surface.canvas;
		this._ensureParticleSimulator();
		this._canvas = targetCanvas;
		this._installContextLifecycleListeners(targetCanvas);
		this._initializeGLContext(targetCanvas);
	}

	public resize(size: RenderSurfaceSize): void {
		const { width, height } = size;
		this._width = toSafeDimension(width);
		this._height = toSafeDimension(height);
		if (this._contextLost || !this._frameServices) return;
		void this._contextWorkQueue.enqueueMaintenance({
			key: "resize",
			label: "frame-resize",
			contextLossPolicy: "reject",
			execute: (scope) => {
				this._postProcessRuntime.invalidateFrameSized();
				scope.services.frame.resize(this._width, this._height);
			},
		}).catch((error) => {
			if (
				error instanceof WebGLContextWorkError &&
				(error.code === "context-lost" || error.code === "destroyed")
			) {
				return;
			}
			Logger.warn(`WebGL deferred resize failed: ${String(error)}`, {
				scope: "WebGLBackend",
			});
		});
	}

	public getAttachments(size: RenderSurfaceSize): {
		width: number;
		height: number;
	} {
		const { width, height } = size;
		return {
			width: toSafeDimension(width),
			height: toSafeDimension(height),
		};
	}

	public beginFrame(context: FrameContext): Promise<void> {
		this._completedFrameCoverage = "full-frame";
		if (!this._frameServices || !this._frameGraphRuntime) {
			return Promise.reject(new Error("WebGL backend has not been initialized."));
		}
		return this._contextWorkQueue.beginFrame("frame-begin", async () => {
			this._executedPasses.clear();
			this._activeContext = context;
			this._prepareFramePassPlan(context);
			await this._frameServices!.warmupCoordinator?.prepareFrameSources?.(context);
			this._particleSimulator?.beginFrame(context);
			this._frameGraphRuntime!.beginFrame(context);
		});
	}

	public executePass(pass: FramePass, context: FrameContext): Promise<void> {
		if (!this._frameServices || !this._frameGraphRuntime) {
			return Promise.reject(new Error("WebGL backend has not been initialized."));
		}
		return this._contextWorkQueue.runFramePass(`frame-pass:${pass.stage}`, async () => {
			this._validatePassDependencies(pass);
			if (pass.stage === "particle-sim") {
				this._frameGraphRuntime!.recordOpaqueGraphStage?.(
					pass.stage,
					"Particle simulation executes outside the logical frame graph.",
				);
				this._particleSimulator?.simulate(
					context,
					this._resolveParticleDeltaTime(context),
				);
				this._particleSimulator?.emitRenderBatches(context);
				this._markPassExecuted(pass.stage);
				return;
			}
			await this._frameGraphRuntime!.executePass(pass, context);
			this._markPassExecuted(pass.stage);
		});
	}

	public skipPass(pass: FramePass): void {
		this._markPassExecuted(pass.stage);
	}

	public readRenderTargetColor(
		id: string,
		attachmentIndex?: number,
		options?: RenderTargetReadbackOptions,
	): Promise<RenderTargetReadbackResult> {
		return this._contextWorkQueue.enqueue({
			label: `render-target-readback:${id}`,
			framePolicy: "idle-only",
			contextLossPolicy: "reject",
			execute: (scope) =>
				scope.services.frame.readCustomRenderTargetColor(
					id,
					attachmentIndex,
					options,
				),
		});
	}

	public endFrame(): Promise<void> {
		if (!this._frameServices || !this._frameGraphRuntime) {
			return Promise.reject(new Error("WebGL backend has not been initialized."));
		}
		const context = this._activeContext;
		if (!context) {
			return Promise.reject(
				new Error("WebGL backend cannot end a frame before beginFrame."),
			);
		}
		return this._contextWorkQueue.endFrame("frame-end", async () => {
			try {
				await this._frameGraphRuntime!.endFrame(context);
				this._particleSimulator?.endFrame();
				this._postProcessRuntime.commitFrame();
				this._frameGraphRuntime!.commitGraphAnalysis?.();
				this._frameServices?.commitTemporalFrame?.();
				this._activeContext = null;
			} catch (error) {
				this._frameGraphRuntime!.abortGraphAnalysis?.(error);
				throw error;
			}
		});
	}

	/** @internal Renderer frame-coordination coverage report. */
	public getCompletedFrameCoverage(): RenderBackendCompletedFrameCoverage {
		return this._completedFrameCoverage;
	}

	public async abortFrame(_error?: unknown): Promise<void> {
		await this._contextWorkQueue.abortFrame("frame-abort", async () => {
			let abortError: unknown = null;
			try {
				await this._postProcessRuntime.abortFrame(_error);
			} catch (error) {
				abortError = error;
			}
			try {
				this._frameGraphRuntime?.abortFrame(_error);
			} catch (error) {
				abortError ??= error;
			}
			try {
				this._particleSimulator?.endFrame();
			} catch (error) {
				abortError ??= error;
			}
			this._frameGraphRuntime?.abortGraphAnalysis?.(_error);
			this._activeContext = null;
			this._executedPasses.clear();
			this._plannedPasses.clear();
			this._plannedPassOrder.clear();
			if (abortError) throw abortError;
		});
	}

	public async warmup(context: FrameContext, options: WarmupOptions = {}): Promise<WarmupReport> {
		const report = createWarmupReport(this.profile.id);
		if (!this._frameServices) {
			throw new Error("WebGL backend has not been initialized.");
		}
		let warmupPostProcessPlan: WarmupPostProcessPlan | undefined;
		let postProcessPlan: PostProcessPlan | undefined;
		if (options.includePostProcess !== false) {
			const graph = this._postProcessRuntime.planWarmup(context);
			postProcessPlan = graph;
			warmupPostProcessPlan = {
				passIds: graph.orderedPasses.map((pass) => pass.id),
				descriptors: graph.orderedPasses.map((pass) => pass.pass),
			};
		}
		const plan = buildWarmupPlan(context, options, warmupPostProcessPlan);
		return this._contextWorkQueue.enqueue({
			label: "webgl-warmup",
			framePolicy: "idle-deferred",
			contextLossPolicy: "reject",
			execute: async (scope) => {
				try {
					const phase = await scope.services.frame.warmupCoordinator.warmup(
						context,
						plan,
						options,
						postProcessPlan,
						scope.signal,
					);
					addWarmupPhase(report, phase);
					this._reportWarmupProgress(options, phase);
				} catch (error) {
					const failedPhase = {
						phase: "webgl-warmup",
						total: 1,
						compiled: 0,
						skipped: 0,
						failed: 1,
						errors: [toShaderCompileError(error, this.profile.id, "WebGLWarmup")],
					};
					addWarmupPhase(report, failedPhase);
					this._reportWarmupProgress(options, failedPhase);
				}
				return finalizeWarmupReport(report);
			},
		});
	}

	public destroy(): void {
		this._contextWorkQueue.destroy();
		this._postProcessRuntime.destroy();
		this._contextServices?.destroy();
		this._contextServices = null;
		this._frameGraphRuntime = null;
		this._particleSimulator = null;
		this._gl = null;
		this._activeContext = null;
		this._debugInfo = WEBGL_DEBUG_INFO_UNINITIALIZED;

		if (this._canvas) {
			if (this._contextLossHandler) {
				this._canvas.removeEventListener("webglcontextlost", this._contextLossHandler);
			}
			if (this._contextRestoreHandler) {
				this._canvas.removeEventListener(
					"webglcontextrestored",
					this._contextRestoreHandler,
				);
			}
		}
		this._contextLossHandler = null;
		this._contextRestoreHandler = null;
	}

	private _ensureParticleSimulator(): void {
		if (this._particleSimulator) {
			return;
		}
		this._particleSimulator = new DefaultParticleSimulator({
			backendTag: this.profile.id,
		});
	}

	private _initializeGLContext(canvas: HTMLCanvasElement): void {
		const gl = canvas.getContext("webgl2", {
			alpha: false,
			antialias: true,
			depth: true,
			stencil: false,
			premultipliedAlpha: false,
			preserveDrawingBuffer: false,
			powerPreference: "high-performance",
		}) as WebGL2RenderingContext | null;
		if (!gl) {
			throw new Error("Failed to acquire WebGL2 context. WebGL backend requires WebGL2.");
		}
		assertWebGLHDRCapabilities(gl);

		this._gl = gl;
		this._postProcessRuntime.destroy();
		this._contextServices?.destroy();
		this._contextServices = new WebGLContextServiceOwner(
			gl,
			this.shaderRuntime,
			this._shaderCompileStage,
			{
				validatePrograms: this._options.validatePrograms === true,
				enableEarlyZPrepass: this._options.enableEarlyZPrepass !== false,
				onProgramCompilePending: () => this._emitProgramCompilePendingEvent(),
				onTextureUploadPending: () => this._emitTextureUploadPendingEvent(),
				postProcessRuntime: this._postProcessRuntime,
			},
		);
		const frameServices = this._contextServices.frame;
		this._frameGraphRuntime = new WebGLFrameGraphRuntime(
			frameServices,
			this._postProcessRuntime,
		);
		this._contextLost = false;
		frameServices.resize(this._width, this._height);
		this._debugInfo = this._createDebugInfo(gl);
		this._contextWorkQueue.bindContext();
	}

	private _createDebugInfo(gl: WebGL2RenderingContext): RenderBackendDebugInfo {
		const debugInfo = getWebGLDebugRendererInfo(gl);
		const raw: Record<string, string | number | boolean> = {};
		let vendor = debugInfo?.vendor;
		let renderer = debugInfo?.renderer;

		if (vendor) {
			raw.unmaskedVendor = vendor;
		} else {
			vendor = getWebGLStringParameter(gl, gl.VENDOR);
			if (vendor) raw.vendor = vendor;
		}
		if (renderer) {
			raw.unmaskedRenderer = renderer;
		} else {
			renderer = getWebGLStringParameter(gl, gl.RENDERER);
			if (renderer) raw.renderer = renderer;
		}

		const device =
			vendor || renderer || Object.keys(raw).length > 0
				? {
						vendor,
						renderer,
						raw: Object.keys(raw).length > 0 ? raw : undefined,
					}
				: undefined;

		return {
			backend: "webgl",
			api: "webgl2",
			available: true,
			device,
			limits: collectWebGLLimits(gl),
			features: collectWebGLExtensions(gl),
		};
	}

	private _reportWarmupProgress(options: WarmupOptions, phase: WarmupPhaseCounters): void {
		options.onProgress?.({
			phase: phase.phase,
			completed: phase.compiled + phase.skipped + phase.failed,
			total: phase.total,
		});
	}

	private _emitProgramCompilePendingEvent(): void {
		this._requireAttachContext().events.emit({
			type: "render-invalidated",
			reason: "postfx",
		});
	}

	private _emitTextureUploadPendingEvent(): void {
		this._requireAttachContext().events.emit({
			type: "render-invalidated",
			reason: "texture",
		});
	}

	private _resolveParticleDeltaTime(context: FrameContext): number {
		const value = context.transient.get(PARTICLE_SIM_DELTA_TIME_SECONDS_KEY);
		if (typeof value !== "number" || !Number.isFinite(value)) {
			return 0;
		}
		return Math.min(MAX_PARTICLE_SIM_DELTA_TIME_SECONDS, Math.max(0, value));
	}

	private _installContextLifecycleListeners(canvas: HTMLCanvasElement): void {
		if (this._contextLossHandler || this._contextRestoreHandler) {
			return;
		}
		this._contextLossHandler = (event: Event) => {
			const webglEvent = event as WebGLContextEvent;
			webglEvent.preventDefault?.();
			const info = {
				reason: "context-lost",
				message: webglEvent.statusMessage,
			};
			this.onDeviceLost(info);
			this._requireAttachContext().events.emit({ type: "device-lost", info });
		};
		this._contextRestoreHandler = () => {
			Logger.warn("WebGL context was restored. Rebuilding WebGL resources.", {
				scope: "WebGLBackend",
			});
			try {
				this.restore();
				this._requireAttachContext().events.emit({ type: "device-restored" });
			} catch (error) {
				Logger.warn(`WebGL context restore failed: ${String(error)}`, {
					scope: "WebGLBackend",
				});
			}
		};

		canvas.addEventListener("webglcontextlost", this._contextLossHandler);
		canvas.addEventListener("webglcontextrestored", this._contextRestoreHandler);
	}

	private _prepareFramePassPlan(context: FrameContext): void {
		this._framePlanner.preparePlan(context, this._getFramePlannerState());
	}

	private _validatePassDependencies(pass: FramePass): void {
		this._framePlanner.validatePassDependencies(pass, this._getFramePlannerState(), {
			reportNonFatalError: (scope, error) =>
				Logger.warn(`[${scope}] ${String(error)}`, {
					scope: "WebGLBackend",
				}),
		});
	}

	private _markPassExecuted(stage: FramePass["stage"]): void {
		this._framePlanner.markPassExecuted(stage, this._getFramePlannerState());
	}

	private _getFramePlannerState(): FramePassPlanValidatorState {
		return {
			executedPasses: this._executedPasses,
			plannedPasses: this._plannedPasses,
			plannedPassOrder: this._plannedPassOrder,
		};
	}

	public getFrameGraphDebugState(): unknown {
		return this._frameGraphRuntime?.getDebugState() ?? null;
	}

	private _requireAttachContext(): RenderBackendAttachContext {
		if (!this._attachContext) {
			throw new Error("WebGLBackend.attach() must be called before initialize().");
		}
		return this._attachContext;
	}
}

function toSafeDimension(value: unknown): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return 1;
	}
	return Math.max(1, Math.floor(value));
}

function getWebGLDebugRendererInfo(
	gl: WebGL2RenderingContext
): { vendor?: string; renderer?: string } | null {
	if (typeof gl.getExtension !== "function") {
		return null;
	}
	let debugInfo: {
		readonly UNMASKED_VENDOR_WEBGL: number;
		readonly UNMASKED_RENDERER_WEBGL: number;
	} | null = null;
	try {
		debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
	} catch {
		return null;
	}
	if (!debugInfo) {
		return null;
	}
	return {
		vendor: getWebGLStringParameter(gl, debugInfo.UNMASKED_VENDOR_WEBGL),
		renderer: getWebGLStringParameter(gl, debugInfo.UNMASKED_RENDERER_WEBGL),
	};
}

function getWebGLStringParameter(
	gl: WebGL2RenderingContext,
	parameter: number | undefined
): string | undefined {
	if (typeof parameter !== "number" || typeof gl.getParameter !== "function") {
		return undefined;
	}
	try {
		const value = gl.getParameter(parameter);
		return typeof value === "string" && value.length > 0 ? value : undefined;
	} catch {
		return undefined;
	}
}

function collectWebGLLimits(gl: WebGL2RenderingContext): Record<string, number> {
	const limits: Record<string, number> = {};
	for (const key of WEBGL_DEBUG_LIMIT_KEYS) {
		const parameter = gl[key];
		if (typeof parameter !== "number" || typeof gl.getParameter !== "function") {
			continue;
		}
		try {
			const value = gl.getParameter(parameter);
			if (typeof value === "number" && Number.isFinite(value)) {
				limits[key] = value;
			}
		} catch {
			continue;
		}
	}
	return limits;
}

function collectWebGLExtensions(gl: WebGL2RenderingContext): readonly string[] {
	if (typeof gl.getSupportedExtensions !== "function") {
		return [];
	}
	try {
		return [...(gl.getSupportedExtensions() ?? [])].sort();
	} catch {
		return [];
	}
}
