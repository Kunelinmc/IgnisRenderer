import {
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
	type FrameContext,
	type FramePass,
} from "../pipeline/types";
import { DefaultParticleSimulator } from "../simulation/particles/DefaultParticleSimulator";
import type {
	IRenderBackend,
	RenderBackendDeviceLostInfo,
	RendererBackendBridge,
	WarmupOptions,
	WarmupReport,
} from "./IRenderBackend";
import type {
	IPostProcessExecutor,
	LogicalGBufferBridge,
	PostProcessBackendAdapter,
	PostProcessPassExecutionContextRequest,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "../postprocess";
import {
	registerPostProcessBackendAdapter,
	unregisterPostProcessBackendAdapter,
} from "../postprocess";
import { WebGLFrameExecutor } from "./webgl/WebGLFrameExecutor";
import {
	ShaderBackendCompileStage,
	DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
	ShaderRuntime,
} from "../shaders/runtime";
import type {
	ShaderDirectiveCompileHook,
	ShaderRuntimeMode,
} from "../shaders/runtime";
import {
	createWebGLShaderSourceFactory,
	type WebGLShaderSourceFactory,
} from "../shaders/webgl/WebGLShaderSourceFactory";
import {
	addWarmupPhase,
	buildWarmupPlan,
	createWarmupReport,
	finalizeWarmupReport,
	toShaderCompileError,
} from "../pipeline/WarmupPlanner";
import { Logger } from "../foundation/Logger";
import {
	FramePassPlanValidator,
	type FramePassPlanValidatorState,
} from "../pipeline/FramePassPlanValidator";

const SUPPORTED_WEBGL_STAGES: readonly FramePass["stage"][] = [
	"shadow",
	"main-opaque",
	"main-transparent",
	"particles",
] as const;
const MAX_PARTICLE_SIM_DELTA_TIME_SECONDS = 0.5;

export interface WebGLBackendOptions {
	shaderMode?: ShaderRuntimeMode;
	directiveHook?: ShaderDirectiveCompileHook | null;
}

type WebGLBackendPassHandler = (
	pass: FramePass,
	context: FrameContext
) => void;

export class WebGLBackend implements IRenderBackend {
	public readonly type = "webgl";
	public readonly frameScheduling = "on-demand";
	public readonly passExecutors = {
		"animation-sim": "shared",
		"particle-sim": "backend",
	} as const;
	public readonly capabilities = {
		sh: true,
		shadows: true,
		reflection: false,
		environment: true,
		clusteredLighting: true,
		oit: true,
	};
	private readonly _postProcessExecutor: IPostProcessExecutor = {
		backend: "webgl",
		createResource: (desc) => this._createPostProcessResource(desc),
		destroyResource: (handle) => this._destroyPostProcessResource(handle),
		getPassExecutionContext: (request) =>
			this._getPostProcessPassExecutionContext(request),
		executePass: (passId, request) =>
			this._executePostProcessPass(passId, request),
	};
	private readonly _postProcessAdapter: PostProcessBackendAdapter = {
		backend: "webgl",
		executor: this._postProcessExecutor,
		createGBufferBridge: (context) => this._createPostProcessGBuffer(context),
	};

	private _canvas: HTMLCanvasElement | null = null;
	private _gl: WebGL2RenderingContext | null = null;
	private _frameExecutor: WebGLFrameExecutor | null = null;
	private _particleSimulator: DefaultParticleSimulator | null = null;
	private _onBackendResourceEvent:
		| RendererBackendBridge["onBackendResourceEvent"]
		| null = null;
	private _onDeviceLost:
		| RendererBackendBridge["onDeviceLost"]
		| null = null;
	private _contextLost = false;
	private _contextLossHandler: ((event: Event) => void) | null = null;
	private _contextRestoreHandler: ((event: Event) => void) | null = null;
	private _width = 1;
	private _height = 1;
	public readonly shaderRuntime: ShaderRuntime;
	private _shaderCompileStage: ShaderBackendCompileStage;
	private _shaderSourceFactory: WebGLShaderSourceFactory;
	private _executedPasses = new Set<FramePass["stage"]>();
	private _plannedPasses = new Set<FramePass["stage"]>();
	private _plannedPassOrder = new Map<FramePass["stage"], number>();
	private readonly _framePlanner = new FramePassPlanValidator("WebGL");
	private readonly _passHandlers: Map<
		FramePass["stage"],
		WebGLBackendPassHandler
	>;

	constructor(options: WebGLBackendOptions = {}) {
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
		this._shaderSourceFactory = createWebGLShaderSourceFactory();
		this._passHandlers = this._createPassHandlers();
		this._ensureParticleSimulator();
		registerPostProcessBackendAdapter(this, this._postProcessAdapter);
	}

	public setRenderer(renderer: RendererBackendBridge): void {
		this._onBackendResourceEvent =
			renderer.onBackendResourceEvent?.bind(renderer) ?? null;
		this._onDeviceLost = renderer.onDeviceLost?.bind(renderer) ?? null;
		this._ensureParticleSimulator();
	}

	private _createPostProcessResource(
		desc: PostProcessResourceDescriptor
	): PostProcessResourceHandle {
		if (!this._frameExecutor) {
			throw new Error(
				"WebGL frame executor is not initialized; cannot create post-process resource."
			);
		}
		return this._frameExecutor.createPostProcessResource(desc);
	}

	private _destroyPostProcessResource(handle: PostProcessResourceHandle): void {
		this._frameExecutor?.destroyPostProcessResource(handle);
	}

	private _createPostProcessGBuffer(context: FrameContext): LogicalGBufferBridge {
		return this._frameExecutor?.createGBufferBridge(context) ?? {
			width: Math.max(1, context.attachments.width),
			height: Math.max(1, context.attachments.height),
			normalSpace: "world",
			depthEncoding: "hardware",
			channels: {},
			worldPosition: {
				source: "derived",
				available: false,
			},
		};
	}

	private _executePostProcessPass(
		passId: string,
		request: PostProcessPassRequest
	): PostProcessPassResult {
		return (
			this._frameExecutor?.executePostProcessPass(passId, request) ??
			{ ran: false }
		);
	}

	private _getPostProcessPassExecutionContext(
		request: PostProcessPassExecutionContextRequest
	): unknown {
		return this._frameExecutor?.getPassExecutionContext(request);
	}

	public async init(canvas: HTMLCanvasElement): Promise<void> {
		this._ensureParticleSimulator();
		this._canvas = canvas;
		this._installContextLifecycleListeners(canvas);
		await this._shaderSourceFactory.prepareAll();
		this._initializeGLContext(canvas);
	}

	public onDeviceLost(info?: RenderBackendDeviceLostInfo): void {
		if (this._contextLost) {
			return;
		}
		this._contextLost = true;
		const detail =
			typeof info?.message === "string" && info.message.length > 0
				? `: ${info.message}`
				: "";
		Logger.warn(
			`WebGL context was lost${detail}. Rendering is paused until context restoration.`,
			{ scope: "WebGLBackend" }
		);
	}

	public restore(canvas?: HTMLCanvasElement): void {
		const targetCanvas = this._resolveRestoreCanvas(canvas);
		this._ensureParticleSimulator();
		this._canvas = targetCanvas;
		this._installContextLifecycleListeners(targetCanvas);
		this._initializeGLContext(targetCanvas);
	}

	public resize(width: number, height: number): void {
		this._width = toSafeDimension(width);
		this._height = toSafeDimension(height);
		if (this._contextLost) return;
		this._emitPostProcessResourceEvent("invalidate", "resize");
		this._frameExecutor?.resize(this._width, this._height);
	}

	public getAttachments(
		width: number,
		height: number
	): {
		width: number;
		height: number;
	} {
		return {
			width: toSafeDimension(width),
			height: toSafeDimension(height),
		};
	}

	public beginFrame(context: FrameContext): void {
		if (!this._frameExecutor) {
			throw new Error("WebGL backend has not been initialized.");
		}
		if (this._contextLost) {
			return;
		}
		this._executedPasses.clear();
		this._prepareFramePassPlan(context);
		this._particleSimulator?.beginFrame(context);
		this._frameExecutor.beginFrame(context);
	}

	public executePass(pass: FramePass, context: FrameContext): void {
		if (!this._frameExecutor) {
			throw new Error("WebGL backend has not been initialized.");
		}
		if (this._contextLost) {
			return;
		}
		const handler = this._passHandlers.get(pass.stage);
		if (!handler) {
			Logger.warn(
				`WebGL backend does not support pass "${pass.stage}" yet; skipping`,
				{ scope: "WebGLBackend" }
			);
			this._markPassExecuted(pass.stage);
			return;
		}
		this._validatePassDependencies(pass);
		handler(pass, context);
		this._markPassExecuted(pass.stage);
	}

	public skipPass(pass: FramePass): void {
		this._markPassExecuted(pass.stage);
	}

	public endFrame(): void {
		if (!this._frameExecutor || this._contextLost) {
			return;
		}
		this._frameExecutor.endFrame();
		this._particleSimulator?.endFrame();
	}

	public abortFrame(_error?: unknown): void {
		if (this._contextLost) {
			return;
		}
		this._frameExecutor?.abortFrame();
		this._particleSimulator?.endFrame();
		this._executedPasses.clear();
		this._plannedPasses.clear();
		this._plannedPassOrder.clear();
	}

	public async warmup(
		context: FrameContext,
		options: WarmupOptions = {}
	): Promise<WarmupReport> {
		const report = createWarmupReport(this.type);
		if (!this._frameExecutor) {
			throw new Error("WebGL backend has not been initialized.");
		}
		const plan = buildWarmupPlan(context, options);
		try {
			const phase = this._frameExecutor.warmup(context, plan);
			addWarmupPhase(report, phase);
		} catch (error) {
			addWarmupPhase(report, {
				phase: "webgl-warmup",
				total: 1,
				compiled: 0,
				skipped: 0,
				failed: 1,
				errors: [toShaderCompileError(error, this.type, "WebGLWarmup")],
			});
		}
		return finalizeWarmupReport(report);
	}

	public destroy(): void {
		this._emitPostProcessResourceEvent("destroy", "destroy");
		this._frameExecutor?.destroy();
		this._frameExecutor = null;
		this._particleSimulator = null;
		this._gl = null;
		unregisterPostProcessBackendAdapter(this);

		if (this._canvas) {
			if (this._contextLossHandler) {
				this._canvas.removeEventListener(
					"webglcontextlost",
					this._contextLossHandler
				);
			}
			if (this._contextRestoreHandler) {
				this._canvas.removeEventListener(
					"webglcontextrestored",
					this._contextRestoreHandler
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
			backendTag: this.type,
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
		}) as WebGL2RenderingContext | null;
		if (!gl) {
			throw new Error(
				"Failed to acquire WebGL2 context. WebGL backend requires WebGL2."
			);
		}

		this._gl = gl;
		this._emitPostProcessResourceEvent("destroy", "context-initialize");
		this._frameExecutor?.destroy();
		this._frameExecutor = new WebGLFrameExecutor(
			gl,
			this.shaderRuntime,
			this._shaderCompileStage,
			this._shaderSourceFactory
		);
		this._contextLost = false;
		this._frameExecutor.resize(this._width, this._height);
	}

	private _emitPostProcessResourceEvent(
		action: "invalidate" | "destroy",
		reason: string
	): void {
		this._onBackendResourceEvent?.({
			resource: "postprocess",
			action,
			backend: "webgl",
			reason,
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
			void this._onDeviceLost?.(info);
		};
		this._contextRestoreHandler = () => {
			Logger.warn(
				"WebGL context was restored. Rebuilding WebGL resources.",
				{ scope: "WebGLBackend" }
			);
			try {
				this.restore();
			} catch (error) {
				Logger.warn(`WebGL context restore failed: ${String(error)}`, {
					scope: "WebGLBackend",
				});
			}
		};

		canvas.addEventListener("webglcontextlost", this._contextLossHandler);
		canvas.addEventListener("webglcontextrestored", this._contextRestoreHandler);
	}

	private _resolveRestoreCanvas(canvas?: HTMLCanvasElement): HTMLCanvasElement {
		const targetCanvas = canvas ?? this._canvas;
		if (!targetCanvas) {
			throw new Error(
				"WebGL backend cannot restore before a canvas has been initialized."
			);
		}
		return targetCanvas;
	}

	private _prepareFramePassPlan(context: FrameContext): void {
		this._framePlanner.preparePlan(context, this._getFramePlannerState());
	}

	private _validatePassDependencies(pass: FramePass): void {
		this._framePlanner.validatePassDependencies(
			pass,
			this._getFramePlannerState(),
			{
				reportNonFatalError: (scope, error) =>
					Logger.warn(`[${scope}] ${String(error)}`, {
						scope: "WebGLBackend",
					}),
			}
		);
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

	private _createPassHandlers(): Map<
		FramePass["stage"],
		WebGLBackendPassHandler
	> {
		const handlers = new Map<FramePass["stage"], WebGLBackendPassHandler>();
		handlers.set("particle-sim", (_pass, context) => {
			this._particleSimulator?.simulate(
				context,
				this._resolveParticleDeltaTime(context)
			);
			this._particleSimulator?.emitRenderBatches(context);
		});
		for (const stage of SUPPORTED_WEBGL_STAGES) {
			handlers.set(stage, (pass, context) => {
				this._frameExecutor?.executePass(pass, context);
			});
		}
		return handlers;
	}
}

function toSafeDimension(value: unknown): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return 1;
	}
	return Math.max(1, Math.floor(value));
}
