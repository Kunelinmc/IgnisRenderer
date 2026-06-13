import {
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
	type FrameContext,
	type FramePass,
	type FrameAttachments,
} from "../pipeline/types";
import { DefaultParticleSimulator } from "../simulation/particles/DefaultParticleSimulator";
import type {
	IRenderBackend,
	IRenderBackendSession,
	RenderBackendDeviceLostInfo,
	RenderBackendProfile,
	RenderBackendSessionContext,
	RenderSurfaceSize,
	WarmupOptions,
	WarmupReport,
} from "./IRenderBackend";
import { WebGLFrameExecutor } from "./webgl/WebGLFrameExecutor";
import { WebGLPostProcessExecutor } from "./webgl/WebGLPostProcessExecutor";
import { BackendPostProcessRuntime } from "../postprocess/BackendPostProcessRuntime";
import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_POINT_LIGHTS,
	MAX_SPOT_LIGHTS,
} from "./constants";
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
	ShaderSource,
	WEBGL_SHADER_PARTS,
} from "../shaders/ShaderSource";
import {
	addWarmupPhase,
	buildWarmupPlan,
	createWarmupReport,
	finalizeWarmupReport,
	toShaderCompileError,
	type WarmupPhaseCounters,
	type WarmupPostProcessPlan,
} from "../pipeline/WarmupPlanner";
import { Logger } from "../foundation/Logger";
import {
	FramePassPlanValidator,
	type FramePassPlanValidatorState,
} from "../pipeline/FramePassPlanValidator";
import {
	createRenderBackendExtensionRegistry,
} from "./BackendExtensions";

const SUPPORTED_WEBGL_STAGES: readonly FramePass["stage"][] = [
	"shadow",
	"main-opaque",
	"main-transparent",
	"particles",
	"postprocess",
] as const;
const MAX_PARTICLE_SIM_DELTA_TIME_SECONDS = 0.5;

export interface WebGLBackendOptions {
	shaderMode?: ShaderRuntimeMode;
	directiveHook?: ShaderDirectiveCompileHook | null;
	validatePrograms?: boolean;
	enableEarlyZPrepass?: boolean;
}

type WebGLBackendPassHandler = (
	pass: FramePass,
	context: FrameContext
) => void | Promise<void>;

export class WebGLBackend implements IRenderBackend, IRenderBackendSession {
	private readonly _options: WebGLBackendOptions;
	private _defaultSession: WebGLBackendSession | null = null;

	public constructor(options: WebGLBackendOptions = {}) {
		this._options = options;
	}

	public createSession(
		context: RenderBackendSessionContext
	): IRenderBackendSession {
		const session = new WebGLBackendSession(this._options, context);
		this._defaultSession = session;
		return session;
	}

	private _getOrEstablishSession(): WebGLBackendSession {
		if (!this._defaultSession) {
			this._defaultSession = new WebGLBackendSession(this._options, {
				surface: { canvas: typeof document !== "undefined" ? document.createElement("canvas") : { width: 320, height: 180 } as any },
				events: { emit: () => {} },
			});
		}
		return this._defaultSession;
	}

	// Delegate properties and methods of IRenderBackendSession
	public get type() { return "webgl"; }
	public get capabilities() { return this._getOrEstablishSession().capabilities; }
	public get frameScheduling() { return this._getOrEstablishSession().frameScheduling; }
	public get extensions() { return this._getOrEstablishSession().extensions; }
	public get profile() { return this._getOrEstablishSession().profile; }

	private _rendererCompat: any = null;
	public get _renderer() { return this._rendererCompat; }
	public set _renderer(val) { this._rendererCompat = val; }

	public setRenderer(renderer: any): void {
		this._rendererCompat = renderer;
	}

	public async init(canvas: HTMLCanvasElement): Promise<void> {
		this.createSession({
			surface: { canvas },
			events: {
				emit: (event) => {
					if (event.type === "device-lost") {
						this._rendererCompat?.onDeviceLost?.(event.info);
					} else if (event.type === "device-restored") {
						this._rendererCompat?.onDeviceRestored?.();
					}
				}
			},
		});
		await this.initialize();
	}

	public initialize(): Promise<void> {
		return this._getOrEstablishSession().initialize();
	}

	public restore(): Promise<void> {
		this._getOrEstablishSession().restore();
		return Promise.resolve();
	}

	public resize(size: RenderSurfaceSize | number, heightParam?: number): void {
		this._getOrEstablishSession().resize(size, heightParam);
	}

	public getAttachments(size: RenderSurfaceSize | number, heightParam?: number): FrameAttachments {
		return this._getOrEstablishSession().getAttachments(size, heightParam);
	}

	public beginFrame(context: FrameContext): void | Promise<void> {
		return this._getOrEstablishSession().beginFrame(context);
	}

	public executePass(pass: FramePass, context: FrameContext): void | Promise<void> {
		return this._getOrEstablishSession().executePass(pass, context);
	}

	public skipPass(pass: FramePass): void {
		this._getOrEstablishSession().skipPass(pass);
	}

	public endFrame(): void | Promise<void> {
		return this._getOrEstablishSession().endFrame();
	}

	public abortFrame(error?: unknown): void | Promise<void> {
		return this._getOrEstablishSession().abortFrame(error);
	}

	public destroy(): void | Promise<void> {
		if (this._defaultSession) {
			return this._defaultSession.destroy();
		}
	}

	public isEarlyZPrepassEnabled(): boolean {
		return this._getOrEstablishSession().isEarlyZPrepassEnabled();
	}

	public onDeviceLost(info?: RenderBackendDeviceLostInfo): void {
		this._getOrEstablishSession().onDeviceLost(info);
	}

	public _resolveParticleDeltaTime(context: FrameContext): number {
		return (this._getOrEstablishSession() as any)._resolveParticleDeltaTime(context);
	}

	public get _frameExecutor(): any {
		return (this._getOrEstablishSession() as any)._frameExecutor;
	}

	public set _frameExecutor(value: any) {
		(this._getOrEstablishSession() as any)._frameExecutor = value;
	}

	public get _particleSimulator(): any {
		return (this._getOrEstablishSession() as any)._particleSimulator;
	}

	public set _particleSimulator(value: any) {
		(this._getOrEstablishSession() as any)._particleSimulator = value;
	}

	public get _contextLost(): boolean {
		return (this._getOrEstablishSession() as any)._contextLost;
	}

	public set _contextLost(value: boolean) {
		(this._getOrEstablishSession() as any)._contextLost = value;
	}

	public get _plannedPasses(): any {
		return (this._getOrEstablishSession() as any)._plannedPasses;
	}

	public get _plannedPassOrder(): any {
		return (this._getOrEstablishSession() as any)._plannedPassOrder;
	}
}

class WebGLBackendSession implements IRenderBackendSession {
	public readonly type = "webgl";
	public readonly frameScheduling = "on-demand";
	public readonly capabilities = {
		sh: true,
		shadows: true,
		reflection: false,
		environment: true,
		postProcess: true,
		clusteredLighting: true,
		oit: true,
		occlusionCulling: false,
	};
	private readonly _postProcessExecutor = new WebGLPostProcessExecutor({
		getFrameExecutor: () => this._frameExecutor,
	});
	private readonly _postProcessRuntime = new BackendPostProcessRuntime({
		executor: this._postProcessExecutor,
		session: this,
		warn: (key, message) =>
			Logger.warn(`[${key}] ${message}`, {
				scope: "WebGLBackend",
				onceKey: key,
			}),
	});
	public get postProcessRuntime(): BackendPostProcessRuntime {
		return this._postProcessRuntime;
	}
	public readonly extensions = createRenderBackendExtensionRegistry([]);
	public readonly profile: RenderBackendProfile = {
		id: "webgl",
		capabilities: this.capabilities,
		frameScheduling: this.frameScheduling,
		shadow: {
			backendKey: "webgl",
			supportsFilterModes: ["pcf", "vsm"],
			supportsDirectionalCSM: true,
			supportsSpotCSM: false,
			supportsPointCSM: false,
			maxDynamicShadowCost: 24,
		},
		lighting: { localizedProbeMode: "backend-local" },
	};

	private readonly _sessionContext: RenderBackendSessionContext;
	private readonly _options: WebGLBackendOptions;
	private _canvas: HTMLCanvasElement | null = null;
	private _gl: WebGL2RenderingContext | null = null;
	private _frameExecutor: WebGLFrameExecutor | null = null;
	private _particleSimulator: DefaultParticleSimulator | null = null;
	private _contextLost = false;
	private _contextLossHandler: ((event: Event) => void) | null = null;
	private _contextRestoreHandler: ((event: Event) => void) | null = null;
	private _width = 1;
	private _height = 1;
	public readonly shaderRuntime: ShaderRuntime;
	private _shaderCompileStage: ShaderBackendCompileStage;
	private _validatePrograms = false;
	private _enableEarlyZPrepass = true;
	private _executedPasses = new Set<FramePass["stage"]>();
	private _plannedPasses = new Set<FramePass["stage"]>();
	private _plannedPassOrder = new Map<FramePass["stage"], number>();
	private readonly _framePlanner = new FramePassPlanValidator("WebGL");
	private readonly _passHandlers: Map<
		FramePass["stage"],
		WebGLBackendPassHandler
	>;

	constructor(
		options: WebGLBackendOptions,
		sessionContext: RenderBackendSessionContext
	) {
		this._options = options;
		this._sessionContext = sessionContext;
		const shaderMode = options.shaderMode ?? "warn";
		this._validatePrograms = options.validatePrograms === true;
		this._enableEarlyZPrepass = options.enableEarlyZPrepass !== false;
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
		this._passHandlers = this._createPassHandlers();
		this._ensureParticleSimulator();
	}

	/**
	 * Reports whether the WebGL backend will run its internal Early Z pre-pass.
	 *
	 * @returns `true` when opaque WebGL frames use depth pre-pass optimization.
	 * @sideEffects None.
	 */
	public isEarlyZPrepassEnabled(): boolean {
		return this._enableEarlyZPrepass;
	}

	public async initialize(): Promise<void> {
		const canvas = this._requireSessionContext().surface.canvas;
		this._ensureParticleSimulator();
		this._canvas = canvas;
		this._installContextLifecycleListeners(canvas);
		await ShaderSource.prepareMany(
			[
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
			]
		);
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
		const detail =
			typeof info?.message === "string" && info.message.length > 0
				? `: ${info.message}`
				: "";
		Logger.warn(
			`WebGL context was lost${detail}. Rendering is paused until context restoration.`,
			{ scope: "WebGLBackend" }
		);
	}

	public restore(): void {
		const targetCanvas = this._requireSessionContext().surface.canvas;
		this._ensureParticleSimulator();
		this._canvas = targetCanvas;
		this._installContextLifecycleListeners(targetCanvas);
		this._initializeGLContext(targetCanvas);
	}

	public resize(size: RenderSurfaceSize | number, heightParam?: number): void {
		let width: number;
		let height: number;
		if (typeof size === "object" && size !== null) {
			width = size.width;
			height = size.height;
		} else {
			width = size as number;
			height = heightParam!;
		}
		this._width = toSafeDimension(width);
		this._height = toSafeDimension(height);
		if (this._contextLost) return;
		this._postProcessRuntime.invalidateFrameSized();
		this._frameExecutor?.resize(this._width, this._height);
	}

	public getAttachments(size: RenderSurfaceSize | number, heightParam?: number): {
		width: number;
		height: number;
	} {
		let width: number;
		let height: number;
		if (typeof size === "object" && size !== null) {
			width = size.width;
			height = size.height;
		} else {
			width = size as number;
			height = heightParam!;
		}
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

	public executePass(pass: FramePass, context: FrameContext): void | Promise<void> {
		if (!this._frameExecutor) {
			throw new Error("WebGL backend has not been initialized.");
		}
		if (this._contextLost) {
			return;
		}
		this._validatePassDependencies(pass);
		const handler = this._passHandlers.get(pass.stage);
		if (!handler) {
			const key = `webgl-pass-unsupported-${pass.stage}`;
			Logger.warn(
				`[${key}] WebGL backend does not support pass "${pass.stage}" yet; skipping`,
				{ scope: "WebGLBackend", onceKey: key }
			);
			this._markPassExecuted(pass.stage);
			return;
		}
		const result = handler(pass, context);
		if (result && typeof (result as Promise<void>).then === "function") {
			return (result as Promise<void>).then(() => {
				this._markPassExecuted(pass.stage);
			});
		}
		this._markPassExecuted(pass.stage);
		return result;
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
		this._postProcessRuntime.commitFrame();
	}

	public async abortFrame(_error?: unknown): Promise<void> {
		if (this._contextLost) {
			return;
		}
		await this._postProcessRuntime.abortFrame(_error);
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
		let warmupPostProcessPlan: WarmupPostProcessPlan | undefined;
		if (options.includePostProcess !== false) {
			const graph = this._postProcessRuntime.compileWarmupGraph(context);
			warmupPostProcessPlan = {
				passIds: graph.orderedPasses.map((pass) => pass.id),
				descriptors: graph.orderedPasses.map((pass) => pass.pass),
			};
		}
		const plan = buildWarmupPlan(context, options, warmupPostProcessPlan);
		try {
			const phase = await this._frameExecutor.warmup(context, plan, options);
			addWarmupPhase(report, phase);
			this._reportWarmupProgress(options, phase);
		} catch (error) {
			const failedPhase = {
				phase: "webgl-warmup",
				total: 1,
				compiled: 0,
				skipped: 0,
				failed: 1,
				errors: [toShaderCompileError(error, this.type, "WebGLWarmup")],
			};
			addWarmupPhase(report, failedPhase);
			this._reportWarmupProgress(options, failedPhase);
		}
		return finalizeWarmupReport(report);
	}

	public destroy(): void {
		this._postProcessRuntime.destroy();
		this._frameExecutor?.destroy();
		this._frameExecutor = null;
		this._particleSimulator = null;
		this._gl = null;

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
		this._postProcessRuntime.destroy();
		this._frameExecutor?.destroy();
		this._frameExecutor = new WebGLFrameExecutor(
			gl,
			this.shaderRuntime,
			this._shaderCompileStage,
			{
				validatePrograms: this._validatePrograms,
				enableEarlyZPrepass: this._enableEarlyZPrepass,
				onProgramCompilePending: () => this._emitProgramCompilePendingEvent(),
				onTextureUploadPending: () => this._emitTextureUploadPendingEvent(),
				postProcessRuntime: this._postProcessRuntime,
			}
		);
		this._contextLost = false;
		this._frameExecutor.resize(this._width, this._height);
	}

	private _reportWarmupProgress(
		options: WarmupOptions,
		phase: WarmupPhaseCounters,
	): void {
		options.onProgress?.({
			phase: phase.phase,
			completed: phase.compiled + phase.skipped + phase.failed,
			total: phase.total,
		});
	}

	private _emitProgramCompilePendingEvent(): void {
		this._sessionContext?.events.emit({
			type: "render-invalidated",
			reason: "postfx",
		});
	}

	private _emitTextureUploadPendingEvent(): void {
		this._sessionContext?.events.emit({
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
			this._sessionContext?.events.emit({ type: "device-lost", info });
		};
		this._contextRestoreHandler = () => {
			Logger.warn(
				"WebGL context was restored. Rebuilding WebGL resources.",
				{ scope: "WebGLBackend" }
			);
			try {
				this.restore();
				this._sessionContext?.events.emit({ type: "device-restored" });
			} catch (error) {
				Logger.warn(`WebGL context restore failed: ${String(error)}`, {
					scope: "WebGLBackend",
				});
			}
		};

		canvas.addEventListener("webglcontextlost", this._contextLossHandler);
		canvas.addEventListener("webglcontextrestored", this._contextRestoreHandler);
	}

	private _requireSessionContext(): RenderBackendSessionContext {
		if (!this._sessionContext) {
			throw new Error(
				"WebGLBackend is a provider. Use createSession() before initialization."
			);
		}
		return this._sessionContext;
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
				if (pass.stage === "postprocess") {
					return this._postProcessRuntime.execute(context);
				}
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
