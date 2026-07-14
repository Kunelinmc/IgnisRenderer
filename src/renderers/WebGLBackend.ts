import {
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
	type FrameContext,
	type FramePass,
	type FrameAttachments,
} from "../pipeline/types";
import { DefaultParticleSimulator } from "../simulation/particles/DefaultParticleSimulator";
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
} from "./IRenderBackend";
import type { RenderTargetReadbackOptions } from "./CustomRenderTargets";
import type { TextureReadbackResult } from "./IComputeRuntime";
import { WebGLFrameExecutor } from "./webgl/WebGLFrameExecutor";
import { WebGLFrameGraphRuntime } from "./webgl/rendergraph/WebGLFrameGraphRuntime";
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
			getFrameExecutor: () => this._frameExecutor,
		}),
		backend: this,
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
		capabilities: {
			sh: true,
			shadows: true,
			reflection: false,
			environment: true,
			postProcess: true,
			clusteredLighting: true,
			oit: true,
			occlusionCulling: false,
			customRenderTargets: true,
			customRenderPasses: true,
			renderTargetReadback: true,
		},
		frameScheduling: "on-demand",
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

	private _attachContext: RenderBackendAttachContext | null = null;
	private _attached = false;
	private readonly _options: WebGLBackendOptions;
	private _canvas: HTMLCanvasElement | null = null;
	private _gl: WebGL2RenderingContext | null = null;
	private _frameExecutor: WebGLFrameExecutor | null = null;
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
	private _validatePrograms = false;
	private _enableEarlyZPrepass = true;
	private _executedPasses = new Set<FramePass["stage"]>();
	private _plannedPasses = new Set<FramePass["stage"]>();
	private _plannedPassOrder = new Map<FramePass["stage"], number>();
	private readonly _framePlanner = new FramePassPlanValidator("WebGL");
	private _debugInfo: RenderBackendDebugInfo = WEBGL_DEBUG_INFO_UNINITIALIZED;

	constructor(options: WebGLBackendOptions = {}) {
		this._options = options;
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
		this._ensureParticleSimulator();
	}

	public attach(context: RenderBackendAttachContext): void {
		if (this._attached) {
			throw new Error("WebGLBackend is already attached to a renderer.");
		}
		this._attachContext = context;
		this._attached = true;
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

	public async initialize(): Promise<void> {
		const canvas = this._requireAttachContext().surface.canvas;
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
		if (this._contextLost) return;
		this._postProcessRuntime.invalidateFrameSized();
		this._frameExecutor?.resize(this._width, this._height);
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

	public beginFrame(context: FrameContext): void {
		this._completedFrameCoverage = "full-frame";
		if (!this._frameExecutor || !this._frameGraphRuntime) {
			throw new Error("WebGL backend has not been initialized.");
		}
		if (this._contextLost) {
			return;
		}
		this._executedPasses.clear();
		this._activeContext = context;
		this._prepareFramePassPlan(context);
		this._particleSimulator?.beginFrame(context);
		this._frameGraphRuntime.beginFrame(context);
	}

	public executePass(pass: FramePass, context: FrameContext): void | Promise<void> {
		if (!this._frameExecutor || !this._frameGraphRuntime) {
			throw new Error("WebGL backend has not been initialized.");
		}
		if (this._contextLost) {
			return;
		}
		this._validatePassDependencies(pass);
		if (pass.stage === "particle-sim") {
			this._particleSimulator?.simulate(context, this._resolveParticleDeltaTime(context));
			this._particleSimulator?.emitRenderBatches(context);
			this._markPassExecuted(pass.stage);
			return;
		}
		const result = this._frameGraphRuntime.executePass(pass, context);
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

	public readRenderTargetColor(
		id: string,
		attachmentIndex?: number,
		options?: RenderTargetReadbackOptions,
	): Promise<TextureReadbackResult> {
		if (!this._frameExecutor) {
			return Promise.reject(new Error("WebGL backend has not been initialized."));
		}
		return this._frameExecutor.readCustomRenderTargetColor(id, attachmentIndex, options);
	}

	public endFrame(): void {
		if (!this._frameExecutor || !this._frameGraphRuntime || this._contextLost) {
			return;
		}
		const context = this._activeContext;
		if (!context) {
			throw new Error("WebGL backend cannot end a frame before beginFrame.");
		}
		this._frameGraphRuntime.endFrame(context);
		this._particleSimulator?.endFrame();
		this._postProcessRuntime.commitFrame();
		this._activeContext = null;
	}

	/** @internal Renderer frame-coordination coverage report. */
	public getCompletedFrameCoverage(): RenderBackendCompletedFrameCoverage {
		return this._completedFrameCoverage;
	}

	public async abortFrame(_error?: unknown): Promise<void> {
		if (this._contextLost) {
			return;
		}
		await this._postProcessRuntime.abortFrame(_error);
		this._frameGraphRuntime?.abortFrame();
		this._particleSimulator?.endFrame();
		this._activeContext = null;
		this._executedPasses.clear();
		this._plannedPasses.clear();
		this._plannedPassOrder.clear();
	}

	public async warmup(context: FrameContext, options: WarmupOptions = {}): Promise<WarmupReport> {
		const report = createWarmupReport(this.profile.id);
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
				errors: [toShaderCompileError(error, this.profile.id, "WebGLWarmup")],
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
			},
		);
		this._frameGraphRuntime = new WebGLFrameGraphRuntime(
			this._frameExecutor,
			this._postProcessRuntime,
		);
		this._contextLost = false;
		this._frameExecutor.resize(this._width, this._height);
		this._debugInfo = this._createDebugInfo(gl);
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
	let debugInfo: any = null;
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
		const parameter = (gl as any)[key];
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
