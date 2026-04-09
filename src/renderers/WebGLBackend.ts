import {
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
	INTERACTION_TRANSIENT_STATE_KEY,
	type FrameContext,
	type FramePass,
	type InteractionTransientState,
} from "../pipeline/types";
import { DefaultParticleSimulator } from "../simulation/particles/DefaultParticleSimulator";
import type {
	IRenderBackend,
	RendererBackendBridge,
	WarmupOptions,
	WarmupReport,
} from "./IRenderBackend";
import { WebGLFrameExecutor } from "./webgl/WebGLFrameExecutor";
import type { WebGLPostProcessPassPlugin } from "./webgl/WebGLPostProcessRuntime";
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

const SUPPORTED_WEBGL_STAGES: readonly FramePass["stage"][] = [
	"shadow",
	"main-opaque",
	"main-transparent",
	"particles",
	"ssao",
	"ssgi",
	"ssr",
	"volumetric",
	"fog",
	"motion-blur",
	"dof",
	"fxaa",
	"interaction-outline",
	"taa",
	"bloom",
	"gamma",
] as const;
const MAX_PARTICLE_SIM_DELTA_TIME_SECONDS = 0.5;
const WEBGL_PASS_DEPENDENCIES = new Map<
	FramePass["stage"],
	readonly FramePass["stage"][]
>([
	["shadow", ["particle-sim"]],
	["main-opaque", ["reflection", "shadow"]],
	["main-transparent", ["main-opaque"]],
	["particles", ["main-transparent"]],
	["ssao", ["particles"]],
	["ssgi", ["ssao"]],
	["taa", ["ssgi", "ssao"]],
	["ssr", ["taa"]],
	["volumetric", ["ssr"]],
	["fog", ["volumetric"]],
	["motion-blur", ["fog"]],
	["dof", ["motion-blur"]],
	["bloom", ["dof"]],
	["fxaa", ["bloom"]],
	["interaction-outline", ["fxaa"]],
	["gamma", ["interaction-outline"]],
]);

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
		skybox: true,
		ssao: true,
		ssgi: false,
		taa: true,
		ssr: false,
		volumetric: false,
		fog: true,
		motionBlur: true,
		dof: true,
		bloom: true,
		clusteredLighting: true,
	};

	private _renderer: RendererBackendBridge | null = null;
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
	private _shaderSourceFactory: WebGLShaderSourceFactory;
	private _executedPasses = new Set<FramePass["stage"]>();
	private _plannedPasses = new Set<FramePass["stage"]>();
	private _plannedPassOrder = new Map<FramePass["stage"], number>();
	private _pendingPostProcessPasses = new Map<string, WebGLPostProcessPassPlugin>();
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
			warn: (_key, message) => this._warn(message),
		});
		this._shaderSourceFactory = createWebGLShaderSourceFactory();
		this._passHandlers = this._createPassHandlers();
	}

	public setRenderer(renderer: RendererBackendBridge): void {
		this._renderer = renderer;
		this._particleSimulator = new DefaultParticleSimulator({
			backendTag: this.type,
		});
	}

	public registerPostProcessPass(pass: WebGLPostProcessPassPlugin): void {
		this._pendingPostProcessPasses.set(pass.id, pass);
		this._frameExecutor?.registerPostProcessPass(pass);
	}

	public unregisterPostProcessPass(id: string): void {
		this._pendingPostProcessPasses.delete(id);
		this._frameExecutor?.unregisterPostProcessPass(id);
	}

	public async init(canvas: HTMLCanvasElement): Promise<void> {
		this._canvas = canvas;
		this._installContextLifecycleListeners(canvas);
		await this._shaderSourceFactory.prepareAll();
		this._initializeGLContext(canvas);
	}

	public resize(width: number, height: number): void {
		this._width = toSafeDimension(width);
		this._height = toSafeDimension(height);
		if (this._contextLost) return;
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
			this._warn(
				`WebGL backend does not support pass "${pass.stage}" yet; skipping`
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
		this._frameExecutor?.destroy();
		this._frameExecutor = new WebGLFrameExecutor(
			gl,
			(_key, message) => this._warn(message),
			this.shaderRuntime,
			this._shaderCompileStage,
			this._shaderSourceFactory
		);
		for (const pass of this._pendingPostProcessPasses.values()) {
			this._frameExecutor.registerPostProcessPass(pass);
		}
		this._contextLost = false;
		this._frameExecutor.resize(this._width, this._height);
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
			(event as WebGLContextEvent).preventDefault?.();
			this._contextLost = true;
			this._warn(
				"WebGL context was lost. Rendering is paused until context restoration."
			);
		};
		this._contextRestoreHandler = () => {
			this._warn(
				"WebGL context was restored. Rebuilding WebGL resources."
			);
			if (!this._canvas) return;
			try {
				this._initializeGLContext(this._canvas);
			} catch (error) {
				this._warn(
					`WebGL context restore failed: ${String(error)}`
				);
			}
		};

		canvas.addEventListener("webglcontextlost", this._contextLossHandler);
		canvas.addEventListener("webglcontextrestored", this._contextRestoreHandler);
	}

	private _warn(message: string): void {
		if (this._renderer?.logger) {
			this._renderer.logger.warn(message, { scope: "WebGLBackend" });
			return;
		}
		Logger.warn(message, { scope: "WebGLBackend" });
	}

	private _prepareFramePassPlan(context: FrameContext): void {
		this._plannedPasses.clear();
		this._plannedPassOrder.clear();
		if (!context?.scene || !context?.features) {
			this._plannedPasses.add("main-opaque");
			this._plannedPassOrder.set("main-opaque", 0);
			return;
		}

		const hasParticleSystems = (context.scene.particleSystems?.length ?? 0) > 0;
		if (hasParticleSystems) {
			this._plannedPasses.add("particle-sim");
		}
		if (
			context.features.enableShadows &&
			context.scene.shadowCasterPackets.length > 0
		) {
			this._plannedPasses.add("shadow");
		}
		if (
			context.features.enableReflection &&
			context.scene.reflectivePackets.length > 0
		) {
			this._plannedPasses.add("reflection");
		}
		this._plannedPasses.add("main-opaque");
		if (context.scene.transparentPackets.length > 0) {
			this._plannedPasses.add("main-transparent");
		}
		if (hasParticleSystems) {
			this._plannedPasses.add("particles");
		}
		if (context.features.enableSSAO) {
			this._plannedPasses.add("ssao");
		}
		if (context.features.enableSSGI) {
			this._plannedPasses.add("ssgi");
		}
		if (context.features.enableTAA) {
			this._plannedPasses.add("taa");
		}
		if (context.features.enableSSR) {
			this._plannedPasses.add("ssr");
		}
		if (context.features.enableVolumetric) {
			this._plannedPasses.add("volumetric");
		}
		if (isFogPostProcessEnabled(context.features)) {
			this._plannedPasses.add("fog");
		}
		if (context.features.enableMotionBlur) {
			this._plannedPasses.add("motion-blur");
		}
		if (context.features.enableDOF) {
			this._plannedPasses.add("dof");
		}
		if (context.features.enableBloom) {
			this._plannedPasses.add("bloom");
		}
		if (context.features.enableFXAA) {
			this._plannedPasses.add("fxaa");
		}
		const interaction = context.transient.get(
			INTERACTION_TRANSIENT_STATE_KEY
		) as InteractionTransientState | null | undefined;
		if ((interaction?.selectedEntityIds?.length ?? 0) > 0) {
			this._plannedPasses.add("interaction-outline");
		}
		if (context.features.enableGamma) {
			this._plannedPasses.add("gamma");
		}
		this._validatePlannedPassGraph();
	}

	private _validatePassDependencies(pass: FramePass): void {
		if (this._plannedPasses.size > 0 && !this._plannedPasses.has(pass.stage)) {
			return;
		}
		const plannedIndex = this._plannedPassOrder.get(pass.stage);
		if (plannedIndex !== undefined) {
			const violated = Array.from(this._executedPasses).some((executedStage) => {
				const index = this._plannedPassOrder.get(executedStage);
				return index !== undefined && index > plannedIndex;
			});
			if (violated) {
				throw new Error(
					`WebGL pass \"${pass.stage}\" execution order violates prevalidated pass plan.`
				);
			}
		}
		const dependencies = this._resolvePassDependencies(pass.stage);
		if (dependencies.length <= 0) {
			return;
		}
		const missing = dependencies.filter(
			(dependency) =>
				this._plannedPasses.has(dependency) &&
				this._isDependencyApplicable(pass.stage, dependency) &&
				!this._executedPasses.has(dependency)
		);
		if (missing.length <= 0) {
			return;
		}
		throw new Error(
			`WebGL pass \"${pass.stage}\" executed before dependencies: ${missing.join(", ")}`
		);
	}

	private _validatePlannedPassGraph(): void {
		const visiting = new Set<FramePass["stage"]>();
		const visited = new Set<FramePass["stage"]>();
		const order: FramePass["stage"][] = [];

		const visit = (stage: FramePass["stage"]): void => {
			if (visited.has(stage)) {
				return;
			}
			if (visiting.has(stage)) {
				throw new Error(
					`WebGL pass plan cycle detected at \"${stage}\" during _prepareFramePassPlan.`
				);
			}
			visiting.add(stage);
			const dependencies = this._resolvePassDependencies(stage);
			for (const dependency of dependencies) {
				if (!this._plannedPasses.has(dependency)) {
					continue;
				}
				visit(dependency);
			}
			visiting.delete(stage);
			visited.add(stage);
			order.push(stage);
		};

		for (const stage of this._plannedPasses) {
			visit(stage);
		}
		for (let i = 0; i < order.length; i++) {
			this._plannedPassOrder.set(order[i], i);
		}
	}

	private _resolvePassDependencies(
		stage: FramePass["stage"]
	): FramePass["stage"][] {
		const dependencies = WEBGL_PASS_DEPENDENCIES.get(stage);
		return dependencies ? Array.from(dependencies) : [];
	}

	private _isDependencyApplicable(
		stage: FramePass["stage"],
		dependency: FramePass["stage"]
	): boolean {
		const stageIndex = this._plannedPassOrder.get(stage);
		const dependencyIndex = this._plannedPassOrder.get(dependency);
		if (stageIndex === undefined || dependencyIndex === undefined) {
			return true;
		}
		if (dependencyIndex < stageIndex) {
			return true;
		}
		this._warn(
			`Ignoring stale dependency \"${dependency}\" for \"${stage}\".`
		);
		return false;
	}

	private _markPassExecuted(stage: FramePass["stage"]): void {
		this._executedPasses.add(stage);
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

function isFogPostProcessEnabled(features: FrameContext["features"]): boolean {
	return (
		features.enableFog &&
		(features.fogOptions?.application ?? "postprocess") !== "scene"
	);
}
