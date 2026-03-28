import {
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
	type FrameContext,
	type FramePass,
} from "../pipeline/types";
import { DefaultParticleSimulator } from "../simulation/particles/DefaultParticleSimulator";
import type {
	IRenderBackend,
	RendererBackendBridge,
	WarmupOptions,
	WarmupReport,
} from "./IRenderBackend";
import { WebGLFrameExecutor } from "./webgl/WebGLFrameExecutor";
import { ShaderRuntime } from "../shaders/runtime";
import {
	addWarmupPhase,
	buildWarmupPlan,
	createWarmupReport,
	finalizeWarmupReport,
	toShaderCompileError,
} from "../pipeline/WarmupPlanner";

const SUPPORTED_WEBGL_STAGES = new Set<FramePass["stage"]>([
	"shadow",
	"main-opaque",
	"main-transparent",
	"particles",
	"ssao",
	"motion-blur",
	"dof",
	"fxaa",
	"taa",
	"bloom",
	"gamma",
]);
const MAX_PARTICLE_SIM_DELTA_TIME_SECONDS = 0.5;

export class WebGLBackend implements IRenderBackend {
	public readonly type = "webgl";
	public readonly frameScheduling = "on-demand";
	public readonly passExecutors = {
		"animation-sim": "shared",
		"particle-sim": "backend",
	} as const;
	public readonly capabilities = {
		sh: false,
		shadows: true,
		reflection: false,
		skybox: true,
		ssao: true,
		taa: true,
		ssr: false,
		volumetric: false,
		motionBlur: true,
		dof: true,
		bloom: true,
		clusteredLighting: false,
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

	constructor() {
		this.shaderRuntime = new ShaderRuntime();
	}

	public setRenderer(renderer: RendererBackendBridge): void {
		this._renderer = renderer;
		this._particleSimulator = new DefaultParticleSimulator({
			backendTag: this.type,
		});
	}

	public async init(canvas: HTMLCanvasElement): Promise<void> {
		this._canvas = canvas;
		this._installContextLifecycleListeners(canvas);
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
		if (pass.stage === "particle-sim") {
			this._particleSimulator?.simulate(
				context,
				this._resolveParticleDeltaTime(context)
			);
			this._particleSimulator?.emitRenderBatches(context);
			return;
		}
		if (!SUPPORTED_WEBGL_STAGES.has(pass.stage)) {
			this._warnOnce(
				`webgl-pass-unsupported-${pass.stage}`,
				`WebGL backend does not support pass "${pass.stage}" yet; skipping`
			);
			return;
		}
		this._frameExecutor.executePass(pass, context);
	}

	public skipPass(_pass: FramePass): void {
		// No pass dependency tracking in WebGLBackend; no-op.
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
				"Failed to acquire WebGL2 context. WebGLBackend v1 requires WebGL2."
			);
		}

		this._gl = gl;
		this._frameExecutor?.destroy();
		this._frameExecutor = new WebGLFrameExecutor(
			gl,
			(key, message) => this._warnOnce(key, message),
			this.shaderRuntime
		);
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
			this._warnOnce(
				"webgl-context-lost",
				"WebGL context was lost. Rendering is paused until context restoration."
			);
		};
		this._contextRestoreHandler = () => {
			this._warnOnce(
				"webgl-context-restored",
				"WebGL context was restored. Rebuilding WebGL resources."
			);
			if (!this._canvas) return;
			try {
				this._initializeGLContext(this._canvas);
			} catch (error) {
				this._warnOnce(
					"webgl-context-restore-failed",
					`WebGL context restore failed: ${String(error)}`
				);
			}
		};

		canvas.addEventListener("webglcontextlost", this._contextLossHandler);
		canvas.addEventListener("webglcontextrestored", this._contextRestoreHandler);
	}

	private _warnOnce(key: string, message: string): void {
		this._renderer?.warnOnce(key, message);
	}
}

function toSafeDimension(value: unknown): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return 1;
	}
	return Math.max(1, Math.floor(value));
}
