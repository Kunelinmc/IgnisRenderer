import type { IRenderBackend } from "../IRenderBackend";
import {
	PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
	PARTICLE_TRANSIENT_BATCHES_KEY,
	type DrawPacket,
	type FrameContext,
	type FramePass,
} from "../../pipeline/types";
import type { SoftwareBackendOptions } from "./SoftwareBackendContracts";
import { Rasterizer } from "./Rasterizer";
import { SoftwarePostProcessExecutor } from "./SoftwarePostProcessExecutor";
import { BackendPostProcessRuntime } from "../../postprocess/BackendPostProcessRuntime";
import type { PostProcessPlan } from "../../postprocess/PostProcessPlanner";
import { SoftwareMainPass } from "./passes/SoftwareMainPass";
import { SoftwareParticlePass } from "./passes/SoftwareParticlePass";
import { SoftwareReflectionPass } from "./passes/SoftwareReflectionPass";
import { SoftwareShadowPass } from "./passes/SoftwareShadowPass";
import { DefaultParticleSimulator } from "../../simulation/particles/DefaultParticleSimulator";
import { Logger } from "../../foundation/Logger";
import {
	createSoftwareFrameServices,
	resetSoftwareFrameServices,
	type SoftwareFrameServices,
	type SoftwarePassContext,
} from "./SoftwareFrameServices";
import type { SoftwareFrameView } from "./SoftwareFrameView";
import type { DisplayOutputState } from "../../rendering/DisplayOutput";

type SoftwareStageHandler = () => void | Promise<void>;

/** @internal Owns Software-specific passes and per-frame execution resources. */
export class SoftwarePassExecutor {
	private readonly _rasterizer = new Rasterizer();
	private readonly _softwarePostProcessExecutor: SoftwarePostProcessExecutor;
	private readonly _services: SoftwareFrameServices;
	private readonly _mainPass: SoftwareMainPass;
	private readonly _particlePass = new SoftwareParticlePass();
	private readonly _shadowPass = new SoftwareShadowPass(this._rasterizer);
	private readonly _reflectionPass: SoftwareReflectionPass;
	private readonly _particleSimulator: DefaultParticleSimulator;
	private readonly _postProcessRuntime: BackendPostProcessRuntime;
	private readonly _stageHandlers = new Map<FramePass["stage"], SoftwareStageHandler>();
	private _particleFrameActive = false;
	private _sourceContext: FrameContext | null = null;
	private _passContext: SoftwarePassContext | null = null;
	private _postProcessPlan: PostProcessPlan | null = null;

	constructor(options: {
		backend: IRenderBackend;
		backendOptions: SoftwareBackendOptions;
		getSceneColor: () => Float32Array;
		getDisplayOutputState: () => DisplayOutputState;
	}) {
		this._softwarePostProcessExecutor = new SoftwarePostProcessExecutor(
			options.getSceneColor,
			options.getDisplayOutputState,
		);
		this._services = createSoftwareFrameServices({
			rasterizer: this._rasterizer,
			postProcess: this._softwarePostProcessExecutor,
		});
		this._reflectionPass = new SoftwareReflectionPass(
			this._rasterizer,
			this._services.reflection,
		);
		this._mainPass = new SoftwareMainPass(this._rasterizer, {
			enableEarlyZPrepass: options.backendOptions.enableEarlyZPrepass,
		});
		this._particleSimulator = new DefaultParticleSimulator({
			backendTag: options.backend.profile.id,
		});
		this._postProcessRuntime = new BackendPostProcessRuntime({
			executor: this._softwarePostProcessExecutor,
			backend: options.backend,
			warn: (key, message) =>
				Logger.warn(`[${key}] ${message}`, {
					scope: "SoftwareBackend",
					onceKey: key,
				}),
		});
		this._createStageHandlers();
	}

	/** Begins coordinator-owned work and resolves the frame's post-process plan. */
	public beginCoordinatorFrame(context: FrameContext): PostProcessPlan {
		if (this._sourceContext) {
			throw new Error("SoftwarePassExecutor already has an active frame.");
		}
		resetSoftwareFrameServices(this._services);
		this._sourceContext = context;
		this._particleSimulator.beginFrame(context);
		this._particleSimulator.emitRenderBatches(context);
		this._particleFrameActive = true;
		this._postProcessPlan = this._postProcessRuntime.planFrame(context);
		return this._postProcessPlan;
	}

	/** Binds the derived Software view after temporal requirements are known. */
	public bindFrame(frame: SoftwareFrameView): SoftwarePassContext {
		if (!this._sourceContext || !this._postProcessPlan) {
			throw new Error("SoftwarePassExecutor.bindFrame() requires frame planning.");
		}
		this._passContext = Object.freeze({ frame, services: this._services });
		this._softwarePostProcessExecutor.bindSoftwareFrame(
			frame,
			this._postProcessPlan.initialColorDomain,
		);
		this._shadowPass.bindSamplers(this._passContext);
		return this._passContext;
	}

	public async execute(pass: FramePass): Promise<void> {
		const source = this._requireSourceContext();
		if (source.customRenderPasses?.has(pass.stage)) {
			const key = "software-custom-render-targets-unsupported";
			Logger.warn(
				`[${key}] Software backend does not support custom render targets or ` +
					`custom render passes yet; skipping pass "${pass.stage}".`,
				{ scope: "SoftwareBackend", onceKey: key },
			);
			return;
		}

		const handler = this._stageHandlers.get(pass.stage);
		if (!handler) {
			const key = `software-pass-unsupported-${pass.stage}`;
			Logger.warn(
				`[${key}] Software backend does not support pass "${pass.stage}" yet; skipping`,
				{ scope: "SoftwareBackend", onceKey: key },
			);
			return;
		}
		await handler();
	}

	private _createStageHandlers(): void {
		this._stageHandlers.set("particle-sim", () => {
			const source = this._requireSourceContext();
			const deltaTimeSeconds = this._resolveParticleDeltaTime(source);
			this._particleSimulator.simulate(source, deltaTimeSeconds);
			this._particleSimulator.emitRenderBatches(source);
			this._services.particles.batches =
				source.transient.get(PARTICLE_TRANSIENT_BATCHES_KEY) ?? [];
			this._services.particles.meshBatches =
				source.transient.get(PARTICLE_MESH_TRANSIENT_BATCHES_KEY) ?? [];
		});
		this._stageHandlers.set("shadow", () => {
			this._shadowPass.render(this._requirePassContext());
		});
		this._stageHandlers.set("reflection", () => {
			this._reflectionPass.render(this._requirePassContext());
		});
		this._stageHandlers.set("main-opaque", async () => {
			const context = this._requirePassContext();
			const packets = this._resolvePacketsForPass(
				context.frame,
				context.frame.scene.opaquePackets,
				"opaque",
			);
			await this._mainPass.render(context, packets, false);
			this._reflectionPass.composite(context, this._resolveOpaqueReflectivePackets(packets));
		});
		this._stageHandlers.set("main-transparent", async () => {
			const context = this._requirePassContext();
			const packets = this._resolvePacketsForPass(
				context.frame,
				context.frame.scene.transparentPackets,
				"transparent",
			);
			await this._mainPass.render(context, packets, true);
		});
		this._stageHandlers.set("particles", () => {
			this._particlePass.render(this._requirePassContext());
		});
		this._stageHandlers.set("postprocess", () => {
			if (!this._postProcessPlan) {
				throw new Error("Software post-process plan is unavailable.");
			}
			return this._postProcessRuntime.execute(
				this._requireSourceContext(),
				this._postProcessPlan,
			);
		});
	}

	private _resolvePacketsForPass(
		frame: SoftwareFrameView,
		packets: DrawPacket[],
		kind: "opaque" | "transparent",
	): DrawPacket[] {
		const spatialIndex = frame.scene.spatialIndex;
		if (!spatialIndex || !frame.incrementalPartial) return packets;
		const source = this._requireSourceContext();
		if (source.incremental.dirtyRects.length === 0) return [];
		return kind === "opaque"
			? spatialIndex.queryOpaquePacketsInRects(source.incremental.dirtyRects)
			: spatialIndex.queryTransparentPacketsInRects(source.incremental.dirtyRects);
	}

	private _resolveOpaqueReflectivePackets(packets: DrawPacket[]): DrawPacket[] {
		return packets.filter(
			(packet) => packet.material.reflectivity > 0 && packet.material.mirrorPlane !== null,
		);
	}

	private _resolveParticleDeltaTime(context: FrameContext): number {
		const value = context.transient.get(PARTICLE_SIM_DELTA_TIME_SECONDS_KEY);
		if (typeof value !== "number" || !Number.isFinite(value)) return 0;
		return Math.max(0, value);
	}

	public endParticleFrame(): void {
		if (!this._particleFrameActive) return;
		this._particleSimulator.endFrame();
		this._particleFrameActive = false;
	}

	public commitFrame(): void {
		this._postProcessRuntime.commitFrame();
		this._finishFrame();
	}

	public async abortFrame(error?: unknown): Promise<void> {
		try {
			await this._postProcessRuntime.abortFrame(error);
		} finally {
			this._finishFrame();
		}
	}

	private _finishFrame(): void {
		this.endParticleFrame();
		this._softwarePostProcessExecutor.unbindSoftwareFrame();
		this._sourceContext = null;
		this._passContext = null;
		this._postProcessPlan = null;
	}

	public getPostProcessDebugState(): unknown {
		return this._postProcessRuntime.getDebugState();
	}

	public get completedFramePreservesOutsideDirtyTiles(): boolean {
		return this._postProcessRuntime.completedFramePreservesOutsideDirtyTiles;
	}

	public get outputColorDomain(): import("../../postprocess/PostProcessPass").PostProcessColorDomain {
		return this._softwarePostProcessExecutor.outputColorDomain;
	}

	public invalidateFrameSized(): void {
		this._postProcessRuntime.invalidateFrameSized();
	}

	public destroy(): void {
		this.endParticleFrame();
		this._softwarePostProcessExecutor.unbindSoftwareFrame();
		this._postProcessRuntime.destroy();
		this._mainPass.destroy();
		this._services.shadow.resources.destroy();
		this._services.reflection.destroy();
		this._sourceContext = null;
		this._passContext = null;
		this._postProcessPlan = null;
	}

	public get reflectionRuntime(): import("./SoftwarePlanarReflectionRuntime").SoftwarePlanarReflectionRuntime {
		return this._reflectionPass.runtime;
	}

	private _requireSourceContext(): FrameContext {
		if (!this._sourceContext) {
			throw new Error("SoftwarePassExecutor requires an active coordinator frame.");
		}
		return this._sourceContext;
	}

	private _requirePassContext(): SoftwarePassContext {
		if (!this._passContext) {
			throw new Error("SoftwarePassExecutor requires a bound Software frame view.");
		}
		return this._passContext;
	}
}
