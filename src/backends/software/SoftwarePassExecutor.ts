import type { IRenderBackend } from "../IRenderBackend";
import type { DrawPacket, FrameContext } from "../../pipeline/types";
import type { SoftwareBackendOptions } from "./types";
import { Rasterizer } from "./Rasterizer";
import { SoftwarePostProcessExecutor } from "./SoftwarePostProcessExecutor";
import { BackendPostProcessRuntime } from "../../postprocess/BackendPostProcessRuntime";
import { SoftwareMainPass } from "./passes/SoftwareMainPass";
import { SoftwareParticlePass } from "./passes/SoftwareParticlePass";
import { SoftwareReflectionPass } from "./passes/SoftwareReflectionPass";
import { SoftwareShadowPass } from "./passes/SoftwareShadowPass";
import { DefaultParticleSimulator } from "../../simulation/particles/DefaultParticleSimulator";
import { Logger } from "../../foundation/Logger";
import {
	createSoftwareShadowSampler,
	getSoftwareShadowRuntimeMap,
} from "./passes/SoftwareShadowPass";

/** @internal Owns Software-specific passes and per-frame execution resources. */
export class SoftwarePassExecutor {
	private readonly _rasterizer = new Rasterizer();
	private readonly _mainPass: SoftwareMainPass;
	private readonly _particlePass = new SoftwareParticlePass();
	private readonly _shadowPass = new SoftwareShadowPass(this._rasterizer);
	private readonly _reflectionPass = new SoftwareReflectionPass(this._rasterizer);
	private readonly _particleSimulator: DefaultParticleSimulator;
	private readonly _postProcessRuntime: BackendPostProcessRuntime;
	private _particleFrameActive = false;

	public constructor(options: {
		backend: IRenderBackend;
		backendOptions: SoftwareBackendOptions;
		getCanvasContext: () => CanvasRenderingContext2D | null;
	}) {
		this._mainPass = new SoftwareMainPass(this._rasterizer, {
			enableEarlyZPrepass: options.backendOptions.enableEarlyZPrepass,
		});
		this._particleSimulator = new DefaultParticleSimulator({
			backendTag: options.backend.profile.id,
		});
		this._postProcessRuntime = new BackendPostProcessRuntime({
			executor: new SoftwarePostProcessExecutor({
				getCanvasContext: options.getCanvasContext,
				getShadowSampler: (context) =>
					createSoftwareShadowSampler(
						context.shadowMaps,
						getSoftwareShadowRuntimeMap(context.transient),
						{ camera: context.viewCamera },
					),
			}),
			backend: options.backend,
			warn: (key, message) =>
				Logger.warn(`[${key}] ${message}`, {
					scope: "SoftwareBackend",
					onceKey: key,
				}),
		});
	}

	public beginFrame(context: FrameContext): void {
		this._particleSimulator.beginFrame(context);
		this._particleFrameActive = true;
	}

	public endParticleFrame(): void {
		if (!this._particleFrameActive) return;
		this._particleSimulator.endFrame();
		this._particleFrameActive = false;
	}

	public simulateParticles(context: FrameContext, deltaTimeSeconds: number): void {
		this._particleSimulator.simulate(context, deltaTimeSeconds);
		this._particleSimulator.emitRenderBatches(context);
	}

	public renderShadows(context: FrameContext): void {
		this._shadowPass.render(context);
	}

	public renderReflections(context: FrameContext): void {
		this._reflectionPass.render(context);
	}

	public async renderOpaque(
		context: FrameContext,
		packets: DrawPacket[],
		reflectivePackets: DrawPacket[],
	): Promise<void> {
		await this._mainPass.render(context, packets, false);
		this._reflectionPass.composite(context, reflectivePackets);
	}

	public async renderTransparent(
		context: FrameContext,
		packets: DrawPacket[],
	): Promise<void> {
		await this._mainPass.render(context, packets, true);
	}

	public renderParticles(context: FrameContext): void {
		this._particlePass.render(context);
	}

	public executePostProcess(context: FrameContext): Promise<void> {
		return this._postProcessRuntime.execute(context);
	}

	public commitFrame(): void {
		this._postProcessRuntime.commitFrame();
	}

	public abortFrame(error?: unknown): Promise<void> {
		return this._postProcessRuntime.abortFrame(error);
	}

	public getPostProcessDebugState(): unknown {
		return this._postProcessRuntime.getDebugState();
	}

	public get completedFramePreservesOutsideDirtyTiles(): boolean {
		return this._postProcessRuntime.completedFramePreservesOutsideDirtyTiles;
	}

	public invalidateFrameSized(): void {
		this._postProcessRuntime.invalidateFrameSized();
	}

	public destroy(): void {
		this.endParticleFrame();
		this._postProcessRuntime.destroy();
		this._mainPass.destroy();
	}
}
