import type { FrameContext, FramePass } from "../../pipeline/types";
import { PARTICLE_SIM_DELTA_TIME_SECONDS_KEY } from "../../pipeline/types";
import type { BackendPostProcessRuntime } from "../../postprocess/BackendPostProcessRuntime";
import type { IParticleSimulator } from "../../simulation/particles/IParticleSimulator";
import type { WebGPUFrameExecutor } from "./WebGPUFrameExecutor";
import type { WebGPURenderResources } from "./WebGPURenderResources";

type ParticleSimulatorWithBatchEmit = IParticleSimulator & {
	simulateAndEmitRenderBatches?: (
		context: FrameContext,
		deltaTimeSeconds: number
	) => Promise<void>;
};

export interface WebGPUBackendPassDispatcherHost {
	readonly frameExecutor: WebGPUFrameExecutor | null;
	readonly particleSimulator: IParticleSimulator | null;
	readonly postProcessRuntime: BackendPostProcessRuntime;
	readonly resources: WebGPURenderResources | null;
}

export class WebGPUBackendPassDispatcher {
	public constructor(private readonly _host: WebGPUBackendPassDispatcherHost) {}

	public executePass(
		pass: FramePass,
		context: FrameContext
	): Promise<void> | void | null {
		switch (pass.stage) {
			case "animation-sim":
				return undefined;
			case "particle-sim":
				return this._executeParticleSimulation(context);
			case "postprocess":
				return this._host.postProcessRuntime.execute(context);
			default:
				return null;
		}
	}

	private async _executeParticleSimulation(context: FrameContext): Promise<void> {
		const deltaTimeSeconds = this._resolveParticleDeltaTime(context);
		const simulator =
			this._host.particleSimulator as ParticleSimulatorWithBatchEmit | null;
		if (simulator?.simulateAndEmitRenderBatches) {
			await simulator.simulateAndEmitRenderBatches(context, deltaTimeSeconds);
		} else {
			simulator?.simulate(context, deltaTimeSeconds);
			simulator?.emitRenderBatches(context);
		}
		const frameResources =
			this._host.frameExecutor?.getPreparedFrameResources();
		if (frameResources) {
			this._host.resources?.updateParticleShadowVolumes?.(
				frameResources,
				context
			);
		}
	}

	private _resolveParticleDeltaTime(context: FrameContext): number {
		const value = context.transient.get(PARTICLE_SIM_DELTA_TIME_SECONDS_KEY);
		if (typeof value !== "number" || !Number.isFinite(value)) {
			return 0;
		}
		return Math.max(0, value);
	}
}
