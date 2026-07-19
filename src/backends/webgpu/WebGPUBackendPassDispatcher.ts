import type { FrameContext, FramePass } from "../../pipeline/types";
import { PARTICLE_SIM_DELTA_TIME_SECONDS_KEY } from "../../pipeline/types";
import type { BackendPostProcessRuntime } from "../../postprocess/BackendPostProcessRuntime";
import type { IParticleSimulator } from "../../simulation/particles/IParticleSimulator";
import type { WebGPUFrameOrchestrator } from "./rendergraph/WebGPUFrameOrchestrator";

type ParticleSimulatorWithBatchEmit = IParticleSimulator & {
	simulateAndEmitRenderBatches?: (
		context: FrameContext,
		deltaTimeSeconds: number
	) => Promise<void>;
};

export interface WebGPUBackendPassDispatcherHost {
	readonly frameOrchestrator: WebGPUFrameOrchestrator | null;
	readonly particleSimulator: IParticleSimulator | null;
	readonly postProcessRuntime: BackendPostProcessRuntime;
}

export class WebGPUBackendPassDispatcher {
	constructor(private readonly _host: WebGPUBackendPassDispatcherHost) {}

	public executePass(pass: FramePass, context: FrameContext): Promise<void> | void | null {
		switch (pass.stage) {
			case "animation-sim":
				return undefined;
			case "particle-sim":
				return this._executeParticleSimulation(context);
			default:
				return null;
		}
	}

	private async _executeParticleSimulation(context: FrameContext): Promise<void> {
		const deltaTimeSeconds = this._resolveParticleDeltaTime(context);
		const simulator = this._host.particleSimulator as ParticleSimulatorWithBatchEmit | null;
		if (simulator?.simulateAndEmitRenderBatches) {
			await simulator.simulateAndEmitRenderBatches(context, deltaTimeSeconds);
		} else {
			simulator?.simulate(context, deltaTimeSeconds);
			simulator?.emitRenderBatches(context);
		}
		this._host.frameOrchestrator?.updateParticleShadowVolumes(context);
	}

	private _resolveParticleDeltaTime(context: FrameContext): number {
		const value = context.transient.get(PARTICLE_SIM_DELTA_TIME_SECONDS_KEY);
		if (typeof value !== "number" || !Number.isFinite(value)) {
			return 0;
		}
		return Math.max(0, value);
	}
}
