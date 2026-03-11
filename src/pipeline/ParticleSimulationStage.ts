import type { FrameContext } from "./types";
import { DefaultParticleSimulator } from "../simulation/particles/DefaultParticleSimulator";

export class ParticleSimulationStage {
	private _simulator = new DefaultParticleSimulator({
		backendTag: "shared-adapter",
	});

	public execute(context: FrameContext, deltaTimeSeconds: number): void {
		this._simulator.beginFrame(context);
		this._simulator.simulate(context, deltaTimeSeconds);
		this._simulator.emitRenderBatches(context);
		this._simulator.endFrame();
	}
}
