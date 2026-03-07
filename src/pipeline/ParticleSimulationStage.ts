import type { FrameContext } from "./types"
import { DefaultParticleSimulator } from "../renderers/particles/DefaultParticleSimulator"

export class ParticleSimulationStage {
	private _simulator = new DefaultParticleSimulator({
		backendTag: "shared-adapter",
	})

	public execute(context: FrameContext, deltaTimeMs: number): void {
		this._simulator.beginFrame(context)
		this._simulator.simulate(context, deltaTimeMs)
		this._simulator.emitRenderBatches(context)
		this._simulator.endFrame()
	}
}

