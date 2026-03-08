import { DefaultAnimationSimulator } from '../simulation/animation/DefaultAnimationSimulator'
import type { AnimationSystem } from '../animation/AnimationSystem'

export interface AnimationSimulationStageContext {
	scene: any
	transient: Map<string, any>
}

export class AnimationSimulationStage {
	private _simulator: DefaultAnimationSimulator

	constructor(animationSystem: AnimationSystem) {
		this._simulator = new DefaultAnimationSimulator(animationSystem)
	}

	public execute(context: AnimationSimulationStageContext, deltaTimeMs: number): void {
		this._simulator.beginFrame(context)
		this._simulator.simulate(context, deltaTimeMs)
		this._simulator.endFrame()
	}
}
