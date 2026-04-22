import { DefaultAnimationSimulator } from "../simulation/animation/DefaultAnimationSimulator";
import type { AnimationSystem } from "../animation/AnimationSystem";
import type { Scene } from "../core/Scene";
import type { TransientStore } from "./types";

export interface AnimationSimulationStageContext {
	scene: Scene;
	transient: TransientStore;
}

export class AnimationSimulationStage {
	private _simulator: DefaultAnimationSimulator;

	constructor(animationSystem: AnimationSystem) {
		this._simulator = new DefaultAnimationSimulator(animationSystem);
	}

	public execute(
		context: AnimationSimulationStageContext,
		deltaTimeMs: number
	): void {
		this._simulator.beginFrame(context);
		this._simulator.simulate(context, deltaTimeMs);
		this._simulator.endFrame();
	}
}
