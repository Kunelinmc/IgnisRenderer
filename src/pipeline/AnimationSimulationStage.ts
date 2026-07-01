import type { AnimationSystem } from "../animation/AnimationSystem";
import type { Scene } from "../core/Scene";
import { AnimationRuntime } from "../simulation/animation/AnimationRuntime";
import type { TransientStore } from "./types";

export interface AnimationSimulationStageContext {
	scene: Scene;
	transient: TransientStore;
}

export class AnimationSimulationStage {
	private _runtime = new AnimationRuntime();
	private _animationSystem: AnimationSystem;

	constructor(animationSystem: AnimationSystem) {
		this._animationSystem = animationSystem;
	}

	public execute(
		context: AnimationSimulationStageContext,
		deltaTimeSeconds: number
	): void {
		this._runtime.update(
			this._animationSystem,
			Math.max(0, deltaTimeSeconds),
			context.transient,
			context.scene
		);
	}
}
