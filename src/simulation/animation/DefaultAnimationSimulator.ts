import type { AnimationSystem } from "../../animation/AnimationSystem";
import { AnimationRuntime } from "./AnimationRuntime";
import type {
	AnimationSimulationContext,
	IAnimationSimulator,
} from "./IAnimationSimulator";

export class DefaultAnimationSimulator implements IAnimationSimulator {
	private _runtime = new AnimationRuntime();
	private _animationSystem: AnimationSystem;

	constructor(animationSystem: AnimationSystem) {
		this._animationSystem = animationSystem;
	}

	public beginFrame(_context: AnimationSimulationContext): void {}

	public simulate(
		context: AnimationSimulationContext,
		deltaTimeMs: number
	): void {
		this._runtime.update(
			this._animationSystem,
			Math.max(0, deltaTimeMs) / 1000,
			context.transient
		);
	}

	public endFrame(): void {}
}
