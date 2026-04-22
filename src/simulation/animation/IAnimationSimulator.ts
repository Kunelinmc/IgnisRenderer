import type { Scene } from "../../core/Scene";
import type { TransientStore } from "../../pipeline/types";

export interface AnimationSimulationContext {
	scene: Scene;
	transient: TransientStore;
}

export interface IAnimationSimulator {
	beginFrame(context: AnimationSimulationContext): void;
	simulate(context: AnimationSimulationContext, deltaTimeMs: number): void;
	endFrame(): void;
}
