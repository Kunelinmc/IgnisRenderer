import type { Scene } from '../../core/Scene'

export interface AnimationSimulationContext {
	scene: Scene
	transient: Map<string, any>
}

export interface IAnimationSimulator {
	beginFrame(context: AnimationSimulationContext): void
	simulate(context: AnimationSimulationContext, deltaTimeMs: number): void
	endFrame(): void
}
