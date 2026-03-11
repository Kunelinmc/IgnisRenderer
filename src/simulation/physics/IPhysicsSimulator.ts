import type {
	PhysicsSimulationContext,
	PhysicsSimulationRequest,
	PhysicsSimulationResult,
} from "./types";

export interface IPhysicsSimulator {
	beginFrame(context: PhysicsSimulationContext): void;
	simulate(
		context: PhysicsSimulationContext,
		request: PhysicsSimulationRequest
	): PhysicsSimulationResult;
	endFrame(): void;
}
