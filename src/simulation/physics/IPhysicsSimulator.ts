import type {
	PhysicsSimulationAsyncContext,
	PhysicsSimulationFrameContext,
	PhysicsSimulationContext,
	PhysicsSimulationRequest,
	PhysicsSimulationResult,
} from "./types";

export interface IPhysicsSimulator {
	beginFrame(context: PhysicsSimulationFrameContext): void;
	simulate(
		context: PhysicsSimulationContext,
		request: PhysicsSimulationRequest
	): PhysicsSimulationResult;
	simulateAsync?(
		context: PhysicsSimulationAsyncContext,
		request: PhysicsSimulationRequest
	): Promise<PhysicsSimulationResult>;
	endFrame(): void;
}
