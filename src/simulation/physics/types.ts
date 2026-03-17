import type {
	PhysicsStepMode,
	PhysicsWorldConfig,
	StepOverride,
} from "../../physics/types";
import type { PhysicsAdapterStepResult } from "../../physics/IPhysicsEngineAdapter";

export interface PhysicsSimulationWorldTarget {
	worldId: string;
	config: PhysicsWorldConfig;
}

export interface PhysicsSimulationFrameContext {
	worlds: PhysicsSimulationWorldTarget[];
}

export interface PhysicsWorldSimulationResult {
	worldId: string;
	mode: PhysicsStepMode;
	substeps: number;
	consumedDeltaSeconds: number;
	steps: PhysicsAdapterStepResult[];
}

export interface PhysicsSimulationResult {
	inputDeltaSeconds: number;
	processedDeltaSeconds: number;
	worldResults: PhysicsWorldSimulationResult[];
}

export interface PhysicsSimulationContext {
	worlds: PhysicsSimulationWorldTarget[];
	stepWorld(worldId: string, deltaSeconds: number): PhysicsAdapterStepResult;
}

export interface PhysicsSimulationAsyncContext {
	worlds: PhysicsSimulationWorldTarget[];
	stepWorld(
		worldId: string,
		deltaSeconds: number
	): Promise<PhysicsAdapterStepResult>;
}

export interface PhysicsSimulationRequest {
	deltaTimeSeconds: number;
	override?: StepOverride;
}
