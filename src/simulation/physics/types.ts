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

export interface PhysicsWorldSimulationResult {
	worldId: string;
	mode: PhysicsStepMode;
	substeps: number;
	consumedDeltaMs: number;
	steps: PhysicsAdapterStepResult[];
}

export interface PhysicsSimulationResult {
	inputDeltaMs: number;
	processedDeltaMs: number;
	worldResults: PhysicsWorldSimulationResult[];
}

export interface PhysicsSimulationContext {
	worlds: PhysicsSimulationWorldTarget[];
	stepWorld(worldId: string, deltaSeconds: number): PhysicsAdapterStepResult;
}

export interface PhysicsSimulationRequest {
	deltaTimeMs: number;
	override?: StepOverride;
}
