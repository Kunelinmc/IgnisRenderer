import type { PhysicsStepConfig, PhysicsStepMode } from "../../physics/types";
import type { IPhysicsSimulator } from "./IPhysicsSimulator";
import type {
	PhysicsSimulationContext,
	PhysicsSimulationRequest,
	PhysicsSimulationResult,
	PhysicsSimulationWorldTarget,
	PhysicsWorldSimulationResult,
} from "./types";
import {
	DEFAULT_FIXED_DELTA_SECONDS,
	DEFAULT_MAX_DELTA_SECONDS,
	DEFAULT_MAX_SUBSTEPS,
} from "./constants";

interface WorldRuntimeState {
	accumulatorSeconds: number;
}

export class DefaultPhysicsSimulator implements IPhysicsSimulator {
	private _runtimeByWorldId = new Map<string, WorldRuntimeState>();

	public beginFrame(context: PhysicsSimulationContext): void {
		const active = new Set(context.worlds.map((world) => world.worldId));
		for (const worldId of this._runtimeByWorldId.keys()) {
			if (!active.has(worldId)) {
				this._runtimeByWorldId.delete(worldId);
			}
		}
	}

	public simulate(
		context: PhysicsSimulationContext,
		request: PhysicsSimulationRequest
	): PhysicsSimulationResult {
		const worldResults: PhysicsWorldSimulationResult[] = [];
		const inputDeltaSeconds = Math.max(0, request.deltaTimeSeconds);
		const scopedWorldIds = new Set(request.override?.worldIds ?? []);
		const useScope = scopedWorldIds.size > 0;

		for (const world of context.worlds) {
			if (useScope && !scopedWorldIds.has(world.worldId)) continue;
			worldResults.push(this._simulateWorld(context, world, request));
		}

		const processedDeltaSeconds = worldResults.reduce((maxValue, item) => {
			return Math.max(maxValue, item.consumedDeltaSeconds);
		}, 0);

		return {
			inputDeltaSeconds,
			processedDeltaSeconds,
			worldResults,
		};
	}

	public endFrame(): void {}

	private _simulateWorld(
		context: PhysicsSimulationContext,
		world: PhysicsSimulationWorldTarget,
		request: PhysicsSimulationRequest
	): PhysicsWorldSimulationResult {
		const config = resolveStepConfig(world.config, request.override);
		const clampedDeltaSeconds = Math.min(
			Math.max(0, request.deltaTimeSeconds),
			config.maxDeltaSeconds
		);
		const mode = config.mode;

		if (mode === "variable") {
			if (clampedDeltaSeconds <= 0) {
				return {
					worldId: world.worldId,
					mode,
					substeps: 0,
					consumedDeltaSeconds: 0,
					steps: [],
				};
			}
			const step = context.stepWorld(world.worldId, clampedDeltaSeconds);
			return {
				worldId: world.worldId,
				mode,
				substeps: 1,
				consumedDeltaSeconds: clampedDeltaSeconds,
				steps: [step],
			};
		}

		const runtime = this._getRuntime(world.worldId);
		runtime.accumulatorSeconds += clampedDeltaSeconds;

		const steps = [];
		let substeps = 0;
		let consumedDeltaSeconds = 0;
		while (
			runtime.accumulatorSeconds >= config.fixedDeltaSeconds &&
			substeps < config.maxSubsteps
		) {
			const step = context.stepWorld(world.worldId, config.fixedDeltaSeconds);
			steps.push(step);
			substeps++;
			consumedDeltaSeconds += config.fixedDeltaSeconds;
			runtime.accumulatorSeconds -= config.fixedDeltaSeconds;
		}

		return {
			worldId: world.worldId,
			mode,
			substeps,
			consumedDeltaSeconds,
			steps,
		};
	}

	private _getRuntime(worldId: string): WorldRuntimeState {
		let runtime = this._runtimeByWorldId.get(worldId);
		if (runtime) return runtime;
		runtime = {
			accumulatorSeconds: 0,
		};
		this._runtimeByWorldId.set(worldId, runtime);
		return runtime;
	}
}

function resolveStepConfig(
	worldConfig: PhysicsStepConfig,
	override: PhysicsStepConfig | undefined
): Required<PhysicsStepConfig> & { mode: PhysicsStepMode } {
	const mode = override?.mode ?? worldConfig.mode ?? "fixed";
	return {
		mode,
		fixedDeltaSeconds: sanitizePositive(
			override?.fixedDeltaSeconds ??
				worldConfig.fixedDeltaSeconds ??
				DEFAULT_FIXED_DELTA_SECONDS,
			DEFAULT_FIXED_DELTA_SECONDS
		),
		maxSubsteps: Math.max(
			1,
			Math.floor(
				override?.maxSubsteps ?? worldConfig.maxSubsteps ?? DEFAULT_MAX_SUBSTEPS
			)
		),
		maxDeltaSeconds: sanitizePositive(
			override?.maxDeltaSeconds ??
				worldConfig.maxDeltaSeconds ??
				DEFAULT_MAX_DELTA_SECONDS,
			DEFAULT_MAX_DELTA_SECONDS
		),
	};
}

function sanitizePositive(value: number, fallback: number): number {
	if (!Number.isFinite(value) || value <= 0) {
		return fallback;
	}
	return value;
}
