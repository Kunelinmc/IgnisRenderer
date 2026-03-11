import type { PhysicsStepConfig, PhysicsStepMode } from "../../physics/types";
import type { IPhysicsSimulator } from "./IPhysicsSimulator";
import type {
	PhysicsSimulationContext,
	PhysicsSimulationRequest,
	PhysicsSimulationResult,
	PhysicsSimulationWorldTarget,
	PhysicsWorldSimulationResult,
} from "./types";

const DEFAULT_FIXED_DELTA_MS = 1000 / 60;
const DEFAULT_MAX_SUBSTEPS = 5;
const DEFAULT_MAX_DELTA_MS = 100;

interface WorldRuntimeState {
	accumulatorMs: number;
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
		const inputDeltaMs = Math.max(0, request.deltaTimeMs);
		const scopedWorldIds = new Set(request.override?.worldIds ?? []);
		const useScope = scopedWorldIds.size > 0;

		for (const world of context.worlds) {
			if (useScope && !scopedWorldIds.has(world.worldId)) continue;
			worldResults.push(this._simulateWorld(context, world, request));
		}

		const processedDeltaMs = worldResults.reduce((maxValue, item) => {
			return Math.max(maxValue, item.consumedDeltaMs);
		}, 0);

		return {
			inputDeltaMs,
			processedDeltaMs,
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
		const clampedDeltaMs = Math.min(
			Math.max(0, request.deltaTimeMs),
			config.maxDeltaMs
		);
		const mode = config.mode;

		if (mode === "variable") {
			if (clampedDeltaMs <= 0) {
				return {
					worldId: world.worldId,
					mode,
					substeps: 0,
					consumedDeltaMs: 0,
					steps: [],
				};
			}
			const step = context.stepWorld(world.worldId, clampedDeltaMs / 1000);
			return {
				worldId: world.worldId,
				mode,
				substeps: 1,
				consumedDeltaMs: clampedDeltaMs,
				steps: [step],
			};
		}

		const runtime = this._getRuntime(world.worldId);
		runtime.accumulatorMs += clampedDeltaMs;

		const steps = [];
		let substeps = 0;
		let consumedDeltaMs = 0;
		while (
			runtime.accumulatorMs >= config.fixedDeltaMs &&
			substeps < config.maxSubsteps
		) {
			const step = context.stepWorld(world.worldId, config.fixedDeltaMs / 1000);
			steps.push(step);
			substeps++;
			consumedDeltaMs += config.fixedDeltaMs;
			runtime.accumulatorMs -= config.fixedDeltaMs;
		}

		return {
			worldId: world.worldId,
			mode,
			substeps,
			consumedDeltaMs,
			steps,
		};
	}

	private _getRuntime(worldId: string): WorldRuntimeState {
		let runtime = this._runtimeByWorldId.get(worldId);
		if (runtime) return runtime;
		runtime = {
			accumulatorMs: 0,
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
		fixedDeltaMs: sanitizePositive(
			override?.fixedDeltaMs ??
				worldConfig.fixedDeltaMs ??
				DEFAULT_FIXED_DELTA_MS,
			DEFAULT_FIXED_DELTA_MS
		),
		maxSubsteps: Math.max(
			1,
			Math.floor(
				override?.maxSubsteps ?? worldConfig.maxSubsteps ?? DEFAULT_MAX_SUBSTEPS
			)
		),
		maxDeltaMs: sanitizePositive(
			override?.maxDeltaMs ?? worldConfig.maxDeltaMs ?? DEFAULT_MAX_DELTA_MS,
			DEFAULT_MAX_DELTA_MS
		),
	};
}

function sanitizePositive(value: number, fallback: number): number {
	if (!Number.isFinite(value) || value <= 0) {
		return fallback;
	}
	return value;
}
