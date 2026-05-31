import {
	FRAME_PASS_DEPENDENCIES,
	type FrameContext,
	type FramePass,
} from "./types";

export interface FramePassPlanValidatorReporter {
	reportNonFatalError(scope: string, error: unknown): void;
}

export interface FramePassPlanValidatorState {
	executedPasses: Set<FramePass["stage"]>;
	plannedPasses: Set<FramePass["stage"]>;
	plannedPassOrder: Map<FramePass["stage"], number>;
}

/**
 * Validates backend pass execution against the renderer-owned frame plan.
 */
export class FramePassPlanValidator {
	private readonly _backendLabel: string;

	/**
	 * Creates a validator with backend-specific diagnostic labels.
	 *
	 * @param backendLabel Human-readable backend name used in thrown errors.
	 * @sideEffects None.
	 */
	public constructor(backendLabel: string) {
		this._backendLabel = backendLabel;
	}

	/**
	 * Loads enabled backend passes from `FrameContext.framePlan`.
	 *
	 * @param context Current frame context.
	 * @param state Mutable validation state owned by the backend.
	 * @returns Nothing.
	 * @sideEffects Clears and rewrites planned pass state.
	 */
	public preparePlan(
		context: FrameContext,
		state: FramePassPlanValidatorState
	): void {
		state.plannedPasses.clear();
		state.plannedPassOrder.clear();

		for (const pass of context.framePlan?.backendPasses ?? []) {
			if (pass.enabled) {
				state.plannedPasses.add(pass.stage);
			}
		}
		if (state.plannedPasses.size <= 0) {
			return;
		}
		this._validatePlannedPassGraph(state);
	}

	/**
	 * Ensures a pass is not executed before applicable planned dependencies.
	 *
	 * @param pass Pass about to execute.
	 * @param state Mutable validation state owned by the backend.
	 * @param reporter Non-fatal diagnostic sink.
	 * @returns Nothing.
	 * @sideEffects May report stale dependency diagnostics.
	 */
	public validatePassDependencies(
		pass: FramePass,
		state: FramePassPlanValidatorState,
		reporter: FramePassPlanValidatorReporter
	): void {
		if (state.plannedPasses.size <= 0) {
			return;
		}
		if (!state.plannedPasses.has(pass.stage)) {
			return;
		}
		const plannedIndex = state.plannedPassOrder.get(pass.stage);
		if (plannedIndex !== undefined) {
			const violated = Array.from(state.executedPasses).some((executedStage) => {
				const index = state.plannedPassOrder.get(executedStage);
				return index !== undefined && index > plannedIndex;
			});
			if (violated) {
				throw new Error(
					`${this._backendLabel} pass "${pass.stage}" execution order violates prevalidated pass plan.`
				);
			}
		}
		const dependencies = this._resolvePassDependencies(pass.stage);
		if (!dependencies || dependencies.length <= 0) {
			return;
		}
		const missing = dependencies.filter(
			(dependency) =>
				state.plannedPasses.has(dependency) &&
				this._isDependencyApplicable(pass.stage, dependency, state, reporter) &&
				!state.executedPasses.has(dependency)
		);
		if (missing.length <= 0) {
			return;
		}
		throw new Error(
			`${this._backendLabel} pass "${pass.stage}" executed before dependencies: ${missing.join(", ")}`
		);
	}

	/**
	 * Marks a backend pass as executed or skipped.
	 *
	 * @param stage Frame pass stage id.
	 * @param state Mutable validation state owned by the backend.
	 * @returns Nothing.
	 * @sideEffects Mutates executed pass state.
	 */
	public markPassExecuted(
		stage: FramePass["stage"],
		state: FramePassPlanValidatorState
	): void {
		state.executedPasses.add(stage);
	}

	private _validatePlannedPassGraph(
		state: FramePassPlanValidatorState
	): void {
		const visiting = new Set<FramePass["stage"]>();
		const visited = new Set<FramePass["stage"]>();
		const order: FramePass["stage"][] = [];

		const visit = (stage: FramePass["stage"]): void => {
			if (visited.has(stage)) {
				return;
			}
			if (visiting.has(stage)) {
				throw new Error(
					`${this._backendLabel} pass plan cycle detected at "${stage}" during frame plan validation.`
				);
			}
			visiting.add(stage);
			const dependencies = this._resolvePassDependencies(stage);
			for (const dependency of dependencies) {
				if (!state.plannedPasses.has(dependency)) {
					continue;
				}
				visit(dependency);
			}
			visiting.delete(stage);
			visited.add(stage);
			order.push(stage);
		};

		for (const stage of state.plannedPasses) {
			visit(stage);
		}
		for (let i = 0; i < order.length; i++) {
			state.plannedPassOrder.set(order[i], i);
		}
	}

	private _resolvePassDependencies(stage: FramePass["stage"]): FramePass["stage"][] {
		const dependencies = FRAME_PASS_DEPENDENCIES.get(stage);
		return dependencies ? Array.from(dependencies) : [];
	}

	private _isDependencyApplicable(
		stage: FramePass["stage"],
		dependency: FramePass["stage"],
		state: FramePassPlanValidatorState,
		reporter: FramePassPlanValidatorReporter
	): boolean {
		const stageIndex = state.plannedPassOrder.get(stage);
		const dependencyIndex = state.plannedPassOrder.get(dependency);
		if (stageIndex === undefined || dependencyIndex === undefined) {
			return true;
		}
		if (dependencyIndex < stageIndex) {
			return true;
		}
		reporter.reportNonFatalError(
			"pass dependency order",
			`Ignoring stale dependency "${dependency}" for "${stage}".`
		);
		return false;
	}
}
