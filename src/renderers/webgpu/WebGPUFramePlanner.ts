import {
	FRAME_PASS_DEPENDENCIES,
	type FrameContext,
	type FramePass,
} from "../../pipeline/types";
import { hasParticleShadowCasters } from "../../pipeline/ParticleShadowVolume";
import type {
	WebGPUFramePlanner,
	WebGPUFramePlannerReporter,
	WebGPUFramePlannerState,
} from "./WebGPUBackendContracts";

export class WebGPUPassPlanner implements WebGPUFramePlanner {
	public preparePlan(context: FrameContext, state: WebGPUFramePlannerState): void {
		state.plannedPasses.clear();
		state.plannedPassOrder.clear();

		const hasParticleSystems = (context.scene.particleSystems?.length ?? 0) > 0;
		if (hasParticleSystems) {
			state.plannedPasses.add("particle-sim");
		}
		if (
			context.features.enableShadows &&
			(context.scene.shadowCasterPackets.length ||
				context.scene.shadowTransmitterPackets.length ||
				hasParticleShadowCasters(context.scene.particleSystems))
		) {
			state.plannedPasses.add("shadow");
		}
		if (context.features.enableReflection && context.scene.reflectivePackets.length) {
			state.plannedPasses.add("reflection");
		}
		state.plannedPasses.add("main-opaque");
		if (context.scene.transparentPackets.length > 0) {
			state.plannedPasses.add("main-transparent");
		}
		if (hasParticleSystems) {
			state.plannedPasses.add("particles");
		}
		this._validatePlannedPassGraph(state);
	}

	public validatePassDependencies(
		pass: FramePass,
		state: WebGPUFramePlannerState,
		reporter: WebGPUFramePlannerReporter
	): void {
		if (state.plannedPasses.size > 0 && !state.plannedPasses.has(pass.stage)) {
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
					`WebGPU pass "${pass.stage}" execution order violates prevalidated pass plan.`
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
			`WebGPU pass "${pass.stage}" executed before dependencies: ${missing.join(", ")}`
		);
	}

	public markPassExecuted(
		stage: FramePass["stage"],
		state: WebGPUFramePlannerState
	): void {
		state.executedPasses.add(stage);
	}

	private _validatePlannedPassGraph(state: WebGPUFramePlannerState): void {
		const visiting = new Set<FramePass["stage"]>();
		const visited = new Set<FramePass["stage"]>();
		const order: FramePass["stage"][] = [];

		const visit = (stage: FramePass["stage"]): void => {
			if (visited.has(stage)) {
				return;
			}
			if (visiting.has(stage)) {
				throw new Error(
					`WebGPU pass plan cycle detected at "${stage}" during _prepareFramePassPlan.`
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
		state: WebGPUFramePlannerState,
		reporter: WebGPUFramePlannerReporter
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
