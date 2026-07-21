import type {
	FrameContext,
} from "../../pipeline/types";
import type {
	RenderBackendProfile,
	WarmupOptions,
	WarmupReport,
} from "../IRenderBackend";
import {
	addWarmupPhase,
	buildWarmupPlan,
	createWarmupReport,
	finalizeWarmupReport,
	toShaderCompileError,
	type WarmupPhaseCounters,
	type WarmupPostProcessPlan,
} from "../../pipeline/WarmupPlanner";
import type { BackendPostProcessRuntime } from "../../postprocess/BackendPostProcessRuntime";
import type { PostProcessPlan } from "../../postprocess/PostProcessPlanner";
import type { WebGPUFrameOrchestrator } from "./rendergraph/WebGPUFrameOrchestrator";
import type { WebGPUFrameServiceOwner } from "./WebGPUFrameServiceOwner";

export interface WebGPUWarmupCoordinatorHost {
	readonly profile: RenderBackendProfile;
	readonly frameOrchestrator: WebGPUFrameOrchestrator | null;
	readonly resources: WebGPUFrameServiceOwner | null;
	readonly postProcessRuntime: BackendPostProcessRuntime;
	setWarmupLogCompilationInfo(enabled: boolean): void;
}

export class WebGPUWarmupCoordinator {
	constructor(private readonly _host: WebGPUWarmupCoordinatorHost) {}

	public async warmup(
		context: FrameContext,
		options: WarmupOptions = {}
	): Promise<WarmupReport> {
		const report = createWarmupReport(this._host.profile.id);
		if (!this._host.resources || !this._host.frameOrchestrator) {
			throw new Error("WebGPU backend has not been initialized.");
		}

		let warmupPostProcessPlan: WarmupPostProcessPlan | undefined;
		let postProcessPlan: PostProcessPlan | undefined;
		if (options.includePostProcess !== false) {
			const graph = this._host.postProcessRuntime.planWarmup(context);
			postProcessPlan = graph;
			warmupPostProcessPlan = {
				passIds: graph.orderedPasses.map((pass) => pass.id),
				descriptors: graph.orderedPasses.map((pass) => pass.pass),
			};
		}
		const plan = buildWarmupPlan(context, options, warmupPostProcessPlan);
		this._host.setWarmupLogCompilationInfo(options.logCompilationInfo === true);
		try {
			const framePhase = await this._host.frameOrchestrator.warmup(
				context,
				plan,
				options,
				postProcessPlan,
			);
			addWarmupPhase(report, framePhase);
			this._reportWarmupProgress(options, framePhase);
			const resourcePhase = await this._host.resources.warmup(
				context,
				plan,
				options
			);
			addWarmupPhase(report, resourcePhase);
			this._reportWarmupProgress(options, resourcePhase);
		} catch (error) {
			const failedPhase = {
				phase: "webgpu-warmup",
				total: 1,
				compiled: 0,
				skipped: 0,
				failed: 1,
				errors: [toShaderCompileError(error, this._host.profile.id, "WebGPUWarmup")],
			};
			addWarmupPhase(report, failedPhase);
			this._reportWarmupProgress(options, failedPhase);
		} finally {
			this._host.setWarmupLogCompilationInfo(false);
		}
		return finalizeWarmupReport(report);
	}

	private _reportWarmupProgress(
		options: WarmupOptions,
		phase: WarmupPhaseCounters,
	): void {
		options.onProgress?.({
			phase: phase.phase,
			completed: phase.compiled + phase.skipped + phase.failed,
			total: phase.total,
		});
	}
}
