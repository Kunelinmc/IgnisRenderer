import type { FrameContext } from "../../pipeline/types";
import {
	toShaderCompileError,
	type WarmupPhaseCounters,
	type WarmupPlan,
} from "../../pipeline/WarmupPlanner";
import { createWarmupYieldController } from "../../pipeline/WarmupScheduler";
import type { PostProcessPlan } from "../../postprocess/PostProcessPlanner";
import type { WarmupOptions } from "../IRenderBackend";

import type { WebGLProgramCompiler, WebGLProgramWarmupHandle } from "./WebGLProgramCompiler";
import {
	WebGLProgramWarmupQueue,
	type WebGLProgramWarmupPriority,
} from "./WebGLProgramWarmupQueue";

export interface WebGLProgramWarmupRequest {
	readonly context: FrameContext;
	readonly plan: WarmupPlan;
	readonly postProcessPlan: PostProcessPlan | null;
}

export interface WebGLProgramWarmupTask {
	readonly label: string;
	readonly priority: WebGLProgramWarmupPriority;
	run(): unknown | Promise<unknown>;
}

/** @internal Static frame-owned source of WebGL warmup work. */
export interface WebGLProgramWarmupContributor {
	collectWarmupTasks(request: WebGLProgramWarmupRequest): readonly WebGLProgramWarmupTask[];
}

export interface WebGLWarmupCoordinatorServices {
	readonly compiler: WebGLProgramCompiler;
	readonly contributors: readonly WebGLProgramWarmupContributor[];
}

/** Compiles WebGL program variants without coupling warmup to frame execution. */
export class WebGLWarmupCoordinator {
	private readonly _services: WebGLWarmupCoordinatorServices;

	constructor(services: WebGLWarmupCoordinatorServices) {
		this._services = services;
	}

	public async warmup(
		context: FrameContext,
		plan: WarmupPlan,
		options: WarmupOptions = {},
		postProcessPlan?: PostProcessPlan,
		signal?: AbortSignal | null,
	): Promise<WarmupPhaseCounters> {
		const yieldController = createWarmupYieldController(options);
		const queue = new WebGLProgramWarmupQueue();
		const request = {
			context,
			plan,
			postProcessPlan: postProcessPlan ?? null,
		};
		for (const contributor of this._services.contributors) {
			for (const task of contributor.collectWarmupTasks(request)) {
				queue.enqueue({
					label: task.label,
					priority: task.priority,
					action: () => this._collectWarmupHandles(() => task.run()),
				});
			}
		}

		const result = await queue.run(yieldController, options, signal);
		return {
			phase: "webgl-programs",
			total: result.handles + result.enqueueFailures,
			compiled: result.compiled,
			skipped: 0,
			failed: result.enqueueFailures + result.failed,
			errors: result.errors.map((entry) =>
				toShaderCompileError(entry.error, "webgl", entry.label),
			),
		};
	}

	private async _collectWarmupHandles(
		action: () => unknown | Promise<unknown>,
	): Promise<WebGLProgramWarmupHandle[]> {
		const compiler = this._services.compiler;
		const mark = compiler.markWarmupHandles();
		const result = await action();
		const logged = compiler.collectWarmupHandlesSince(mark);
		if (logged.length > 0) return logged;
		if (isWebGLProgramWarmupHandle(result)) return [result];
		if (Array.isArray(result) && result.every(isWebGLProgramWarmupHandle)) {
			return result;
		}
		return [];
	}
}

function isWebGLProgramWarmupHandle(value: unknown): value is WebGLProgramWarmupHandle {
	return (
		typeof value === "object" &&
		value !== null &&
		"label" in value &&
		typeof (value as { isComplete?: unknown }).isComplete === "function" &&
		typeof (value as { finalize?: unknown }).finalize === "function"
	);
}
