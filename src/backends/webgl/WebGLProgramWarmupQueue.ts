import type { WarmupOptions } from "../IRenderBackend";
import type { WebGLProgramWarmupHandle } from "./WebGLProgramCompiler";
import type { WarmupYieldController } from "../../pipeline/WarmupScheduler";

export type WebGLProgramWarmupPriority = "core" | "optional" | "postprocess";

export interface WebGLProgramWarmupQueueItem {
	readonly label: string;
	readonly priority: WebGLProgramWarmupPriority;
	readonly action: () =>
		| readonly WebGLProgramWarmupHandle[]
		| Promise<readonly WebGLProgramWarmupHandle[]>;
}

export interface WebGLProgramWarmupQueueOptions {
	/**
	 * Optional hard cap on finalizations per scheduling slice. When omitted,
	 * only `timeBudgetMs` bounds a slice.
	 */
	readonly maxFinalizesPerSlice?: number;
	/** Wall-clock budget per scheduling slice; defaults to 2 ms. */
	readonly timeBudgetMs?: number;
	/** Monotonic clock used for budget checks; injectable for tests. */
	readonly now?: () => number;
	readonly waitForSlice?: () => Promise<void>;
}

export interface WebGLProgramWarmupQueueError {
	readonly label: string;
	readonly error: unknown;
}

export interface WebGLProgramWarmupQueueResult {
	readonly handles: number;
	readonly enqueueFailures: number;
	readonly compiled: number;
	readonly failed: number;
	readonly errors: WebGLProgramWarmupQueueError[];
}

const WEBGL_WARMUP_PRIORITY_ORDER: Record<WebGLProgramWarmupPriority, number> = {
	core: 0,
	optional: 1,
	postprocess: 2,
};

const DEFAULT_SLICE_TIME_BUDGET_MS = 2;

interface PendingWarmupHandle {
	readonly label: string;
	readonly handle: WebGLProgramWarmupHandle;
}

interface QueuedWarmupItem extends WebGLProgramWarmupQueueItem {
	readonly order: number;
}

/**
 * Coordinates WebGL program warmup so expensive status/link finalization is
 * spread across browser scheduling slices bounded by a wall-clock budget
 * (default 2 ms) with an optional hard per-slice count cap.
 *
 * @internal WebGL backend warmup infrastructure only.
 */
export class WebGLProgramWarmupQueue {
	private readonly _items: QueuedWarmupItem[] = [];
	private readonly _maxFinalizesPerSlice: number | null;
	private readonly _timeBudgetMs: number;
	private readonly _now: () => number;
	private readonly _waitForSlice: () => Promise<void>;
	private _order = 0;

	public constructor(options: WebGLProgramWarmupQueueOptions = {}) {
		this._maxFinalizesPerSlice =
			options.maxFinalizesPerSlice !== undefined
				? Math.max(1, Math.floor(options.maxFinalizesPerSlice))
				: null;
		this._timeBudgetMs = Math.max(
			0,
			options.timeBudgetMs ?? DEFAULT_SLICE_TIME_BUDGET_MS
		);
		this._now = options.now ?? (() => performance.now());
		this._waitForSlice = options.waitForSlice ?? waitForWebGLWarmupSlice;
	}

	public enqueue(item: WebGLProgramWarmupQueueItem): void {
		this._items.push({
			...item,
			order: this._order++,
		});
	}

	public async run(
		yieldController: WarmupYieldController,
		_options: WarmupOptions = {},
		signal?: AbortSignal | null,
	): Promise<WebGLProgramWarmupQueueResult> {
		const pending = new Map<string, PendingWarmupHandle>();
		const errors: WebGLProgramWarmupQueueError[] = [];
		let enqueueFailures = 0;
		let compiled = 0;
		let failed = 0;

		for (const item of this._sortedItems()) {
			assertWarmupNotAborted(signal);
			try {
				const handles = await item.action();
				assertWarmupNotAborted(signal);
				for (const handle of handles) {
					pending.set(handle.label, {
						label: handle.label,
						handle,
					});
				}
			} catch (error) {
				enqueueFailures++;
				errors.push({ label: item.label, error });
			}

			const progress = await this._finalizeReadyHandles(pending, errors);
			assertWarmupNotAborted(signal);
			compiled += progress.compiled;
			failed += progress.failed;
			if (progress.compiled > 0 || progress.failed > 0) {
				await this._waitForSlice();
				assertWarmupNotAborted(signal);
			}
			await yieldController.yieldIfNeeded();
			assertWarmupNotAborted(signal);
		}

		while (pending.size > 0) {
			assertWarmupNotAborted(signal);
			const progress = await this._finalizeReadyHandles(pending, errors);
			assertWarmupNotAborted(signal);
			compiled += progress.compiled;
			failed += progress.failed;
			if (progress.compiled === 0 && progress.failed === 0) {
				await this._waitForSlice();
				assertWarmupNotAborted(signal);
				continue;
			}
			if (pending.size > 0) {
				await this._waitForSlice();
				assertWarmupNotAborted(signal);
			}
			await yieldController.yieldIfNeeded();
			assertWarmupNotAborted(signal);
		}

		return {
			handles: compiled + failed,
			enqueueFailures,
			compiled,
			failed,
			errors,
		};
	}

	private _sortedItems(): QueuedWarmupItem[] {
		return [...this._items].sort((a, b) => {
			const priorityDelta =
				WEBGL_WARMUP_PRIORITY_ORDER[a.priority] -
				WEBGL_WARMUP_PRIORITY_ORDER[b.priority];
			return priorityDelta || a.order - b.order;
		});
	}

	private async _finalizeReadyHandles(
		pending: Map<string, PendingWarmupHandle>,
		errors: WebGLProgramWarmupQueueError[]
	): Promise<{ compiled: number; failed: number }> {
		let compiled = 0;
		let failed = 0;
		const sliceStartedAt = this._now();
		for (const entry of Array.from(pending.values())) {
			if (
				this._maxFinalizesPerSlice !== null &&
				compiled + failed >= this._maxFinalizesPerSlice
			) {
				break;
			}
			if (
				compiled + failed > 0 &&
				this._now() - sliceStartedAt >= this._timeBudgetMs
			) {
				break;
			}
			let complete = false;
			try {
				complete = entry.handle.isComplete();
			} catch (error) {
				pending.delete(entry.label);
				errors.push({ label: entry.label, error });
				failed++;
				continue;
			}
			if (!complete) {
				continue;
			}
			try {
				entry.handle.finalize();
				compiled++;
			} catch (error) {
				errors.push({ label: entry.label, error });
				failed++;
			}
			pending.delete(entry.label);
		}
		return { compiled, failed };
	}
}

function assertWarmupNotAborted(signal?: AbortSignal | null): void {
	if (!signal?.aborted) return;
	const error = new Error("WebGL program warmup was aborted.");
	error.name = "AbortError";
	(error as { cause?: unknown }).cause = signal.reason;
	throw error;
}

function waitForWebGLWarmupSlice(): Promise<void> {
	return new Promise((resolve) => {
		const requestFrame = (globalThis as {
			requestAnimationFrame?: (callback: () => void) => unknown;
		}).requestAnimationFrame;
		if (typeof requestFrame === "function") {
			requestFrame(() => resolve());
			return;
		}
		setTimeout(resolve, 0);
	});
}
