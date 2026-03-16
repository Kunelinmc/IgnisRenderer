const FALLBACK_WORKER_POOL_SIZE = 4;

function resolveDefaultWorkerPoolSize(): number {
	const navigatorValue = (
		globalThis as {
			navigator?: {
				hardwareConcurrency?: number;
			};
		}
	).navigator?.hardwareConcurrency;
	if (!Number.isFinite(navigatorValue)) {
		return FALLBACK_WORKER_POOL_SIZE;
	}
	return Math.max(1, Math.floor(navigatorValue as number));
}

export const DEFAULT_WORKER_POOL_ID = "default";
export const DEFAULT_WORKER_POOL_SIZE = resolveDefaultWorkerPoolSize();
export const DEFAULT_WORKER_MAX_QUEUE_SIZE = 1024;
export const DEFAULT_WORKER_TASK_TIMEOUT_MS = 0;
export const DEFAULT_WORKER_RESTART_ON_FAILURE = true;
export const GLOBAL_WORKER_SCHEDULER_KEY = "__ignisGlobalWorkerScheduler__";
