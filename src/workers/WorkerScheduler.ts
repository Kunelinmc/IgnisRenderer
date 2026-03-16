import {
	DEFAULT_WORKER_MAX_QUEUE_SIZE,
	DEFAULT_WORKER_POOL_ID,
	DEFAULT_WORKER_POOL_SIZE,
	DEFAULT_WORKER_RESTART_ON_FAILURE,
	DEFAULT_WORKER_TASK_TIMEOUT_MS,
	GLOBAL_WORKER_SCHEDULER_KEY,
} from "./constants";
import {
	postMessageWorkerTransportPlugin,
	resolveWorkerTransportPlugin,
} from "./transports";
import type {
	WorkerErrorEventLike,
	WorkerLike,
	WorkerMessageEventLike,
	WorkerRuntimeCapabilities,
	WorkerPoolOptions,
	WorkerPoolStats,
	WorkerSchedulerStats,
	WorkerTransportPlugin,
	WorkerTaskEnvelope,
	WorkerTaskResultEnvelope,
	WorkerTaskScheduleOptions,
} from "./types";

interface ScheduledTask {
	id: number;
	sequence: number;
	priority: number;
	payload: unknown;
	transfer: Transferable[];
	timeoutMs: number;
	queued: boolean;
	settled: boolean;
	worker: WorkerState | null;
	resolve: (value: unknown) => void;
	reject: (error: unknown) => void;
	timeoutHandle: ReturnType<typeof setTimeout> | null;
	removeAbortListener: (() => void) | null;
}

interface WorkerState {
	index: number;
	worker: WorkerLike;
	busy: boolean;
	taskId: number | null;
	messageHandler: (event: WorkerMessageEventLike<unknown>) => void;
	errorHandler: (event: WorkerErrorEventLike) => void;
}

function createAbortError(): Error {
	const error = new Error("Task was aborted");
	error.name = "AbortError";
	return error;
}

function createTimeoutError(poolId: string, taskId: number): Error {
	return new Error(
		`Worker task timed out in pool "${poolId}" (task #${taskId})`
	);
}

function normalizeError(error: unknown): Error {
	if (error instanceof Error) return error;
	return new Error(String(error));
}

function normalizeWorkerEventError(event: WorkerErrorEventLike): Error {
	const directMessage =
		typeof event?.message === "string" && event.message.length > 0 ?
			event.message
		:	null;
	if (directMessage) {
		return new Error(directMessage);
	}
	if (event?.error instanceof Error) {
		return event.error;
	}
	return new Error("Worker emitted an unknown error");
}

class WorkerPool {
	private _id: string;
	private _createWorker: WorkerPoolOptions["createWorker"];
	private _size: number;
	private _maxQueueSize: number;
	private _defaultTimeoutMs: number;
	private _restartOnFailure: boolean;
	private _closed: boolean;
	private _taskIdCounter: number;
	private _taskSequenceCounter: number;
	private _dispatchCursor: number;
	private _workers: WorkerState[];
	private _queue: ScheduledTask[];
	private _inFlight: Map<number, ScheduledTask>;
	private _transportPlugin: WorkerTransportPlugin;
	private _runtimeCapabilities: WorkerRuntimeCapabilities;

	public constructor(options: WorkerPoolOptions) {
		if (!options.id || options.id.trim().length === 0) {
			throw new Error("WorkerPool requires a non-empty id");
		}
		if (typeof options.createWorker !== "function") {
			throw new Error(
				`WorkerPool "${options.id}" requires a createWorker factory`
			);
		}
		this._id = options.id;
		this._createWorker = options.createWorker;
		this._size = Math.max(
			1,
			Math.floor(options.size ?? DEFAULT_WORKER_POOL_SIZE)
		);
		this._maxQueueSize = Math.max(
			1,
			Math.floor(options.maxQueueSize ?? DEFAULT_WORKER_MAX_QUEUE_SIZE)
		);
		this._defaultTimeoutMs = Math.max(
			0,
			Math.floor(options.defaultTimeoutMs ?? DEFAULT_WORKER_TASK_TIMEOUT_MS)
		);
		this._restartOnFailure =
			options.restartOnFailure ?? DEFAULT_WORKER_RESTART_ON_FAILURE;
		this._closed = false;
		this._taskIdCounter = 0;
		this._taskSequenceCounter = 0;
		this._dispatchCursor = 0;
		this._workers = [];
		this._queue = [];
		this._inFlight = new Map();
		const configuredPlugins =
			options.transportPlugin ? [options.transportPlugin]
			: options.transportPlugins && options.transportPlugins.length > 0 ?
				options.transportPlugins
			:	[postMessageWorkerTransportPlugin];
		const selection = resolveWorkerTransportPlugin(
			configuredPlugins,
			options.runtimeCapabilities
		);
		this._transportPlugin = selection.plugin;
		this._runtimeCapabilities = selection.capabilities;

		for (let i = 0; i < this._size; i++) {
			this._workers.push(this._createWorkerState(i));
		}
	}

	/**
	 * Schedules a task on this pool.
	 */
	public schedule<TResult = unknown, TPayload = unknown>(
		payload: TPayload,
		options: WorkerTaskScheduleOptions = {}
	): Promise<TResult> {
		this._assertOpen();
		if (this._workers.length === 0) {
			throw new Error(
				`WorkerPool "${this._id}" has no available workers to run tasks`
			);
		}
		if (this._queue.length >= this._maxQueueSize) {
			throw new Error(
				`WorkerPool "${this._id}" queue exceeded max size (${this._maxQueueSize})`
			);
		}

		const signal = options.signal ?? null;
		if (signal?.aborted) {
			return Promise.reject(createAbortError());
		}

		const timeoutMs = this._resolveTimeoutMs(options.timeoutMs);
		const transfer = options.transfer ?? [];
		const priority = Math.floor(options.priority ?? 0);

		return new Promise<TResult>((resolve, reject) => {
			const task: ScheduledTask = {
				id: ++this._taskIdCounter,
				sequence: ++this._taskSequenceCounter,
				priority,
				payload,
				transfer,
				timeoutMs,
				queued: true,
				settled: false,
				worker: null,
				resolve,
				reject,
				timeoutHandle: null,
				removeAbortListener: null,
			};

			if (signal) {
				const onAbort = () => {
					this._cancelTask(task, createAbortError(), false);
				};
				signal.addEventListener("abort", onAbort, { once: true });
				task.removeAbortListener = () => {
					signal.removeEventListener("abort", onAbort);
				};
			}

			this._enqueue(task);
			this._drainQueue();
		});
	}

	/**
	 * Returns runtime stats for this pool.
	 */
	public getStats(): WorkerPoolStats {
		const busyWorkers = this._workers.reduce(
			(count, worker) => count + (worker.busy ? 1 : 0),
			0
		);
		return {
			id: this._id,
			workerCount: this._workers.length,
			busyWorkers,
			idleWorkers: Math.max(0, this._workers.length - busyWorkers),
			queuedTasks: this._queue.length,
			inFlightTasks: this._inFlight.size,
			closed: this._closed,
			transportPluginId: this._transportPlugin.id,
			transportMode: this._transportPlugin.mode,
			sharedArrayBufferEnabled: this._runtimeCapabilities.sharedArrayBuffer,
		};
	}

	/**
	 * Shuts down the pool and rejects all pending tasks.
	 */
	public shutdown(): void {
		if (this._closed) return;
		this._closed = true;

		const shutdownError = new Error(
			`WorkerPool "${this._id}" has been shut down`
		);
		for (const task of this._queue.splice(0, this._queue.length)) {
			this._rejectTask(task, shutdownError, true);
		}
		for (const task of this._inFlight.values()) {
			this._rejectTask(task, shutdownError, true);
		}
		this._inFlight.clear();

		for (const workerState of this._workers.splice(0, this._workers.length)) {
			this._detachWorker(workerState);
			try {
				workerState.worker.terminate();
			} catch (error) {
				// Ignore failures while shutting down the pool.
			}
		}
	}

	private _resolveTimeoutMs(timeoutOverride?: number): number {
		if (!Number.isFinite(timeoutOverride)) return this._defaultTimeoutMs;
		return Math.max(0, Math.floor(timeoutOverride as number));
	}

	private _assertOpen(): void {
		if (!this._closed) return;
		throw new Error(`WorkerPool "${this._id}" is closed`);
	}

	private _enqueue(task: ScheduledTask): void {
		let insertAt = this._queue.length;
		for (let i = 0; i < this._queue.length; i++) {
			const queuedTask = this._queue[i];
			if (task.priority > queuedTask.priority) {
				insertAt = i;
				break;
			}
		}
		this._queue.splice(insertAt, 0, task);
	}

	private _drainQueue(): void {
		if (this._closed) return;
		while (this._queue.length > 0) {
			const workerState = this._nextIdleWorker();
			if (!workerState) return;
			const task = this._queue.shift();
			if (!task || task.settled) continue;
			this._startTask(workerState, task);
		}
	}

	private _nextIdleWorker(): WorkerState | null {
		const total = this._workers.length;
		if (total === 0) return null;
		for (let i = 0; i < total; i++) {
			const index = (this._dispatchCursor + i) % total;
			const workerState = this._workers[index];
			if (workerState.busy) continue;
			this._dispatchCursor = (index + 1) % total;
			return workerState;
		}
		return null;
	}

	private _startTask(workerState: WorkerState, task: ScheduledTask): void {
		task.queued = false;
		task.worker = workerState;
		workerState.busy = true;
		workerState.taskId = task.id;
		this._inFlight.set(task.id, task);

		if (task.timeoutMs > 0) {
			task.timeoutHandle = setTimeout(() => {
				this._cancelTask(task, createTimeoutError(this._id, task.id), true);
			}, task.timeoutMs);
		}

		try {
			const message: WorkerTaskEnvelope<unknown> = {
				id: task.id,
				payload: task.payload,
			};
			const encoded = this._transportPlugin.encodeTask(message);
			const transfer = [...(encoded.transfer ?? [])];
			for (const item of task.transfer) {
				if (transfer.includes(item)) continue;
				transfer.push(item);
			}
			workerState.worker.postMessage(
				encoded.message,
				transfer.length > 0 ? transfer : undefined
			);
		} catch (error) {
			const normalized = normalizeError(error);
			this._cancelTask(task, normalized, true);
		}
	}

	private _cancelTask(
		task: ScheduledTask,
		reason: Error,
		recycleWorker: boolean
	): void {
		if (task.settled) return;
		const assignedWorker = task.worker;
		if (task.queued) {
			const queueIndex = this._queue.findIndex((item) => item.id === task.id);
			if (queueIndex >= 0) {
				this._queue.splice(queueIndex, 1);
			}
			this._rejectTask(task, reason, true);
		} else {
			if (recycleWorker) {
				this._inFlight.delete(task.id);
				this._rejectTask(task, reason, true);
			} else {
				// Keep the task attached to the worker until a terminal message
				// arrives, so the worker isn't reused while still executing.
				this._rejectTask(task, reason, false);
			}
		}
		if (recycleWorker && assignedWorker) {
			this._recycleWorker(assignedWorker, true);
		}
		this._drainQueue();
	}

	private _resolveTask(
		workerState: WorkerState,
		envelope: WorkerTaskResultEnvelope<unknown>
	): void {
		const task = this._inFlight.get(envelope.id);
		if (!task) return;
		if (task.worker !== workerState) return;

		this._inFlight.delete(task.id);
		this._clearTaskLifecycle(task);
		task.worker = null;
		workerState.taskId = null;
		workerState.busy = false;

		if (!task.settled) {
			task.settled = true;
			if (typeof envelope.error === "string" && envelope.error.length > 0) {
				task.reject(new Error(envelope.error));
			} else {
				task.resolve(envelope.result);
			}
		}
		this._drainQueue();
	}

	private _rejectTask(
		task: ScheduledTask,
		reason: Error,
		releaseWorker: boolean
	): void {
		if (task.settled) return;
		task.settled = true;
		this._clearTaskLifecycle(task);
		task.reject(reason);
		if (releaseWorker && task.worker) {
			task.worker.taskId = null;
			task.worker.busy = false;
			task.worker = null;
		}
	}

	private _clearTaskLifecycle(task: ScheduledTask): void {
		if (task.timeoutHandle) {
			clearTimeout(task.timeoutHandle);
			task.timeoutHandle = null;
		}
		if (task.removeAbortListener) {
			task.removeAbortListener();
			task.removeAbortListener = null;
		}
	}

	private _handleWorkerError(
		workerState: WorkerState,
		event: WorkerErrorEventLike
	): void {
		const reason = normalizeWorkerEventError(event);
		const taskId = workerState.taskId;
		if (typeof taskId === "number") {
			const task = this._inFlight.get(taskId);
			if (task) {
				this._inFlight.delete(taskId);
				this._rejectTask(task, reason, true);
			}
		}
		this._recycleWorker(workerState, true);
		this._drainQueue();
	}

	private _recycleWorker(workerState: WorkerState, failed: boolean): void {
		const index = this._workers.indexOf(workerState);
		if (index < 0) return;

		this._detachWorker(workerState);
		try {
			workerState.worker.terminate();
		} catch (error) {
			// Ignore worker terminate failures.
		}

		if (this._closed) {
			this._workers.splice(index, 1);
			return;
		}
		if (failed && !this._restartOnFailure) {
			this._workers.splice(index, 1);
			if (this._dispatchCursor >= this._workers.length) {
				this._dispatchCursor = 0;
			}
			return;
		}

		try {
			this._workers[index] = this._createWorkerState(workerState.index);
		} catch (error) {
			this._workers.splice(index, 1);
			if (this._dispatchCursor >= this._workers.length) {
				this._dispatchCursor = 0;
			}
		}
	}

	private _createWorkerState(workerIndex: number): WorkerState {
		const worker = this._createWorker(workerIndex, this._id);
		if (!worker) {
			throw new Error(
				`WorkerPool "${this._id}" createWorker returned an invalid worker`
			);
		}
		const state: WorkerState = {
			index: workerIndex,
			worker,
			busy: false,
			taskId: null,
			messageHandler: () => undefined,
			errorHandler: () => undefined,
		};

		state.messageHandler = (event) => {
			const data = this._transportPlugin.decodeResult(event?.data);
			if (!data || typeof data.id !== "number") {
				return;
			}
			this._resolveTask(state, {
				id: data.id,
				result: data.result,
				error: data.error,
			});
		};
		state.errorHandler = (event) => {
			this._handleWorkerError(state, event);
		};

		if (
			typeof worker.addEventListener === "function" &&
			typeof worker.removeEventListener === "function"
		) {
			worker.addEventListener("message", state.messageHandler);
			worker.addEventListener("error", state.errorHandler);
		} else {
			worker.onmessage = state.messageHandler;
			worker.onerror = state.errorHandler;
		}

		return state;
	}

	private _detachWorker(workerState: WorkerState): void {
		const worker = workerState.worker;
		if (
			typeof worker.removeEventListener === "function" &&
			typeof worker.addEventListener === "function"
		) {
			worker.removeEventListener("message", workerState.messageHandler);
			worker.removeEventListener("error", workerState.errorHandler);
		}
		worker.onmessage = null;
		worker.onerror = null;
	}
}

export interface WorkerSchedulerOptions {
	defaultPoolId?: string;
}

export class WorkerScheduler {
	private _defaultPoolId: string | null;
	private _pools: Map<string, WorkerPool>;

	public constructor(options: WorkerSchedulerOptions = {}) {
		this._defaultPoolId = options.defaultPoolId ?? DEFAULT_WORKER_POOL_ID;
		this._pools = new Map();
	}

	/**
	 * Returns the shared global WorkerScheduler singleton.
	 */
	public static getGlobal(): WorkerScheduler {
		const scope = globalThis as typeof globalThis & {
			[key: string]: WorkerScheduler | undefined;
		};
		if (!scope[GLOBAL_WORKER_SCHEDULER_KEY]) {
			scope[GLOBAL_WORKER_SCHEDULER_KEY] = new WorkerScheduler();
		}
		return scope[GLOBAL_WORKER_SCHEDULER_KEY]!;
	}

	/**
	 * Registers a new worker pool.
	 */
	public registerPool(options: WorkerPoolOptions): void {
		if (this._pools.has(options.id)) {
			throw new Error(`Worker pool "${options.id}" is already registered`);
		}
		this._pools.set(options.id, new WorkerPool(options));
	}

	/**
	 * Checks whether a pool id has been registered.
	 */
	public hasPool(poolId: string): boolean {
		return this._pools.has(poolId);
	}

	/**
	 * Lists all registered pool ids.
	 */
	public listPoolIds(): string[] {
		return [...this._pools.keys()];
	}

	/**
	 * Sets the default pool id used by scheduleOnDefault.
	 */
	public setDefaultPoolId(poolId: string | null): void {
		if (poolId !== null && !this._pools.has(poolId)) {
			throw new Error(`Cannot set unknown worker pool "${poolId}" as default`);
		}
		this._defaultPoolId = poolId;
	}

	/**
	 * Gets the current default pool id.
	 */
	public getDefaultPoolId(): string | null {
		return this._defaultPoolId;
	}

	/**
	 * Schedules a task on a specific pool.
	 */
	public schedule<TResult = unknown, TPayload = unknown>(
		poolId: string,
		payload: TPayload,
		options: WorkerTaskScheduleOptions = {}
	): Promise<TResult> {
		const pool = this._pools.get(poolId);
		if (!pool) {
			throw new Error(`Unknown worker pool "${poolId}"`);
		}
		return pool.schedule<TResult, TPayload>(payload, options);
	}

	/**
	 * Schedules a task on the currently configured default pool.
	 */
	public scheduleOnDefault<TResult = unknown, TPayload = unknown>(
		payload: TPayload,
		options: WorkerTaskScheduleOptions = {}
	): Promise<TResult> {
		if (!this._defaultPoolId) {
			throw new Error("No default worker pool is configured");
		}
		return this.schedule<TResult, TPayload>(
			this._defaultPoolId,
			payload,
			options
		);
	}

	/**
	 * Removes a pool and shuts down all tasks running in it.
	 */
	public unregisterPool(poolId: string): void {
		const pool = this._pools.get(poolId);
		if (!pool) return;
		pool.shutdown();
		this._pools.delete(poolId);
		if (this._defaultPoolId === poolId) {
			this._defaultPoolId = null;
		}
	}

	/**
	 * Shuts down all pools and clears the scheduler state.
	 */
	public shutdownAll(): void {
		for (const pool of this._pools.values()) {
			pool.shutdown();
		}
		this._pools.clear();
		this._defaultPoolId = null;
	}

	/**
	 * Returns stats for all pools.
	 */
	public getStats(): WorkerSchedulerStats {
		return {
			defaultPoolId: this._defaultPoolId,
			pools: [...this._pools.values()].map((pool) => pool.getStats()),
		};
	}

	/**
	 * Returns stats for a single pool, if it exists.
	 */
	public getPoolStats(poolId: string): WorkerPoolStats | null {
		const pool = this._pools.get(poolId);
		return pool ? pool.getStats() : null;
	}
}

export const globalWorkerScheduler = WorkerScheduler.getGlobal();
