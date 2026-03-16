export interface WorkerMessageEventLike<TData = unknown> {
	data: TData;
}

export interface WorkerErrorEventLike {
	message?: string;
	error?: unknown;
}

export interface WorkerLike {
	onmessage: ((event: WorkerMessageEventLike<unknown>) => void) | null;
	onerror: ((event: WorkerErrorEventLike) => void) | null;
	postMessage(message: unknown, transfer?: Transferable[]): void;
	terminate(): void;
	addEventListener?(
		type: "message",
		listener: (event: WorkerMessageEventLike<unknown>) => void
	): void;
	addEventListener?(
		type: "error",
		listener: (event: WorkerErrorEventLike) => void
	): void;
	removeEventListener?(
		type: "message",
		listener: (event: WorkerMessageEventLike<unknown>) => void
	): void;
	removeEventListener?(
		type: "error",
		listener: (event: WorkerErrorEventLike) => void
	): void;
}

export interface WorkerTaskEnvelope<TPayload = unknown> {
	id: number;
	payload: TPayload;
}

export interface WorkerTaskResultEnvelope<TResult = unknown> {
	id: number;
	result?: TResult;
	error?: string;
}

export interface WorkerTaskScheduleOptions {
	priority?: number;
	timeoutMs?: number;
	signal?: AbortSignal | null;
	transfer?: Transferable[];
}

export interface WorkerPoolOptions {
	id: string;
	createWorker: (workerIndex: number, poolId: string) => WorkerLike;
	size?: number;
	maxQueueSize?: number;
	defaultTimeoutMs?: number;
	restartOnFailure?: boolean;
}

export interface WorkerPoolStats {
	id: string;
	workerCount: number;
	busyWorkers: number;
	idleWorkers: number;
	queuedTasks: number;
	inFlightTasks: number;
	closed: boolean;
}

export interface WorkerSchedulerStats {
	defaultPoolId: string | null;
	pools: WorkerPoolStats[];
}
