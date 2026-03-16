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

export type WorkerTransportMode = "post-message" | "shared-array-buffer";

export interface WorkerRuntimeCapabilities {
	sharedArrayBuffer: boolean;
	crossOriginIsolated: boolean;
}

export interface WorkerTransportEncodedMessage {
	message: unknown;
	transfer?: Transferable[];
}

export interface WorkerTransportPlugin {
	readonly id: string;
	readonly mode: WorkerTransportMode;
	isSupported?(capabilities: WorkerRuntimeCapabilities): boolean;
	encodeTask(
		envelope: WorkerTaskEnvelope<unknown>
	): WorkerTransportEncodedMessage;
	decodeTask(data: unknown): WorkerTaskEnvelope<unknown> | null;
	encodeResult(
		envelope: WorkerTaskResultEnvelope<unknown>
	): WorkerTransportEncodedMessage;
	decodeResult(data: unknown): WorkerTaskResultEnvelope<unknown> | null;
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
	transportPlugin?: WorkerTransportPlugin;
	transportPlugins?: WorkerTransportPlugin[];
	runtimeCapabilities?: Partial<WorkerRuntimeCapabilities>;
}

export interface WorkerPoolStats {
	id: string;
	workerCount: number;
	busyWorkers: number;
	idleWorkers: number;
	queuedTasks: number;
	inFlightTasks: number;
	closed: boolean;
	transportPluginId: string;
	transportMode: WorkerTransportMode;
	sharedArrayBufferEnabled: boolean;
}

export interface WorkerSchedulerStats {
	defaultPoolId: string | null;
	pools: WorkerPoolStats[];
}
