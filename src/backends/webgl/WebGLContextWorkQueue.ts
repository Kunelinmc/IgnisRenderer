import { WebGLContextWorkError } from "../../foundation/Error";

export type WebGLContextWorkFramePolicy = "between-passes" | "idle-only";
export type WebGLContextLossPolicy = "retain-pending" | "reject";

export interface WebGLContextWorkScope<TServices> {
	readonly generation: number;
	readonly services: TServices;
	readonly signal: AbortSignal;
}

export interface WebGLContextWorkRequest<T, TServices> {
	label: string;
	framePolicy: WebGLContextWorkFramePolicy;
	contextLossPolicy: WebGLContextLossPolicy;
	signal?: AbortSignal | null;
	execute(scope: WebGLContextWorkScope<TServices>): T | Promise<T>;
}

export interface WebGLContextWorkQueueOptions<TServices> {
	resolveServices(): TServices | null;
	restoreBaseline(
		scope: WebGLContextWorkScope<TServices>,
		frameActive: boolean,
	): void | Promise<void>;
}

export interface WebGLContextWorkDebugSnapshot {
	readonly state: "not-initialized" | "ready" | "context-lost" | "destroyed";
	readonly generation: number;
	readonly activeLabel: string | null;
	readonly activeStage: "frame-begin" | "frame-pass" | "auxiliary" | "frame-end" | "frame-abort" | null;
	readonly frameState:
		| "idle"
		| "beginning"
		| "between-passes"
		| "active-pass"
		| "ending"
		| "abort-required"
		| "aborting";
	readonly pendingCount: number;
	readonly pendingFrameCount: number;
}

interface Deferred<T> {
	promise: Promise<T>;
	resolve(value: T | PromiseLike<T>): void;
	reject(reason?: unknown): void;
}

interface PendingAuxiliary<T, TServices> {
	request: WebGLContextWorkRequest<T, TServices>;
	deferred: Deferred<T>;
	frameId: number | null;
	abortHandler: (() => void) | null;
}

interface PendingFrame<TServices> {
	label: string;
	execute(scope: WebGLContextWorkScope<TServices>): void | Promise<void>;
	deferred: Deferred<void>;
}

interface ActiveExecution {
	controller: AbortController;
	lossDeferred: Deferred<never>;
}

/** @internal Owned by one WebGL backend for its complete device lifetime. */
export class WebGLContextWorkQueue<TServices> {
	private readonly _options: WebGLContextWorkQueueOptions<TServices>;
	private _state: WebGLContextWorkDebugSnapshot["state"] = "not-initialized";
	private _generation = 0;
	private _activeLabel: string | null = null;
	private _activeStage: WebGLContextWorkDebugSnapshot["activeStage"] = null;
	private _activeExecution: ActiveExecution | null = null;
	private _frameState: WebGLContextWorkDebugSnapshot["frameState"] = "idle";
	private _frameId = 0;
	private _pendingAuxiliary: PendingAuxiliary<unknown, TServices>[] = [];
	private _pendingFrames: PendingFrame<TServices>[] = [];
	private _frameBoundaryWaiters: Deferred<void>[] = [];
	private _allowAuxiliaryBeforeNextFrame = false;
	private _scheduling = false;

	public constructor(options: WebGLContextWorkQueueOptions<TServices>) {
		this._options = options;
	}

	public bindContext(): number {
		this._assertNotDestroyed("bind-context");
		this._generation++;
		this._state = "ready";
		this._schedule();
		return this._generation;
	}

	public suspend(): void {
		if (this._state === "destroyed" || this._state === "context-lost") return;
		this._state = "context-lost";
		const error = new WebGLContextWorkError("context-lost", this._activeLabel ?? undefined);
		if (this._activeExecution) {
			this._activeExecution.controller.abort(error);
			this._activeExecution.lossDeferred.reject(error);
		}
		for (let index = this._pendingAuxiliary.length - 1; index >= 0; index--) {
			const item = this._pendingAuxiliary[index];
			if (item.request.contextLossPolicy === "reject") {
				this._pendingAuxiliary.splice(index, 1);
				this._detachAbort(item);
				item.deferred.reject(new WebGLContextWorkError("context-lost", item.request.label));
			}
		}
		while (this._pendingFrames.length > 0) {
			const frame = this._pendingFrames.shift()!;
			frame.deferred.reject(new WebGLContextWorkError("context-lost", frame.label));
		}
		if (this._frameState !== "idle") this._frameState = "abort-required";
	}

	public beginFrame(
		label: string,
		execute: (scope: WebGLContextWorkScope<TServices>) => void | Promise<void>,
	): Promise<void> {
		try {
			this._assertReady(label);
		} catch (error) {
			return Promise.reject(error);
		}
		if (this._frameState !== "idle" || this._pendingFrames.length > 0) {
			return Promise.reject(new WebGLContextWorkError("active-frame", label));
		}
		const deferred = createDeferred<void>();
		this._pendingFrames.push({ label, execute, deferred });
		this._schedule();
		return deferred.promise;
	}

	public async runFramePass(
		label: string,
		execute: (scope: WebGLContextWorkScope<TServices>) => void | Promise<void>,
	): Promise<void> {
		this._assertReady(label);
		if (this._frameState !== "between-passes") {
			throw new WebGLContextWorkError(
				this._frameState === "active-pass" ? "active-pass" : "active-frame",
				label,
			);
		}
		await this._waitForFrameBoundaryAuxiliary();
		this._assertReady(label);
		this._frameState = "active-pass";
		try {
			await this._runContextExecution("frame-pass", label, execute);
			this._frameState = "between-passes";
		} catch (error) {
			this._frameState = "abort-required";
			throw error;
		}
	}

	public async endFrame(
		label: string,
		execute: (scope: WebGLContextWorkScope<TServices>) => void | Promise<void>,
	): Promise<void> {
		this._assertReady(label);
		if (this._frameState !== "between-passes") {
			throw new WebGLContextWorkError("active-frame", label);
		}
		await this._waitForFrameBoundaryAuxiliary();
		this._assertReady(label);
		this._frameState = "ending";
		try {
			await this._runContextExecution("frame-end", label, execute);
			this._releaseFrame();
		} catch (error) {
			this._frameState = "abort-required";
			throw error;
		}
	}

	public async abortFrame(
		label: string,
		execute?: () => void | Promise<void>,
	): Promise<void> {
		this._assertNotDestroyed(label);
		if (this._frameState === "idle") return;
		this._frameState = "aborting";
		this._activeLabel = label;
		this._activeStage = "frame-abort";
		let error: unknown = null;
		try {
			await execute?.();
		} catch (caught) {
			error = caught;
		} finally {
			this._activeLabel = null;
			this._activeStage = null;
			this._releaseFrame();
		}
		if (error) throw error;
	}

	public enqueue<T>(request: WebGLContextWorkRequest<T, TServices>): Promise<T> {
		try {
			this._assertNotDestroyed(request.label);
			if (request.signal?.aborted) return Promise.reject(createAbortError(request.signal.reason));
			if (this._state === "not-initialized") {
				throw new WebGLContextWorkError("not-initialized", request.label);
			}
			if (this._frameState === "active-pass") {
				throw new WebGLContextWorkError("active-pass", request.label);
			}
			if (
				this._frameState !== "idle" &&
				this._frameState !== "between-passes"
			) {
				throw new WebGLContextWorkError("active-frame", request.label);
			}
			if (request.framePolicy === "idle-only" && this._frameState !== "idle") {
				throw new WebGLContextWorkError("active-frame", request.label);
			}
			if (this._state === "context-lost" && request.contextLossPolicy === "reject") {
				throw new WebGLContextWorkError("context-lost", request.label);
			}
		} catch (error) {
			return Promise.reject(error);
		}

		const deferred = createDeferred<T>();
		const item: PendingAuxiliary<T, TServices> = {
			request,
			deferred,
			frameId: this._frameState === "idle" ? null : this._frameId,
			abortHandler: null,
		};
		if (request.signal) {
			item.abortHandler = () => {
				const index = this._pendingAuxiliary.indexOf(
					item as PendingAuxiliary<unknown, TServices>,
				);
				if (index >= 0) {
					this._pendingAuxiliary.splice(index, 1);
					deferred.reject(createAbortError(request.signal?.reason));
					this._notifyFrameBoundary();
				}
			};
			request.signal.addEventListener("abort", item.abortHandler, { once: true });
		}
		this._pendingAuxiliary.push(item as PendingAuxiliary<unknown, TServices>);
		this._schedule();
		return deferred.promise;
	}

	public destroy(): void {
		if (this._state === "destroyed") return;
		this._state = "destroyed";
		const error = new WebGLContextWorkError("destroyed", this._activeLabel ?? undefined);
		if (this._activeExecution) {
			this._activeExecution.controller.abort(error);
			this._activeExecution.lossDeferred.reject(error);
		}
		for (const item of this._pendingAuxiliary.splice(0)) {
			this._detachAbort(item);
			item.deferred.reject(new WebGLContextWorkError("destroyed", item.request.label));
		}
		for (const frame of this._pendingFrames.splice(0)) {
			frame.deferred.reject(new WebGLContextWorkError("destroyed", frame.label));
		}
		for (const waiter of this._frameBoundaryWaiters.splice(0)) waiter.reject(error);
		this._frameState = "idle";
	}

	public getDebugSnapshot(): WebGLContextWorkDebugSnapshot {
		return {
			state: this._state,
			generation: this._generation,
			activeLabel: this._activeLabel,
			activeStage: this._activeStage,
			frameState: this._frameState,
			pendingCount: this._pendingAuxiliary.length,
			pendingFrameCount: this._pendingFrames.length,
		};
	}

	private _schedule(): void {
		if (this._scheduling) return;
		this._scheduling = true;
		queueMicrotask(() => {
			this._scheduling = false;
			void this._drain().catch(() => undefined);
		});
	}

	private async _drain(): Promise<void> {
		if (this._state !== "ready" || this._activeExecution) return;
		if (this._frameState === "between-passes") {
			const item = this._pendingAuxiliary.find((candidate) => candidate.frameId === this._frameId);
			if (item) {
				this._removeAuxiliary(item);
				await this._runAuxiliary(item, true);
				this._schedule();
			} else {
				this._notifyFrameBoundary();
			}
			return;
		}
		if (this._frameState !== "idle") return;

		const auxiliary = this._pendingAuxiliary.find((candidate) => candidate.frameId === null);
		const frame = this._pendingFrames[0];
		if (auxiliary && (!frame || this._allowAuxiliaryBeforeNextFrame)) {
			this._allowAuxiliaryBeforeNextFrame = false;
			this._removeAuxiliary(auxiliary);
			await this._runAuxiliary(auxiliary, false);
			this._schedule();
			return;
		}
		if (frame) {
			this._pendingFrames.shift();
			this._frameId++;
			this._frameState = "beginning";
			try {
				await this._runContextExecution("frame-begin", frame.label, frame.execute);
				this._frameState = "between-passes";
				frame.deferred.resolve();
				this._schedule();
			} catch (error) {
				this._frameState = "abort-required";
				frame.deferred.reject(error);
			}
			return;
		}
		if (auxiliary) {
			this._removeAuxiliary(auxiliary);
			await this._runAuxiliary(auxiliary, false);
			this._schedule();
		}
	}

	private async _runAuxiliary(
		item: PendingAuxiliary<unknown, TServices>,
		frameActive: boolean,
	): Promise<void> {
		this._detachAbort(item);
		try {
			const value = await this._runContextExecution(
				"auxiliary",
				item.request.label,
				async (scope) => {
					try {
						return await item.request.execute(scope);
					} finally {
						await this._options.restoreBaseline(scope, frameActive);
					}
				},
				item.request.signal,
			);
			item.deferred.resolve(value);
		} catch (error) {
			item.deferred.reject(error);
		} finally {
			this._notifyFrameBoundary();
		}
	}

	private async _runContextExecution<T>(
		stage: NonNullable<WebGLContextWorkDebugSnapshot["activeStage"]>,
		label: string,
		execute: (scope: WebGLContextWorkScope<TServices>) => T | Promise<T>,
		userSignal?: AbortSignal | null,
	): Promise<T> {
		this._assertReady(label);
		const services = this._options.resolveServices();
		if (!services) throw new WebGLContextWorkError("not-initialized", label);
		const controller = new AbortController();
		const lossDeferred = createDeferred<never>();
		const active = { controller, lossDeferred };
		const abortHandler = userSignal
			? () => {
					const error = createAbortError(userSignal.reason);
					controller.abort(error);
					lossDeferred.reject(error);
				}
			: null;
		userSignal?.addEventListener("abort", abortHandler!, { once: true });
		if (userSignal?.aborted) abortHandler?.();
		this._activeExecution = active;
		this._activeLabel = label;
		this._activeStage = stage;
		const execution = Promise.resolve().then(() =>
			execute({ generation: this._generation, services, signal: controller.signal }),
		);
		try {
			return await Promise.race([execution, lossDeferred.promise]);
		} finally {
			void execution.catch(() => undefined).finally(() => {
				if (abortHandler) userSignal?.removeEventListener("abort", abortHandler);
				if (this._activeExecution !== active) return;
				this._activeExecution = null;
				this._activeLabel = null;
				this._activeStage = null;
				this._schedule();
			});
		}
	}

	private async _waitForFrameBoundaryAuxiliary(): Promise<void> {
		while (
			this._pendingAuxiliary.some((item) => item.frameId === this._frameId) ||
			this._activeStage === "auxiliary"
		) {
			const waiter = createDeferred<void>();
			this._frameBoundaryWaiters.push(waiter);
			this._schedule();
			await waiter.promise;
		}
	}

	private _notifyFrameBoundary(): void {
		if (this._pendingAuxiliary.some((item) => item.frameId === this._frameId)) return;
		for (const waiter of this._frameBoundaryWaiters.splice(0)) waiter.resolve();
	}

	private _releaseFrame(): void {
		for (let index = this._pendingAuxiliary.length - 1; index >= 0; index--) {
			const item = this._pendingAuxiliary[index];
			if (item.frameId === this._frameId) item.frameId = null;
		}
		this._frameState = "idle";
		this._allowAuxiliaryBeforeNextFrame = true;
		this._notifyFrameBoundary();
		this._schedule();
	}

	private _removeAuxiliary(item: PendingAuxiliary<unknown, TServices>): void {
		const index = this._pendingAuxiliary.indexOf(item);
		if (index >= 0) this._pendingAuxiliary.splice(index, 1);
	}

	private _detachAbort(item: PendingAuxiliary<unknown, TServices>): void {
		if (item.abortHandler && item.request.signal) {
			item.request.signal.removeEventListener("abort", item.abortHandler);
		}
		item.abortHandler = null;
	}

	private _assertReady(label: string): void {
		this._assertNotDestroyed(label);
		if (this._state === "not-initialized") {
			throw new WebGLContextWorkError("not-initialized", label);
		}
		if (this._state === "context-lost") {
			throw new WebGLContextWorkError("context-lost", label);
		}
	}

	private _assertNotDestroyed(label: string): void {
		if (this._state === "destroyed") {
			throw new WebGLContextWorkError("destroyed", label);
		}
	}
}

function createDeferred<T>(): Deferred<T> {
	let resolve!: Deferred<T>["resolve"];
	let reject!: Deferred<T>["reject"];
	const promise = new Promise<T>((promiseResolve, promiseReject) => {
		resolve = promiseResolve;
		reject = promiseReject;
	});
	return { promise, resolve, reject };
}

function createAbortError(reason?: unknown): Error {
	if (reason instanceof Error && reason.name === "AbortError") return reason;
	const error = new Error("WebGL context work was aborted.");
	error.name = "AbortError";
	(error as { cause?: unknown }).cause = reason;
	return error;
}
