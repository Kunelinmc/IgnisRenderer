import type {
	IRenderBackend,
	RenderBackendEvent,
} from "./IRenderBackend";
import type { RenderFrameResult } from "./Renderer";

/**
 * Manages the lifetime, event notifications, and state of a single backend.
 * Exposes synchronous lifecycle states and handles context/device recovery.
 *
 * @internal Used by Renderer to separate public APIs from backend management.
 */
export class RendererRuntime {
	private readonly _backend: IRenderBackend;
	private readonly _onBackendEvent: (event: RenderBackendEvent) => void;
	private _initialized = false;
	private _destroyed = false;
	private _activeFramePromise: Promise<RenderFrameResult> | null = null;
	private _destroyPromise: Promise<void> | null = null;

	constructor(
		backend: IRenderBackend,
		onBackendEvent: (event: RenderBackendEvent) => void
	) {
		this._backend = backend;
		this._onBackendEvent = onBackendEvent;
	}

	public get backend(): IRenderBackend {
		return this._backend;
	}

	public get isInitialized(): boolean {
		return this._initialized;
	}

	public get isDestroyed(): boolean {
		return this._destroyed;
	}

	public get activeFramePromise(): Promise<RenderFrameResult> | null {
		return this._activeFramePromise;
	}

	public set activeFramePromise(promise: Promise<RenderFrameResult> | null) {
		this._activeFramePromise = promise;
	}

	/**
	 * Asserts that the runtime is initialized and not destroyed.
	 */
	public assertReady(operation: string): void {
		if (this._destroyed || this._destroyPromise) {
			throw new Error(`Renderer.${operation}() cannot run after destroy().`);
		}
		if (!this._initialized) {
			throw new Error(`Renderer.${operation}() requires initialize() to complete first.`);
		}
	}

	/**
	 * Asserts that the runtime has not been destroyed.
	 */
	public assertNotDestroyed(operation: string): void {
		if (this._destroyed || this._destroyPromise) {
			throw new Error(`Renderer.${operation}() cannot run after destroy().`);
		}
	}

	public async initialize(): Promise<void> {
		this.assertNotDestroyed("initialize");
		if (this._initialized) return;
		await this._backend.initialize();
		this._initialized = true;
	}

	public async restore(): Promise<void> {
		this.assertReady("restore");
		await this._backend.restore();
	}

	public async destroy(): Promise<void> {
		if (this._destroyPromise) return this._destroyPromise;
		this._destroyPromise = this._destroyInternal();
		return this._destroyPromise;
	}

	private async _destroyInternal(): Promise<void> {
		if (this._destroyed) return;
		const activeFrame = this._activeFramePromise;
		if (activeFrame) {
			try {
				await activeFrame;
			} catch {
				// Frame cleanup is handled by the render path before destruction.
			}
		}
		await this._backend.destroy();
		this._destroyed = true;
		this._initialized = false;
	}

	public handleBackendEvent(event: RenderBackendEvent): void {
		this._onBackendEvent(event);
	}
}
