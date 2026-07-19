import type { RenderBackendCompletedFrameCoverage, RenderSurfaceSize } from "../IRenderBackend";
import type { FrameContext } from "../../pipeline/types";

/** @internal Owns the Software backend's active frame session and deferred resize. */
export class SoftwareFrameRuntime {
	private _activeContext: FrameContext | null = null;
	private _completedCoverage: RenderBackendCompletedFrameCoverage = "full-frame";
	private _pendingResize: RenderSurfaceSize | null = null;

	public begin(context: FrameContext): void {
		this._activeContext = context;
		this._completedCoverage = "full-frame";
	}

	public requireActive(
		context: FrameContext | null,
		operation: "executePass" | "skipPass" | "endFrame",
	): FrameContext {
		if (!this._activeContext || !context) {
			throw new Error(`SoftwareBackend.${operation}() requires an active frame.`);
		}
		if (context !== this._activeContext) {
			throw new Error(`SoftwareBackend.${operation}() received a foreign frame context.`);
		}
		return context;
	}

	public get activeContext(): FrameContext | null {
		return this._activeContext;
	}

	public complete(preserveDirtyTiles: boolean): void {
		this._completedCoverage = preserveDirtyTiles ? "dirty-tiles" : "full-frame";
		this._activeContext = null;
	}

	public abort(): void {
		this._activeContext = null;
	}

	public queueResize(size: RenderSurfaceSize): void {
		this._pendingResize = { width: size.width, height: size.height };
	}

	public consumePendingResize(): RenderSurfaceSize | null {
		const pending = this._pendingResize;
		this._pendingResize = null;
		return pending;
	}

	public clear(): void {
		this._activeContext = null;
		this._pendingResize = null;
		this._completedCoverage = "full-frame";
	}

	public get completedCoverage(): RenderBackendCompletedFrameCoverage {
		return this._completedCoverage;
	}
}
