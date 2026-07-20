import { Texture, type TextureBaseParams } from "./Texture";

type CanvasTextureContext2D =
	| CanvasRenderingContext2D
	| OffscreenCanvasRenderingContext2D;
type CanvasTextureCanvas = HTMLCanvasElement | OffscreenCanvas;
type CanvasMutationMethod = (...args: unknown[]) => unknown;

interface ContextMutationPatchState {
	listeners: Set<CanvasTexture>;
	originals: Map<string, CanvasMutationMethod>;
}

const CANVAS_MUTATING_METHODS = [
	"clearRect",
	"fillRect",
	"strokeRect",
	"drawImage",
	"putImageData",
	"fillText",
	"strokeText",
	"fill",
	"stroke",
	"reset",
] as const;

export interface CanvasTextureParams extends TextureBaseParams {
	context: CanvasTextureContext2D;
	/**
	 * Auto-registers this texture into the global dynamic-texture update loop.
	 */
	autoUpdate?: boolean;
	/**
	 * Tracks common mutating Canvas2D calls and only refreshes pixels when dirty.
	 */
	trackContextMutations?: boolean;
	/**
	 * Optional throttling for mutation-driven updates.
	 */
	minUpdateIntervalMs?: number;
}

/**
 * Dynamic texture that pulls pixels from a Canvas 2D rendering context.
 */
export class CanvasTexture extends Texture {
	public readonly context: CanvasTextureContext2D;
	public readonly canvas: CanvasTextureCanvas;
	public readonly autoUpdate: boolean;
	public readonly trackContextMutations: boolean;

	private static _contextMutationPatches = new WeakMap<
		CanvasTextureContext2D,
		ContextMutationPatchState
	>();

	private _isDisposed: boolean;
	private _forceRefresh: boolean;
	private _dirty: boolean;
	private _lastUpdateTimeMs: number;
	private _minUpdateIntervalMs: number;

	/**
	 * Creates a dynamic canvas texture from one parameter object.
	 */
	constructor(params: CanvasTextureParams) {
		if (!params || typeof params !== "object" || !("context" in params)) {
			throw new TypeError("CanvasTexture requires a parameter object.");
		}
		const { context } = params;
		const canvas = CanvasTexture._resolveCanvas(context);
		super({
			width: Math.max(0, canvas.width | 0),
			height: Math.max(0, canvas.height | 0),
			colorSpace: params.colorSpace,
			label: params.label,
			usageHint: params.usageHint,
		});

		this.context = context;
		this.canvas = canvas;
		this.autoUpdate = params.autoUpdate !== false;
		this.trackContextMutations = params.trackContextMutations !== false;
		this._isDisposed = false;
		this._forceRefresh = true;
		this._dirty = true;
		this._lastUpdateTimeMs = -Infinity;
		this._minUpdateIntervalMs = Math.max(0, params.minUpdateIntervalMs ?? 0);

		if (this.autoUpdate) {
			this._registerAsDynamicTexture();
		}
		if (this.trackContextMutations) {
			this._attachContextMutationTracking();
		}

		this.update(0);
	}

	/**
	 * Manually marks canvas pixels as dirty for the next update tick.
	 */
	public invalidate(): void {
		if (this._isDisposed) return;
		this._forceRefresh = true;
		this._dirty = true;
	}

	public override update(timeMs: number = 0): boolean {
		if (this._isDisposed) {
			return false;
		}

		const width = this.canvas.width | 0;
		const height = this.canvas.height | 0;
		if (width <= 0 || height <= 0) {
			return false;
		}

		const dimensionsChanged = width !== this.width || height !== this.height;
		if (!dimensionsChanged && !this._forceRefresh && !this._dirty) {
			return false;
		}
		if (
			!dimensionsChanged &&
			!this._forceRefresh &&
			this._minUpdateIntervalMs > 0
		) {
			const elapsed = timeMs - this._lastUpdateTimeMs;
			if (
				Number.isFinite(elapsed) &&
				elapsed >= 0 &&
				elapsed < this._minUpdateIntervalMs
			) {
				return false;
			}
		}

		try {
			const frameData = this.context.getImageData(0, 0, width, height).data;
			if (
				!(this.data instanceof Uint8ClampedArray) ||
				this.data.length !== frameData.length
			) {
				this.data = new Uint8ClampedArray(frameData);
			} else {
				this.data.set(frameData);
			}
		} catch (error) {
			const reason =
				error instanceof Error ? error.message : "Unknown canvas error";
			throw new Error(
				`CanvasTexture failed to read canvas pixels. Ensure canvas CORS/canvas access is allowed: ${reason}`
			);
		}

		this.width = width;
		this.height = height;
		this.mipmaps = this.data ? [this.data] : [];
		this._forceRefresh = false;
		this._dirty = false;
		this._lastUpdateTimeMs =
			Number.isFinite(timeMs) ? timeMs : performance.now();
		this.markNeedsUpdate();
		return true;
	}

	public override dispose(): void {
		if (this._isDisposed) return;
		this._isDisposed = true;
		this._detachContextMutationTracking();
		super.dispose();
	}

	private _markDirtyFromContextMutation(): void {
		if (this._isDisposed) return;
		this._dirty = true;
	}

	private _attachContextMutationTracking(): void {
		let state = CanvasTexture._contextMutationPatches.get(this.context);
		if (!state) {
			state = {
				listeners: new Set<CanvasTexture>(),
				originals: new Map<string, CanvasMutationMethod>(),
			};
			CanvasTexture._contextMutationPatches.set(this.context, state);

			const context = this.context as unknown as Record<string, unknown>;
			for (const methodName of CANVAS_MUTATING_METHODS) {
				const original = context[methodName] as
					| CanvasMutationMethod
					| undefined;
				if (typeof original !== "function") {
					continue;
				}
				try {
					state.originals.set(methodName, original);
					const listeners = state.listeners;
					context[methodName] = (...args: unknown[]) => {
						for (const listener of listeners) {
							listener._markDirtyFromContextMutation();
						}
						return original.apply(this.context, args);
					};
				} catch {
					state.originals.delete(methodName);
				}
			}
		}
		state.listeners.add(this);
	}

	private _detachContextMutationTracking(): void {
		const state = CanvasTexture._contextMutationPatches.get(this.context);
		if (!state) {
			return;
		}
		state.listeners.delete(this);
		if (state.listeners.size > 0) {
			return;
		}

		const context = this.context as unknown as Record<string, unknown>;
		for (const [methodName, original] of state.originals) {
			try {
				context[methodName] = original;
			} catch {}
		}
		state.originals.clear();
		CanvasTexture._contextMutationPatches.delete(this.context);
	}

	private static _resolveCanvas(
		context: CanvasTextureContext2D
	): CanvasTextureCanvas {
		if (!context || typeof (context as any).getImageData !== "function") {
			throw new Error(
				"CanvasTexture requires a valid CanvasRenderingContext2D"
			);
		}
		const canvas = (context as any).canvas as CanvasTextureCanvas | undefined;
		if (
			!canvas ||
			typeof canvas.width !== "number" ||
			typeof canvas.height !== "number"
		) {
			throw new Error(
				"CanvasTexture requires a context with a valid backing canvas"
			);
		}
		return canvas;
	}
}
