import { computeHaltonJitterNDC } from "../../maths/Misc";

export interface TemporalJitterStateRequest {
	readonly enabled: boolean;
	readonly isOrthographic: boolean;
	readonly width: number;
	readonly height: number;
	readonly jitterScale?: number;
	readonly reset?: boolean;
}

/**
 * Current and previous temporal jitter values captured for one render frame.
 */
export interface TemporalJitterFrameState {
	/** Current frame camera jitter in NDC units. */
	readonly currentJitter: readonly [number, number];
	/** Previous frame camera jitter in NDC units. */
	readonly previousJitter: readonly [number, number];
}

/** @internal Restorable temporal jitter state for backend frame transactions. */
export interface TemporalJitterCheckpoint {
	readonly frameIndex: number;
	readonly current: readonly [number, number];
	readonly enabledLastFrame: boolean;
}

/**
 * Tracks current and previous temporal jitter for scene rendering.
 */
export class TemporalJitterState {
	private _frameIndex = 0;
	private _current: [number, number] = [0, 0];
	private _enabledLastFrame = false;

	/**
	 * Resets the temporal jitter sequence.
	 *
	 * @returns Nothing.
	 * @sideEffects Clears current and previous jitter history.
	 */
	public reset(): void {
		this._frameIndex = 0;
		this._current = [0, 0];
		this._enabledLastFrame = false;
	}

	/** @internal Captures state before beginning a backend frame transaction. */
	public createCheckpoint(): TemporalJitterCheckpoint {
		return {
			frameIndex: this._frameIndex,
			current: [this._current[0], this._current[1]],
			enabledLastFrame: this._enabledLastFrame,
		};
	}

	/** @internal Restores a checkpoint after a backend frame abort. */
	public restoreCheckpoint(checkpoint: TemporalJitterCheckpoint): void {
		this._frameIndex = checkpoint.frameIndex;
		this._current = [checkpoint.current[0], checkpoint.current[1]];
		this._enabledLastFrame = checkpoint.enabledLastFrame;
	}

	/**
	 * Advances and returns packed current/previous jitter.
	 *
	 * @param request Current frame jitter inputs.
	 * @returns `[currentX, currentY, previousX, previousY]` in NDC units.
	 * @sideEffects Advances the Halton sequence when enabled.
	 */
	public next(
		request: TemporalJitterStateRequest
	): [number, number, number, number] {
		if (request.reset) {
			this.reset();
		}
		if (
			!request.enabled ||
			request.isOrthographic ||
			request.width <= 0 ||
			request.height <= 0
		) {
			this.reset();
			return [0, 0, 0, 0];
		}

		const previous = this._enabledLastFrame ? this._current : [0, 0];
		const next = computeHaltonJitterNDC(
			this._frameIndex,
			request.width,
			request.height,
			request.jitterScale
		);
		this._frameIndex++;
		this._current = next;
		this._enabledLastFrame = true;
		return [next[0], next[1], previous[0], previous[1]];
	}

	/**
	 * Advances and returns current/previous jitter as a frame state snapshot.
	 *
	 * @param request Current frame jitter inputs.
	 * @returns Current and previous jitter in NDC units.
	 * @sideEffects Advances the Halton sequence when enabled.
	 */
	public nextFrameState(
		request: TemporalJitterStateRequest
	): TemporalJitterFrameState {
		const jitter = this.next(request);
		return {
			currentJitter: [jitter[0], jitter[1]],
			previousJitter: [jitter[2], jitter[3]],
		};
	}
}
