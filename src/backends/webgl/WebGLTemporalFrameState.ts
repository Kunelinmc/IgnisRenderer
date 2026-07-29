import type { FrameContext } from "../../pipeline/types";
import type { FramePreparationRequirements } from "../../pipeline/FrameRequirements";
import { TemporalFrameState } from "../cross/TemporalFrameState";
import { toColumnMajorMat4 } from "./WebGLFrameMath";

/** @internal Transactional temporal camera uniforms for one WebGL device. */
export class WebGLTemporalFrameState {
	private readonly _state = new TemporalFrameState();
	private readonly _jitterCurrentPrev = new Float32Array(4);
	private _previousViewProjection: Float32Array | null = null;

	public get jitterCurrentPrev(): Float32Array {
		return this._jitterCurrentPrev;
	}

	public get previousViewProjection(): Float32Array | null {
		return this._previousViewProjection;
	}

	public beginFrame(
		context: FrameContext,
		frameRequirements: FramePreparationRequirements,
	): void {
		const snapshot = this._state.beginFrame({
			camera: context.viewCamera,
			width: context.attachments.width,
			height: context.attachments.height,
			frameRequirements,
			reset: context.incremental.temporalHistoryReset,
		});
		this._jitterCurrentPrev.set(snapshot.jitterCurrentPrev);
		this._previousViewProjection = snapshot.previousViewProjection ?
			toColumnMajorMat4(snapshot.previousViewProjection) : null;
	}

	public commitFrame(): void {
		this._state.commitFrame();
	}

	public abortFrame(): void {
		this._state.abortFrame();
		this._jitterCurrentPrev.fill(0);
	}

	public reset(): void {
		this._state.reset();
		this._jitterCurrentPrev.fill(0);
		this._previousViewProjection = null;
	}
}
