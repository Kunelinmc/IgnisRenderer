import type { Vec4Tuple } from "../../maths/Vector4";
import { CameraType, type Camera } from "../../cameras/Camera";
import type { Matrix4 } from "../../maths/Matrix4";
import type { FramePreparationRequirements } from "../../pipeline/FrameRequirements";
import {
	TemporalJitterState,
	type TemporalJitterCheckpoint,
} from "./TemporalJitterState";

export interface TemporalFrameSnapshot {
	readonly jitterCurrentPrev: Readonly<Vec4Tuple>;
	readonly previousViewProjection: Matrix4 | null;
}

export interface TemporalFrameBeginRequest {
	readonly camera: Camera;
	readonly width: number;
	readonly height: number;
	readonly frameRequirements: FramePreparationRequirements;
	readonly reset?: boolean;
}

/**
 * Tracks tentative camera-temporal state across one backend frame transaction.
 *
 * @internal Owned by backend frame services. Post-process histories remain
 * owned by `BackendPostProcessRuntime`.
 */
export class TemporalFrameState {
	private readonly _jitter = new TemporalJitterState();
	private _previousViewProjection: Matrix4 | null = null;
	private _pendingViewProjection: Matrix4 | null = null;
	private _checkpoint: TemporalJitterCheckpoint | null = null;
	private _snapshot: TemporalFrameSnapshot = {
		jitterCurrentPrev: [0, 0, 0, 0],
		previousViewProjection: null,
	};

	public beginFrame(request: TemporalFrameBeginRequest): TemporalFrameSnapshot {
		if (this._checkpoint) {
			throw new Error("Temporal frame state already has an active transaction.");
		}
		this._checkpoint = this._jitter.createCheckpoint();
		const requirement = request.frameRequirements.cameraJitter;
		const jitterCurrentPrev = this._jitter.next({
			enabled: !!requirement,
			isOrthographic: request.camera.type === CameraType.Orthographic,
			width: request.width,
			height: request.height,
			jitterScale: requirement?.scale,
			reset: request.reset,
		});
		this._pendingViewProjection = request.camera.viewProjectionMatrix.clone();
		this._snapshot = {
			jitterCurrentPrev,
			previousViewProjection:
				request.reset ? null : this._previousViewProjection,
		};
		return this._snapshot;
	}

	public get snapshot(): TemporalFrameSnapshot {
		return this._snapshot;
	}

	public commitFrame(): void {
		if (!this._checkpoint) return;
		this._previousViewProjection = this._pendingViewProjection;
		this._pendingViewProjection = null;
		this._checkpoint = null;
	}

	public abortFrame(): void {
		if (this._checkpoint) {
			this._jitter.restoreCheckpoint(this._checkpoint);
		}
		this._pendingViewProjection = null;
		this._checkpoint = null;
		this._snapshot = {
			jitterCurrentPrev: [0, 0, 0, 0],
			previousViewProjection: this._previousViewProjection,
		};
	}

	public reset(): void {
		this._jitter.reset();
		this._previousViewProjection = null;
		this._pendingViewProjection = null;
		this._checkpoint = null;
		this._snapshot = {
			jitterCurrentPrev: [0, 0, 0, 0],
			previousViewProjection: null,
		};
	}
}
