import type { AnimationInterpolation, AnimationTrackBinding } from "./types";

export interface KeyframeTrackParams {
	name?: string;
	binding: AnimationTrackBinding;
	times: ArrayLike<number>;
	values: ArrayLike<number>;
	valueSize: number;
	interpolation?: AnimationInterpolation;
}

export class KeyframeTrack {
	public readonly name: string;
	public readonly binding: AnimationTrackBinding;
	public readonly times: Float32Array;
	public readonly values: Float32Array;
	public readonly valueSize: number;
	public readonly interpolation: AnimationInterpolation;

	constructor(params: KeyframeTrackParams) {
		this.name =
			params.name ?? `${params.binding.targetPath}:${params.binding.property}`;
		this.binding = params.binding;
		this.times = new Float32Array(params.times);
		this.values = new Float32Array(params.values);
		this.valueSize = Math.max(1, Math.floor(params.valueSize));
		this.interpolation = params.interpolation ?? "linear";
	}

	public get frameCount(): number {
		return this.times.length;
	}
}
