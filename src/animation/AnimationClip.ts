import { KeyframeTrack } from "./KeyframeTrack";

export interface AnimationClipParams {
	name: string;
	duration: number;
	tracks?: KeyframeTrack[];
}

export class AnimationClip {
	public readonly name: string;
	public readonly duration: number;
	public readonly tracks: KeyframeTrack[];

	constructor(params: AnimationClipParams) {
		this.name = params.name;
		this.duration = Math.max(0, params.duration);
		this.tracks = params.tracks ? [...params.tracks] : [];
	}

	public addTrack(track: KeyframeTrack): this {
		this.tracks.push(track);
		return this;
	}
}
