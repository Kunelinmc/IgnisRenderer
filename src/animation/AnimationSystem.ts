import type { Node } from "../core/Node";
import { AnimationMixer } from "./AnimationMixer";

export class AnimationSystem {
	private _mixers = new Set<AnimationMixer>();

	public createMixer(root: Node): AnimationMixer {
		const mixer = new AnimationMixer({ root });
		this._mixers.add(mixer);
		return mixer;
	}

	public addMixer(mixer: AnimationMixer): AnimationMixer {
		this._mixers.add(mixer);
		return mixer;
	}

	public removeMixer(mixer: AnimationMixer): boolean {
		return this._mixers.delete(mixer);
	}

	public clear(): void {
		this._mixers.clear();
	}

	public get mixers(): AnimationMixer[] {
		return Array.from(this._mixers);
	}

	public hasActiveActions(): boolean {
		for (const mixer of this._mixers) {
			if (mixer.hasActiveActions()) return true;
		}
		return false;
	}
}
