import { AnimationClip } from "./AnimationClip";

interface FadeState {
	startWeight: number;
	targetWeight: number;
	duration: number;
	elapsed: number;
}

export interface AnimationActionOptions {
	weight?: number;
	speed?: number;
	loop?: boolean;
	repetitions?: number;
	additive?: boolean;
}

export class AnimationAction {
	public readonly clip: AnimationClip;
	public time: number;
	public weight: number;
	public speed: number;
	public loop: boolean;
	public repetitions: number;
	public enabled: boolean;
	public paused: boolean;
	public additive: boolean;
	public finished: boolean;
	private _loopCount: number;
	private _fadeState: FadeState | null;

	constructor(clip: AnimationClip, options: AnimationActionOptions = {}) {
		this.clip = clip;
		this.time = 0;
		this.weight = options.weight ?? 1;
		this.speed = options.speed ?? 1;
		this.loop = options.loop ?? true;
		this.repetitions = options.repetitions ?? Number.POSITIVE_INFINITY;
		this.enabled = true;
		this.paused = false;
		this.additive = options.additive ?? false;
		this.finished = false;
		this._loopCount = 0;
		this._fadeState = null;
	}

	public play(): this {
		this.enabled = true;
		this.paused = false;
		this.finished = false;
		return this;
	}

	public stop(): this {
		this.enabled = false;
		this.finished = true;
		return this;
	}

	public reset(): this {
		this.time = 0;
		this._loopCount = 0;
		this.finished = false;
		return this;
	}

	public setEffectiveWeight(weight: number): this {
		this.weight = Math.max(0, weight);
		return this;
	}

	public setEffectiveTimeScale(speed: number): this {
		this.speed = speed;
		return this;
	}

	public fadeTo(targetWeight: number, duration: number): this {
		this._fadeState = {
			startWeight: this.weight,
			targetWeight: Math.max(0, targetWeight),
			duration: Math.max(1e-6, duration),
			elapsed: 0,
		};
		return this;
	}

	public fadeIn(duration: number): this {
		this.weight = 0;
		return this.fadeTo(1, duration);
	}

	public fadeOut(duration: number): this {
		return this.fadeTo(0, duration);
	}

	public crossFadeTo(target: AnimationAction, duration: number): this {
		target.reset().play().fadeIn(duration);
		this.fadeOut(duration);
		return this;
	}

	public update(deltaSeconds: number): void {
		if (!this.enabled || this.paused || this.finished) return;

		if (this._fadeState) {
			this._fadeState.elapsed += Math.max(0, deltaSeconds);
			const t = Math.min(1, this._fadeState.elapsed / this._fadeState.duration);
			this.weight =
				this._fadeState.startWeight +
				(this._fadeState.targetWeight - this._fadeState.startWeight) * t;
			if (t >= 1) {
				this._fadeState = null;
				if (this.weight <= 0) {
					this.enabled = false;
				}
			}
		}

		const duration = this.clip.duration;
		if (duration <= 0) return;

		this.time += deltaSeconds * this.speed;
		if (this.loop) {
			while (this.time >= duration) {
				this.time -= duration;
				this._loopCount++;
				if (this._loopCount >= this.repetitions) {
					this.finished = true;
					this.enabled = false;
					this.time = duration;
					break;
				}
			}
			while (this.time < 0) {
				this.time += duration;
			}
			return;
		}

		if (this.time >= duration) {
			this.time = duration;
			this.finished = true;
			this.enabled = false;
		}
		if (this.time < 0) {
			this.time = 0;
		}
	}
}
