import type { BlendTreeChildWeight } from "./types";

export interface BlendTree1DChild {
	clipName: string;
	threshold: number;
}

export interface BlendTree1DOptions {
	name: string;
	parameter: string;
	children: BlendTree1DChild[];
}

export class BlendTree1D {
	public readonly name: string;
	public readonly parameter: string;
	public readonly children: BlendTree1DChild[];

	constructor(options: BlendTree1DOptions) {
		this.name = options.name;
		this.parameter = options.parameter;
		this.children = [...options.children].sort(
			(left, right) => left.threshold - right.threshold
		);
	}

	public evaluate(parameterValue: number): BlendTreeChildWeight[] {
		if (this.children.length === 0) return [];
		if (this.children.length === 1) {
			return [{ clipName: this.children[0].clipName, weight: 1 }];
		}

		if (parameterValue <= this.children[0].threshold) {
			return [{ clipName: this.children[0].clipName, weight: 1 }];
		}
		const last = this.children[this.children.length - 1];
		if (parameterValue >= last.threshold) {
			return [{ clipName: last.clipName, weight: 1 }];
		}

		for (let i = 0; i < this.children.length - 1; i++) {
			const left = this.children[i];
			const right = this.children[i + 1];
			if (parameterValue < left.threshold || parameterValue > right.threshold) {
				continue;
			}
			const range = Math.max(1e-6, right.threshold - left.threshold);
			const t = (parameterValue - left.threshold) / range;
			return [
				{ clipName: left.clipName, weight: 1 - t },
				{ clipName: right.clipName, weight: t },
			];
		}

		return [];
	}
}
