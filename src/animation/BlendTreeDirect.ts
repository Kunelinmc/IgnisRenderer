import type { BlendTreeChildWeight } from "./types";

export interface BlendTreeDirectChild {
	clipName: string;
	parameter: string;
}

export interface BlendTreeDirectOptions {
	name: string;
	children: BlendTreeDirectChild[];
}

export class BlendTreeDirect {
	public readonly name: string;
	public readonly children: BlendTreeDirectChild[];

	constructor(options: BlendTreeDirectOptions) {
		this.name = options.name;
		this.children = [...options.children];
	}

	public evaluate(
		parameterValues: ReadonlyMap<string, number | boolean>
	): BlendTreeChildWeight[] {
		const weights: BlendTreeChildWeight[] = [];
		let total = 0;

		for (const child of this.children) {
			const rawValue = parameterValues.get(child.parameter);
			const numeric =
				typeof rawValue === "boolean"
					? rawValue
						? 1
						: 0
					: typeof rawValue === "number"
						? rawValue
						: 0;
			const weight = Math.max(0, numeric);
			total += weight;
			weights.push({ clipName: child.clipName, weight });
		}

		if (total <= 1e-6 && weights.length > 0) {
			weights[0].weight = 1;
			for (let i = 1; i < weights.length; i++) {
				weights[i].weight = 0;
			}
			return weights;
		}

		if (total > 1e-6) {
			for (const weight of weights) {
				weight.weight /= total;
			}
		}

		return weights;
	}
}
