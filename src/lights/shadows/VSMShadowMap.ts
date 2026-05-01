import type { ShadowMapBaseOptions } from "./types";
import { SingleShadowMap } from "./SingleShadowMap";

export interface VSMShadowMapOptions extends ShadowMapBaseOptions {
	momentBias?: number;
	bleedReduction?: number;
}

/**
 * VSM v1 currently keeps runtime fallback behavior to PCF across backends.
 * The map still preserves VSM-specific parameters for future backend upgrades.
 */
export class VSMShadowMap extends SingleShadowMap {
	public readonly kind = "vsm" as const;
	public momentBias: number;
	public bleedReduction: number;

	constructor(options: VSMShadowMapOptions = {}) {
		super({
			...options,
			sampling: {
				...(options.sampling ?? {}),
				filterMode: "vsm",
			},
		});
		this.momentBias = resolveFinite(options.momentBias, 0.0005);
		this.bleedReduction = resolveFinite(options.bleedReduction, 0.1);
	}
}

function resolveFinite(value: unknown, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return value;
}
