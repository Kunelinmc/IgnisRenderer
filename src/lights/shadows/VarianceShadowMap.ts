import { LightType } from "..";
import type { ShadowConfig } from "./ShadowMapping";
import type { ShadowMapBaseOptions } from "./types";
import { SingleShadowMap } from "./SingleShadowMap";

export interface VarianceShadowMapOptions extends ShadowMapBaseOptions {
	momentBias?: number;
	bleedReduction?: number;
	minVariance?: number;
}

/**
 * VarianceShadowMap v1 currently keeps runtime fallback behavior to PCF across backends.
 * The map still preserves VSM-specific parameters for future backend upgrades.
 */
export class VarianceShadowMap extends SingleShadowMap {
	public override readonly kind = "variance" as const;
	private _momentBias: number;
	private _bleedReduction: number;
	private _minVariance: number;

	constructor(options: VarianceShadowMapOptions = {}) {
		super({
			...options,
			sampling: {
				...(options.sampling ?? {}),
				filterMode: "vsm",
			},
		});
		this._momentBias = DEFAULT_VSM_MOMENT_BIAS;
		this._bleedReduction = DEFAULT_VSM_BLEED_REDUCTION;
		this._minVariance = DEFAULT_VSM_MIN_VARIANCE;
		this.momentBias = options.momentBias ?? DEFAULT_VSM_MOMENT_BIAS;
		this.bleedReduction =
			options.bleedReduction ?? DEFAULT_VSM_BLEED_REDUCTION;
		this.minVariance = options.minVariance ?? DEFAULT_VSM_MIN_VARIANCE;
	}

	public get momentBias(): number {
		return this._momentBias;
	}

	public set momentBias(value: number) {
		this._momentBias = clampFinite(
			value,
			DEFAULT_VSM_MOMENT_BIAS,
			0,
			1
		);
	}

	public get bleedReduction(): number {
		return this._bleedReduction;
	}

	public set bleedReduction(value: number) {
		this._bleedReduction = clampFinite(
			value,
			DEFAULT_VSM_BLEED_REDUCTION,
			0,
			1
		);
	}

	public get minVariance(): number {
		return this._minVariance;
	}

	public set minVariance(value: number) {
		this._minVariance = clampFinite(
			value,
			DEFAULT_VSM_MIN_VARIANCE,
			1e-8,
			1
		);
	}

	public setVarianceParameters(options: {
		momentBias?: number;
		bleedReduction?: number;
		minVariance?: number;
	}): this {
		if (options.momentBias !== undefined) {
			this.momentBias = options.momentBias;
		}
		if (options.bleedReduction !== undefined) {
			this.bleedReduction = options.bleedReduction;
		}
		if (options.minVariance !== undefined) {
			this.minVariance = options.minVariance;
		}
		return this;
	}

	public override toLegacyShadowConfig(
		lightType: LightType,
		overrides?: {
			size?: number;
		}
	): ShadowConfig {
		const config = super.toLegacyShadowConfig(lightType, overrides);
		if (config.strategy !== "single-map") {
			return config;
		}
		return {
			...config,
			params: {
				...(config.params ?? {}),
				shadowMomentBias: this.momentBias,
				shadowBleedReduction: this.bleedReduction,
				shadowMinVariance: this.minVariance,
			},
		};
	}
}

const DEFAULT_VSM_MOMENT_BIAS = 0.0005;
const DEFAULT_VSM_BLEED_REDUCTION = 0.1;
const DEFAULT_VSM_MIN_VARIANCE = 0.00002;

function clampFinite(
	value: unknown,
	fallback: number,
	min: number,
	max: number
): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return Math.min(max, Math.max(min, value));
}
