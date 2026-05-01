import { IdGenerator } from "../../foundation/IdGenerator";
import { LightType } from "..";
import type {
	CSMShadowConfig,
	ShadowConfig,
	ShadowParams,
	SingleMapShadowConfig,
} from "../ShadowMapping";
import type {
	ShadowBiasSettings,
	ShadowBoundLightType,
	ShadowFilterMode,
	ShadowMapBaseOptions,
	ShadowSamplingSettings,
} from "./types";

const DEFAULT_SHADOW_SIZE = 1024;

export abstract class ShadowMapBase {
	public readonly id: string;
	public enabled: boolean;
	public priority: number;
	public size: number;
	public bias: ShadowBiasSettings;
	public sampling: ShadowSamplingSettings;

	public abstract readonly kind: "single" | "vsm" | "csm";

	protected constructor(options: ShadowMapBaseOptions = {}) {
		this.id = options.id ?? IdGenerator.nextId("shadow");
		this.enabled = options.enabled !== false;
		this.priority = toFiniteNumber(options.priority, 0);
		this.size = Math.max(
			1,
			Math.floor(toFiniteNumber(options.size, DEFAULT_SHADOW_SIZE))
		);
		this.bias = {
			constant: toFiniteNumber(options.bias?.constant, 0),
			slope: toFiniteNumber(options.bias?.slope, 0.01),
			normal: toFiniteNumber(options.bias?.normal, 0.01),
			normalMin: toFiniteNumber(options.bias?.normalMin, 0.01),
			texel: toFiniteNumber(options.bias?.texel, 1.0),
			max: toFiniteNumber(options.bias?.max, 0.1),
		};
		this.sampling = {
			filterMode: options.sampling?.filterMode ?? "pcf",
			pcfRadius: toFiniteNumber(options.sampling?.pcfRadius, 1),
			strength: toFiniteNumber(options.sampling?.strength, 1),
			radius: toFiniteNumber(options.sampling?.radius, 0),
			samples: Math.max(1, Math.floor(toFiniteNumber(options.sampling?.samples, 16))),
			searchSamples: Math.max(
				1,
				Math.floor(toFiniteNumber(options.sampling?.searchSamples, 16))
			),
		};
	}

	public get filterMode(): ShadowFilterMode {
		return this.sampling.filterMode ?? "pcf";
	}

	public resolveBoundLightType(lightType: LightType): ShadowBoundLightType {
		switch (lightType) {
			case LightType.Directional:
				return "directional";
			case LightType.Point:
				return "point";
			case LightType.Spot:
				return "spot";
			default:
				return "rectArea";
		}
	}

	public estimateCost(
		lightType: LightType,
		size: number = this.size,
		cascadeCount: number = 1
	): number {
		const normalizedSize = Math.max(1, size) / 1024;
		const perSliceCost = normalizedSize * normalizedSize;
		const boundType = this.resolveBoundLightType(lightType);
		const sliceMultiplier =
			boundType === "point" ?
				Math.max(1, cascadeCount) * 6
			:	Math.max(1, cascadeCount);
		return perSliceCost * sliceMultiplier;
	}

	public abstract toLegacyShadowConfig(
		lightType: LightType,
		overrides?: {
			size?: number;
			cascadeCount?: number;
		}
	): ShadowConfig;

	protected resolveShadowParams(): ShadowParams {
		return {
			shadowBias: this.bias.constant,
			shadowSlopeBias: this.bias.slope,
			shadowNormalBias: this.bias.normal,
			shadowNormalBiasMin: this.bias.normalMin,
			shadowTexelBias: this.bias.texel,
			shadowMaxBias: this.bias.max,
			shadowPCF: this.sampling.pcfRadius,
			shadowStrength: this.sampling.strength,
			shadowRadius: this.sampling.radius,
			shadowSamples: this.sampling.samples,
			shadowSearchSamples: this.sampling.searchSamples,
		};
	}

	protected createSingleMapLegacyConfig(sizeOverride?: number): SingleMapShadowConfig {
		const size = Math.max(1, Math.floor(sizeOverride ?? this.size));
		return {
			strategy: "single-map",
			size,
			priority: this.priority,
			params: this.resolveShadowParams(),
		};
	}

	protected createCSMLegacyConfig(
		cascadeCount: number,
		options: {
			size?: number;
			lambda?: number;
			maxDistance?: number;
			blendRatio?: number;
			stabilize?: boolean;
		}
	): CSMShadowConfig {
		const size = Math.max(1, Math.floor(options.size ?? this.size));
		return {
			strategy: "csm",
			size,
			priority: this.priority,
			params: this.resolveShadowParams(),
			cascadeCount: clampCascadeCount(cascadeCount),
			splitMode: "practical",
			lambda: toFiniteNumber(options.lambda, 0.65),
			maxDistance: options.maxDistance,
			blendRatio: toFiniteNumber(options.blendRatio, 0.1),
			stabilize: options.stabilize !== false,
		};
	}
}

function toFiniteNumber(value: unknown, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return value;
}

function clampCascadeCount(value: number): 1 | 2 | 3 | 4 {
	if (value <= 1) return 1;
	if (value <= 2) return 2;
	if (value >= 4) return 4;
	return 3;
}
