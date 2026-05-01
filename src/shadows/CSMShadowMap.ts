import { LightType } from "../lights";
import type { ShadowConfig } from "../lights/ShadowMapping";
import { ShadowMapBase } from "./ShadowMapBase";
import type {
	ShadowBoundLightType,
	ShadowCSMDefaults,
	ShadowCSMOptions,
} from "./types";

const DEFAULT_CASCADE_COUNTS: ShadowCSMDefaults = {
	directional: 4,
	spot: 3,
	point: 2,
};

export class CSMShadowMap extends ShadowMapBase {
	public readonly kind = "csm" as const;
	public cascadeCounts: ShadowCSMDefaults;
	public lambda: number;
	public maxDistance?: number;
	public blendRatio: number;
	public stabilize: boolean;

	constructor(options: ShadowCSMOptions = {}) {
		super(options);
		this.cascadeCounts = {
			directional: clampCascadeCount(
				resolveFinite(options.cascadeCounts?.directional, 4)
			),
			spot: clampCascadeCount(resolveFinite(options.cascadeCounts?.spot, 3)),
			point: clampCascadeCount(resolveFinite(options.cascadeCounts?.point, 2)),
		};
		this.lambda = resolveFinite(options.lambda, 0.65);
		this.maxDistance =
			typeof options.maxDistance === "number" &&
			Number.isFinite(options.maxDistance) ?
				Math.max(0.01, options.maxDistance)
			:	undefined;
		this.blendRatio = resolveFinite(options.blendRatio, 0.1);
		this.stabilize = options.stabilize !== false;
	}

	public getCascadeCountForLightType(lightType: LightType): number {
		const boundType = this.resolveBoundLightType(lightType);
		return this.getCascadeCountForBoundType(boundType);
	}

	public getCascadeCountForBoundType(boundType: ShadowBoundLightType): number {
		switch (boundType) {
			case "directional":
				return this.cascadeCounts.directional;
			case "spot":
				return this.cascadeCounts.spot;
			case "point":
				return this.cascadeCounts.point;
			default:
				return 2;
		}
	}

	public override toLegacyShadowConfig(
		lightType: LightType,
		overrides?: {
			size?: number;
			cascadeCount?: number;
		}
	): ShadowConfig {
		const cascadeCount =
			overrides?.cascadeCount ?? this.getCascadeCountForLightType(lightType);
		return this.createCSMLegacyConfig(cascadeCount, {
			size: overrides?.size,
			lambda: this.lambda,
			maxDistance: this.maxDistance,
			blendRatio: this.blendRatio,
			stabilize: this.stabilize,
		});
	}
}

function resolveFinite(value: unknown, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return value;
}

function clampCascadeCount(value: number): number {
	return Math.max(1, Math.min(4, Math.floor(value)));
}

export { DEFAULT_CASCADE_COUNTS };
