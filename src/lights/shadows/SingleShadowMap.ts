import { LightType } from "..";
import type { ShadowConfig } from "../ShadowMapping";
import { ShadowMapBase } from "./ShadowMapBase";
import type { ShadowMapBaseOptions } from "./types";

export class SingleShadowMap extends ShadowMapBase {
	public readonly kind: "single" | "vsm" = "single";

	constructor(options: ShadowMapBaseOptions = {}) {
		super(options);
	}

	public override toLegacyShadowConfig(
		_lightType: LightType,
		overrides?: {
			size?: number;
		}
	): ShadowConfig {
		return this.createSingleMapLegacyConfig(overrides?.size);
	}
}
