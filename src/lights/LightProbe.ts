import { SH } from "../maths/SH";
import { Light, LightType } from "./Light";
import type { SHCoefficients } from "../maths/types";
import type { Texture } from "../core/Texture";

/**
 * LightProbe stores spherical harmonics coefficients and optional prefiltered map.
 * Baking/projection from environment maps lives in pipeline helpers.
 */
export class LightProbe extends Light<LightType.LightProbe> {
	public sh: SHCoefficients;
	public prefilteredMap: Texture | null = null;

	constructor(
		sh: SHCoefficients | null = null,
		intensity = 1.0,
		prefilteredMap: Texture | null = null
	) {
		super(LightType.LightProbe, { intensity });
		this.sh = sh ? JSON.parse(JSON.stringify(sh)) : SH.empty();
		this.prefilteredMap = prefilteredMap;
	}

	public copy(source: LightProbe | SHCoefficients): LightProbe {
		const sourceSH = source instanceof LightProbe ? source.sh : source;
		const sourceIntensity =
			source instanceof LightProbe ? source.intensity : this.intensity;

		for (let i = 0; i < this.sh.length; i++) {
			this.sh[i].r = sourceSH[i].r;
			this.sh[i].g = sourceSH[i].g;
			this.sh[i].b = sourceSH[i].b;
		}

		this.intensity = sourceIntensity;

		if (source instanceof LightProbe) {
			this.prefilteredMap = source.prefilteredMap;
		}

		return this;
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.sh = this.sh.map((coefficient) => ({
			r: coefficient.r,
			g: coefficient.g,
			b: coefficient.b,
		}));
		target.prefilteredMap = this.prefilteredMap;
	}
}
