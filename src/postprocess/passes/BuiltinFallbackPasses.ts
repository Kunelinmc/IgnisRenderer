import {
	DEFAULT_COLOR_FILTER_OPTIONS,
	DEFAULT_DOF_OPTIONS,
	DEFAULT_MOTION_BLUR_OPTIONS,
	type ColorFilterOptions,
	type DOFOptions,
	type MotionBlurOptions,
} from "../../pipeline/types";
import {
	PostProcessPass,
	type PostProcessPassConfig,
} from "../PostProcessPass";
import type {
	PostProcessPassImplementation,
	PostProcessPassRequirements,
} from "../types";

const FALLBACK_IMPLEMENTATION: PostProcessPassImplementation = {};

function fallbackImplementations(): PostProcessPassConfig["implementations"] {
	return {
		software: FALLBACK_IMPLEMENTATION,
		webgpu: FALLBACK_IMPLEMENTATION,
		webgl: FALLBACK_IMPLEMENTATION,
	};
}

function mergeOptions<TOptions extends object>(
	defaults: TOptions,
	options: Readonly<Partial<TOptions>>
): TOptions {
	return {
		...defaults,
		...options,
	};
}

export {
	BloomPass,
	type BloomPassConfig,
} from "./BloomPass";
export {
	FogPass,
	type FogPassConfig,
} from "./FogPass";
export {
	VolumetricLightingPass,
	type VolumetricLightingPassConfig,
} from "./VolumetricLightingPass";

export interface MotionBlurPassConfig
	extends Omit<
		PostProcessPassConfig<MotionBlurOptions>,
		"id" | "placement" | "implementations"
	> {}

/**
 * Stateful logical motion blur pass.
 */
export class MotionBlurPass extends PostProcessPass<
	MotionBlurOptions,
	MotionBlurOptions
> {
	public constructor(config: MotionBlurPassConfig = {}) {
		super({
			...config,
			id: "motion-blur",
			placement: "camera",
			implementations: fallbackImplementations(),
		});
	}

	public override normalizeOptions(): MotionBlurOptions {
		return mergeOptions(DEFAULT_MOTION_BLUR_OPTIONS, this.getRawOptions());
	}

	public override getRequirements(): PostProcessPassRequirements {
		return { gBuffer: ["depth", "motion"] };
	}
}

export interface DepthOfFieldPassConfig
	extends Omit<
		PostProcessPassConfig<DOFOptions>,
		"id" | "placement" | "implementations"
	> {}

/**
 * Stateful logical depth of field pass.
 */
export class DepthOfFieldPass extends PostProcessPass<DOFOptions, DOFOptions> {
	public constructor(config: DepthOfFieldPassConfig = {}) {
		super({
			...config,
			id: "dof",
			placement: "camera",
			implementations: fallbackImplementations(),
		});
	}

	public override normalizeOptions(): DOFOptions {
		return mergeOptions(DEFAULT_DOF_OPTIONS, this.getRawOptions());
	}

	public override getRequirements(): PostProcessPassRequirements {
		return { gBuffer: ["depth"] };
	}
}

/**
 * Stateful logical tone mapping pass.
 */
export class ToneMappingPass extends PostProcessPass<
	Record<string, never>,
	Record<string, never>
> {
	public constructor(
		config: Omit<
			PostProcessPassConfig<Record<string, never>>,
			"id" | "placement" | "implementations"
		> = {}
	) {
		super({
			...config,
			id: "tonemap",
			placement: "hdr",
			implementations: fallbackImplementations(),
		});
	}
}

export interface ColorFilterPassConfig
	extends Omit<
		PostProcessPassConfig<ColorFilterOptions>,
		"id" | "placement" | "implementations"
	> {}

/**
 * Stateful logical color filter pass.
 */
export class ColorFilterPass extends PostProcessPass<
	ColorFilterOptions,
	ColorFilterOptions
> {
	public constructor(config: ColorFilterPassConfig = {}) {
		super({
			...config,
			id: "color-filter",
			placement: "ldr",
			implementations: fallbackImplementations(),
		});
	}

	public override normalizeOptions(): ColorFilterOptions {
		return mergeOptions(DEFAULT_COLOR_FILTER_OPTIONS, this.getRawOptions());
	}
}

/**
 * Stateful logical interaction outline pass.
 */
export class InteractionOutlinePass extends PostProcessPass<
	Record<string, never>,
	Record<string, never>
> {
	public constructor(
		config: Omit<
			PostProcessPassConfig<Record<string, never>>,
			"id" | "placement" | "implementations"
		> = {}
	) {
		super({
			...config,
			id: "interaction-outline",
			placement: "overlay",
			implementations: fallbackImplementations(),
		});
	}
}

/**
 * Stateful logical gamma correction pass.
 */
export class GammaPass extends PostProcessPass<
	Record<string, never>,
	Record<string, never>
> {
	public constructor(
		config: Omit<
			PostProcessPassConfig<Record<string, never>>,
			"id" | "placement" | "implementations"
		> = {}
	) {
		super({
			...config,
			id: "gamma",
			placement: "present",
			implementations: fallbackImplementations(),
		});
	}
}
