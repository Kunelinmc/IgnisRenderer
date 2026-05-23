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

export const MOTION_BLUR_PASS_ID = "motion-blur";
export const DEPTH_OF_FIELD_PASS_ID = "dof";
export const TONE_MAPPING_PASS_ID = "tonemap";
export const COLOR_FILTER_PASS_ID = "color-filter";
export const INTERACTION_OUTLINE_PASS_ID = "interaction-outline";
export const GAMMA_PASS_ID = "gamma";

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
		| "id"
		| "builtIn"
		| "capabilityId"
		| "warningLabel"
		| "placement"
		| "order"
		| "implementations"
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
			id: MOTION_BLUR_PASS_ID,
			builtIn: true,
			capabilityId: MOTION_BLUR_PASS_ID,
			warningLabel: "motion blur",
			placement: "camera",
			order: 400,
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
		| "id"
		| "builtIn"
		| "capabilityId"
		| "warningLabel"
		| "placement"
		| "order"
		| "implementations"
	> {}

/**
 * Stateful logical depth of field pass.
 */
export class DepthOfFieldPass extends PostProcessPass<DOFOptions, DOFOptions> {
	public constructor(config: DepthOfFieldPassConfig = {}) {
		super({
			...config,
			id: DEPTH_OF_FIELD_PASS_ID,
			builtIn: true,
			capabilityId: DEPTH_OF_FIELD_PASS_ID,
			warningLabel: "depth of field",
			placement: "camera",
			order: 410,
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
			| "id"
			| "builtIn"
			| "capabilityId"
			| "warningLabel"
			| "placement"
			| "order"
			| "implementations"
		> = {}
	) {
		super({
			...config,
			id: TONE_MAPPING_PASS_ID,
			builtIn: true,
			capabilityId: TONE_MAPPING_PASS_ID,
			warningLabel: "tone mapping",
			placement: "hdr",
			order: 600,
			implementations: fallbackImplementations(),
		});
	}
}

export interface ColorFilterPassConfig
	extends Omit<
		PostProcessPassConfig<ColorFilterOptions>,
		| "id"
		| "builtIn"
		| "capabilityId"
		| "warningLabel"
		| "placement"
		| "order"
		| "implementations"
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
			id: COLOR_FILTER_PASS_ID,
			builtIn: true,
			capabilityId: COLOR_FILTER_PASS_ID,
			warningLabel: "color filter",
			placement: "ldr",
			order: 700,
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
			| "id"
			| "builtIn"
			| "capabilityId"
			| "warningLabel"
			| "placement"
			| "order"
			| "implementations"
		> = {}
	) {
		super({
			...config,
			id: INTERACTION_OUTLINE_PASS_ID,
			builtIn: true,
			capabilityId: INTERACTION_OUTLINE_PASS_ID,
			warningLabel: "interaction outline",
			placement: "overlay",
			order: 800,
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
			| "id"
			| "builtIn"
			| "capabilityId"
			| "warningLabel"
			| "placement"
			| "order"
			| "implementations"
		> = {}
	) {
		super({
			...config,
			id: GAMMA_PASS_ID,
			builtIn: true,
			capabilityId: GAMMA_PASS_ID,
			warningLabel: "gamma correction",
			placement: "present",
			order: 900,
			implementations: fallbackImplementations(),
		});
	}
}
