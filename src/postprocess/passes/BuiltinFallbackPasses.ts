import {
	DEFAULT_BLOOM_OPTIONS,
	DEFAULT_COLOR_FILTER_OPTIONS,
	DEFAULT_DOF_OPTIONS,
	DEFAULT_FOG_OPTIONS,
	DEFAULT_MOTION_BLUR_OPTIONS,
	DEFAULT_VOLUMETRIC_OPTIONS,
	type BloomOptions,
	type ColorFilterOptions,
	type DOFOptions,
	type FogOptions,
	type MotionBlurOptions,
	type VolumetricOptions,
} from "../../pipeline/types";
import {
	PostProcessPass,
	type PostProcessPassConfig,
} from "../PostProcessPass";
import type {
	PostProcessHistoryDescriptor,
	IPostProcessExecutor,
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassRequirements,
	PostProcessPassResult,
} from "../types";

const DEFAULT_HISTORY_USAGE = ["sampled", "storage", "render-target"] as const;
const MOTION_HISTORY_USAGE = ["sampled", "copy-dst", "render-target"] as const;
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

export interface VolumetricLightingPassConfig
	extends Omit<
		PostProcessPassConfig<VolumetricOptions>,
		"id" | "placement" | "implementations"
	> {}

/**
 * Stateful logical volumetric lighting pass.
 */
export class VolumetricLightingPass extends PostProcessPass<
	VolumetricOptions,
	VolumetricOptions
> {
	public constructor(config: VolumetricLightingPassConfig = {}) {
		super({
			...config,
			id: "volumetric",
			placement: "atmosphere",
			implementations: fallbackImplementations(),
		});
	}

	public override normalizeOptions(): VolumetricOptions {
		return mergeOptions(DEFAULT_VOLUMETRIC_OPTIONS, this.getRawOptions());
	}

	public override getRequirements(): PostProcessPassRequirements {
		return { gBuffer: ["depth", "motion"] };
	}

	public override getHistoryDescriptors(): readonly PostProcessHistoryDescriptor[] {
		return [
			{ id: "volumetric", usage: DEFAULT_HISTORY_USAGE },
			{ id: "volumetric-reservoir", usage: DEFAULT_HISTORY_USAGE },
			{ id: "motion", usage: MOTION_HISTORY_USAGE },
		];
	}
}

export interface FogPassConfig
	extends Omit<
		PostProcessPassConfig<FogOptions>,
		"id" | "placement" | "implementations"
	> {}

/**
 * Stateful logical fog pass.
 */
export class FogPass extends PostProcessPass<FogOptions, FogOptions> {
	public constructor(config: FogPassConfig = {}) {
		super({
			...config,
			id: "fog",
			placement: "atmosphere",
			implementations: fallbackImplementations(),
		});
	}

	public override normalizeOptions(): FogOptions {
		return mergeOptions(DEFAULT_FOG_OPTIONS, this.getRawOptions());
	}

	public override getRequirements(): PostProcessPassRequirements {
		return { gBuffer: ["depth"] };
	}

	public override execute(
		request: PostProcessPassRequest<FogOptions>,
		context: unknown,
		executor: IPostProcessExecutor
	): PostProcessPassResult | Promise<PostProcessPassResult> {
		if ((request.options.application ?? "postprocess") === "scene") {
			return { ran: false };
		}
		return super.execute(request, context, executor);
	}
}

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

export interface BloomPassConfig
	extends Omit<
		PostProcessPassConfig<BloomOptions>,
		"id" | "placement" | "implementations"
	> {}

/**
 * Stateful logical bloom pass.
 */
export class BloomPass extends PostProcessPass<BloomOptions, BloomOptions> {
	public constructor(config: BloomPassConfig = {}) {
		super({
			...config,
			id: "bloom",
			placement: "hdr",
			implementations: fallbackImplementations(),
		});
	}

	public override normalizeOptions(): BloomOptions {
		return mergeOptions(DEFAULT_BLOOM_OPTIONS, this.getRawOptions());
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
