import type {
	BloomOptions,
	ColorFilterOptions,
	DOFOptions,
	FogOptions,
	MotionBlurOptions,
	SSAOOptions,
	SSGIOptions,
	SSROptions,
	TAAOptions,
	VolumetricOptions,
} from "./types";
import type {
	PostProcessPass,
	PostProcessPassRegistrySnapshot,
} from "../postprocess/PostProcessPass";
export {
	DEFAULT_POST_PROCESS_CAPABILITIES,
	POST_PROCESS_PASS_IDS,
	PostProcessPassRegistry,
	getEnabledCustomPostProcessPassIds,
	hasEnabledCustomPostProcessPass,
	isBuiltInPostProcessPassId,
	type PostProcessCapabilities,
	type PostProcessPassConfig,
	type PostProcessPassId,
} from "../postprocess/PostProcessPass";

export interface PostProcessOptionsMap {
	ssao: SSAOOptions;
	ssgi: SSGIOptions;
	taa: TAAOptions;
	ssr: SSROptions;
	volumetric: VolumetricOptions;
	fog: FogOptions;
	"motion-blur": MotionBlurOptions;
	dof: DOFOptions;
	bloom: BloomOptions;
	tonemap: Record<string, never>;
	"color-filter": ColorFilterOptions;
	fxaa: Record<string, never>;
	"interaction-outline": Record<string, never>;
	gamma: Record<string, never>;
}

export type ResolvedPostProcessOptionsMap = {
	[K in keyof PostProcessOptionsMap]: PostProcessOptionsMap[K];
} & Record<string, unknown>;

export type ResolvedPostProcessState = PostProcessPassRegistrySnapshot;
export type PostProcessCustomPassDescriptor = PostProcessPass;

/**
 * Returns whether fog should execute as a post-process pass for this frame.
 *
 * @param postProcess Per-frame post-process registry snapshot.
 * @returns `true` when `fog` is enabled and configured for post-process mode.
 * @sideEffects None.
 */
export function isFogPostProcessEnabled(
	postProcess: PostProcessPassRegistrySnapshot
): boolean {
	return (
		postProcess.isEnabled("fog") &&
		(postProcess.getOptions<FogOptions>("fog")?.application ?? "postprocess") !==
			"scene"
	);
}
