import type {
	FogOptions,
} from "./types";
import type {
	PostProcessPass,
	PostProcessPassRegistrySnapshot,
} from "../postprocess/PostProcessPass";
export {
	PostProcessPassRegistry,
	getEnabledCustomPostProcessPassIds,
	hasEnabledCustomPostProcessPass,
	type PostProcessCapabilities,
	type PostProcessPassConfig,
	type PostProcessPassId,
} from "../postprocess/PostProcessPass";

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
