import type { FramePassStage } from "../pipeline/types";
import type { PostProcessGrade } from "../pipeline/incremental";
import type { PostProcessPlacement } from "./ordering";

export interface BuiltinPostProcessOrderMetadata {
	readonly id: string;
	readonly placement: PostProcessPlacement;
	readonly order: number;
}

export interface BuiltinPostProcessIncrementalMetadata {
	readonly firstPass: FramePassStage | null;
	readonly grade: PostProcessGrade;
	readonly inflationRadius: number;
	readonly fallbackScale?: number;
	readonly enabledPredicate?: "fog-postprocess";
}

export interface BuiltinPostProcessMetadata
	extends BuiltinPostProcessOrderMetadata {
	readonly incremental: BuiltinPostProcessIncrementalMetadata;
}

type BuiltinPostProcessMetadataList = readonly BuiltinPostProcessMetadata[];

export const BUILTIN_POST_PROCESS_METADATA: BuiltinPostProcessMetadataList = [
	{
		id: "ssao",
		placement: "spatial",
		order: 100,
		incremental: {
			firstPass: "ssao",
			grade: "standard",
			inflationRadius: 8,
		},
	},
	{
		id: "ssgi",
		placement: "spatial",
		order: 110,
		incremental: {
			firstPass: "ssgi",
			grade: "standard",
			inflationRadius: 12,
		},
	},
	{
		id: "taa",
		placement: "temporal",
		order: 200,
		incremental: {
			firstPass: "taa",
			grade: "cinematic",
			inflationRadius: 8,
		},
	},
	{
		id: "ssr",
		placement: "temporal",
		order: 210,
		incremental: {
			firstPass: "ssr",
			grade: "cinematic",
			inflationRadius: 16,
		},
	},
	{
		id: "ssrefraction",
		placement: "temporal",
		order: 215,
		incremental: {
			firstPass: "ssrefraction",
			grade: "cinematic",
			inflationRadius: 16,
		},
	},
	{
		id: "volumetric",
		placement: "atmosphere",
		order: 300,
		incremental: {
			firstPass: "volumetric",
			grade: "cinematic",
			inflationRadius: 16,
		},
	},
	{
		id: "fog",
		placement: "atmosphere",
		order: 310,
		incremental: {
			firstPass: "fog",
			grade: "cinematic",
			inflationRadius: 20,
			enabledPredicate: "fog-postprocess",
		},
	},
	{
		id: "motion-blur",
		placement: "camera",
		order: 400,
		incremental: {
			firstPass: "motion-blur",
			grade: "cinematic",
			inflationRadius: 24,
		},
	},
	{
		id: "dof",
		placement: "camera",
		order: 410,
		incremental: {
			firstPass: "dof",
			grade: "cinematic",
			inflationRadius: 32,
		},
	},
	{
		id: "bloom",
		placement: "hdr",
		order: 500,
		incremental: {
			firstPass: "bloom",
			grade: "standard",
			inflationRadius: 48,
		},
	},
	{
		id: "tonemap",
		placement: "hdr",
		order: 600,
		incremental: {
			firstPass: "tonemap",
			grade: "light",
			inflationRadius: 0,
		},
	},
	{
		id: "color-filter",
		placement: "ldr",
		order: 700,
		incremental: {
			firstPass: "color-filter",
			grade: "light",
			inflationRadius: 2,
		},
	},
	{
		id: "fxaa",
		placement: "ldr",
		order: 710,
		incremental: {
			firstPass: "fxaa",
			grade: "light",
			inflationRadius: 2,
		},
	},
	{
		id: "interaction-outline",
		placement: "overlay",
		order: 800,
		incremental: {
			firstPass: "interaction-outline",
			grade: "light",
			inflationRadius: 2,
		},
	},
	{
		id: "gamma",
		placement: "present",
		order: 900,
		incremental: {
			firstPass: "gamma",
			grade: "light",
			inflationRadius: 0,
		},
	},
] as const;

const BUILTIN_POST_PROCESS_METADATA_BY_ID = new Map(
	BUILTIN_POST_PROCESS_METADATA.map((metadata) => [metadata.id, metadata])
);

/**
 * Returns built-in post-process ordering metadata for a pass id.
 *
 * @param id Built-in pass id.
 * @returns Ordering metadata, or `null` when the id is not a built-in pass.
 * @sideEffects None.
 */
export function getBuiltinPostProcessOrderMetadata(
	id: string
): BuiltinPostProcessOrderMetadata | null {
	const metadata = BUILTIN_POST_PROCESS_METADATA_BY_ID.get(id);
	if (!metadata) {
		return null;
	}
	return {
		id: metadata.id,
		placement: metadata.placement,
		order: metadata.order,
	};
}

/**
 * Resolves built-in ordering metadata and throws when the id is unknown.
 *
 * @param id Built-in pass id.
 * @returns Ordering metadata for use in built-in pass constructors.
 * @sideEffects None.
 */
export function getRequiredBuiltinPostProcessOrderMetadata(
	id: string
): BuiltinPostProcessOrderMetadata {
	const metadata = getBuiltinPostProcessOrderMetadata(id);
	if (!metadata) {
		throw new Error(`Unknown built-in post-process pass "${id}".`);
	}
	return metadata;
}
