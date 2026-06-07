import {
	BUILTIN_POST_PROCESS_METADATA,
	getBuiltinPostProcessOrderMetadata,
} from "./builtinMetadata";

export const POST_PROCESS_PLACEMENTS = [
	"spatial",
	"temporal",
	"atmosphere",
	"camera",
	"hdr",
	"ldr",
	"overlay",
	"present",
] as const;

export type PostProcessPlacement = (typeof POST_PROCESS_PLACEMENTS)[number];

export const DEFAULT_POST_PROCESS_PLACEMENT: PostProcessPlacement = "overlay";

export interface BuiltinPostProcessOrderEntry {
	readonly id: string;
	readonly placement: PostProcessPlacement;
	readonly order: number;
}

export const BUILTIN_POST_PROCESS_ORDER: readonly BuiltinPostProcessOrderEntry[] =
	BUILTIN_POST_PROCESS_METADATA.map((metadata) => ({
		id: metadata.id,
		placement: metadata.placement,
		order: metadata.order,
	}));

const CUSTOM_PLACEMENT_ORDER: Record<PostProcessPlacement, number> = {
	spatial: 120,
	temporal: 220,
	atmosphere: 320,
	camera: 420,
	hdr: 550,
	ldr: 650,
	overlay: 850,
	present: 890,
};

/**
 * Returns static ordering metadata for a built-in post-process pass.
 *
 * @param id Candidate pass id.
 * @returns Built-in ordering metadata or `null`.
 * @sideEffects None.
 */
export function getBuiltinPostProcessOrder(
	id: string
): BuiltinPostProcessOrderEntry | null {
	return getBuiltinPostProcessOrderMetadata(id);
}

/**
 * Returns the stable insertion order for custom passes in a placement bucket.
 *
 * @param placement Placement bucket requested by a custom pass.
 * @returns Numeric base order used by `PostProcessPipeline`.
 * @sideEffects None.
 */
export function getCustomPostProcessPlacementOrder(
	placement: PostProcessPlacement
): number {
	return CUSTOM_PLACEMENT_ORDER[placement];
}

/**
 * Returns whether a value names a supported post-process placement bucket.
 *
 * @param value Candidate placement value.
 * @returns `true` when the value is a supported placement.
 * @sideEffects None.
 */
export function isPostProcessPlacement(
	value: unknown
): value is PostProcessPlacement {
	return (
		typeof value === "string" &&
		(POST_PROCESS_PLACEMENTS as readonly string[]).includes(value)
	);
}
