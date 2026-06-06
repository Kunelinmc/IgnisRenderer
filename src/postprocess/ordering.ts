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

export const BUILTIN_POST_PROCESS_ORDER: readonly BuiltinPostProcessOrderEntry[] = [
	{ id: "ssao", placement: "spatial", order: 100 },
	{ id: "ssgi", placement: "spatial", order: 110 },
	{ id: "taa", placement: "temporal", order: 200 },
	{ id: "ssr", placement: "temporal", order: 210 },
	{ id: "ssrefraction", placement: "temporal", order: 215 },
	{ id: "volumetric", placement: "atmosphere", order: 300 },
	{ id: "fog", placement: "atmosphere", order: 310 },
	{ id: "motion-blur", placement: "camera", order: 400 },
	{ id: "dof", placement: "camera", order: 410 },
	{ id: "bloom", placement: "hdr", order: 500 },
	{ id: "tonemap", placement: "hdr", order: 600 },
	{ id: "color-filter", placement: "ldr", order: 700 },
	{ id: "fxaa", placement: "ldr", order: 710 },
	{ id: "interaction-outline", placement: "overlay", order: 800 },
	{ id: "gamma", placement: "present", order: 900 },
] as const;

const PASS_OWNED_BUILTIN_ORDER_BY_ID = new Map(
	BUILTIN_POST_PROCESS_ORDER.map((entry) => [entry.id, entry])
);

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
 * Registers pass-owned ordering metadata for a built-in post-process pass.
 *
 * @param entry Built-in pass id, placement bucket, and absolute execution order.
 * @returns The same metadata entry for reuse in pass constructors.
 * @sideEffects Adds or confirms the entry in the built-in order lookup table.
 */
export function defineBuiltinPostProcessOrder<
	TEntry extends BuiltinPostProcessOrderEntry,
>(entry: TEntry): TEntry {
	if (!entry.id) {
		throw new Error("Built-in post-process pass id is required.");
	}
	if (!isPostProcessPlacement(entry.placement)) {
		throw new Error(
			`Built-in post-process pass "${entry.id}" has invalid placement.`
		);
	}
	if (!Number.isFinite(entry.order)) {
		throw new Error(
			`Built-in post-process pass "${entry.id}" requires a finite order.`
		);
	}
	const current = PASS_OWNED_BUILTIN_ORDER_BY_ID.get(entry.id);
	if (
		current &&
		(current.placement !== entry.placement || current.order !== entry.order)
	) {
		throw new Error(
			`Built-in post-process pass "${entry.id}" has conflicting order metadata.`
		);
	}
	PASS_OWNED_BUILTIN_ORDER_BY_ID.set(entry.id, entry);
	return entry;
}

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
	return PASS_OWNED_BUILTIN_ORDER_BY_ID.get(id) ?? null;
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
