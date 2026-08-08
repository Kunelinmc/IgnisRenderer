import type { ShadowCastingLight } from "../../lights";
import {
	createShadowRenderSet,
	ensureShadowRenderSetMatchesConfig,
	type ShadowConfig,
	type ShadowRenderSet,
	type ShadowRenderSetOptions,
} from "../../lights/shadows/ShadowMapping";
import type { ShadowFramePlan } from "./ShadowFramePlan";

export interface LegacyShadowPlanEntry {
	readonly light: ShadowCastingLight;
	readonly config: ShadowConfig;
	readonly options: ShadowRenderSetOptions;
}

const PUBLISHED_LEGACY_SHADOW_MAPS = new WeakMap<
	ShadowFramePlan,
	Map<ShadowCastingLight, ShadowRenderSet>
>();

/**
 * Temporary compatibility bridge for backends being migrated to direct plan use.
 *
 * @internal This adapter owns legacy mutable metadata outside the immutable plan.
 */
export class LegacyShadowPlanAdapter {
	private readonly _renderSets = new Map<ShadowCastingLight, ShadowRenderSet>();

	public reconcile(
		entries: readonly LegacyShadowPlanEntry[]
	): Map<ShadowCastingLight, ShadowRenderSet> {
		const active = new Set(entries.map((entry) => entry.light));
		for (const light of this._renderSets.keys()) {
			if (!active.has(light)) this._renderSets.delete(light);
		}

		for (const entry of entries) {
			const existing = this._renderSets.get(entry.light);
			const renderSet = existing ?
				ensureShadowRenderSetMatchesConfig(existing, entry.config, entry.options)
			: createShadowRenderSet(entry.config, entry.options);
			this._renderSets.set(entry.light, renderSet);
		}
		return this._renderSets;
	}

	public publish(
		plan: ShadowFramePlan,
		shadowMaps: Map<ShadowCastingLight, ShadowRenderSet>
	): void {
		PUBLISHED_LEGACY_SHADOW_MAPS.set(plan, shadowMaps);
	}

	public getShadowMaps(
		plan: ShadowFramePlan
	): Map<ShadowCastingLight, ShadowRenderSet> {
		return resolveLegacyShadowMaps(plan);
	}
}

/** @internal Resolves compatibility metadata for a backend still in migration. */
export function resolveLegacyShadowMaps(
	plan: ShadowFramePlan | null | undefined
): Map<ShadowCastingLight, ShadowRenderSet> {
	return plan ? PUBLISHED_LEGACY_SHADOW_MAPS.get(plan) ?? new Map() : new Map();
}

/**
 * Creates a minimal plan for direct backend tests during the migration window.
 *
 * @internal Renderer-driven frames must use `ShadowPlanner` instead.
 */
export function createLegacyShadowFramePlan(
	shadowMaps: Map<ShadowCastingLight, ShadowRenderSet>
): ShadowFramePlan {
	const lights = Array.from(shadowMaps, ([light, renderSet]) => ({
		light,
		lightId: light.id,
		definition: Object.freeze({
			id: `legacy:${light.id}`,
			kind: "single" as const,
			enabled: true,
			projection: Object.freeze({ technique: "single" as const }),
			storagePreference: renderSet.storageMode,
			resolution: renderSet.size,
			bias: Object.freeze({
				constant: 0,
				slope: 0,
				normal: 0,
				normalMin: 0,
				texel: 0,
				max: 0,
			}),
			sampling: Object.freeze({
				filterMode: "pcf" as const,
				pcfRadius: 1,
				strength: 1,
				radius: 0,
				samples: 1,
				searchSamples: 1,
			}),
			priority: 0,
			revision: 0,
		}),
		requestedTechnique: "single" as const,
		effectiveTechnique: "single" as const,
		requestedCascadeCount: 1,
		effectiveCascadeCount: 1,
		requestedResolution: renderSet.size,
		effectiveResolution: renderSet.size,
		sampling: Object.freeze({
			filterMode: "pcf" as const,
			pcfRadius: 1,
			strength: 1,
			radius: 0,
			samples: 1,
			searchSamples: 1,
		}),
		filterMode: "pcf" as const,
		storage: renderSet.storageMode,
		priority: 0,
		cost: 0,
		score: 0,
		slices: Object.freeze([]),
	}));
	const jobs = lights.map((light, lightIndex) => ({
		id: `${light.definition.id}:${light.storage}`,
		lightIndex,
		technique: light.storage,
		sliceIndices: Object.freeze([] as number[]),
	}));
	const plan: ShadowFramePlan = Object.freeze({
		revision: 0,
		lights: Object.freeze(lights.map((light) => Object.freeze(light))),
		jobs: Object.freeze(jobs.map((job) => Object.freeze(job))),
		diagnostics: Object.freeze([]),
		hasRasterWork: jobs.length > 0,
		hasTransmissionWork: false,
		hasPagedWork: jobs.some((job) => job.technique === "paged"),
	});
	PUBLISHED_LEGACY_SHADOW_MAPS.set(plan, shadowMaps);
	return plan;
}
