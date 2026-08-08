import { LightType, type SceneLight, type ShadowCastingLight } from "../../lights";
import {
	type IShadowBackendCapabilities,
	type ShadowDefinitionSnapshot,
	type ShadowFilterMode,
	type ShadowMapBase,
} from "../../lights/shadows";
import type {
	PagedShadowLayoutMetadata,
	ShadowConfig,
	ShadowRenderSet,
} from "../../lights/shadows/ShadowMapping";
import type { ShadowManager } from "../../lights/shadows/ShadowManager";
import type { IVector3 } from "../../maths/types";
import {
	resolveShadowCasterBounds,
	updateShadowMapMetadata,
} from "../ShadowMetadata";
import type { ShadowStrategyCamera, SceneBounds } from "../../lights/shadows/types";
import {
	LegacyShadowPlanAdapter,
	type LegacyShadowPlanEntry,
} from "./LegacyShadowPlanAdapter";
import type {
	PreparedShadowLight,
	PreparedShadowSlice,
	ShadowCasterIntent,
	ShadowDiagnostic,
	ShadowFramePlan,
	ShadowRenderJob,
} from "./ShadowFramePlan";

interface ShadowPlanCandidate {
	readonly light: ShadowCastingLight;
	readonly definition: ShadowMapBase;
	readonly snapshot: ShadowDefinitionSnapshot;
	readonly score: number;
	readonly requestedFilter: ShadowFilterMode;
	readonly filterMode: ShadowFilterMode;
	readonly storage: "atlas" | "paged";
	readonly requestedCascadeCount: number;
	readonly cascadeCount: number;
	readonly size: number;
	readonly cost: number;
	readonly config: ShadowConfig;
	readonly adapterEntry: LegacyShadowPlanEntry;
}

export interface ShadowPlannerOptions {
	readonly manager: ShadowManager;
	readonly lights: readonly SceneLight[];
	readonly capabilities: IShadowBackendCapabilities;
	readonly camera: ShadowStrategyCamera;
	readonly cameraPosition: IVector3 | null;
	readonly sceneBounds: SceneBounds;
	readonly casterIntent: ShadowCasterIntent;
	readonly enableShadows: boolean;
	readonly hasTransmissionCasters: boolean;
	readonly needsAtlasFallback: boolean;
}

/**
 * Resolves all backend-neutral shadow choices into one immutable frame plan.
 *
 * @internal `FrameCoordinator` owns one planner per attached backend.
 */
export class ShadowPlanner {
	private readonly _legacyAdapter = new LegacyShadowPlanAdapter();
	private _revision = 0;

	public plan(options: ShadowPlannerOptions): ShadowFramePlan {
		const diagnostics: ShadowDiagnostic[] = [];
		if (!options.enableShadows) {
			return this._publish([], [], diagnostics, false, false);
		}

		const candidates = this._collectCandidates(options, diagnostics);
		const selected = this._applyBudget(
			candidates,
			options.capabilities,
			diagnostics
		);
		const shadowMaps = this._legacyAdapter.reconcile(
			selected.map((candidate) => candidate.adapterEntry)
		);
		// Particle positions are not available until simulation. Keep the full
		// scene bounds in that case instead of tightening the projection around
		// mesh packets and clipping particle casters.
		const meshCasterBounds = resolveShadowCasterBounds(
			options.casterIntent.meshPackets.slice(),
			options.sceneBounds,
			options.camera
		);
		const casterBounds = options.casterIntent.particleBounds ?
			mergeSceneBounds(meshCasterBounds, options.casterIntent.particleBounds)
		: options.casterIntent.hasParticleCasters ?
			options.sceneBounds
		: meshCasterBounds;
		const preparedLights: PreparedShadowLight[] = [];

		for (const candidate of selected) {
			const renderSet = shadowMaps.get(candidate.light);
			if (!renderSet) continue;
			updateShadowMapMetadata(renderSet, candidate.light, casterBounds, {
				camera: options.camera,
				requestedConfig: candidate.config,
				skipCapabilityFallback: true,
			});
			const slices = this._prepareSlices(renderSet);
			if (slices.length === 0) {
				diagnostics.push(this._diagnostic(
					"invalid-projection",
					candidate,
					`Shadow projection for light ${candidate.light.id} produced no valid slices.`
				));
				continue;
			}
			preparedLights.push(Object.freeze({
				light: candidate.light,
				lightId: candidate.light.id,
				definition: candidate.snapshot,
				requestedTechnique: candidate.snapshot.projection.technique,
				effectiveTechnique: candidate.config.strategy === "csm" ?
					"cascaded"
				: "single",
				requestedCascadeCount: candidate.requestedCascadeCount,
				effectiveCascadeCount: candidate.cascadeCount,
				requestedResolution: candidate.snapshot.resolution,
				effectiveResolution: candidate.size,
				sampling: candidate.snapshot.sampling,
				fallbackReason: this._resolveFallbackReason(candidate),
				filterMode: candidate.filterMode,
				storage: candidate.storage,
				priority: candidate.snapshot.priority,
				cost: candidate.cost,
				score: candidate.score,
				slices: Object.freeze(slices),
			}));
		}

		const hasCasters =
			options.casterIntent.meshPackets.length > 0 ||
			options.casterIntent.hasParticleCasters ||
			options.hasTransmissionCasters;
		const jobs = hasCasters ?
			this._createJobs(preparedLights, options.needsAtlasFallback)
		: [];
		const hasTransmissionWork =
			options.hasTransmissionCasters &&
			options.capabilities.supportsTransmission !== false &&
			jobs.length > 0;
		if (
			options.hasTransmissionCasters &&
			options.capabilities.supportsTransmission === false &&
			selected[0]
		) {
			diagnostics.push(this._diagnostic(
				"transmission-unsupported",
				selected[0],
				`Backend ${options.capabilities.backendKey} does not support shadow transmission.`
			));
		}
		return this._publish(
			preparedLights,
			jobs,
			diagnostics,
			hasTransmissionWork,
			true,
			jobs.length > 0 ? shadowMaps : new Map()
		);
	}

	public get legacyAdapter(): LegacyShadowPlanAdapter {
		return this._legacyAdapter;
	}

	private _collectCandidates(
		options: ShadowPlannerOptions,
		diagnostics: ShadowDiagnostic[]
	): ShadowPlanCandidate[] {
		const candidates: ShadowPlanCandidate[] = [];
		for (const light of options.lights) {
			if (!isShadowCastingLight(light)) continue;
			const definition = options.manager.getBoundShadowMap(light);
			if (!definition || !definition.enabled) continue;
			const snapshot = definition.snapshot();
			if (!isBuiltinKind(snapshot.kind)) {
				diagnostics.push({
					code: "custom-kind-deprecated",
					severity: "warning",
					lightId: light.id,
					definitionId: snapshot.id,
					message: `Custom shadow kind ${snapshot.kind} is deprecated; planner converted it to a built-in descriptor.`,
				});
			}
			if (!this._supportsLight(light, options.capabilities)) {
				diagnostics.push({
					code: "unsupported-light-type",
					severity: "warning",
					lightId: light.id,
					definitionId: snapshot.id,
					message: `Backend ${options.capabilities.backendKey} does not support ${light.type} shadows.`,
				});
				continue;
			}

			const requestedCascadeCount = Math.max(
				1,
				definition.resolveCascadeCount(light.type)
			);
			const supportsCascaded = this._supportsCascaded(light, options.capabilities);
			const cascadeCount = supportsCascaded ? requestedCascadeCount : 1;
			if (snapshot.projection.technique === "cascaded" && !supportsCascaded) {
				diagnostics.push({
					code: "projection-fallback",
					severity: "warning",
					lightId: light.id,
					definitionId: snapshot.id,
					message: `Backend ${options.capabilities.backendKey} resolved cascaded shadows to a single projection for light ${light.id}.`,
				});
			}

			const requestedFilter = definition.filterMode;
			const filterMode = options.capabilities.supportsFilterModes.includes(
				requestedFilter
			) ? requestedFilter : "pcf";
			if (filterMode !== requestedFilter) {
				diagnostics.push({
					code: "filter-fallback",
					severity: "warning",
					lightId: light.id,
					definitionId: snapshot.id,
					message: `Backend ${options.capabilities.backendKey} resolved ${requestedFilter} to ${filterMode} for light ${light.id}.`,
				});
			}

			const storage = this._resolveStorage(
				definition,
				options.capabilities,
				light,
				diagnostics
			);
			const size = definition.size;
			const config = this._resolveConfig(
				definition,
				light,
				size,
				cascadeCount,
				snapshot.projection.technique === "cascaded" && !supportsCascaded
			);
			const cost = definition.estimateCost(light.type, size, cascadeCount);
			const adapterEntry = this._createAdapterEntry(
				light,
				definition,
				storage,
				config
			);
			candidates.push({
				light,
				definition,
				snapshot,
				score: this._score(light, definition.priority, options.cameraPosition),
				requestedFilter,
				filterMode,
				storage,
				requestedCascadeCount,
				cascadeCount,
				size,
				cost,
				config,
				adapterEntry,
			});
		}
		candidates.sort((left, right) =>
			right.definition.priority - left.definition.priority ||
			right.light.intensity - left.light.intensity ||
			right.score - left.score ||
			left.light.id.localeCompare(right.light.id)
		);
		return this._applyCapabilityLimits(
			candidates,
			options.capabilities,
			diagnostics
		);
	}

	private _applyCapabilityLimits(
		candidates: readonly ShadowPlanCandidate[],
		capabilities: IShadowBackendCapabilities,
		diagnostics: ShadowDiagnostic[]
	): ShadowPlanCandidate[] {
		const counts = new Map<string, number>();
		const cascadedCounts = new Map<string, number>();
		const accepted: ShadowPlanCandidate[] = [];
		for (const candidate of candidates) {
			const key = candidate.definition.resolveBoundLightType(candidate.light.type);
			const limits = capabilities.lightTypes?.[key];
			if (!limits) {
				accepted.push(candidate);
				continue;
			}
			const count = counts.get(key) ?? 0;
			const cascadedCount = cascadedCounts.get(key) ?? 0;
			const exceedsLightLimit = count >= limits.maxLights;
			const exceedsCascadedLimit =
				candidate.cascadeCount > 1 &&
				cascadedCount >= limits.maxCascadedLights;
			if (exceedsLightLimit || exceedsCascadedLimit) {
				diagnostics.push(this._diagnostic(
					"capability-limit",
					candidate,
					`Backend ${capabilities.backendKey} shadow count limit disabled light ${candidate.light.id}.`
				));
				continue;
			}
			counts.set(key, count + 1);
			if (candidate.cascadeCount > 1) {
				cascadedCounts.set(key, cascadedCount + 1);
			}
			accepted.push(candidate);
		}
		return accepted;
	}

	private _applyBudget(
		candidates: readonly ShadowPlanCandidate[],
		capabilities: IShadowBackendCapabilities,
		diagnostics: ShadowDiagnostic[]
	): ShadowPlanCandidate[] {
		const budget =
			typeof capabilities.maxDynamicShadowCost === "number" &&
			capabilities.maxDynamicShadowCost > 0 ?
				capabilities.maxDynamicShadowCost
			: Number.POSITIVE_INFINITY;
		let consumed = 0;
		const selected: ShadowPlanCandidate[] = [];
		for (const candidate of candidates) {
			if (consumed + candidate.cost <= budget) {
				selected.push(candidate);
				consumed += candidate.cost;
				continue;
			}
			const degraded = this._degrade(candidate, Math.max(0, budget - consumed));
			if (!degraded) {
				diagnostics.push(this._diagnostic(
					"budget-disabled",
					candidate,
					`Shadow for light ${candidate.light.id} was disabled by the dynamic shadow budget.`
				));
				continue;
			}
			diagnostics.push(this._diagnostic(
				"budget-degraded",
				degraded,
				`Shadow for light ${candidate.light.id} was degraded to ${degraded.cascadeCount} cascade(s) at ${degraded.size}px.`
			));
			selected.push(degraded);
			consumed += degraded.cost;
		}
		return selected;
	}

	private _degrade(
		candidate: ShadowPlanCandidate,
		remainingBudget: number
	): ShadowPlanCandidate | null {
		if (remainingBudget <= 0) return null;
		for (let cascades = candidate.cascadeCount; cascades >= 1; cascades--) {
			const degraded = this._withResolution(candidate, candidate.size, cascades);
			if (degraded.cost <= remainingBudget) return degraded;
		}
		for (let size = Math.floor(candidate.size / 2); size >= 128; size /= 2) {
			const degraded = this._withResolution(candidate, Math.floor(size), 1);
			if (degraded.cost <= remainingBudget) return degraded;
		}
		return null;
	}

	private _withResolution(
		candidate: ShadowPlanCandidate,
		size: number,
		cascadeCount: number
	): ShadowPlanCandidate {
		const config = this._resolveConfig(
			candidate.definition,
			candidate.light,
			size,
			cascadeCount,
			candidate.snapshot.projection.technique === "cascaded" &&
				candidate.config.strategy === "single-map"
		);
		return {
			...candidate,
			size,
			cascadeCount,
			cost: candidate.definition.estimateCost(
				candidate.light.type,
				size,
				cascadeCount
			),
			config,
			adapterEntry: this._createAdapterEntry(
				candidate.light,
				candidate.definition,
				candidate.storage,
				config
			),
		};
	}

	private _resolveConfig(
		definition: ShadowMapBase,
		light: ShadowCastingLight,
		size: number,
		cascadeCount: number,
		forceSingle: boolean
	): ShadowConfig {
		const requested = definition.toLegacyShadowConfig(light.type, {
			size,
			cascadeCount,
		});
		if (!forceSingle) return requested;
		return {
			strategy: "single-map",
			size,
			priority: definition.priority,
			params: requested.params,
		};
	}

	private _createAdapterEntry(
		light: ShadowCastingLight,
		definition: ShadowMapBase,
		storage: "atlas" | "paged",
		config: ShadowConfig
	): LegacyShadowPlanEntry {
		return {
			light,
			config,
			options: storage === "paged" ? {
				storageMode: "paged",
				paged: resolvePagedLayoutMetadata(definition),
			} : { storageMode: "atlas" },
		};
	}

	private _prepareSlices(renderSet: {
		readonly slices: readonly {
			readonly index: number;
			readonly splitNear: number;
			readonly splitFar: number;
			readonly shadowMap: {
				readonly size: number;
				readonly viewMatrix: PreparedShadowSlice["view"] | null;
				readonly projectionMatrix: PreparedShadowSlice["projection"] | null;
				readonly viewProjectionMatrix: PreparedShadowSlice["viewProjection"] | null;
				readonly latestLightDir: IVector3;
			};
		}[];
	}): PreparedShadowSlice[] {
		const slices: PreparedShadowSlice[] = [];
		for (const slice of renderSet.slices) {
			const shadowMap = slice.shadowMap;
			if (!shadowMap.viewMatrix || !shadowMap.projectionMatrix ||
				!shadowMap.viewProjectionMatrix) continue;
			slices.push(Object.freeze({
				index: slice.index,
				resolution: shadowMap.size,
				view: freezeMatrix(shadowMap.viewMatrix.clone()),
				projection: freezeMatrix(shadowMap.projectionMatrix.clone()),
				viewProjection: freezeMatrix(shadowMap.viewProjectionMatrix.clone()),
				lightDirection: Object.freeze({ ...shadowMap.latestLightDir }),
				splitNear: slice.splitNear,
				splitFar: slice.splitFar,
			}));
		}
		return slices;
	}

	private _createJobs(
		lights: readonly PreparedShadowLight[],
		needsAtlasFallback: boolean
	): ShadowRenderJob[] {
		const jobs: ShadowRenderJob[] = [];
		for (let lightIndex = 0; lightIndex < lights.length; lightIndex++) {
			const light = lights[lightIndex];
			const sliceIndices = Object.freeze(light.slices.map((slice) => slice.index));
			jobs.push(Object.freeze({
				id: `${light.definition.id}:${light.storage}`,
				lightIndex,
				technique: light.storage,
				sliceIndices,
			}));
			if (light.storage === "paged" && needsAtlasFallback) {
				jobs.push(Object.freeze({
					id: `${light.definition.id}:atlas-fallback`,
					lightIndex,
					technique: "atlas-fallback",
					sliceIndices,
				}));
			}
		}
		return jobs;
	}

	private _publish(
		lights: PreparedShadowLight[],
		jobs: ShadowRenderJob[],
		diagnostics: ShadowDiagnostic[],
		hasTransmissionWork: boolean,
		publishLegacy: boolean,
		shadowMaps = new Map<ShadowCastingLight, ShadowRenderSet>()
	): ShadowFramePlan {
		const plan: ShadowFramePlan = Object.freeze({
			revision: ++this._revision,
			lights: Object.freeze(lights),
			jobs: Object.freeze(jobs),
			diagnostics: Object.freeze(diagnostics.map((item) => Object.freeze(item))),
			hasRasterWork: jobs.length > 0,
			hasTransmissionWork,
			hasPagedWork: jobs.some((job) => job.technique === "paged"),
		});
		if (publishLegacy) {
			this._legacyAdapter.publish(
				plan,
				shadowMaps
			);
		}
		return plan;
	}

	private _supportsLight(
		light: ShadowCastingLight,
		capabilities: IShadowBackendCapabilities
	): boolean {
		const key = lightTypeKey(light);
		const explicit = capabilities.lightTypes?.[key];
		if (explicit) return explicit.projections.length > 0 && explicit.maxLights > 0;
		if (light.type === LightType.Point) return capabilities.supportsPointCSM;
		return light.type !== LightType.RectArea;
	}

	private _supportsCascaded(
		light: ShadowCastingLight,
		capabilities: IShadowBackendCapabilities
	): boolean {
		const explicit = capabilities.lightTypes?.[lightTypeKey(light)];
		if (explicit) return explicit.projections.includes("cascaded");
		if (light.type === LightType.Directional) return capabilities.supportsDirectionalCSM;
		if (light.type === LightType.Spot) return capabilities.supportsSpotCSM;
		if (light.type === LightType.Point) return capabilities.supportsPointCSM;
		return false;
	}

	private _resolveStorage(
		definition: ShadowMapBase,
		capabilities: IShadowBackendCapabilities,
		light: ShadowCastingLight,
		diagnostics: ShadowDiagnostic[]
	): "atlas" | "paged" {
		if (definition.kind !== "paged-shadow") return "atlas";
		const explicit = capabilities.lightTypes?.[lightTypeKey(light)];
		const layout = resolvePagedLayoutMetadata(definition);
		const pageSizeRange = capabilities.pagedShadowPageSizeRange;
		const hasCompletePagedSupport =
			light.type === LightType.Directional &&
			capabilities.supportsPagedShadowRendering === true &&
			(explicit?.storage.includes("paged") ?? true) &&
			!!layout &&
			typeof capabilities.maxPagedShadowPages === "number" &&
			capabilities.maxPagedShadowPages > 0 &&
			Array.isArray(pageSizeRange) &&
			pageSizeRange.length === 2 &&
			pageSizeRange[0] > 0 &&
			pageSizeRange[1] >= pageSizeRange[0] &&
			layout.physicalPageCount <= capabilities.maxPagedShadowPages &&
			layout.pageSize >= pageSizeRange[0] &&
			layout.pageSize <= pageSizeRange[1];
		if (hasCompletePagedSupport) return "paged";
		diagnostics.push({
			code: "storage-fallback",
			severity: "warning",
			lightId: light.id,
			definitionId: definition.id,
			message: `Backend ${capabilities.backendKey} resolved paged shadows to atlas storage for light ${light.id}.`,
		});
		return "atlas";
	}

	private _resolveFallbackReason(
		candidate: ShadowPlanCandidate
	): ShadowDiagnostic["code"] | undefined {
		if (candidate.filterMode !== candidate.requestedFilter) return "filter-fallback";
		if (candidate.storage !== candidate.snapshot.storagePreference) {
			return "storage-fallback";
		}
		if (
			candidate.cascadeCount !== candidate.requestedCascadeCount ||
			candidate.size !== candidate.snapshot.resolution ||
			(candidate.snapshot.projection.technique === "cascaded" &&
				candidate.config.strategy === "single-map")
		) {
			return candidate.size !== candidate.snapshot.resolution ?
				"budget-degraded"
			: "projection-fallback";
		}
		return undefined;
	}

	private _score(
		light: ShadowCastingLight,
		priority: number,
		cameraPosition: IVector3 | null
	): number {
		let score = priority * 1000 + light.intensity * 10;
		if (!cameraPosition || light.type === LightType.Directional) return score;
		const position = light.getWorldPosition();
		const dx = position.x - cameraPosition.x;
		const dy = position.y - cameraPosition.y;
		const dz = position.z - cameraPosition.z;
		score += 1 / Math.max(1, Math.sqrt(dx * dx + dy * dy + dz * dz));
		return score;
	}

	private _diagnostic(
		code: ShadowDiagnostic["code"],
		candidate: ShadowPlanCandidate,
		message: string
	): ShadowDiagnostic {
		return {
			code,
			severity: "warning",
			lightId: candidate.light.id,
			definitionId: candidate.snapshot.id,
			message,
		};
	}
}

function isShadowCastingLight(light: SceneLight): light is ShadowCastingLight {
	return light.type === LightType.Directional || light.type === LightType.Point ||
		light.type === LightType.Spot || light.type === LightType.RectArea;
}

function isBuiltinKind(kind: string): boolean {
	return kind === "single" || kind === "variance" || kind === "cascaded" ||
		kind === "paged-shadow";
}

function lightTypeKey(
	light: ShadowCastingLight
): "directional" | "point" | "spot" | "rectArea" {
	if (light.type === LightType.Directional) return "directional";
	if (light.type === LightType.Point) return "point";
	if (light.type === LightType.Spot) return "spot";
	return "rectArea";
}

function resolvePagedLayoutMetadata(
	definition: ShadowMapBase
): PagedShadowLayoutMetadata | undefined {
	const resolver = (definition as {
		toLayoutMetadata?: () => PagedShadowLayoutMetadata;
	}).toLayoutMetadata;
	return typeof resolver === "function" ? resolver.call(definition) : undefined;
}

function freezeMatrix(matrix: import("../../maths/Matrix4").Matrix4) {
	matrix.elements = matrix.elements.map((row) =>
		Object.freeze([...row]) as unknown as number[]
	);
	Object.freeze(matrix.elements);
	return Object.freeze(matrix);
}

function mergeSceneBounds(
	left: SceneBounds,
	right: Readonly<{ center: IVector3; radius: number }>
): SceneBounds {
	const leftRadius = Math.max(0, left.radius);
	const rightRadius = Math.max(0, right.radius);
	const minX = Math.min(left.center.x - leftRadius, right.center.x - rightRadius);
	const minY = Math.min(left.center.y - leftRadius, right.center.y - rightRadius);
	const minZ = Math.min(left.center.z - leftRadius, right.center.z - rightRadius);
	const maxX = Math.max(left.center.x + leftRadius, right.center.x + rightRadius);
	const maxY = Math.max(left.center.y + leftRadius, right.center.y + rightRadius);
	const maxZ = Math.max(left.center.z + leftRadius, right.center.z + rightRadius);
	const center = {
		x: (minX + maxX) * 0.5,
		y: (minY + maxY) * 0.5,
		z: (minZ + maxZ) * 0.5,
	};
	return {
		center,
		radius: Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) * 0.5,
	};
}
