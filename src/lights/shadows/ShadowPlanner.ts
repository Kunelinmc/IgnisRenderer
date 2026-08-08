import { LightType, type SceneLight, type ShadowCastingLight } from "..";
import {
	type IShadowBackendCapabilities,
	type ShadowDefinitionSnapshot,
	type ShadowFilterMode,
	type ShadowProjectionConfig,
	type ShadowProjectionSliceState,
} from "./types";
import { CascadedShadowMap } from "./CascadedShadowMap";
import type { ShadowManager } from "./ShadowManager";
import type { ShadowMapBase } from "./ShadowMapBase";
import { SingleShadowMap } from "./SingleShadowMap";
import type { IVector3 } from "../../maths/types";
import { Matrix4 } from "../../maths/Matrix4";
import { resolveShadowCasterBounds } from "./ShadowCasterBounds";
import type { ShadowStrategyCamera, SceneBounds } from "./types";
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
	readonly projection: ShadowProjectionConfig;
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

interface ShadowProjectionHistory {
	readonly definitionRevision: number;
	readonly signature: string;
	readonly slices: ShadowProjectionSliceState[];
}

/** @internal Per-renderer history consumed by the static shadow planner. */
export interface ShadowPlannerState {
	revision: number;
	readonly projectionStates: Map<ShadowCastingLight, ShadowProjectionHistory>;
}

/**
 * Resolves all backend-neutral shadow choices into one immutable frame plan.
 *
 * @internal `FrameCoordinator` owns the state passed to the static planner.
 */
export class ShadowPlanner {
	/**
	 * Creates isolated cross-frame state for one renderer coordinator.
	 *
	 * @internal `FrameCoordinator` owns this state for its attached backend.
	 */
	public static createState(): ShadowPlannerState {
		return {
			revision: 0,
			projectionStates: new Map(),
		};
	}

	/**
	 * Resolves one immutable, backend-neutral shadow frame plan.
	 *
	 * @param options Current scene, camera, caster, and backend capability input.
	 * @param state Per-renderer projection history and plan revision state.
	 * @returns The immutable shadow plan consumed by backend runtimes.
	 * @internal `FrameCoordinator` is the owning caller.
	 */
	public static plan(
		options: ShadowPlannerOptions,
		state: ShadowPlannerState,
	): ShadowFramePlan {
		const diagnostics: ShadowDiagnostic[] = [];
		if (!options.enableShadows) {
			state.projectionStates.clear();
			return ShadowPlanner._publish(state, [], [], diagnostics, false, false);
		}

		const candidates = ShadowPlanner._collectCandidates(options, diagnostics);
		const selected = ShadowPlanner._applyBudget(
			candidates,
			options.capabilities,
			diagnostics
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
		const activeLights = new Set<ShadowCastingLight>();

		for (const candidate of selected) {
			activeLights.add(candidate.light);
			const slices = ShadowPlanner._prepareSlices(
				state,
				candidate,
				casterBounds,
				options.camera,
			);
			if (slices.length === 0) {
				diagnostics.push(ShadowPlanner._diagnostic(
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
				effectiveTechnique: candidate.projection.technique === "cascaded" ?
					"cascaded"
				: "single",
				requestedCascadeCount: candidate.requestedCascadeCount,
				effectiveCascadeCount: candidate.cascadeCount,
				requestedResolution: candidate.snapshot.resolution,
				effectiveResolution: candidate.size,
				sampling: candidate.snapshot.sampling,
				fallbackReason: ShadowPlanner._resolveFallbackReason(candidate),
				filterMode: candidate.filterMode,
				storage: candidate.storage,
				pagedSettings: candidate.storage === "paged" ?
					candidate.snapshot.pagedSettings : undefined,
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
			ShadowPlanner._createJobs(preparedLights, options.needsAtlasFallback)
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
			diagnostics.push(ShadowPlanner._diagnostic(
				"transmission-unsupported",
				selected[0],
				`Backend ${options.capabilities.backendKey} does not support shadow transmission.`
			));
		}
		ShadowPlanner._trimProjectionStates(state, activeLights);
		return ShadowPlanner._publish(
			state,
			preparedLights,
			jobs,
			diagnostics,
			hasTransmissionWork,
			true,
		);
	}


	private static _collectCandidates(
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
			if (!ShadowPlanner._supportsLight(light, options.capabilities)) {
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
			const supportsCascaded = ShadowPlanner._supportsCascaded(
				light,
				options.capabilities,
			);
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

			const storage = ShadowPlanner._resolveStorage(
				definition,
				options.capabilities,
				light,
				diagnostics
			);
			const size = definition.size;
			const projection = ShadowPlanner._resolveProjection(
				snapshot,
				size,
				cascadeCount,
				snapshot.projection.technique === "cascaded" && !supportsCascaded
			);
			const cost = definition.estimateCost(light.type, size, cascadeCount);
			candidates.push({
				light,
				definition,
				snapshot,
				score: ShadowPlanner._score(
					light,
					definition.priority,
					options.cameraPosition,
				),
				requestedFilter,
				filterMode,
				storage,
				requestedCascadeCount,
				cascadeCount,
				size,
				cost,
				projection,
			});
		}
		candidates.sort((left, right) =>
			right.definition.priority - left.definition.priority ||
			right.light.intensity - left.light.intensity ||
			right.score - left.score ||
			left.light.id.localeCompare(right.light.id)
		);
		return ShadowPlanner._applyCapabilityLimits(
			candidates,
			options.capabilities,
			diagnostics
		);
	}

	private static _applyCapabilityLimits(
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
				diagnostics.push(ShadowPlanner._diagnostic(
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

	private static _applyBudget(
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
			const degraded = ShadowPlanner._degrade(
				candidate,
				Math.max(0, budget - consumed),
			);
			if (!degraded) {
				diagnostics.push(ShadowPlanner._diagnostic(
					"budget-disabled",
					candidate,
					`Shadow for light ${candidate.light.id} was disabled by the dynamic shadow budget.`
				));
				continue;
			}
			diagnostics.push(ShadowPlanner._diagnostic(
				"budget-degraded",
				degraded,
				`Shadow for light ${candidate.light.id} was degraded to ${degraded.cascadeCount} cascade(s) at ${degraded.size}px.`
			));
			selected.push(degraded);
			consumed += degraded.cost;
		}
		return selected;
	}

	private static _degrade(
		candidate: ShadowPlanCandidate,
		remainingBudget: number
	): ShadowPlanCandidate | null {
		if (remainingBudget <= 0) return null;
		for (let cascades = candidate.cascadeCount; cascades >= 1; cascades--) {
			const degraded = ShadowPlanner._withResolution(
				candidate,
				candidate.size,
				cascades,
			);
			if (degraded.cost <= remainingBudget) return degraded;
		}
		for (let size = Math.floor(candidate.size / 2); size >= 128; size /= 2) {
			const degraded = ShadowPlanner._withResolution(
				candidate,
				Math.floor(size),
				1,
			);
			if (degraded.cost <= remainingBudget) return degraded;
		}
		return null;
	}

	private static _withResolution(
		candidate: ShadowPlanCandidate,
		size: number,
		cascadeCount: number
	): ShadowPlanCandidate {
		const projection = ShadowPlanner._resolveProjection(
			candidate.snapshot,
			size,
			cascadeCount,
			candidate.snapshot.projection.technique === "cascaded" &&
				candidate.projection.technique === "single"
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
			projection,
		};
	}

	private static _resolveProjection(
		snapshot: ShadowDefinitionSnapshot,
		size: number,
		cascadeCount: number,
		forceSingle: boolean
	): ShadowProjectionConfig {
		const requested = snapshot.projection;
		return {
			technique: forceSingle ? "single" : requested.technique,
			resolution: Math.max(1, Math.floor(size)),
			cascadeCount: forceSingle ? 1 : Math.max(1, Math.floor(cascadeCount)),
			lambda: Math.max(0, Math.min(1, requested.lambda ?? 0.65)),
			maxDistance: requested.maxDistance,
			blendRatio: Math.max(0, Math.min(1, requested.blendRatio ?? 0.1)),
			stabilize: requested.stabilize !== false,
		};
	}

	private static _prepareSlices(
		state: ShadowPlannerState,
		candidate: ShadowPlanCandidate,
		sceneBounds: SceneBounds,
		camera: ShadowStrategyCamera,
	): PreparedShadowSlice[] {
		const sliceCount = resolveProjectionSliceCount(
			candidate.light,
			candidate.projection.cascadeCount,
			candidate.projection.technique,
		);
		const signature = [
			candidate.snapshot.revision,
			candidate.projection.technique,
			candidate.projection.resolution,
			sliceCount,
		].join(":");
		let projectionHistory = state.projectionStates.get(candidate.light);
		if (!projectionHistory || projectionHistory.signature !== signature) {
			projectionHistory = {
				definitionRevision: candidate.snapshot.revision,
				signature,
				slices: Array.from({ length: sliceCount }, (_, index) => ({
					index,
					resolution: resolveProjectionSliceResolution(candidate.projection),
					stabilizedBoundsRadius: null,
					csmStableCenterLightX: null,
					csmStableCenterLightY: null,
					csmStableLightDir: null,
				})),
			};
			state.projectionStates.set(candidate.light, projectionHistory);
		}
		const descriptors = candidate.projection.technique === "cascaded" ?
			CascadedShadowMap.buildSlices({
				light: candidate.light,
				slices: projectionHistory.slices,
				config: candidate.projection,
				sceneBounds,
				camera,
			}) :
			SingleShadowMap.buildSlices({
				light: candidate.light,
				slices: projectionHistory.slices,
				config: candidate.projection,
				sceneBounds,
				camera,
			});
		return descriptors.map((slice, index) => Object.freeze({
			index,
			resolution: projectionHistory.slices[index]?.resolution ??
				candidate.projection.resolution,
			view: freezeMatrix(slice.view.clone()),
			projection: freezeMatrix(slice.projection.clone()),
			viewProjection: freezeMatrix(Matrix4.multiply(slice.projection, slice.view)),
			lightDirection: Object.freeze({ ...slice.lightDir }),
			splitNear: slice.splitNear,
			splitFar: slice.splitFar,
		}));
	}

	private static _trimProjectionStates(
		state: ShadowPlannerState,
		activeLights: ReadonlySet<ShadowCastingLight>,
	): void {
		for (const light of state.projectionStates.keys()) {
			if (!activeLights.has(light)) state.projectionStates.delete(light);
		}
	}

	private static _createJobs(
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

	private static _publish(
		state: ShadowPlannerState,
		lights: PreparedShadowLight[],
		jobs: ShadowRenderJob[],
		diagnostics: ShadowDiagnostic[],
		hasTransmissionWork: boolean,
		_publishLegacy: boolean,
	): ShadowFramePlan {
		const plan: ShadowFramePlan = Object.freeze({
			revision: ++state.revision,
			lights: Object.freeze(lights),
			jobs: Object.freeze(jobs),
			diagnostics: Object.freeze(diagnostics.map((item) => Object.freeze(item))),
			hasRasterWork: jobs.length > 0,
			hasTransmissionWork,
			hasPagedWork: jobs.some((job) => job.technique === "paged"),
		});
		return plan;
	}

	private static _supportsLight(
		light: ShadowCastingLight,
		capabilities: IShadowBackendCapabilities
	): boolean {
		const key = lightTypeKey(light);
		const explicit = capabilities.lightTypes?.[key];
		if (explicit) return explicit.projections.length > 0 && explicit.maxLights > 0;
		if (light.type === LightType.Point) return capabilities.supportsPointCSM;
		return light.type !== LightType.RectArea;
	}

	private static _supportsCascaded(
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

	private static _resolveStorage(
		definition: ShadowMapBase,
		capabilities: IShadowBackendCapabilities,
		light: ShadowCastingLight,
		diagnostics: ShadowDiagnostic[]
	): "atlas" | "paged" {
		if (definition.kind !== "paged-shadow") return "atlas";
		const explicit = capabilities.lightTypes?.[lightTypeKey(light)];
		const layout = definition.snapshot().pagedSettings;
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

	private static _resolveFallbackReason(
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
				candidate.projection.technique === "single")
		) {
			return candidate.size !== candidate.snapshot.resolution ?
				"budget-degraded"
			: "projection-fallback";
		}
		return undefined;
	}

	private static _score(
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

	private static _diagnostic(
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

function resolveProjectionSliceCount(
	light: ShadowCastingLight,
	cascadeCount: number,
	technique: ShadowProjectionConfig["technique"],
): number {
	if (technique !== "cascaded") return 1;
	return light.type === LightType.Point ? Math.max(1, cascadeCount) * 6 :
		Math.max(1, cascadeCount);
}

function resolveProjectionSliceResolution(config: ShadowProjectionConfig): number {
	return config.technique === "cascaded" ?
		Math.max(1, Math.floor(config.resolution / 2)) : config.resolution;
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
