import type {
	ShadowBoundLightType,
	ShadowFilterMode,
	ShadowProjectionPreference,
	ShadowStoragePreference,
} from "./types";

interface ShadowPlannerLightPolicy {
	readonly projections: readonly ShadowProjectionPreference[];
	readonly storage: readonly ShadowStoragePreference[];
	readonly maxLights: number;
	readonly maxCascadedLights: number;
}

/** @internal Fixed shadow-planning policy owned by `ShadowPlanner`. */
export interface ShadowPlannerBackendPolicy {
	readonly filterModes: Readonly<
		Record<"atlas" | "paged", readonly ShadowFilterMode[]>
	>;
	readonly lightTypes: Partial<Record<ShadowBoundLightType, ShadowPlannerLightPolicy>>;
	readonly supportsTransmission: boolean;
	readonly maxDynamicShadowCost: number;
	readonly supportsPagedShadowRendering?: boolean;
	readonly maxPagedShadowPages?: number;
	readonly pagedShadowPageSizeRange?: readonly [number, number];
}

const SOFTWARE_POLICY: ShadowPlannerBackendPolicy = {
	filterModes: { atlas: ["pcf", "pcss"], paged: ["pcf"] },
	lightTypes: {
		directional: {
			projections: ["single", "cascaded"],
			storage: ["atlas"],
			maxLights: 4,
			maxCascadedLights: 1,
		},
		spot: {
			projections: ["single", "cascaded"],
			storage: ["atlas"],
			maxLights: 8,
			maxCascadedLights: 8,
		},
		point: {
			projections: ["single", "cascaded"],
			storage: ["atlas"],
			maxLights: 16,
			maxCascadedLights: 16,
		},
	},
	supportsTransmission: true,
	maxDynamicShadowCost: 20,
};

const WEBGL_POLICY: ShadowPlannerBackendPolicy = {
	filterModes: { atlas: ["pcf", "pcss"], paged: ["pcf"] },
	lightTypes: {
		directional: {
			projections: ["single", "cascaded"],
			storage: ["atlas"],
			maxLights: 4,
			maxCascadedLights: 1,
		},
		spot: {
			projections: ["single"],
			storage: ["atlas"],
			maxLights: 8,
			maxCascadedLights: 0,
		},
		point: {
			projections: [],
			storage: [],
			maxLights: 0,
			maxCascadedLights: 0,
		},
	},
	supportsTransmission: true,
	maxDynamicShadowCost: 24,
};

const WEBGPU_POLICY: ShadowPlannerBackendPolicy = {
	filterModes: { atlas: ["pcf", "pcss"], paged: ["pcf"] },
	lightTypes: {
		directional: {
			projections: ["single", "cascaded"],
			storage: ["atlas", "paged"],
			maxLights: 4,
			maxCascadedLights: 1,
		},
		spot: {
			projections: ["single"],
			storage: ["atlas"],
			maxLights: 8,
			maxCascadedLights: 0,
		},
		point: {
			projections: [],
			storage: [],
			maxLights: 0,
			maxCascadedLights: 0,
		},
	},
	supportsTransmission: true,
	maxDynamicShadowCost: 48,
	supportsPagedShadowRendering: true,
	maxPagedShadowPages: 2048,
	pagedShadowPageSizeRange: [64, 256],
};

// Unknown backends receive only behavior shared by every built-in backend.
const CROSS_BACKEND_POLICY: ShadowPlannerBackendPolicy = {
	filterModes: { atlas: ["pcf"], paged: ["pcf"] },
	lightTypes: {
		directional: {
			projections: ["single", "cascaded"],
			storage: ["atlas"],
			maxLights: 4,
			maxCascadedLights: 1,
		},
		spot: {
			projections: ["single"],
			storage: ["atlas"],
			maxLights: 8,
			maxCascadedLights: 0,
		},
		point: {
			projections: [],
			storage: [],
			maxLights: 0,
			maxCascadedLights: 0,
		},
	},
	supportsTransmission: true,
	maxDynamicShadowCost: 20,
};

/** @internal Selects the fixed planner-owned policy for a backend identifier. */
export function resolveShadowPlannerBackendPolicy(
	backendKey: string,
): ShadowPlannerBackendPolicy {
	switch (backendKey) {
		case "software":
			return SOFTWARE_POLICY;
		case "webgl":
			return WEBGL_POLICY;
		case "webgpu":
			return WEBGPU_POLICY;
		default:
			return CROSS_BACKEND_POLICY;
	}
}
