import type {
	ShadowBoundLightType,
	ShadowFilterMode,
	ShadowProjectionPreference,
} from "./types";

interface ShadowPlannerLightPolicy {
	readonly projections: readonly ShadowProjectionPreference[];
	readonly maxLights: number;
	readonly maxCascadedLights: number;
}

/** @internal Fixed shadow-planning policy owned by `ShadowPlanner`. */
export interface ShadowPlannerBackendPolicy {
	readonly filterModes: readonly ShadowFilterMode[];
	readonly lightTypes: Partial<Record<ShadowBoundLightType, ShadowPlannerLightPolicy>>;
	readonly supportsTransmission: boolean;
	readonly maxDynamicShadowCost: number;
}

const SOFTWARE_POLICY: ShadowPlannerBackendPolicy = {
	filterModes: ["pcf", "pcss"],
	lightTypes: {
		directional: {
			projections: ["single", "cascaded"],
			maxLights: 4,
			maxCascadedLights: 1,
		},
		spot: {
			projections: ["single", "cascaded"],
			maxLights: 8,
			maxCascadedLights: 8,
		},
		point: {
			projections: ["single", "cascaded"],
			maxLights: 16,
			maxCascadedLights: 16,
		},
	},
	supportsTransmission: true,
	maxDynamicShadowCost: 20,
};

const WEBGL_POLICY: ShadowPlannerBackendPolicy = {
	filterModes: ["pcf", "pcss"],
	lightTypes: {
		directional: {
			projections: ["single", "cascaded"],
			maxLights: 4,
			maxCascadedLights: 1,
		},
		spot: {
			projections: ["single"],
			maxLights: 8,
			maxCascadedLights: 0,
		},
		point: {
			projections: [],
			maxLights: 0,
			maxCascadedLights: 0,
		},
	},
	supportsTransmission: true,
	maxDynamicShadowCost: 24,
};

const WEBGPU_POLICY: ShadowPlannerBackendPolicy = {
	filterModes: ["pcf", "pcss"],
	lightTypes: {
		directional: {
			projections: ["single", "cascaded"],
			maxLights: 4,
			maxCascadedLights: 1,
		},
		spot: {
			projections: ["single"],
			maxLights: 8,
			maxCascadedLights: 0,
		},
		point: {
			projections: [],
			maxLights: 0,
			maxCascadedLights: 0,
		},
	},
	supportsTransmission: true,
	maxDynamicShadowCost: 48,
};

// Unknown backends receive only behavior shared by every built-in backend.
const CROSS_BACKEND_POLICY: ShadowPlannerBackendPolicy = {
	filterModes: ["pcf"],
	lightTypes: {
		directional: {
			projections: ["single", "cascaded"],
			maxLights: 4,
			maxCascadedLights: 1,
		},
		spot: {
			projections: ["single"],
			maxLights: 8,
			maxCascadedLights: 0,
		},
		point: {
			projections: [],
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
