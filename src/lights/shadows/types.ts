import type { Matrix4 } from "../../maths/Matrix4";
import type { IVector3 } from "../../maths/types";
import type { ShadowCastingLight } from "..";
import type { ShadowMapBase } from "./ShadowMapBase";

export type ShadowMapKind = "single" | "cascaded";
export type ShadowFilterMode = "pcf" | "pcss";
export type ShadowSamplingQuality = "low" | "medium" | "high";
export type ShadowBoundLightType = "directional" | "point" | "spot" | "rectArea";

export interface ShadowBiasSettings {
	constant?: number;
	slope?: number;
	normal?: number;
	normalMin?: number;
	texel?: number;
	max?: number;
}

export interface ShadowSamplingSettings {
	quality?: ShadowSamplingQuality;
}

export type ShadowProjectionPreference = "single" | "cascaded";

export interface CascadedShadowMapDefaults {
	directional: number;
	spot: number;
	point: number;
}

export interface ShadowProjectionSnapshot {
	readonly technique: ShadowProjectionPreference;
	readonly cascadeCounts?: Readonly<CascadedShadowMapDefaults>;
	readonly lambda?: number;
	readonly maxDistance?: number;
	readonly blendRatio?: number;
	readonly stabilize?: boolean;
}

/** @internal Immutable authoring input consumed by `ShadowPlanner`. */
export interface ShadowDefinitionSnapshot {
	readonly id: string;
	readonly kind: ShadowMapKind;
	readonly enabled: boolean;
	readonly projection: Readonly<ShadowProjectionSnapshot>;
	readonly resolution: number;
	readonly bias: Readonly<Required<ShadowBiasSettings>>;
	readonly filterMode: ShadowFilterMode;
	readonly sampling: Readonly<Required<ShadowSamplingSettings>>;
	readonly strength: number;
	readonly priority: number;
	readonly revision: number;
}

/** @internal Listener used by `ShadowManager` to observe shared definitions. */
export type ShadowDefinitionListener = (definition: ShadowMapBase) => void;

export interface SceneBounds {
	center: IVector3;
	radius: number;
}

export interface ShadowStrategyCamera {
	type?: string;
	near?: number;
	far?: number;
	fov?: number;
	aspectRatio?: number;
	size?: number;
	left?: number | null;
	right?: number | null;
	bottom?: number | null;
	top?: number | null;
	up?: IVector3;
	position?: IVector3;
	getBounds?: () => { left: number; right: number; bottom: number; top: number };
	getWorldPosition?: (target?: IVector3) => IVector3;
	getWorldDirection?: (localDirection: IVector3, target?: IVector3) => IVector3;
}

/** @internal Mutable stabilization data retained only by `ShadowPlanner`. */
export interface ShadowProjectionSliceState {
	index: number;
	resolution: number;
	stabilizedBoundsRadius: number | null;
	csmStableCenterLightX: number | null;
	csmStableCenterLightY: number | null;
	csmStableLightDir: IVector3 | null;
}

/** @internal Normalized planner-only projection request. */
export interface ShadowProjectionConfig {
	readonly technique: ShadowProjectionPreference;
	readonly resolution: number;
	readonly cascadeCount: number;
	readonly lambda: number;
	readonly maxDistance?: number;
	readonly blendRatio: number;
	readonly stabilize: boolean;
}

export interface ShadowSliceDescriptor {
	view: Matrix4;
	projection: Matrix4;
	lightDir: IVector3;
	splitNear: number;
	splitFar: number;
}

/** @internal Projection builder input owned by `ShadowPlanner`. */
export interface ShadowStrategyBuildContext {
	light: ShadowCastingLight;
	slices: ShadowProjectionSliceState[];
	config: ShadowProjectionConfig;
	sceneBounds: SceneBounds;
	camera?: ShadowStrategyCamera | null;
}
