import type { Matrix4 } from "../../maths/Matrix4";
import type { IVector3 } from "../../maths/types";
import type { ShadowCastingLight } from "..";
import type { ShadowMapBase } from "./ShadowMapBase";
import type {
	ShadowConfig,
	ShadowRenderSet,
	ShadowStrategyType,
} from "./ShadowMapping";

export type BuiltinShadowMapKind = "single" | "variance" | "cascaded" | "paged-shadow";
export type ShadowMapKind = BuiltinShadowMapKind | (string & {});
export type ShadowFilterMode = "pcf" | "vsm";
export type ShadowBoundLightType =
	| "directional"
	| "point"
	| "spot"
	| "rectArea";

export interface ShadowBiasSettings {
	constant?: number;
	slope?: number;
	normal?: number;
	normalMin?: number;
	texel?: number;
	max?: number;
}

export interface ShadowSamplingSettings {
	filterMode?: ShadowFilterMode;
	pcfRadius?: number;
	strength?: number;
	radius?: number;
	samples?: number;
	searchSamples?: number;
}

export interface ShadowMapBaseOptions {
	id?: string;
	enabled?: boolean;
	priority?: number;
	size?: number;
	bias?: ShadowBiasSettings;
	sampling?: ShadowSamplingSettings;
}

export type ShadowProjectionPreference = "single" | "cascaded";
export type ShadowStoragePreference = "atlas" | "paged";

export interface ShadowProjectionSnapshot {
	readonly technique: ShadowProjectionPreference;
	readonly cascadeCounts?: Readonly<CascadedShadowMapDefaults>;
	readonly lambda?: number;
	readonly maxDistance?: number;
	readonly blendRatio?: number;
	readonly stabilize?: boolean;
}

/**
 * Immutable settings captured from a public shadow definition.
 *
 * @internal Owned by the shadow planning subsystem. Consumers should configure
 * shadows through `ShadowMapBase` and its concrete subclasses.
 */
export interface ShadowDefinitionSnapshot {
	readonly id: string;
	readonly kind: ShadowMapKind;
	readonly enabled: boolean;
	readonly projection: Readonly<ShadowProjectionSnapshot>;
	readonly storagePreference: ShadowStoragePreference;
	readonly resolution: number;
	readonly bias: Readonly<Required<ShadowBiasSettings>>;
	readonly sampling: Readonly<Required<ShadowSamplingSettings>>;
	readonly priority: number;
	readonly revision: number;
}

/** @internal Listener used by `ShadowManager` to observe shared definitions. */
export type ShadowDefinitionListener = (
	definition: ShadowMapBase
) => void;

export interface CascadedShadowMapDefaults {
	directional: number;
	spot: number;
	point: number;
}

export interface CascadedShadowMapOptions extends ShadowMapBaseOptions {
	cascadeCounts?: Partial<CascadedShadowMapDefaults>;
	lambda?: number;
	maxDistance?: number;
	blendRatio?: number;
	stabilize?: boolean;
}

export type PagedShadowFeedbackMode = "conservative" | "screen-feedback";

export interface PagedShadowMapOptions extends ShadowMapBaseOptions {
	virtualResolution?: number;
	pageSize?: number;
	physicalPageCount?: number;
	clipmapLevels?: number;
	maxPagesPerFrame?: number;
	cacheFrames?: number;
	feedbackMode?: PagedShadowFeedbackMode;
	cascadeCounts?: Partial<CascadedShadowMapDefaults>;
	lambda?: number;
	maxDistance?: number;
	blendRatio?: number;
	stabilize?: boolean;
}

export interface ShadowRuntimeSlice {
	index: number;
	view: Matrix4 | null;
	projection: Matrix4 | null;
	viewProjection: Matrix4 | null;
	lightDir: IVector3;
	splitNear: number;
	splitFar: number;
}

export interface ShadowBindingRecord {
	light: ShadowCastingLight;
	shadowMapId: string;
	shadowMapKind: ShadowMapKind;
	filterMode: ShadowFilterMode;
	priority: number;
	renderSet: ShadowRenderSet;
	cost: number;
	score: number;
}

export interface ShadowBudgetDecision {
	size: number;
	cascadeCount?: number;
	enabled: boolean;
	cost: number;
}

export interface IShadowBackendCapabilities {
	backendKey: string;
	supportsFilterModes: ShadowFilterMode[];
	/** Explicit projection, storage, and count limits for each light type. */
	lightTypes?: Partial<Record<ShadowBoundLightType, {
		readonly projections: readonly ShadowProjectionPreference[];
		readonly storage: readonly ShadowStoragePreference[];
		readonly maxLights: number;
		readonly maxCascadedLights: number;
	}>>;
	/** Whether RGB transmission visibility is produced by the shadow runtime. */
	supportsTransmission?: boolean;
	/** @deprecated Use `lightTypes.directional.projections`. */
	supportsDirectionalCSM: boolean;
	/** @deprecated Use `lightTypes.spot.projections`. */
	supportsSpotCSM: boolean;
	/** @deprecated Use `lightTypes.point.projections`. */
	supportsPointCSM: boolean;
	maxDynamicShadowCost?: number;
	supportsPagedShadows?: boolean;
	supportsPagedShadowRendering?: boolean;
	maxPagedShadowPages?: number;
	pagedShadowPageSizeRange?: [number, number];
}

export interface ShadowSliceAllocation {
	lightId: string;
	shadowMapId: string;
	sliceIndex: number;
	tileX: number;
	tileY: number;
	size: number;
}

export interface IShadowSliceAllocator {
	allocate(records: ShadowBindingRecord[]): ShadowSliceAllocation[];
}

export interface ShadowPassSliceInput {
	light: ShadowCastingLight;
	record: ShadowBindingRecord;
	allocation: ShadowSliceAllocation;
}

export interface IShadowPassExecutor {
	executeShadowPass(slices: ShadowPassSliceInput[]): void | Promise<void>;
}

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
	getBounds?: () => {
		left: number;
		right: number;
		bottom: number;
		top: number;
	};
	getWorldPosition?: (target?: IVector3) => IVector3;
	getWorldDirection?: (localDirection: IVector3, target?: IVector3) => IVector3;
}

export interface ShadowSliceDescriptor {
	view: Matrix4;
	projection: Matrix4;
	lightDir: IVector3;
	splitNear: number;
	splitFar: number;
}

export interface ShadowStrategyBuildContext {
	light: ShadowCastingLight;
	renderSet: ShadowRenderSet;
	config: ShadowConfig;
	sceneBounds: SceneBounds;
	camera?: ShadowStrategyCamera | null;
}

export interface IShadowStrategyProvider {
	readonly type: ShadowStrategyType;
	supports(light: ShadowCastingLight): boolean;
	build(context: ShadowStrategyBuildContext): ShadowSliceDescriptor[];
}
