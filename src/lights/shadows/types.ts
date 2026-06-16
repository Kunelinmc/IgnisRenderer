import type { Matrix4 } from "../../maths/Matrix4";
import type { IVector3 } from "../../maths/types";
import type { ShadowCastingLight } from "..";
import type {
	ShadowConfig,
	ShadowRenderSet,
	ShadowStrategyType,
} from "./ShadowMapping";

export type BuiltinShadowMapKind = "single" | "vsm" | "csm";
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

export interface ShadowCSMDefaults {
	directional: number;
	spot: number;
	point: number;
}

export interface ShadowCSMOptions extends ShadowMapBaseOptions {
	cascadeCounts?: Partial<ShadowCSMDefaults>;
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
	supportsDirectionalCSM: boolean;
	supportsSpotCSM: boolean;
	supportsPointCSM: boolean;
	maxDynamicShadowCost?: number;
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
