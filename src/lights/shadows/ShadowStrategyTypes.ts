import type { Matrix4 } from "../../maths/Matrix4";
import type { IVector3 } from "../../maths/types";
import type { ShadowCastingLight } from "..";
import type {
	ShadowConfig,
	ShadowRenderSet,
	ShadowStrategyType,
} from "./ShadowMapping";

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
	up?: IVector3;
	position?: IVector3;
	getWorldPosition?: (target?: IVector3) => IVector3;
	getWorldDirection?: (
		localDirection: IVector3,
		target?: IVector3
	) => IVector3;
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
