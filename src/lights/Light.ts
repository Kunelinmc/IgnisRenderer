import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import type { RGB } from "../foundation/Color";
import { Node, type NodeParams } from "../core/Node";
import type {
	CSMShadowConfig,
	ShadowConfig,
	SingleMapShadowConfig,
} from "./ShadowMapping";

export enum LightType {
	Ambient = "ambient",
	Directional = "directional",
	Point = "point",
	Spot = "spot",
	LightProbe = "lightProbe",
	ReflectionProbe = "reflectionProbe",
	RectArea = "rectArea",
}

export interface ShadowCameraResult {
	view: Matrix4;
	projection: Matrix4;
	lightDir: IVector3;
}

export interface LightParams extends NodeParams {
	color?: RGB;
	intensity?: number;
	castShadow?: boolean;
	shadow?: ShadowConfig;
}

export abstract class Light<TType extends LightType = LightType> extends Node {
	public readonly type: TType;
	public color: RGB;
	public intensity: number;
	public castShadow: boolean;
	public shadow?: ShadowConfig;

	protected constructor(type: TType, params: LightParams = {}) {
		super({
			...params,
			idPrefix: "light",
		});
		this.type = type;
		this.color = params.color ?? { r: 255, g: 255, b: 255 };
		this.intensity = params.intensity ?? 1;
		this.castShadow = params.castShadow ?? false;
		this.shadow = params.shadow;
	}

	public setShadowStrategy(config: ShadowConfig): this {
		this.shadow = config;
		this.castShadow = true;
		return this;
	}

	public setSingleMapShadow(config: Omit<SingleMapShadowConfig, "strategy"> = {}): this {
		this.shadow = {
			strategy: "single-map",
			...config,
		};
		this.castShadow = true;
		return this;
	}

	public setCSMShadow(config: Omit<CSMShadowConfig, "strategy"> = {}): this {
		this.shadow = {
			strategy: "csm",
			...config,
		};
		this.castShadow = true;
		return this;
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.color = {
			r: this.color.r,
			g: this.color.g,
			b: this.color.b,
		};
		target.intensity = this.intensity;
		target.castShadow = this.castShadow;
		target.shadow =
			this.shadow ?
				(JSON.parse(JSON.stringify(this.shadow)) as ShadowConfig)
			: 	undefined;
	}
}
