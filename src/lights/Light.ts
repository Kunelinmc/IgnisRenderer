import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import type { RGB } from "../utils/Color";
import { Node, type NodeParams } from "../core/Node";

export enum LightType {
	Ambient = "ambient",
	Directional = "directional",
	Point = "point",
	Spot = "spot",
	LightProbe = "lightProbe",
	RectArea = "rectArea",
}

export interface ShadowCameraResult {
	view: Matrix4;
	projection: Matrix4;
	lightDir: IVector3;
}

export interface ShadowCaster {
	setupShadowCamera(ctx: {
		sceneBounds: { center: IVector3; radius: number };
		worldMatrix: Matrix4;
	}): ShadowCameraResult | null;
}

export interface LightParams extends NodeParams {
	color?: RGB;
	intensity?: number;
	castShadow?: boolean;
}

export abstract class Light<TType extends LightType = LightType> extends Node {
	public readonly type: TType;
	public color: RGB;
	public intensity: number;
	public castShadow: boolean;
	public shadow?: ShadowCaster;

	protected constructor(type: TType, params: LightParams = {}) {
		super({
			...params,
			idPrefix: "light",
		});
		this.type = type;
		this.color = params.color ?? { r: 255, g: 255, b: 255 };
		this.intensity = params.intensity ?? 1;
		this.castShadow = params.castShadow ?? false;
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
	}
}
