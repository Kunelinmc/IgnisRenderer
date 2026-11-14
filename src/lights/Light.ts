import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import { Node, type NodeParams } from "../core/Node";

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

export interface LightParams extends NodeParams {}

export abstract class Light<TType extends LightType = LightType> extends Node {
	public readonly type: TType;

	protected constructor(type: TType, params: LightParams = {}) {
		super({
			...params,
			idPrefix: "light",
		});
		this.type = type;
	}
}
