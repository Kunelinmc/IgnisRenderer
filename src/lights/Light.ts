import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import type { RGB } from "../utils/Color";
import { IdGenerator } from "../utils/IdGenerator";

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

export interface LightParams {
	color?: RGB;
	intensity?: number;
	castShadow?: boolean;
}

export abstract class Light<TType extends LightType = LightType> {
	public readonly id: string;
	public readonly type: TType;
	public color: RGB;
	public intensity: number;
	public castShadow: boolean;

	/**
	 * World matrix of the light, updated once per frame if needed.
	 */
	private _worldMatrix: Matrix4 = Matrix4.identity();

	public get worldMatrix(): Matrix4 {
		return this._worldMatrix;
	}

	protected constructor(type: TType, params: LightParams = {}) {
		this.id = IdGenerator.nextId("light");
		this.type = type;
		this.color = params.color ?? { r: 255, g: 255, b: 255 };
		this.intensity = params.intensity ?? 1.0;
		this.castShadow = params.castShadow ?? false;
	}

	/**
	 * Update the world matrix for this light.
	 * This should be called once per frame if the light or its parent transforms.
	 */
	public updateWorldMatrix(matrix: Matrix4): void {
		this._worldMatrix = matrix;
	}

	/**
	 * Reference to shadow caster logic if this light supports it.
	 */
	public shadow?: ShadowCaster;
}
