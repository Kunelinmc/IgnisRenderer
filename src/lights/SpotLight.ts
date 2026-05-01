import type { IVector3 } from "../maths/types";
import {
	Light,
	LightType,
	type LightParams,
} from "./Light";

export interface SpotLightParams extends LightParams {
	position?: IVector3;
	direction?: IVector3;
	outerAngle?: number;
	innerAngle?: number;
	penumbra?: number;
	range?: number;
}

export class SpotLight extends Light<LightType.Spot> {
	public direction: IVector3;
	public outerAngle: number;
	public innerAngle?: number;
	public penumbra: number;
	public range: number;

	constructor(params: SpotLightParams = {}) {
		super(LightType.Spot, params);
		if (params.position) {
			this.position.copy(params.position);
		}
		this.direction = params.direction ?? { x: 0, y: -1, z: 0 };
		this.outerAngle = params.outerAngle ?? Math.PI / 4;
		this.innerAngle = params.innerAngle;
		this.penumbra = params.penumbra ?? 0;
		this.range = params.range ?? 1000;
	}

	/**
	 * Resolves the light origin in world space.
	 */
	public getWorldLightPosition(out?: IVector3): IVector3 {
		return this.getWorldPosition(out);
	}

	/**
	 * Resolves the spotlight direction in world space.
	 */
	public getWorldLightDirection(out?: IVector3): IVector3 {
		return this.getWorldDirection(this.direction, out);
	}

	/**
	 * Resolves the effective inner cone angle.
	 */
	public getInnerAngle(): number {
		return this.innerAngle ?? this.outerAngle * (1 - this.penumbra);
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.direction = {
			x: this.direction.x,
			y: this.direction.y,
			z: this.direction.z,
		};
		target.outerAngle = this.outerAngle;
		target.innerAngle = this.innerAngle;
		target.penumbra = this.penumbra;
		target.range = this.range;
	}
}
