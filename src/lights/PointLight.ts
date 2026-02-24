import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import {
	Light,
	LightType,
	type LightParams,
	type LightContribution,
	type SurfacePoint,
} from "./Light";

export interface PointLightParams extends LightParams {
	position?: IVector3;
	pos?: IVector3;
	range?: number;
}

export class PointLight extends Light<LightType.Point> {
	public position: IVector3;
	public range: number;

	constructor(params: PointLightParams = {}) {
		super(LightType.Point, params);
		this.position = params.position ?? params.pos ?? { x: 0, y: 0, z: 0 };
		this.range = params.range ?? 1000;
	}

	public computeContribution(surface: SurfacePoint): LightContribution | null {
		const position = this._requireSurfacePosition(surface)
		let lightPos = this.position;
		const p = Matrix4.transformPoint(this.worldMatrix, lightPos);
		lightPos = { x: p.x, y: p.y, z: p.z };

		const dx = lightPos.x - position.x;
		const dy = lightPos.y - position.y;
		const dz = lightPos.z - position.z;
		const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

		if (distance > this.range) return null;

		// Physically-based distance attenuation: inverse square law with smooth windowing
		// This uses the formula: attenuation = max(0, 1 - (d/r)^4)^2 / (d^2 + 1)
		// The +1 in the denominator prevents the singularity at d=0.
		const distanceSq = distance * distance;
		const rangeSq = this.range * this.range;
		const rangeFactor = distanceSq / rangeSq;
		const smoothFactor = Math.max(0, 1 - rangeFactor * rangeFactor);
		const attenuation = (smoothFactor * smoothFactor) / (distanceSq + 1.0);

		return {
			type: "direct",
			color: this.color,
			intensity: this.intensity * attenuation,
			direction:
				distance > 0 ?
					{ x: dx / distance, y: dy / distance, z: dz / distance }
				:	{ x: 0, y: 1, z: 0 },
		};
	}
}
