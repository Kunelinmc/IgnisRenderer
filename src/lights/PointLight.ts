import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import {
	Light,
	LightType,
	type LightParams,
	type LightContribution,
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

	public computeContribution(point: IVector3): LightContribution | null {
		let lightPos = this.position;
		const p = Matrix4.transformPoint(this.worldMatrix, lightPos);
		lightPos = { x: p.x, y: p.y, z: p.z };

		const dx = lightPos.x - point.x;
		const dy = lightPos.y - point.y;
		const dz = lightPos.z - point.z;
		const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

		if (distance > this.range) return null;

		// Smooth distance attenuation: ensures light hits 0 at distance === range
		const distanceFactor = distance / this.range;
		const smoothFactor = Math.max(0, 1 - distanceFactor * distanceFactor);

		// Combine with standard attenuation model
		const constant = 1.0;
		const linear = 0.007;
		const quadratic = 0.0002;
		const attenuation =
			(1.0 / (constant + linear * distance + quadratic * distance * distance)) *
			(smoothFactor * smoothFactor);

		return {
			type: "direct",
			color: {
				r: this.color.r * this.intensity * attenuation,
				g: this.color.g * this.intensity * attenuation,
				b: this.color.b * this.intensity * attenuation,
			},
			direction:
				distance > 0 ?
					{ x: dx / distance, y: dy / distance, z: dz / distance }
				:	{ x: 0, y: 1, z: 0 },
		};
	}
}
