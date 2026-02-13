import { Vector3 } from "../maths/Vector3";
import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import {
	Light,
	LightType,
	type LightParams,
	type LightContribution,
	type ShadowCaster,
	type ShadowCameraResult,
} from "./Light";

export interface DirectionalLightParams extends LightParams {
	dir?: IVector3;
}

class DirectionalShadowCaster implements ShadowCaster {
	constructor(private light: DirectionalLight) {}

	setupShadowCamera(ctx: {
		sceneBounds: { center: IVector3; radius: number };
		worldMatrix: Matrix4;
	}): ShadowCameraResult | null {
		const { sceneBounds, worldMatrix } = ctx;
		let dir = this.light.dir;
		dir = Matrix4.transformDirection(worldMatrix, dir);
		dir = Vector3.normalize(dir);

		const center = sceneBounds.center;
		const radius = sceneBounds.radius;

		const lightPos = {
			x: center.x - dir.x * radius * 1.5,
			y: center.y - dir.y * radius * 1.5,
			z: center.z - dir.z * radius * 1.5,
		};

		const up =
			Math.abs(dir.y) > 0.9 ? { x: 1, y: 0, z: 0 } : { x: 0, y: 1, z: 0 };
		const view = Matrix4.lookAt(lightPos, center, up);

		const size = radius * 1.2;
		const projection = Matrix4.ortho(-size, size, -size, size, 0, radius * 3);

		return { view, projection, lightDir: dir };
	}
}

export class DirectionalLight extends Light<LightType.Directional> {
	public dir: IVector3;

	constructor(params: DirectionalLightParams = {}) {
		super(LightType.Directional, params);
		this.dir = params.dir ?? { x: 0, y: -1, z: 0 };
		this.shadow = new DirectionalShadowCaster(this);
		this.castShadow = true;
	}

	public computeContribution(_point: IVector3): LightContribution {
		let dir = this.dir;
		dir = Matrix4.transformDirection(this.worldMatrix, dir);
		dir = Vector3.normalize(dir);

		return {
			type: "direct",
			color: {
				r: this.color.r * this.intensity,
				g: this.color.g * this.intensity,
				b: this.color.b * this.intensity,
			},
			direction: { x: -dir.x, y: -dir.y, z: -dir.z },
		};
	}
}
