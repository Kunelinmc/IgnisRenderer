import { Matrix4 } from "../maths/Matrix4";
import { Vector3 } from "../maths/Vector3";
import type { IVector3 } from "../maths/types";
import {
	Light,
	LightType,
	type LightParams,
	type ShadowCameraResult,
	type ShadowCaster,
} from "./Light";

export interface DirectionalLightParams extends LightParams {
	direction?: IVector3;
}

class DirectionalShadowCaster implements ShadowCaster {
	constructor(private light: DirectionalLight) {}

	setupShadowCamera(ctx: {
		sceneBounds: { center: IVector3; radius: number };
		worldMatrix: Matrix4;
	}): ShadowCameraResult | null {
		const { sceneBounds, worldMatrix } = ctx;
		let direction = Matrix4.transformDirection(
			worldMatrix,
			this.light.direction
		);
		direction = Vector3.normalize(direction);

		const { center, radius } = sceneBounds;
		const shadowDistance = radius * 1.5;
		const lightPos = Vector3.sub(
			center,
			Vector3.scale(direction, shadowDistance)
		);
		const up =
			Math.abs(direction.y) < 0.999
				? { x: 0, y: 1, z: 0 }
				: { x: 0, y: 0, z: 1 };
		const view = Matrix4.lookAt(lightPos, center, up);
		const size = radius * 1.2;
		const projection = Matrix4.ortho(
			-size,
			size,
			-size,
			size,
			0,
			shadowDistance * 2
		);

		return { view, projection, lightDir: direction };
	}
}

export class DirectionalLight extends Light<LightType.Directional> {
	public direction: IVector3;

	constructor(params: DirectionalLightParams = {}) {
		super(LightType.Directional, params);
		this.direction = params.direction ?? { x: 0, y: -1, z: 0 };
		this.shadow = new DirectionalShadowCaster(this);
		this.castShadow = true;
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.direction = {
			x: this.direction.x,
			y: this.direction.y,
			z: this.direction.z,
		};
	}
}
