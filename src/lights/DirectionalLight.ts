import { Vector3 } from "../maths/Vector3";
import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import {
	Light,
	LightType,
	type LightParams,
	type LightContribution,
	type SurfacePoint,
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

		const { center, radius } = sceneBounds;

		// Offset light position far enough to include the whole scene volume.
		// Using 1.5x radius provides a buffer to avoid clipping.
		const shadowDistance = radius * 1.5;
		const lightPos = Vector3.sub(center, Vector3.scale(dir, shadowDistance));

		// Use world Y as up by default, flip to Z if direction is nearly vertical.
		// Threshold increased from 0.9 to 0.999 to prevent premature popping.
		const up =
			Math.abs(dir.y) < 0.999 ? { x: 0, y: 1, z: 0 } : { x: 0, y: 0, z: 1 };
		const view = Matrix4.lookAt(lightPos, center, up);

		const size = radius * 1.2; // Provide a margin to ensure full scene coverage
		const projection = Matrix4.ortho(
			-size,
			size,
			-size,
			size,
			0,
			shadowDistance * 2
		);

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

	public computeContribution(_surface: SurfacePoint): LightContribution {
		let dir = this.dir;
		dir = Matrix4.transformDirection(this.worldMatrix, dir);
		dir = Vector3.normalize(dir);

		return {
			type: "direct",
			color: this.color,
			intensity: this.intensity,
			direction: { x: -dir.x, y: -dir.y, z: -dir.z },
		};
	}
}
