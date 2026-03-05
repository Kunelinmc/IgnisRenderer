import { Vector3 } from "../maths/Vector3";
import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import { ShadowConstants } from "../core/pipeline/constants";
import {
	Light,
	LightType,
	type LightParams,
	type ShadowCaster,
	type ShadowCameraResult,
} from "./Light";

export interface AreaLightParams extends LightParams {
	position?: IVector3;
	width?: number;
	height?: number;
	rotation?: IVector3; // Euler angles in radians
	range?: number;
}

class AreaShadowCaster implements ShadowCaster {
	constructor(private light: AreaLight) {}

	setupShadowCamera(ctx: {
		sceneBounds: { center: IVector3; radius: number };
		worldMatrix: Matrix4;
	}): ShadowCameraResult | null {
		const { worldMatrix } = ctx;
		const pos = this.light.position;
		const center = Matrix4.transformPoint(worldMatrix, pos);

		const rotMat = Matrix4.rotationFromEuler(
			this.light.rotation.x,
			this.light.rotation.y,
			this.light.rotation.z
		);

		const localNormal = { x: 0, y: 1, z: 0 };
		const normal = Vector3.normalize(
			Matrix4.transformDirection(
				worldMatrix,
				Matrix4.transformDirection(rotMat, localNormal)
			)
		);

		// Direction is towards the light emission (normal direction)
		const dir = normal;
		const target = {
			x: center.x + dir.x,
			y: center.y + dir.y,
			z: center.z + dir.z,
		};
		const up =
			Math.abs(dir.y) < 0.999 ? { x: 0, y: 1, z: 0 } : { x: 0, y: 0, z: 1 };

		const view = Matrix4.lookAt(center, target, up);

		// Perspective for area light, using a wide FOV
		const far = this.light.range;
		const near = ShadowConstants.MIN_SHADOW_NEAR;
		const projection = Matrix4.perspective(120, 1, near, far);

		return {
			view,
			projection,
			lightDir: dir,
		};
	}
}

export class AreaLight extends Light<LightType.RectArea> {
	public position: IVector3;
	public width: number;
	public height: number;
	public rotation: IVector3;
	public range: number;

	constructor(params: AreaLightParams = {}) {
		super(LightType.RectArea, params);
		this.position = params.position ?? { x: 0, y: 0, z: 0 };
		this.width = params.width ?? 100;
		this.height = params.height ?? 100;
		this.rotation = params.rotation ?? { x: 0, y: 0, z: 0 };
		this.range = params.range ?? 1000;
		this.shadow = new AreaShadowCaster(this);
		this.castShadow = params.castShadow ?? true;
	}
}
