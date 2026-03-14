import { Matrix4 } from "../maths/Matrix4";
import { Vector3 } from "../maths/Vector3";
import type { IVector3 } from "../maths/types";
import { ShadowConstants } from "../core/constants";
import {
	Light,
	LightType,
	type LightParams,
	type ShadowCameraResult,
	type ShadowCaster,
} from "./Light";

export interface AreaLightParams extends LightParams {
	position?: IVector3;
	width?: number;
	height?: number;
	range?: number;
}

class AreaShadowCaster implements ShadowCaster {
	constructor(private light: AreaLight) {}

	setupShadowCamera(ctx: {
		sceneBounds: { center: IVector3; radius: number };
		worldMatrix: Matrix4;
	}): ShadowCameraResult | null {
		const center = Matrix4.transformPoint(ctx.worldMatrix, {
			x: 0,
			y: 0,
			z: 0,
		});
		const direction = Vector3.normalize(
			Matrix4.transformDirection(ctx.worldMatrix, { x: 0, y: 1, z: 0 })
		);
		const target = {
			x: center.x + direction.x,
			y: center.y + direction.y,
			z: center.z + direction.z,
		};
		const up =
			Math.abs(direction.y) < 0.999
				? { x: 0, y: 1, z: 0 }
				: { x: 0, y: 0, z: 1 };

		const view = Matrix4.lookAt(center, target, up);
		const far = Math.max(this.light.range, ShadowConstants.MIN_SHADOW_FAR);
		const near = ShadowConstants.MIN_SHADOW_NEAR;
		const projection = Matrix4.perspective(120, 1, near, far);

		return {
			view,
			projection,
			lightDir: direction,
		};
	}
}

export class AreaLight extends Light<LightType.RectArea> {
	public width: number;
	public height: number;
	public range: number;

	constructor(params: AreaLightParams = {}) {
		super(LightType.RectArea, params);
		if (params.position) {
			this.position.copy(params.position);
		}
		this.width = params.width ?? 100;
		this.height = params.height ?? 100;
		this.range = params.range ?? 1000;
		this.shadow = new AreaShadowCaster(this);
		this.castShadow = params.castShadow ?? true;
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.width = this.width;
		target.height = this.height;
		target.range = this.range;
	}
}
