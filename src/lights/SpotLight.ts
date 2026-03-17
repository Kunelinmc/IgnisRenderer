import { Matrix4 } from "../maths/Matrix4";
import { Vector3 } from "../maths/Vector3";
import type { IVector3 } from "../maths/types";
import { ShadowConstants } from "./constants";
import {
	Light,
	LightType,
	type LightParams,
	type ShadowCameraResult,
	type ShadowCaster,
} from "./Light";

export interface SpotLightParams extends LightParams {
	position?: IVector3;
	direction?: IVector3;
	angle?: number;
	innerAngle?: number;
	penumbra?: number;
	range?: number;
}

class SpotShadowCaster implements ShadowCaster {
	constructor(private light: SpotLight) {}

	setupShadowCamera(ctx: {
		sceneBounds: { center: IVector3; radius: number };
		worldMatrix: Matrix4;
	}): ShadowCameraResult | null {
		const position = Matrix4.transformPoint(ctx.worldMatrix, {
			x: 0,
			y: 0,
			z: 0,
		});
		let direction = Matrix4.transformDirection(
			ctx.worldMatrix,
			this.light.direction
		);
		direction = Vector3.normalize(direction);
		const target = {
			x: position.x + direction.x,
			y: position.y + direction.y,
			z: position.z + direction.z,
		};
		const up =
			Math.abs(direction.y) < 0.999
				? { x: 0, y: 1, z: 0 }
				: { x: 0, y: 0, z: 1 };
		const view = Matrix4.lookAt(position, target, up);

		const distanceToCenter = Vector3.length(
			Vector3.sub(position, ctx.sceneBounds.center)
		);
		const autoFar = distanceToCenter + ctx.sceneBounds.radius;
		let far = Math.min(this.light.range, Math.max(autoFar, 0));
		far = Math.max(ShadowConstants.MIN_SHADOW_FAR, far);

		const nearCandidate = distanceToCenter - ctx.sceneBounds.radius;
		const near = Math.max(
			ShadowConstants.MIN_SHADOW_NEAR,
			Math.min(nearCandidate, far - ShadowConstants.SHADOW_NEAR_FAR_GAP)
		);

		const projection = Matrix4.perspective(
			this.light.angle * 2 * (180 / Math.PI),
			1,
			near,
			far
		);

		return { view, projection, lightDir: direction };
	}
}

export class SpotLight extends Light<LightType.Spot> {
	public direction: IVector3;
	public angle: number;
	public innerAngle?: number;
	public penumbra: number;
	public range: number;

	constructor(params: SpotLightParams = {}) {
		super(LightType.Spot, params);
		if (params.position) {
			this.position.copy(params.position);
		}
		this.direction = params.direction ?? { x: 0, y: -1, z: 0 };
		this.angle = params.angle ?? Math.PI / 4;
		this.innerAngle = params.innerAngle;
		this.penumbra = params.penumbra ?? 0;
		this.range = params.range ?? 1000;
		this.shadow = new SpotShadowCaster(this);
		this.castShadow = true;
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.direction = {
			x: this.direction.x,
			y: this.direction.y,
			z: this.direction.z,
		};
		target.angle = this.angle;
		target.innerAngle = this.innerAngle;
		target.penumbra = this.penumbra;
		target.range = this.range;
	}
}
