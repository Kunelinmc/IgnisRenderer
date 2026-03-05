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

export interface SpotLightParams extends LightParams {
	position?: IVector3;
	dir?: IVector3;
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
		const { worldMatrix } = ctx;
		let pos = this.light.position;
		let dir = this.light.dir;

		const p = Matrix4.transformPoint(worldMatrix, pos);
		pos = { x: p.x, y: p.y, z: p.z };
		dir = Matrix4.transformDirection(worldMatrix, dir);

		dir = Vector3.normalize(dir);
		const target = { x: pos.x + dir.x, y: pos.y + dir.y, z: pos.z + dir.z };
		// Use world Y as up by default, flip to Z if direction is nearly vertical.
		// Threshold increased from 0.9 to 0.999 to prevent premature popping.
		const up =
			Math.abs(dir.y) < 0.999 ? { x: 0, y: 1, z: 0 } : { x: 0, y: 0, z: 1 };

		const view = Matrix4.lookAt(pos, target, up);

		// Use light range if specified, otherwise calculate from scene bounds
		const distanceToCenter = Vector3.length(
			Vector3.sub(pos, ctx.sceneBounds.center)
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

		return { view, projection, lightDir: dir };
	}
}

export class SpotLight extends Light<LightType.Spot> {
	public position: IVector3;
	public dir: IVector3;
	public angle: number;
	public innerAngle?: number;
	public penumbra: number;
	public range: number;

	constructor(params: SpotLightParams = {}) {
		super(LightType.Spot, params);
		this.position = params.position ?? { x: 0, y: 0, z: 0 };
		this.dir = params.dir ?? { x: 0, y: -1, z: 0 };
		this.angle = params.angle ?? Math.PI / 4;
		this.innerAngle = params.innerAngle;
		this.penumbra = params.penumbra ?? 0;
		this.range = params.range ?? 1000;
		this.shadow = new SpotShadowCaster(this);
		this.castShadow = true;
	}
}
