import { Vector3 } from "../maths/Vector3";
import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import { ShadowConstants } from "../core/Constants";
import {
	Light,
	LightType,
	type LightParams,
	type LightContribution,
	type SurfacePoint,
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

	public computeContribution(surface: SurfacePoint): LightContribution | null {
		const position = this._requireSurfacePosition(surface);
		let lightPos = this.position;
		let lightDir = this.dir;

		const p = Matrix4.transformPoint(this.worldMatrix, lightPos);
		lightPos = { x: p.x, y: p.y, z: p.z };
		lightDir = Matrix4.transformDirection(this.worldMatrix, lightDir);

		lightDir = Vector3.normalize(lightDir);

		const dx = lightPos.x - position.x;
		const dy = lightPos.y - position.y;
		const dz = lightPos.z - position.z;
		const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

		if (distance > this.range) return null;

		const L =
			distance > 0 ?
				{ x: dx / distance, y: dy / distance, z: dz / distance }
			:	{ x: 0, y: 1, z: 0 };
		const lightToPoint = { x: -L.x, y: -L.y, z: -L.z };
		const cosTheta = Vector3.dot(lightToPoint, lightDir);

		// angle is the outer cutoff, innerAngle (or penumbra) defines the inner cutoff
		const outerCutoff = Math.cos(this.angle);
		let iAngle = this.innerAngle;
		if (iAngle === undefined) {
			iAngle = this.angle * (1 - this.penumbra);
		}
		const innerCutoff = Math.cos(iAngle);
		const epsilon = innerCutoff - outerCutoff;

		if (cosTheta < outerCutoff) return null;

		const spotIntensity = Math.max(
			0,
			Math.min(1, (cosTheta - outerCutoff) / (epsilon || 1e-6))
		);

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
			intensity: this.intensity * attenuation * spotIntensity,
			direction: L,
		};
	}
}
