import { Vector3 } from "../maths/Vector3";
import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import { ShadowConstants } from "../core/constants";
import {
	Light,
	LightType,
	type LightParams,
	type LightContribution,
	type SurfacePoint,
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

	public computeContribution(surface: SurfacePoint): LightContribution | null {
		const surfacePos = this._requireSurfacePosition(surface);
		const m = this.worldMatrix;

		// World space center
		const center = Matrix4.transformPoint(m, this.position);

		// Local basis rotated by the light's own rotation
		const rotMat = Matrix4.rotationFromEuler(
			this.rotation.x,
			this.rotation.y,
			this.rotation.z
		);

		// We define the area light in local space:
		// Width is along X axis, Height is along Z axis.
		// Normal of the emission is the Y axis.
		const localRight = { x: 1, y: 0, z: 0 };
		const localUp = { x: 0, y: 0, z: 1 };
		const localNormal = { x: 0, y: 1, z: 0 };

		let right = Matrix4.transformDirection(rotMat, localRight);
		let up = Matrix4.transformDirection(rotMat, localUp);
		let normal = Matrix4.transformDirection(rotMat, localNormal);

		// Transform bases to world space
		right = Vector3.normalize(Matrix4.transformDirection(m, right));
		up = Vector3.normalize(Matrix4.transformDirection(m, up));
		normal = Vector3.normalize(Matrix4.transformDirection(m, normal));

		const relPos = Vector3.sub(surfacePos, center);
		const distToPlane = Vector3.dot(relPos, normal);

		// If surface point is behind the light plane (relative to emission normal)
		if (distToPlane <= 0) return null;

		// Project relative position onto the rectangle's plane basis
		const projX = Vector3.dot(relPos, right);
		const projY = Vector3.dot(relPos, up);

		// Clamp to rectangle bounds
		const halfW = this.width / 2;
		const halfH = this.height / 2;
		const clampedX = Math.max(-halfW, Math.min(halfW, projX));
		const clampedY = Math.max(-halfH, Math.min(halfH, projY));

		// Find the closest point on the rectangle to the shaded point
		const closestPoint = Vector3.add(
			center,
			Vector3.add(Vector3.scale(right, clampedX), Vector3.scale(up, clampedY))
		);

		const L_vec = Vector3.sub(closestPoint, surfacePos);
		const distance = Vector3.length(L_vec);

		if (distance > this.range) return null;

		const L = Vector3.normalize(L_vec);

		// Angle of incidence at the light source plane
		// (L points from surface to light, normal points away from light)
		const cosLight = Math.max(0, Vector3.dot(Vector3.scale(normal, -1), L));

		const distanceSq = distance * distance;
		const attenuation =
			((this.width * this.height) / 100) * (cosLight / (distanceSq + 1.0));

		return {
			type: "direct",
			color: this.color,
			intensity: this.intensity * attenuation,
			direction: L,
		};
	}
}
