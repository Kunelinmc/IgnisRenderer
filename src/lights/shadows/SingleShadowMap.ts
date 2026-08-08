import { Matrix4 } from "../../maths/Matrix4";
import { Vector3 } from "../../maths/Vector3";
import type { IVector3 } from "../../maths/types";
import {
	LightType,
	type DirectionalLight,
	type ShadowCastingLight,
	type SpotLight,
} from "..";
import {
	MIN_SHADOW_FAR,
	MIN_SHADOW_NEAR,
	SHADOW_NEAR_FAR_GAP,
} from "../constants";
import { ShadowMapBase, type ShadowMapBaseOptions } from "./ShadowMapBase";
import type {
	SceneBounds,
	ShadowSliceDescriptor,
	ShadowStrategyBuildContext,
} from "./types";

type PositionalShadowLight = ShadowCastingLight & {
	direction?: IVector3;
	outerAngle?: number;
	range?: number;
};

const DEFAULT_POSITIONAL_SHADOW_FOV_DEGREES = 120;
const MIN_POSITIONAL_SHADOW_FOV_DEGREES = 1;
const MAX_POSITIONAL_SHADOW_FOV_DEGREES = 175;
const DEFAULT_POSITIONAL_SHADOW_DIRECTION = { x: 0, y: -1, z: 0 };
const DEFAULT_SHADOW_PERSPECTIVE_ASPECT_RATIO = 1;

export class SingleShadowMap extends ShadowMapBase {
	public readonly kind: "single" | "variance" = "single";

	constructor(options: ShadowMapBaseOptions = {}) {
		super(options);
	}

	public static buildSlices(
		context: ShadowStrategyBuildContext
	): ShadowSliceDescriptor[] {
		const perspectiveAspectRatio = SingleShadowMap.resolvePerspectiveAspectRatio(
			context.camera?.aspectRatio
		);
		switch (context.light.type) {
			case LightType.Directional:
				return [
					SingleShadowMap.buildDirectionalSlice(
						context.light as DirectionalLight,
						context.sceneBounds
					),
				];
			case LightType.Spot:
				return [
					SingleShadowMap.buildSpotSlice(
						context.light as SpotLight,
						context.sceneBounds,
						perspectiveAspectRatio
					),
				];
			default:
				return [
					SingleShadowMap.buildPositionalSlice(
						context.light as PositionalShadowLight,
						context.sceneBounds
					),
				];
		}
	}

	public static buildDirectionalSlice(
		light: DirectionalLight,
		sceneBounds: SceneBounds
	): ShadowSliceDescriptor {
		const direction = SingleShadowMap.resolveDirectionalWorldDirection(light);
		const { center, radius } = sceneBounds;
		const shadowDistance = radius * 1.5;
		const lightPos = Vector3.sub(center, Vector3.scale(direction, shadowDistance));
		const view = Matrix4.lookAt(
			lightPos,
			center,
			SingleShadowMap.chooseUpVector(direction)
		);
		const size = radius * 1.2;
		const projection = Matrix4.ortho(
			-size,
			size,
			-size,
			size,
			0,
			Math.max(MIN_SHADOW_FAR, shadowDistance * 2)
		);
		return {
			view,
			projection,
			lightDir: direction,
			splitNear: 0,
			splitFar: 1,
		};
	}

	public static buildSpotSlice(
		light: SpotLight,
		sceneBounds: SceneBounds,
		aspectRatio = DEFAULT_SHADOW_PERSPECTIVE_ASPECT_RATIO
	): ShadowSliceDescriptor {
		const position = Matrix4.transformPoint(light.worldMatrix, {
			x: 0,
			y: 0,
			z: 0,
		});
		let direction = Matrix4.transformDirection(light.worldMatrix, light.direction);
		direction = Vector3.normalize(direction);
		const target = {
			x: position.x + direction.x,
			y: position.y + direction.y,
			z: position.z + direction.z,
		};
		const view = Matrix4.lookAt(
			position,
			target,
			SingleShadowMap.chooseUpVector(direction)
		);

		const distanceToCenter = Vector3.length(
			Vector3.sub(position, sceneBounds.center)
		);
		const autoFar = distanceToCenter + sceneBounds.radius;
		let far = Math.min(light.range, Math.max(autoFar, 0));
		far = Math.max(MIN_SHADOW_FAR, far);
		const nearCandidate = distanceToCenter - sceneBounds.radius;
		const near = Math.max(
			MIN_SHADOW_NEAR,
			Math.min(nearCandidate, far - SHADOW_NEAR_FAR_GAP)
		);

		const projection = Matrix4.perspective(
			light.outerAngle * 2 * (180 / Math.PI),
			SingleShadowMap.resolvePerspectiveAspectRatio(aspectRatio),
			near,
			far
		);

		return {
			view,
			projection,
			lightDir: direction,
			splitNear: near,
			splitFar: far,
		};
	}

	public static buildPositionalSlice(
		light: PositionalShadowLight,
		sceneBounds: SceneBounds,
		aspectRatio = DEFAULT_SHADOW_PERSPECTIVE_ASPECT_RATIO
	): ShadowSliceDescriptor {
		const position = Matrix4.transformPoint(light.worldMatrix, {
			x: 0,
			y: 0,
			z: 0,
		});
		const direction = SingleShadowMap.resolvePositionalShadowDirection(
			light,
			position,
			sceneBounds
		);
		const target = {
			x: position.x + direction.x,
			y: position.y + direction.y,
			z: position.z + direction.z,
		};
		const view = Matrix4.lookAt(
			position,
			target,
			SingleShadowMap.chooseUpVector(direction)
		);
		const distanceToCenter = Vector3.length(
			Vector3.sub(position, sceneBounds.center)
		);
		const autoFar = distanceToCenter + sceneBounds.radius;
		const far = SingleShadowMap.resolvePositionalShadowRange(light, autoFar);
		const nearCandidate = distanceToCenter - sceneBounds.radius;
		const near = Math.max(
			MIN_SHADOW_NEAR,
			Math.min(nearCandidate, far - SHADOW_NEAR_FAR_GAP)
		);
		const projection = Matrix4.perspective(
			SingleShadowMap.resolvePositionalShadowFov(light),
			SingleShadowMap.resolvePerspectiveAspectRatio(aspectRatio),
			near,
			far
		);

		return {
			view,
			projection,
			lightDir: direction,
			splitNear: near,
			splitFar: far,
		};
	}

	public static resolveDirectionalWorldDirection(light: DirectionalLight): IVector3 {
		const transformed = Matrix4.transformDirection(light.worldMatrix, light.direction);
		const normalized = Vector3.normalize(transformed);
		return { x: normalized.x, y: normalized.y, z: normalized.z };
	}

	public static chooseUpVector(direction: IVector3): IVector3 {
		if (Math.abs(direction.y) < 0.999) {
			return { x: 0, y: 1, z: 0 };
		}
		return { x: 0, y: 0, z: 1 };
	}

	public static resolvePerspectiveAspectRatio(aspectRatio: unknown): number {
		if (
			typeof aspectRatio === "number" &&
			Number.isFinite(aspectRatio) &&
			aspectRatio > 0
		) {
			return aspectRatio;
		}
		return DEFAULT_SHADOW_PERSPECTIVE_ASPECT_RATIO;
	}

	private static resolvePositionalShadowDirection(
		light: PositionalShadowLight,
		position: IVector3,
		sceneBounds: SceneBounds
	): IVector3 {
		const localDirection = light.direction;
		if (localDirection) {
			const transformed = Matrix4.transformDirection(
				light.worldMatrix,
				localDirection
			);
			const normalized = Vector3.normalize(transformed);
			return {
				x: normalized.x,
				y: normalized.y,
				z: normalized.z,
			};
		}

		const toCenter = Vector3.sub(sceneBounds.center, position);
		const lengthSquared =
			toCenter.x * toCenter.x +
			toCenter.y * toCenter.y +
			toCenter.z * toCenter.z;
		if (lengthSquared <= 1e-8) {
			return {
				x: DEFAULT_POSITIONAL_SHADOW_DIRECTION.x,
				y: DEFAULT_POSITIONAL_SHADOW_DIRECTION.y,
				z: DEFAULT_POSITIONAL_SHADOW_DIRECTION.z,
			};
		}
		const normalized = Vector3.normalize(toCenter);
		return {
			x: normalized.x,
			y: normalized.y,
			z: normalized.z,
		};
	}

	private static resolvePositionalShadowFov(light: PositionalShadowLight): number {
		const outerAngle = light.outerAngle;
		if (
			typeof outerAngle === "number" &&
			Number.isFinite(outerAngle) &&
			outerAngle > 0
		) {
			return Math.max(
				MIN_POSITIONAL_SHADOW_FOV_DEGREES,
				Math.min(
					MAX_POSITIONAL_SHADOW_FOV_DEGREES,
					outerAngle * 2 * (180 / Math.PI)
				)
			);
		}
		return DEFAULT_POSITIONAL_SHADOW_FOV_DEGREES;
	}

	private static resolvePositionalShadowRange(
		light: PositionalShadowLight,
		autoFar: number
	): number {
		const range = light.range;
		if (typeof range === "number" && Number.isFinite(range) && range > 0) {
			return Math.min(range, Math.max(MIN_SHADOW_FAR, autoFar));
		}
		return Math.max(MIN_SHADOW_FAR, autoFar);
	}
}
