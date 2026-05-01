import { Matrix4 } from "../../maths/Matrix4";
import { Vector3 } from "../../maths/Vector3";
import type { IVector3 } from "../../maths/types";
import { LightType, type DirectionalLight, type PointLight, type SpotLight } from "..";
import { MIN_SHADOW_NEAR, SHADOW_NEAR_FAR_GAP } from "../constants";
import type { CSMShadowConfig, ShadowConfig } from "./ShadowMapping";
import { ShadowMapBase } from "./ShadowMapBase";
import { SingleShadowMap } from "./SingleShadowMap";
import type {
	SceneBounds,
	ShadowSliceDescriptor,
	ShadowStrategyBuildContext,
	ShadowStrategyCamera,
	ShadowBoundLightType,
	ShadowCSMDefaults,
	ShadowCSMOptions,
} from "./types";

const DEFAULT_CASCADE_COUNTS: ShadowCSMDefaults = {
	directional: 4,
	spot: 3,
	point: 2,
};

const _tmpCamPos = { x: 0, y: 0, z: 0 };
const _tmpCamForward = { x: 0, y: 0, z: -1 };
const _tmpCamUp = { x: 0, y: 1, z: 0 };

export class CSMShadowMap extends ShadowMapBase {
	public readonly kind = "csm" as const;
	public cascadeCounts: ShadowCSMDefaults;
	public lambda: number;
	public maxDistance?: number;
	public blendRatio: number;
	public stabilize: boolean;

	constructor(options: ShadowCSMOptions = {}) {
		super(options);
		this.cascadeCounts = {
			directional: clampCascadeCount(
				resolveFinite(options.cascadeCounts?.directional, 4)
			),
			spot: clampCascadeCount(resolveFinite(options.cascadeCounts?.spot, 3)),
			point: clampCascadeCount(resolveFinite(options.cascadeCounts?.point, 2)),
		};
		this.lambda = resolveFinite(options.lambda, 0.65);
		this.maxDistance =
			typeof options.maxDistance === "number" &&
			Number.isFinite(options.maxDistance) ?
				Math.max(0.01, options.maxDistance)
			:	undefined;
		this.blendRatio = resolveFinite(options.blendRatio, 0.1);
		this.stabilize = options.stabilize !== false;
	}

	public getCascadeCountForLightType(lightType: LightType): number {
		const boundType = this.resolveBoundLightType(lightType);
		return this.getCascadeCountForBoundType(boundType);
	}

	public getCascadeCountForBoundType(boundType: ShadowBoundLightType): number {
		switch (boundType) {
			case "directional":
				return this.cascadeCounts.directional;
			case "spot":
				return this.cascadeCounts.spot;
			case "point":
				return this.cascadeCounts.point;
			default:
				return 2;
		}
	}

	public override toLegacyShadowConfig(
		lightType: LightType,
		overrides?: {
			size?: number;
			cascadeCount?: number;
		}
	): ShadowConfig {
		const cascadeCount =
			overrides?.cascadeCount ?? this.getCascadeCountForLightType(lightType);
		return this.createCSMLegacyConfig(cascadeCount, {
			size: overrides?.size,
			lambda: this.lambda,
			maxDistance: this.maxDistance,
			blendRatio: this.blendRatio,
			stabilize: this.stabilize,
		});
	}

	public static buildSlices(
		context: ShadowStrategyBuildContext
	): ShadowSliceDescriptor[] {
		const config = context.config as CSMShadowConfig;
		if (context.light.type === LightType.Spot) {
			return CSMShadowMap.buildSpotCascadeSlices(
				context.light as SpotLight,
				context.sceneBounds,
				config
			);
		}
		if (context.light.type === LightType.Point) {
			return CSMShadowMap.buildPointCubeCascadeSlices(
				context.light as PointLight,
				context.sceneBounds,
				config
			);
		}
		if (context.light.type !== LightType.Directional) {
			return [SingleShadowMap.buildPositionalSlice(context.light, context.sceneBounds)];
		}
		if (!context.camera) {
			return [
				SingleShadowMap.buildDirectionalSlice(
					context.light as DirectionalLight,
					context.sceneBounds
				),
			];
		}

		const cameraNear = Math.max(0.01, context.camera.near ?? 0.1);
		const cameraFarLimit =
			typeof context.camera.far === "number" && Number.isFinite(context.camera.far) ?
				Math.max(cameraNear + SHADOW_NEAR_FAR_GAP, context.camera.far)
			:	Math.max(cameraNear + SHADOW_NEAR_FAR_GAP, context.sceneBounds.radius * 4);
		const cameraFar =
			typeof config.maxDistance === "number" && Number.isFinite(config.maxDistance) ?
				Math.max(
					cameraNear + SHADOW_NEAR_FAR_GAP,
					Math.min(cameraFarLimit, config.maxDistance)
				)
			:	cameraFarLimit;

		const cascadeCount = Math.max(1, Math.min(4, config.cascadeCount ?? 4));
		const lambda = Math.max(0, Math.min(1, config.lambda ?? 0.65));
		const splits = CSMShadowMap.calculatePracticalCascadeSplits(
			cascadeCount,
			cameraNear,
			cameraFar,
			lambda
		);
		const direction = SingleShadowMap.resolveDirectionalWorldDirection(
			context.light as DirectionalLight
		);
		const stabilize = config.stabilize !== false;
		const shadowMapSize =
			context.renderSet.slices[0]?.shadowMap.size ??
			Math.max(1, Math.floor(context.renderSet.size / 2));
		const cornersBuffer: IVector3[] = new Array(8);
		const slices: ShadowSliceDescriptor[] = [];

		for (let index = 0; index < cascadeCount; index++) {
			const splitNear = splits[index];
			const splitFar = splits[index + 1];
			const hasCorners = CSMShadowMap.computePerspectiveFrustumCorners(
				context.camera,
				splitNear,
				splitFar,
				cornersBuffer
			);
			if (!hasCorners) {
				return [];
			}

			slices.push(
				CSMShadowMap.buildDirectionalCascadeSlice(
					direction,
					cornersBuffer,
					shadowMapSize,
					stabilize,
					splitNear,
					splitFar,
					context.sceneBounds
				)
			);
		}

		return slices;
	}

	private static calculatePracticalCascadeSplits(
		cascadeCount: number,
		cameraNear: number,
		cameraFar: number,
		lambda: number
	): number[] {
		const splits: number[] = [cameraNear];
		for (let index = 1; index < cascadeCount; index++) {
			const t = index / cascadeCount;
			const logarithmic = cameraNear * Math.pow(cameraFar / cameraNear, t);
			const uniform = cameraNear + (cameraFar - cameraNear) * t;
			splits.push(logarithmic * lambda + uniform * (1 - lambda));
		}
		splits.push(cameraFar);
		return splits;
	}

	private static buildSpotCascadeSlices(
		light: SpotLight,
		sceneBounds: SceneBounds,
		config: CSMShadowConfig
	): ShadowSliceDescriptor[] {
		const baseSlice = SingleShadowMap.buildSpotSlice(light, sceneBounds);
		const cascadeCount = Math.max(1, Math.min(4, config.cascadeCount ?? 3));
		const lambda = Math.max(0, Math.min(1, config.lambda ?? 0.65));
		const splits = CSMShadowMap.calculatePracticalCascadeSplits(
			cascadeCount,
			baseSlice.splitNear,
			baseSlice.splitFar,
			lambda
		);
		const slices: ShadowSliceDescriptor[] = [];
		for (let index = 0; index < cascadeCount; index++) {
			const splitNear = splits[index];
			const splitFar = splits[index + 1];
			slices.push({
				view: baseSlice.view,
				projection: Matrix4.perspective(
					light.outerAngle * 2 * (180 / Math.PI),
					1,
					splitNear,
					splitFar
				),
				lightDir: baseSlice.lightDir,
				splitNear,
				splitFar,
			});
		}
		return slices;
	}

	private static buildPointCubeCascadeSlices(
		light: PointLight,
		sceneBounds: SceneBounds,
		config: CSMShadowConfig
	): ShadowSliceDescriptor[] {
		const position = Matrix4.transformPoint(light.worldMatrix, {
			x: 0,
			y: 0,
			z: 0,
		});
		const distanceToCenter = Vector3.length(
			Vector3.sub(position, sceneBounds.center)
		);
		const autoFar = distanceToCenter + sceneBounds.radius;
		const far = Math.max(
			0.01,
			Math.min(Math.max(0.01, autoFar), Math.max(0.01, light.range))
		);
		const near = MIN_SHADOW_NEAR;
		const cascadeCount = Math.max(1, Math.min(4, config.cascadeCount ?? 2));
		const lambda = Math.max(0, Math.min(1, config.lambda ?? 0.65));
		const splits = CSMShadowMap.calculatePracticalCascadeSplits(
			cascadeCount,
			near,
			far,
			lambda
		);
		const faces: Array<{ direction: IVector3; up: IVector3 }> = [
			{ direction: { x: 1, y: 0, z: 0 }, up: { x: 0, y: 1, z: 0 } },
			{ direction: { x: -1, y: 0, z: 0 }, up: { x: 0, y: 1, z: 0 } },
			{ direction: { x: 0, y: 1, z: 0 }, up: { x: 0, y: 0, z: -1 } },
			{ direction: { x: 0, y: -1, z: 0 }, up: { x: 0, y: 0, z: 1 } },
			{ direction: { x: 0, y: 0, z: 1 }, up: { x: 0, y: 1, z: 0 } },
			{ direction: { x: 0, y: 0, z: -1 }, up: { x: 0, y: 1, z: 0 } },
		];
		const slices: ShadowSliceDescriptor[] = [];
		for (let cascadeIndex = 0; cascadeIndex < cascadeCount; cascadeIndex++) {
			const splitNear = splits[cascadeIndex];
			const splitFar = splits[cascadeIndex + 1];
			for (const face of faces) {
				const target = {
					x: position.x + face.direction.x,
					y: position.y + face.direction.y,
					z: position.z + face.direction.z,
				};
				slices.push({
					view: Matrix4.lookAt(position, target, face.up),
					projection: Matrix4.perspective(90, 1, splitNear, splitFar),
					lightDir: face.direction,
					splitNear,
					splitFar,
				});
			}
		}

		return slices;
	}

	private static resolveCameraPosition(
		camera: ShadowStrategyCamera | null | undefined
	): IVector3 | null {
		if (!camera) {
			return null;
		}
		if (typeof camera.getWorldPosition === "function") {
			return camera.getWorldPosition(_tmpCamPos);
		}
		if (!camera.position) {
			return null;
		}
		return camera.position;
	}

	private static resolveCameraDirection(
		camera: ShadowStrategyCamera | null | undefined,
		localDirection: IVector3,
		fallback: IVector3,
		target: IVector3
	): IVector3 {
		if (camera && typeof camera.getWorldDirection === "function") {
			return camera.getWorldDirection(localDirection, target);
		}
		target.x = fallback.x;
		target.y = fallback.y;
		target.z = fallback.z;
		return target;
	}

	private static computePerspectiveFrustumCorners(
		camera: ShadowStrategyCamera,
		splitNear: number,
		splitFar: number,
		output: IVector3[]
	): boolean {
		const cameraPosition = CSMShadowMap.resolveCameraPosition(camera);
		if (!cameraPosition) {
			return false;
		}

		const fov =
			typeof camera.fov === "number" && Number.isFinite(camera.fov) ?
				camera.fov
			:	60;
		const aspect =
			typeof camera.aspectRatio === "number" &&
			Number.isFinite(camera.aspectRatio) &&
			camera.aspectRatio > 0 ?
				camera.aspectRatio
			:	16 / 9;

		const forward = CSMShadowMap.resolveCameraDirection(
			camera,
			{ x: 0, y: 0, z: -1 },
			{ x: 0, y: 0, z: -1 },
			_tmpCamForward
		);
		const upVector = CSMShadowMap.resolveCameraDirection(
			camera,
			camera.up ?? { x: 0, y: 1, z: 0 },
			camera.up ?? { x: 0, y: 1, z: 0 },
			_tmpCamUp
		);
		const right = Vector3.normalize(Vector3.cross(forward, upVector));
		const up = Vector3.normalize(Vector3.cross(right, forward));

		if (camera.type === "orthographic") {
			const bounds = CSMShadowMap.resolveOrthographicBounds(camera, aspect);
			const nearCenter = Vector3.add(
				cameraPosition,
				Vector3.scale(forward, splitNear)
			);
			const farCenter = Vector3.add(
				cameraPosition,
				Vector3.scale(forward, splitFar)
			);
			const left = Vector3.scale(right, bounds.left);
			const rightOffset = Vector3.scale(right, bounds.right);
			const top = Vector3.scale(up, bounds.top);
			const bottom = Vector3.scale(up, bounds.bottom);
			const corners: IVector3[] = [
				Vector3.add(Vector3.add(nearCenter, left), top),
				Vector3.add(Vector3.add(nearCenter, rightOffset), top),
				Vector3.add(Vector3.add(nearCenter, left), bottom),
				Vector3.add(Vector3.add(nearCenter, rightOffset), bottom),
				Vector3.add(Vector3.add(farCenter, left), top),
				Vector3.add(Vector3.add(farCenter, rightOffset), top),
				Vector3.add(Vector3.add(farCenter, left), bottom),
				Vector3.add(Vector3.add(farCenter, rightOffset), bottom),
			];

			for (let index = 0; index < corners.length; index++) {
				output[index] = corners[index];
			}
			return true;
		}

		const halfTan = Math.tan((fov * Math.PI) / 360);
		const nearHalfHeight = splitNear * halfTan;
		const nearHalfWidth = nearHalfHeight * aspect;
		const farHalfHeight = splitFar * halfTan;
		const farHalfWidth = farHalfHeight * aspect;

		const nearCenter = Vector3.add(
			cameraPosition,
			Vector3.scale(forward, splitNear)
		);
		const farCenter = Vector3.add(cameraPosition, Vector3.scale(forward, splitFar));

		const nearRight = Vector3.scale(right, nearHalfWidth);
		const nearUp = Vector3.scale(up, nearHalfHeight);
		const farRight = Vector3.scale(right, farHalfWidth);
		const farUp = Vector3.scale(up, farHalfHeight);

		const corners: IVector3[] = [
			Vector3.add(Vector3.sub(nearCenter, nearRight), nearUp),
			Vector3.add(Vector3.add(nearCenter, nearRight), nearUp),
			Vector3.sub(Vector3.sub(nearCenter, nearRight), nearUp),
			Vector3.sub(Vector3.add(nearCenter, nearRight), nearUp),
			Vector3.add(Vector3.sub(farCenter, farRight), farUp),
			Vector3.add(Vector3.add(farCenter, farRight), farUp),
			Vector3.sub(Vector3.sub(farCenter, farRight), farUp),
			Vector3.sub(Vector3.add(farCenter, farRight), farUp),
		];

		for (let index = 0; index < corners.length; index++) {
			output[index] = corners[index];
		}
		return true;
	}

	private static resolveOrthographicBounds(
		camera: ShadowStrategyCamera,
		aspect: number
	): {
		left: number;
		right: number;
		bottom: number;
		top: number;
	} {
		if (typeof camera.getBounds === "function") {
			const bounds = camera.getBounds();
			if (
				Number.isFinite(bounds.left) &&
				Number.isFinite(bounds.right) &&
				Number.isFinite(bounds.bottom) &&
				Number.isFinite(bounds.top)
			) {
				return bounds;
			}
		}

		const size =
			typeof camera.size === "number" && Number.isFinite(camera.size) ?
				Math.max(0.001, camera.size)
			:	100;
		const halfHeight = size * 0.5;
		const halfWidth = halfHeight * aspect;
		return {
			left:
				typeof camera.left === "number" && Number.isFinite(camera.left) ?
					camera.left
				:	-halfWidth,
			right:
				typeof camera.right === "number" && Number.isFinite(camera.right) ?
					camera.right
				:	halfWidth,
			bottom:
				typeof camera.bottom === "number" && Number.isFinite(camera.bottom) ?
					camera.bottom
				:	-halfHeight,
			top:
				typeof camera.top === "number" && Number.isFinite(camera.top) ?
					camera.top
				:	halfHeight,
		};
	}

	private static buildDirectionalCascadeSlice(
		lightDir: IVector3,
		cascadeCorners: IVector3[],
		shadowMapSize: number,
		stabilize: boolean,
		splitNear: number,
		splitFar: number,
		sceneBounds?: SceneBounds
	): ShadowSliceDescriptor {
		let centerX = 0;
		let centerY = 0;
		let centerZ = 0;
		for (const corner of cascadeCorners) {
			centerX += corner.x;
			centerY += corner.y;
			centerZ += corner.z;
		}
		centerX /= cascadeCorners.length;
		centerY /= cascadeCorners.length;
		centerZ /= cascadeCorners.length;

		let maxRadiusSquared = 0;
		for (const corner of cascadeCorners) {
			const dx = corner.x - centerX;
			const dy = corner.y - centerY;
			const dz = corner.z - centerZ;
			maxRadiusSquared = Math.max(
				maxRadiusSquared,
				dx * dx + dy * dy + dz * dz
			);
		}
		const radius = Math.max(0.001, Math.sqrt(maxRadiusSquared));
		if (stabilize) {
			const span = Math.max(0.001, radius * 2);
			const texelSize = span / Math.max(1, shadowMapSize);
			if (texelSize > 0) {
				const lightBackward = {
					x: -lightDir.x,
					y: -lightDir.y,
					z: -lightDir.z,
				};
				const lightUpReference = SingleShadowMap.chooseUpVector(lightDir);
				const lightRight = Vector3.normalize(
					Vector3.cross(lightUpReference, lightBackward)
				);
				const lightUp = Vector3.normalize(
					Vector3.cross(lightBackward, lightRight)
				);
				const centerLightX =
					centerX * lightRight.x +
					centerY * lightRight.y +
					centerZ * lightRight.z;
				const centerLightY =
					centerX * lightUp.x + centerY * lightUp.y + centerZ * lightUp.z;
				const snappedCenterLightX =
					Math.round(centerLightX / texelSize) * texelSize;
				const snappedCenterLightY =
					Math.round(centerLightY / texelSize) * texelSize;
				const deltaX = snappedCenterLightX - centerLightX;
				const deltaY = snappedCenterLightY - centerLightY;

				centerX += lightRight.x * deltaX + lightUp.x * deltaY;
				centerY += lightRight.y * deltaX + lightUp.y * deltaY;
				centerZ += lightRight.z * deltaX + lightUp.z * deltaY;
			}
		}
		let lightDistance = radius * 2;
		if (
			sceneBounds &&
			Number.isFinite(sceneBounds.radius) &&
			sceneBounds.radius > 0
		) {
			const dx = sceneBounds.center.x - centerX;
			const dy = sceneBounds.center.y - centerY;
			const dz = sceneBounds.center.z - centerZ;
			const upstreamDistance =
				-(dx * lightDir.x + dy * lightDir.y + dz * lightDir.z) +
				sceneBounds.radius;
			if (Number.isFinite(upstreamDistance) && upstreamDistance > 0) {
				lightDistance = Math.max(
					lightDistance,
					upstreamDistance + MIN_SHADOW_NEAR + SHADOW_NEAR_FAR_GAP
				);
			}
		}
		const lightPosition = {
			x: centerX - lightDir.x * lightDistance,
			y: centerY - lightDir.y * lightDistance,
			z: centerZ - lightDir.z * lightDistance,
		};
		const center = { x: centerX, y: centerY, z: centerZ };
		const view = Matrix4.lookAt(
			lightPosition,
			center,
			SingleShadowMap.chooseUpVector(lightDir)
		);

		let minX = Infinity;
		let maxX = -Infinity;
		let minY = Infinity;
		let maxY = -Infinity;
		let minZ = Infinity;
		let maxZ = -Infinity;
		for (const corner of cascadeCorners) {
			const lightSpace = Matrix4.transformPoint(view, corner);
			minX = Math.min(minX, lightSpace.x);
			maxX = Math.max(maxX, lightSpace.x);
			minY = Math.min(minY, lightSpace.y);
			maxY = Math.max(maxY, lightSpace.y);
			minZ = Math.min(minZ, lightSpace.z);
			maxZ = Math.max(maxZ, lightSpace.z);
		}
		if (
			sceneBounds &&
			Number.isFinite(sceneBounds.radius) &&
			sceneBounds.radius > 0
		) {
			const lightSpaceCenter = Matrix4.transformPoint(view, sceneBounds.center);
			minZ = Math.min(minZ, lightSpaceCenter.z - sceneBounds.radius);
			maxZ = Math.max(maxZ, lightSpaceCenter.z + sceneBounds.radius);
		}

		let spanForPadding = 0;
		if (stabilize) {
			const halfSpan = radius;
			const span = Math.max(0.001, halfSpan * 2);
			minX = -halfSpan;
			maxX = halfSpan;
			minY = -halfSpan;
			maxY = halfSpan;
			spanForPadding = span;
		} else {
			const spanX = Math.max(0.001, maxX - minX);
			const spanY = Math.max(0.001, maxY - minY);
			const span = Math.max(spanX, spanY);
			const halfSpan = span * 0.5;
			const centerLSX = (minX + maxX) * 0.5;
			const centerLSY = (minY + maxY) * 0.5;
			minX = centerLSX - halfSpan;
			maxX = centerLSX + halfSpan;
			minY = centerLSY - halfSpan;
			maxY = centerLSY + halfSpan;
			spanForPadding = span;
		}

		const depthPadding = Math.max(10, spanForPadding * 0.5);
		const near = Math.max(MIN_SHADOW_NEAR, -maxZ - depthPadding);
		const far = Math.max(near + SHADOW_NEAR_FAR_GAP, -minZ + depthPadding);
		const projection = Matrix4.ortho(minX, maxX, minY, maxY, near, far);

		return {
			view,
			projection,
			lightDir,
			splitNear,
			splitFar,
		};
	}
}

function resolveFinite(value: unknown, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return value;
}

function clampCascadeCount(value: number): number {
	return Math.max(1, Math.min(4, Math.floor(value)));
}

export { DEFAULT_CASCADE_COUNTS };
