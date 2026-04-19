import { Matrix4 } from "../maths/Matrix4";
import { Vector3 } from "../maths/Vector3";
import type { IVector3 } from "../maths/types";
import {
	MIN_SHADOW_FAR,
	MIN_SHADOW_NEAR,
	SHADOW_NEAR_FAR_GAP,
} from "../lights/constants";
import {
	LightType,
	type DirectionalLight,
	type ShadowCastingLight,
	type SpotLight,
} from "../lights";
import type {
	CSMShadowConfig,
	ShadowConfig,
	ShadowRenderSet,
	ShadowStrategyType,
} from "../lights/ShadowMapping";

export interface SceneBounds {
	center: IVector3;
	radius: number;
}

export interface ShadowStrategyCamera {
	type?: string;
	near?: number;
	far?: number;
	fov?: number;
	aspectRatio?: number;
	up?: IVector3;
	position?: IVector3;
	getWorldPosition?: (target?: IVector3) => IVector3;
	getWorldDirection?: (localDirection: IVector3, target?: IVector3) => IVector3;
}

export interface ShadowSliceDescriptor {
	view: Matrix4;
	projection: Matrix4;
	lightDir: IVector3;
	splitNear: number;
	splitFar: number;
}

export interface ShadowStrategyBuildContext {
	light: ShadowCastingLight;
	renderSet: ShadowRenderSet;
	config: ShadowConfig;
	sceneBounds: SceneBounds;
	camera?: ShadowStrategyCamera | null;
}

export interface IShadowStrategyProvider {
	readonly type: ShadowStrategyType;
	supports(light: ShadowCastingLight): boolean;
	build(context: ShadowStrategyBuildContext): ShadowSliceDescriptor[];
}

export interface ShadowBackendCapabilities {
	backendKey: string;
	supportsSingleMap: boolean;
	supportsDirectionalCSM: boolean;
	maxCsmDirectionalLights: number;
}

const _tmpCamPos = { x: 0, y: 0, z: 0 };
const _tmpCamForward = { x: 0, y: 0, z: -1 };
const _tmpCamUp = { x: 0, y: 1, z: 0 };

function resolveCameraPosition(camera: ShadowStrategyCamera | null | undefined): IVector3 | null {
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

function resolveCameraDirection(
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

function resolveDirectionalWorldDirection(light: DirectionalLight): IVector3 {
	const transformed = Matrix4.transformDirection(light.worldMatrix, light.direction);
	const normalized = Vector3.normalize(transformed);
	return { x: normalized.x, y: normalized.y, z: normalized.z };
}

function chooseUpVector(direction: IVector3): IVector3 {
	if (Math.abs(direction.y) < 0.999) {
		return { x: 0, y: 1, z: 0 };
	}
	return { x: 0, y: 0, z: 1 };
}

class SingleMapShadowStrategyProvider implements IShadowStrategyProvider {
	public readonly type: ShadowStrategyType = "single-map";

	public supports(_light: ShadowCastingLight): boolean {
		return true;
	}

	public build(context: ShadowStrategyBuildContext): ShadowSliceDescriptor[] {
		switch (context.light.type) {
			case LightType.Directional:
				return [this._buildDirectional(context.light as DirectionalLight, context.sceneBounds)];
			case LightType.Spot:
				return [this._buildSpot(context.light as SpotLight, context.sceneBounds)];
			default:
				return [];
		}
	}

	private _buildDirectional(
		light: DirectionalLight,
		sceneBounds: SceneBounds
	): ShadowSliceDescriptor {
		const direction = resolveDirectionalWorldDirection(light);
		const { center, radius } = sceneBounds;
		const shadowDistance = radius * 1.5;
		const lightPos = Vector3.sub(center, Vector3.scale(direction, shadowDistance));
		const view = Matrix4.lookAt(lightPos, center, chooseUpVector(direction));
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

	private _buildSpot(
		light: SpotLight,
		sceneBounds: SceneBounds
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
		const view = Matrix4.lookAt(position, target, chooseUpVector(direction));

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
			1,
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
}

function calculatePracticalCascadeSplits(
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

function computePerspectiveFrustumCorners(
	camera: ShadowStrategyCamera,
	splitNear: number,
	splitFar: number,
	output: IVector3[]
): boolean {
	const cameraPosition = resolveCameraPosition(camera);
	if (!cameraPosition) {
		return false;
	}

	const fov = typeof camera.fov === "number" && Number.isFinite(camera.fov) ?
		camera.fov
	: 	60;
	const aspect =
		typeof camera.aspectRatio === "number" &&
		Number.isFinite(camera.aspectRatio) &&
		camera.aspectRatio > 0 ?
			camera.aspectRatio
		: 	16 / 9;

	const forward = resolveCameraDirection(
		camera,
		{ x: 0, y: 0, z: -1 },
		{ x: 0, y: 0, z: -1 },
		_tmpCamForward
	);
	const upVector = resolveCameraDirection(
		camera,
		camera.up ?? { x: 0, y: 1, z: 0 },
		camera.up ?? { x: 0, y: 1, z: 0 },
		_tmpCamUp
	);
	const right = Vector3.normalize(Vector3.cross(forward, upVector));
	const up = Vector3.normalize(Vector3.cross(right, forward));

	const halfTan = Math.tan((fov * Math.PI) / 360);
	const nearHalfHeight = splitNear * halfTan;
	const nearHalfWidth = nearHalfHeight * aspect;
	const farHalfHeight = splitFar * halfTan;
	const farHalfWidth = farHalfHeight * aspect;

	const nearCenter = Vector3.add(cameraPosition, Vector3.scale(forward, splitNear));
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

function buildDirectionalCascadeSlice(
	lightDir: IVector3,
	cascadeCorners: IVector3[],
	shadowMapSize: number,
	stabilize: boolean,
	splitNear: number,
	splitFar: number
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
		maxRadiusSquared = Math.max(maxRadiusSquared, dx * dx + dy * dy + dz * dz);
	}
	const radius = Math.max(0.001, Math.sqrt(maxRadiusSquared));
	const lightPosition = {
		x: centerX - lightDir.x * (radius * 2),
		y: centerY - lightDir.y * (radius * 2),
		z: centerZ - lightDir.z * (radius * 2),
	};
	const center = { x: centerX, y: centerY, z: centerZ };
	const view = Matrix4.lookAt(lightPosition, center, chooseUpVector(lightDir));

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

	let spanX = Math.max(0.001, maxX - minX);
	let spanY = Math.max(0.001, maxY - minY);
	const span = Math.max(spanX, spanY);
	const halfSpan = span * 0.5;
	let centerLSX = (minX + maxX) * 0.5;
	let centerLSY = (minY + maxY) * 0.5;
	if (stabilize) {
		const texelSize = span / Math.max(1, shadowMapSize);
		centerLSX = Math.floor(centerLSX / texelSize) * texelSize;
		centerLSY = Math.floor(centerLSY / texelSize) * texelSize;
	}
	minX = centerLSX - halfSpan;
	maxX = centerLSX + halfSpan;
	minY = centerLSY - halfSpan;
	maxY = centerLSY + halfSpan;
	spanX = span;
	spanY = span;

	const depthPadding = Math.max(10, span * 0.5);
	const near = Math.max(MIN_SHADOW_NEAR, -maxZ - depthPadding);
	const far = Math.max(near + SHADOW_NEAR_FAR_GAP, -minZ + depthPadding);
	const projection = Matrix4.ortho(minX, maxX, minY, maxY, near, far);

	return {
		view,
		projection,
		lightDir: lightDir,
		splitNear,
		splitFar,
	};
}

class CSMShadowStrategyProvider implements IShadowStrategyProvider {
	public readonly type: ShadowStrategyType = "csm";

	public supports(light: ShadowCastingLight): boolean {
		return light.type === LightType.Directional;
	}

	public build(context: ShadowStrategyBuildContext): ShadowSliceDescriptor[] {
		if (context.light.type !== LightType.Directional) {
			return [];
		}
		if (!context.camera) {
			return [];
		}

		const config = context.config as CSMShadowConfig;
		const cameraNear = Math.max(0.01, context.camera.near ?? 0.1);
		const cameraFarLimit =
			typeof context.camera.far === "number" && Number.isFinite(context.camera.far) ?
				Math.max(cameraNear + SHADOW_NEAR_FAR_GAP, context.camera.far)
			: 	Math.max(cameraNear + SHADOW_NEAR_FAR_GAP, context.sceneBounds.radius * 4);
		const cameraFar =
			typeof config.maxDistance === "number" && Number.isFinite(config.maxDistance) ?
				Math.max(cameraNear + SHADOW_NEAR_FAR_GAP, Math.min(cameraFarLimit, config.maxDistance))
			: 	cameraFarLimit;

		const cascadeCount = Math.max(2, Math.min(4, config.cascadeCount ?? 4));
		const lambda = Math.max(0, Math.min(1, config.lambda ?? 0.65));
		const splits = calculatePracticalCascadeSplits(
			cascadeCount,
			cameraNear,
			cameraFar,
			lambda
		);
		const direction = resolveDirectionalWorldDirection(
			context.light as DirectionalLight
		);
		const stabilize = config.stabilize !== false;
		const shadowMapSize =
			context.renderSet.slices[0]?.shadowMap.size ?? Math.max(1, Math.floor(context.renderSet.size / 2));
		const cornersBuffer: IVector3[] = new Array(8);
		const slices: ShadowSliceDescriptor[] = [];

		for (let index = 0; index < cascadeCount; index++) {
			const splitNear = splits[index];
			const splitFar = splits[index + 1];
			const hasCorners = computePerspectiveFrustumCorners(
				context.camera,
				splitNear,
				splitFar,
				cornersBuffer
			);
			if (!hasCorners) {
				return [];
			}

			slices.push(
				buildDirectionalCascadeSlice(
					direction,
					cornersBuffer,
					shadowMapSize,
					stabilize,
					splitNear,
					splitFar
				)
			);
		}

		return slices;
	}
}

export class ShadowStrategyRegistry {
	private readonly _providers = new Map<ShadowStrategyType, IShadowStrategyProvider>();

	public register(provider: IShadowStrategyProvider): this {
		this._providers.set(provider.type, provider);
		return this;
	}

	public get(type: ShadowStrategyType): IShadowStrategyProvider | null {
		return this._providers.get(type) ?? null;
	}

	public build(context: ShadowStrategyBuildContext): ShadowSliceDescriptor[] {
		const provider = this.get(context.config.strategy);
		if (!provider || !provider.supports(context.light)) {
			return [];
		}
		return provider.build(context);
	}
}

const _defaultRegistry = new ShadowStrategyRegistry()
	.register(new SingleMapShadowStrategyProvider())
	.register(new CSMShadowStrategyProvider());

export function getDefaultShadowStrategyRegistry(): ShadowStrategyRegistry {
	return _defaultRegistry;
}

function resolveShadowPriority(light: ShadowCastingLight): number {
	if (!light.shadow) {
		return 0;
	}
	const priority = (light.shadow as { priority?: unknown }).priority;
	if (typeof priority !== "number" || !Number.isFinite(priority)) {
		return 0;
	}
	return priority;
}

export function selectCSMDirectionalLights(
	lights: ShadowCastingLight[],
	maxCount: number
): Set<ShadowCastingLight> {
	const requested = lights.filter(
		(light) => light.type === LightType.Directional && light.shadow?.strategy === "csm"
	);
	if (requested.length <= 0 || maxCount <= 0) {
		return new Set();
	}

	requested.sort((left, right) => {
		const priorityDelta = resolveShadowPriority(right) - resolveShadowPriority(left);
		if (priorityDelta !== 0) {
			return priorityDelta;
		}
		const intensityDelta = right.intensity - left.intensity;
		if (intensityDelta !== 0) {
			return intensityDelta;
		}
		return left.id.localeCompare(right.id);
	});

	return new Set(requested.slice(0, maxCount));
}
