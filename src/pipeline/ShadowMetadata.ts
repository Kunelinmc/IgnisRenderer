import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import type { ShadowCastingLight } from "../lights";
import { ShadowMap } from "../lights/ShadowMapping";
import type { DrawPacket } from "./types";

interface SceneBounds {
	center: IVector3;
	radius: number;
}

interface ShadowBoundsCamera {
	isSphereInFrustum?: (center: IVector3, radius: number) => boolean;
	getWorldPosition?: (target?: IVector3) => IVector3;
	position?: IVector3;
}

const _tmpShadowBoundsCameraPosition: IVector3 = { x: 0, y: 0, z: 0 };

function hasFiniteRadius(bounds: SceneBounds): boolean {
	return Number.isFinite(bounds.radius) && bounds.radius > 1e-6;
}

function resolveCameraPosition(camera: ShadowBoundsCamera): IVector3 | null {
	if (typeof camera.getWorldPosition === "function") {
		return camera.getWorldPosition(_tmpShadowBoundsCameraPosition);
	}

	const { position } = camera;
	if (!position) return null;
	if (
		!Number.isFinite(position.x) ||
		!Number.isFinite(position.y) ||
		!Number.isFinite(position.z)
	) {
		return null;
	}

	_tmpShadowBoundsCameraPosition.x = position.x;
	_tmpShadowBoundsCameraPosition.y = position.y;
	_tmpShadowBoundsCameraPosition.z = position.z;
	return _tmpShadowBoundsCameraPosition;
}

function resolveShadowBoundsPackets(
	shadowCasterPackets: DrawPacket[],
	camera?: ShadowBoundsCamera | null
): DrawPacket[] {
	if (!camera || shadowCasterPackets.length === 0) {
		return shadowCasterPackets;
	}
	if (typeof camera.isSphereInFrustum !== "function") {
		return shadowCasterPackets;
	}

	const visiblePackets: DrawPacket[] = [];
	const cameraPosition = resolveCameraPosition(camera);
	let nearestPacket: DrawPacket | null = null;
	let nearestDistanceSquared = Infinity;

	for (const packet of shadowCasterPackets) {
		const center = packet.worldBounds.center;
		const radius = Math.max(0, packet.worldBounds.radius);
		const inFrustum = camera.isSphereInFrustum(center, radius);

		if (inFrustum) {
			visiblePackets.push(packet);
			continue;
		}

		if (!cameraPosition) continue;

		const dx = center.x - cameraPosition.x;
		const dy = center.y - cameraPosition.y;
		const dz = center.z - cameraPosition.z;
		const distanceSquared = dx * dx + dy * dy + dz * dz;
		if (distanceSquared < nearestDistanceSquared) {
			nearestDistanceSquared = distanceSquared;
			nearestPacket = packet;
		}
	}

	if (visiblePackets.length > 0) {
		return visiblePackets;
	}

	if (nearestPacket) {
		return [nearestPacket];
	}

	return shadowCasterPackets;
}

export function resolveShadowCasterBounds(
	shadowCasterPackets: DrawPacket[],
	fallbackBounds: SceneBounds,
	camera?: ShadowBoundsCamera | null
): SceneBounds {
	const packetsForBounds = resolveShadowBoundsPackets(
		shadowCasterPackets,
		camera
	);

	if (packetsForBounds.length === 0 || !hasFiniteRadius(fallbackBounds)) {
		return fallbackBounds;
	}

	let minX = Infinity;
	let minY = Infinity;
	let minZ = Infinity;
	let maxX = -Infinity;
	let maxY = -Infinity;
	let maxZ = -Infinity;

	for (const packet of packetsForBounds) {
		const center = packet.worldBounds.center;
		const radius = Math.max(0, packet.worldBounds.radius);
		minX = Math.min(minX, center.x - radius);
		minY = Math.min(minY, center.y - radius);
		minZ = Math.min(minZ, center.z - radius);
		maxX = Math.max(maxX, center.x + radius);
		maxY = Math.max(maxY, center.y + radius);
		maxZ = Math.max(maxZ, center.z + radius);
	}

	if (!Number.isFinite(minX) || !Number.isFinite(maxX)) {
		return fallbackBounds;
	}

	const center = {
		x: (minX + maxX) * 0.5,
		y: (minY + maxY) * 0.5,
		z: (minZ + maxZ) * 0.5,
	};
	const sizeX = maxX - minX;
	const sizeY = maxY - minY;
	const sizeZ = maxZ - minZ;
	const radius = Math.sqrt(sizeX * sizeX + sizeY * sizeY + sizeZ * sizeZ) * 0.5;

	if (!Number.isFinite(radius) || radius <= 1e-6) {
		return fallbackBounds;
	}

	return { center, radius };
}

function resetShadowMapMetadata(shadowMap: ShadowMap): void {
	shadowMap.viewMatrix = null;
	shadowMap.projectionMatrix = null;
	shadowMap.viewProjectionMatrix = null;
	shadowMap.latestLightDir = { x: 0, y: -1, z: 0 };
	shadowMap.stabilizedBoundsRadius = null;
}

const SHADOW_RADIUS_SHRINK_BLEND = 0.12;

function resolveStabilizedShadowRadius(
	shadowMap: ShadowMap,
	radius: number
): number {
	const safeRadius = Number.isFinite(radius) ? Math.max(radius, 1e-6) : 1e-6;
	const previousRadius = shadowMap.stabilizedBoundsRadius;
	if (!Number.isFinite(previousRadius) || previousRadius === null || previousRadius <= 1e-6) {
		shadowMap.stabilizedBoundsRadius = safeRadius;
		return safeRadius;
	}

	if (safeRadius >= previousRadius) {
		shadowMap.stabilizedBoundsRadius = safeRadius;
		return safeRadius;
	}

	const stabilizedRadius =
		previousRadius + (safeRadius - previousRadius) * SHADOW_RADIUS_SHRINK_BLEND;
	const clampedRadius = Math.max(safeRadius, stabilizedRadius);
	shadowMap.stabilizedBoundsRadius = clampedRadius;
	return clampedRadius;
}

export function syncShadowMapRegistry(
	shadowMaps: Map<ShadowCastingLight, ShadowMap>,
	activeLights: ShadowCastingLight[]
): void {
	for (const [light] of shadowMaps) {
		if (!activeLights.includes(light)) {
			shadowMaps.delete(light);
		}
	}

	for (const light of activeLights) {
		if (!shadowMaps.has(light)) {
			shadowMaps.set(light, new ShadowMap());
		}
	}
}

export function updateShadowMapMetadata(
	shadowMap: ShadowMap,
	light: ShadowCastingLight,
	sceneBounds: SceneBounds
): void {
	if (!light.shadow) {
		resetShadowMapMetadata(shadowMap);
		return;
	}

	const stabilizedSceneBounds: SceneBounds = {
		center: sceneBounds.center,
		radius: resolveStabilizedShadowRadius(shadowMap, sceneBounds.radius),
	};

	const config = light.shadow.setupShadowCamera({
		sceneBounds: stabilizedSceneBounds,
		worldMatrix: light.worldMatrix,
	});
	if (!config) {
		resetShadowMapMetadata(shadowMap);
		return;
	}

	shadowMap.viewMatrix = config.view;
	shadowMap.projectionMatrix = config.projection;
	shadowMap.latestLightDir = config.lightDir;
	shadowMap.viewProjectionMatrix = Matrix4.multiply(
		config.projection,
		config.view
	);
}
