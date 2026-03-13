import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import type { ShadowCastingLight } from "../lights";
import { ShadowMap } from "../lights/ShadowMapping";
import type { DrawPacket } from "./types";

interface SceneBounds {
	center: IVector3;
	radius: number;
}

function hasFiniteRadius(bounds: SceneBounds): boolean {
	return Number.isFinite(bounds.radius) && bounds.radius > 1e-6;
}

export function resolveShadowCasterBounds(
	shadowCasterPackets: DrawPacket[],
	fallbackBounds: SceneBounds
): SceneBounds {
	if (shadowCasterPackets.length === 0 || !hasFiniteRadius(fallbackBounds)) {
		return fallbackBounds;
	}

	let minX = Infinity;
	let minY = Infinity;
	let minZ = Infinity;
	let maxX = -Infinity;
	let maxY = -Infinity;
	let maxZ = -Infinity;

	for (const packet of shadowCasterPackets) {
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

	const config = light.shadow.setupShadowCamera({
		sceneBounds,
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
