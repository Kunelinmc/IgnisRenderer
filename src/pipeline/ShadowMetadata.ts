import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import type { ShadowCastingLight } from "../lights";
import {
	createShadowRenderSet,
	ensureShadowRenderSetMatchesConfig,
	normalizeShadowConfig,
	shadowConfigSignature,
	ShadowMap,
	type ShadowConfig,
	type ShadowRenderSet,
} from "../lights/ShadowMapping";
import type { DrawPacket } from "./types";
import {
	getDefaultShadowStrategyRegistry,
	type ShadowBackendCapabilities,
	type ShadowStrategyCamera,
	type SceneBounds,
} from "./ShadowStrategyRegistry";

interface ShadowBoundsCamera {
	isSphereInFrustum?: (center: IVector3, radius: number) => boolean;
	getWorldPosition?: (target?: IVector3) => IVector3;
	position?: IVector3;
}

export interface ShadowMetadataUpdateOptions {
	camera?: ShadowStrategyCamera | null;
	backendCapabilities?: ShadowBackendCapabilities;
	allowCSMDirectionalLights?: Set<ShadowCastingLight> | null;
	onWarning?: (key: string, message: string) => void;
}

const _tmpShadowBoundsCameraPosition: IVector3 = { x: 0, y: 0, z: 0 };
const _strategyRegistry = getDefaultShadowStrategyRegistry();

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

function resetRenderSetMetadata(renderSet: ShadowRenderSet): void {
	for (const slice of renderSet.slices) {
		resetShadowMapMetadata(slice.shadowMap);
		slice.splitNear = 0;
		slice.splitFar = 0;
		slice.atlasRect = null;
	}
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

function resolveEffectiveConfig(
	light: ShadowCastingLight,
	renderSet: ShadowRenderSet,
	options?: ShadowMetadataUpdateOptions
): ShadowConfig {
	const requested = normalizeShadowConfig(light.shadow);
	const capabilities = options?.backendCapabilities;
	let effective = requested;

	if (requested.strategy === "csm") {
		const supportsCSM =
			!capabilities || capabilities.supportsDirectionalCSM === true;
		const needsDirectionalBudgetSelection = light.type === "directional";
		const selectedForCSM =
			!needsDirectionalBudgetSelection ||
			!options?.allowCSMDirectionalLights ||
			options.allowCSMDirectionalLights.has(light);

		if (!supportsCSM || !selectedForCSM) {
			effective = {
				strategy: "single-map",
				size: requested.size,
				params: requested.params,
				priority: requested.priority,
			};
			if (typeof options?.onWarning === "function") {
				const key = `shadow-strategy-fallback-${capabilities?.backendKey ?? "generic"}-${light.id}`;
				options.onWarning(
					key,
					`Light ${light.id} requested csm shadows but backend ${capabilities?.backendKey ?? "generic"} uses single-map fallback.`
				);
			}
		}
	}

	renderSet.requestedStrategyType = requested.strategy;
	renderSet.effectiveStrategyType = effective.strategy;
	return effective;
}

function reconfigureRenderSet(renderSet: ShadowRenderSet, config: ShadowConfig): void {
	const resolved = normalizeShadowConfig(config);
	const signature = shadowConfigSignature(resolved);
	if (renderSet.configSignature === signature && renderSet.effectiveStrategyType === resolved.strategy) {
		renderSet.resolvedConfig = resolved;
		renderSet.size = resolved.size ?? renderSet.size;
		return;
	}

	const rebuilt = createShadowRenderSet(resolved);
	renderSet.resolvedConfig = rebuilt.resolvedConfig;
	renderSet.configSignature = rebuilt.configSignature;
	renderSet.size = rebuilt.size;
	renderSet.slices = rebuilt.slices;
	renderSet.effectiveStrategyType = resolved.strategy;
	renderSet.metadataVersion++;
}

export function syncShadowMapRegistry(
	shadowMaps: Map<ShadowCastingLight, ShadowRenderSet>,
	activeLights: ShadowCastingLight[]
): void {
	for (const [light] of shadowMaps) {
		if (!activeLights.includes(light)) {
			shadowMaps.delete(light);
		}
	}

	for (const light of activeLights) {
		const existing = shadowMaps.get(light);
		if (!existing) {
			shadowMaps.set(light, createShadowRenderSet(light.shadow));
			continue;
		}

		const next = ensureShadowRenderSetMatchesConfig(existing, light.shadow);
		if (next !== existing) {
			shadowMaps.set(light, next);
		}
	}
}

export function updateShadowMapMetadata(
	renderSet: ShadowRenderSet,
	light: ShadowCastingLight,
	sceneBounds: SceneBounds,
	options?: ShadowMetadataUpdateOptions
): void {
	if (!light.shadow) {
		resetRenderSetMetadata(renderSet);
		return;
	}

	const effectiveConfig = resolveEffectiveConfig(light, renderSet, options);
	reconfigureRenderSet(renderSet, effectiveConfig);

	const stabilizedSceneBounds: SceneBounds = {
		center: sceneBounds.center,
		radius:
			renderSet.slices.length > 0 ?
				resolveStabilizedShadowRadius(
					renderSet.slices[0].shadowMap,
					sceneBounds.radius
				)
			:	sceneBounds.radius,
	};

	const descriptors = _strategyRegistry.build({
		light,
		renderSet,
		config: effectiveConfig,
		sceneBounds: stabilizedSceneBounds,
		camera: options?.camera ?? null,
	});

	if (descriptors.length <= 0) {
		resetRenderSetMetadata(renderSet);
		return;
	}

	for (let index = 0; index < renderSet.slices.length; index++) {
		const slice = renderSet.slices[index];
		const descriptor = descriptors[index];
		if (!descriptor) {
			resetShadowMapMetadata(slice.shadowMap);
			slice.splitNear = 0;
			slice.splitFar = 0;
			slice.atlasRect = null;
			continue;
		}
		slice.shadowMap.viewMatrix = descriptor.view;
		slice.shadowMap.projectionMatrix = descriptor.projection;
		slice.shadowMap.latestLightDir = descriptor.lightDir;
		slice.shadowMap.viewProjectionMatrix = Matrix4.multiply(
			descriptor.projection,
			descriptor.view
		);
		slice.splitNear = descriptor.splitNear;
		slice.splitFar = descriptor.splitFar;
		slice.atlasRect = null;
	}

	renderSet.metadataVersion++;
}
