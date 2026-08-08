import type { IVector3 } from "../../maths/types";
import type { SceneBounds } from "../../lights/shadows";
import type { DrawPacket } from "../types";

interface ShadowBoundsCamera {
	isSphereInFrustum?: (center: IVector3, radius: number) => boolean;
	getWorldPosition?: (target?: IVector3) => IVector3;
	position?: IVector3;
}

const _cameraPosition: IVector3 = { x: 0, y: 0, z: 0 };

export function resolveShadowCasterBounds(
	shadowCasterPackets: DrawPacket[],
	fallbackBounds: SceneBounds,
	camera?: ShadowBoundsCamera | null,
): SceneBounds {
	const packets = resolveBoundsPackets(shadowCasterPackets, camera);
	if (packets.length === 0 || !Number.isFinite(fallbackBounds.radius) ||
		fallbackBounds.radius <= 1e-6) return fallbackBounds;
	let minX = Infinity;
	let minY = Infinity;
	let minZ = Infinity;
	let maxX = -Infinity;
	let maxY = -Infinity;
	let maxZ = -Infinity;
	for (const packet of packets) {
		const { center } = packet.worldBounds;
		const radius = Math.max(0, packet.worldBounds.radius);
		minX = Math.min(minX, center.x - radius);
		minY = Math.min(minY, center.y - radius);
		minZ = Math.min(minZ, center.z - radius);
		maxX = Math.max(maxX, center.x + radius);
		maxY = Math.max(maxY, center.y + radius);
		maxZ = Math.max(maxZ, center.z + radius);
	}
	const radius = Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) * 0.5;
	if (!Number.isFinite(radius) || radius <= 1e-6) return fallbackBounds;
	return {
		center: { x: (minX + maxX) * 0.5, y: (minY + maxY) * 0.5, z: (minZ + maxZ) * 0.5 },
		radius,
	};
}

function resolveBoundsPackets(
	packets: DrawPacket[],
	camera?: ShadowBoundsCamera | null,
): DrawPacket[] {
	if (!camera?.isSphereInFrustum || packets.length === 0) return packets;
	const position = camera.getWorldPosition?.(_cameraPosition) ?? camera.position ?? null;
	const visible: DrawPacket[] = [];
	let nearest: DrawPacket | null = null;
	let nearestDistance = Infinity;
	for (const packet of packets) {
		const { center } = packet.worldBounds;
		if (camera.isSphereInFrustum(center, Math.max(0, packet.worldBounds.radius))) {
			visible.push(packet);
			continue;
		}
		if (!position) continue;
		const distance = Math.hypot(center.x - position.x, center.y - position.y, center.z - position.z);
		if (distance < nearestDistance) {
			nearest = packet;
			nearestDistance = distance;
		}
	}
	return visible.length > 0 ? visible : nearest ? [nearest] : packets;
}
