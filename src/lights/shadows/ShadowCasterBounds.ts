import type { IVector3 } from "../../maths/types";
import type { DrawSubmission } from "../../pipeline/types";
import type { SceneBounds } from "./types";

interface ShadowBoundsCamera {
	isSphereInFrustum?: (center: IVector3, radius: number) => boolean;
	getWorldPosition?: (target?: IVector3) => IVector3;
	position?: IVector3;
}

const _cameraPosition: IVector3 = { x: 0, y: 0, z: 0 };

export function resolveShadowCasterBounds(
	shadowCasterSubmissions: readonly DrawSubmission[],
	fallbackBounds: SceneBounds,
	camera?: ShadowBoundsCamera | null,
): SceneBounds {
	const submissions = resolveBoundsSubmissions(shadowCasterSubmissions, camera);
	if (submissions.length === 0 || !Number.isFinite(fallbackBounds.radius) ||
		fallbackBounds.radius <= 1e-6) return fallbackBounds;
	let minX = Infinity;
	let minY = Infinity;
	let minZ = Infinity;
	let maxX = -Infinity;
	let maxY = -Infinity;
	let maxZ = -Infinity;
	for (const submission of submissions) {
		const { center } = submission.worldBounds;
		const radius = Math.max(0, submission.worldBounds.radius);
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

function resolveBoundsSubmissions(
	submissions: readonly DrawSubmission[],
	camera?: ShadowBoundsCamera | null,
): readonly DrawSubmission[] {
	if (!camera?.isSphereInFrustum || submissions.length === 0) return submissions;
	const position = camera.getWorldPosition?.(_cameraPosition) ?? camera.position ?? null;
	const visible: DrawSubmission[] = [];
	let nearest: DrawSubmission | null = null;
	let nearestDistance = Infinity;
	for (const submission of submissions) {
		const { center } = submission.worldBounds;
		if (
			camera.isSphereInFrustum(
				center,
				Math.max(0, submission.worldBounds.radius),
			)
		) {
			visible.push(submission);
			continue;
		}
		if (!position) continue;
		const distance = Math.hypot(center.x - position.x, center.y - position.y, center.z - position.z);
		if (distance < nearestDistance) {
			nearest = submission;
			nearestDistance = distance;
		}
	}
	return visible.length > 0 ? visible : nearest ? [nearest] : submissions;
}
