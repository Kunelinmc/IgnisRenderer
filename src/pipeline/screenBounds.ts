import type { Camera } from "../cameras/Camera";
import type { Matrix4 } from "../maths/Matrix4";
import type { BoundingSphere } from "../core/types";
import type { DirtyRect } from "./incremental";

/**
 * Computes a conservative screen-space rect for a world-space bounding sphere.
 *
 * @param packet Object carrying world-space bounds.
 * @param camera Camera used to project the bounds.
 * @param viewportWidth Width of the target viewport in pixels.
 * @param viewportHeight Height of the target viewport in pixels.
 * @returns Clamped pixel rect when the bounds project into the viewport.
 * @sideEffects None.
 */
export function computePacketScreenRect(
	packet: { worldBounds: BoundingSphere },
	camera: Camera,
	viewportWidth: number,
	viewportHeight: number
): DirtyRect | null {
	const worldCenter = packet.worldBounds.center;
	const worldRadius = packet.worldBounds.radius;
	if (
		!Number.isFinite(worldCenter.x) ||
		!Number.isFinite(worldCenter.y) ||
		!Number.isFinite(worldCenter.z) ||
		!Number.isFinite(worldRadius) ||
		worldRadius <= 0
	) {
		return null;
	}

	const center = projectToScreen(
		camera.viewProjectionMatrix,
		worldCenter.x,
		worldCenter.y,
		worldCenter.z,
		viewportWidth,
		viewportHeight
	);
	if (!center) {
		return null;
	}

	const right = camera.getWorldDirection(
		{ x: 1, y: 0, z: 0 },
		{ x: 0, y: 0, z: 0 }
	);
	const up = camera.getWorldDirection(
		{ x: 0, y: 1, z: 0 },
		{ x: 0, y: 0, z: 0 }
	);

	const rightPoint = projectToScreen(
		camera.viewProjectionMatrix,
		worldCenter.x + right.x * worldRadius,
		worldCenter.y + right.y * worldRadius,
		worldCenter.z + right.z * worldRadius,
		viewportWidth,
		viewportHeight
	);
	const upPoint = projectToScreen(
		camera.viewProjectionMatrix,
		worldCenter.x + up.x * worldRadius,
		worldCenter.y + up.y * worldRadius,
		worldCenter.z + up.z * worldRadius,
		viewportWidth,
		viewportHeight
	);
	if (!rightPoint || !upPoint) {
		return null;
	}

	const radiusX = Math.max(1, Math.abs(rightPoint.x - center.x));
	const radiusY = Math.max(1, Math.abs(upPoint.y - center.y));
	const minX = center.x - radiusX - 1;
	const minY = center.y - radiusY - 1;
	const maxX = center.x + radiusX + 1;
	const maxY = center.y + radiusY + 1;

	const x = Math.max(0, Math.floor(minX));
	const y = Math.max(0, Math.floor(minY));
	const width = Math.min(viewportWidth, Math.ceil(maxX)) - x;
	const height = Math.min(viewportHeight, Math.ceil(maxY)) - y;
	if (width <= 0 || height <= 0) {
		return null;
	}
	return {
		x,
		y,
		width,
		height,
	};
}

function projectToScreen(
	viewProjection: Matrix4,
	x: number,
	y: number,
	z: number,
	viewportWidth: number,
	viewportHeight: number
): { x: number; y: number; depth: number } | null {
	const matrix = viewProjection.elements;
	const clipX =
		matrix[0][0] * x +
		matrix[0][1] * y +
		matrix[0][2] * z +
		matrix[0][3];
	const clipY =
		matrix[1][0] * x +
		matrix[1][1] * y +
		matrix[1][2] * z +
		matrix[1][3];
	const clipZ =
		matrix[2][0] * x +
		matrix[2][1] * y +
		matrix[2][2] * z +
		matrix[2][3];
	const clipW =
		matrix[3][0] * x +
		matrix[3][1] * y +
		matrix[3][2] * z +
		matrix[3][3];

	if (!Number.isFinite(clipW) || Math.abs(clipW) < 1e-8) {
		return null;
	}

	const invW = 1 / clipW;
	const ndcX = clipX * invW;
	const ndcY = clipY * invW;
	const ndcZ = clipZ * invW;
	if (!Number.isFinite(ndcX) || !Number.isFinite(ndcY) || !Number.isFinite(ndcZ)) {
		return null;
	}

	return {
		x: (ndcX * 0.5 + 0.5) * viewportWidth,
		y: (0.5 - ndcY * 0.5) * viewportHeight,
		depth: ndcZ,
	};
}
