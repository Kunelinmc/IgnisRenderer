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

	const right = camera.getWorldDirection(
		{ x: 1, y: 0, z: 0 },
		{ x: 0, y: 0, z: 0 }
	);
	const up = camera.getWorldDirection(
		{ x: 0, y: 1, z: 0 },
		{ x: 0, y: 0, z: 0 }
	);
	const forward = camera.getWorldDirection(
		{ x: 0, y: 0, z: -1 },
		{ x: 0, y: 0, z: 0 }
	);
	let minX = Number.POSITIVE_INFINITY;
	let minY = Number.POSITIVE_INFINITY;
	let maxX = Number.NEGATIVE_INFINITY;
	let maxY = Number.NEGATIVE_INFINITY;
	let positiveWCount = 0;
	let nonPositiveWCount = 0;

	// A camera-aligned cube conservatively encloses the sphere. With positive
	// clip-W throughout the cube, projected extrema occur at its vertices.
	for (const sideX of [-1, 1]) {
		for (const sideY of [-1, 1]) {
			for (const sideZ of [-1, 1]) {
				const projected = projectToScreen(
					camera.viewProjectionMatrix,
					worldCenter.x +
						(right.x * sideX + up.x * sideY + forward.x * sideZ) *
							worldRadius,
					worldCenter.y +
						(right.y * sideX + up.y * sideY + forward.y * sideZ) *
							worldRadius,
					worldCenter.z +
						(right.z * sideX + up.z * sideY + forward.z * sideZ) *
							worldRadius,
					viewportWidth,
					viewportHeight
				);
				if (!projected) {
					return makeFullViewportRect(viewportWidth, viewportHeight);
				}
				if (projected.clipW <= 1e-8) {
					nonPositiveWCount++;
					continue;
				}
				positiveWCount++;
				minX = Math.min(minX, projected.x);
				minY = Math.min(minY, projected.y);
				maxX = Math.max(maxX, projected.x);
				maxY = Math.max(maxY, projected.y);
			}
		}
	}
	if (positiveWCount === 0) return null;
	if (nonPositiveWCount > 0) {
		return makeFullViewportRect(viewportWidth, viewportHeight);
	}
	minX -= 1;
	minY -= 1;
	maxX += 1;
	maxY += 1;

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
): { x: number; y: number; depth: number; clipW: number } | null {
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

	if (!Number.isFinite(clipW) || Math.abs(clipW) < 1e-12) {
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
		clipW,
	};
}

function makeFullViewportRect(
	viewportWidth: number,
	viewportHeight: number
): DirtyRect {
	return {
		x: 0,
		y: 0,
		width: Math.max(1, Math.floor(viewportWidth)),
		height: Math.max(1, Math.floor(viewportHeight)),
	};
}
