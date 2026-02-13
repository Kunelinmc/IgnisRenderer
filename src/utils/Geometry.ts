/**
 * Geometry utility functions
 */

import { rotatePoint } from "../maths/Common";
import type { IVector3 } from "../maths/types";

export interface DepthInfo {
	avg: number;
	min: number;
	max: number;
	range: number;
}

/**
 * Calculate face depth information for sorting
 */
export function calculateFaceDepth(
	vertices: IVector3[],
	rotation: IVector3
): DepthInfo {
	let totalZ = 0;
	let minZ = Infinity;
	let maxZ = -Infinity;

	const rotatedVertices = vertices.map((vert) =>
		rotatePoint(vert, rotation.x, rotation.y, rotation.z)
	);

	for (const rotated of rotatedVertices) {
		totalZ += rotated.z;
		minZ = Math.min(minZ, rotated.z);
		maxZ = Math.max(maxZ, rotated.z);
	}

	const avgZ = totalZ / (vertices.length || 1);
	return { avg: avgZ, min: minZ, max: maxZ, range: maxZ - minZ };
}
