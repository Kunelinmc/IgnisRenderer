import type { IVector3 } from "./types";
import { Vector3 } from "./Vector3";

/**
 * Hammersley sequence for generate quasi-random points on a 2D plane.
 */
export function hammersley(i: number, n: number): { x: number; y: number } {
	let bits = i;
	bits = (bits << 16) | (bits >>> 16);
	bits = ((bits & 0x55555555) << 1) | ((bits & 0xaaaaaaaa) >>> 1);
	bits = ((bits & 0x33333333) << 2) | ((bits & 0xcccccccc) >>> 2);
	bits = ((bits & 0x0f0f0f0f) << 4) | ((bits & 0xf0f0f0f0) >>> 4);
	bits = ((bits & 0x00ff00ff) << 8) | ((bits & 0xff00ff00) >>> 8);

	const fraction = (bits >>> 0) * 2.3283064365386963e-10;
	return { x: i / n, y: fraction };
}

/**
 * Normal Distribution Function (GGX/Trowbridge-Reitz) importance sampling.
 */
export function importanceSampleGGX(
	xi: { x: number; y: number },
	N: IVector3,
	roughness: number
): IVector3 {
	const a = roughness * roughness;
	const a2 = a * a;
	const phi = 2.0 * Math.PI * xi.x;
	const cosTheta = Math.sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
	const sinTheta = Math.sqrt(1.0 - cosTheta * cosTheta);

	const H = {
		x: Math.cos(phi) * sinTheta,
		y: Math.sin(phi) * sinTheta,
		z: cosTheta,
	};

	// Local to world space
	if (N.x === 0 && N.y === 0 && Math.abs(N.z) > 0.9999) return H;

	// Use a stable tangent space construction
	const up =
		Math.abs(N.y) < 0.999 ? { x: 0, y: 1, z: 0 } : { x: 0, y: 0, z: 1 };
	const tangent = Vector3.normalize(Vector3.cross(up, N));
	const bitangent = Vector3.cross(N, tangent);

	const worldH = {
		x: tangent.x * H.x + bitangent.x * H.y + N.x * H.z,
		y: tangent.y * H.x + bitangent.y * H.y + N.y * H.z,
		z: tangent.z * H.x + bitangent.z * H.y + N.z * H.z,
	};

	return Vector3.normalize(worldH);
}
