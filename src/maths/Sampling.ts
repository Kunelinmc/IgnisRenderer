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
/**
 * GGX VNDF (Visible Normal Distribution Function) importance sampling.
 * Based on "Sampling the GGX Distribution of Visible Normals" by Eric Heitz.
 */
export function importanceSampleGGX_VNDF(
	xi: { x: number; y: number },
	V: IVector3,
	N: IVector3,
	roughness: number
): IVector3 {
	const alpha = roughness * roughness;

	// 1. Construct tangent space
	const up =
		Math.abs(N.y) < 0.999 ? { x: 0, y: 1, z: 0 } : { x: 0, y: 0, z: 1 };
	const tangent = Vector3.normalize(Vector3.cross(up, N));
	const bitangent = Vector3.cross(N, tangent);

	// 2. Transform View direction to local space
	const Ve = {
		x: Vector3.dot(V, tangent),
		y: Vector3.dot(V, bitangent),
		z: Vector3.dot(V, N),
	};

	// 3. Section 3.2: transforming the view direction to the stretched cosine hemisphere
	const Vh = Vector3.normalize({
		x: alpha * Ve.x,
		y: alpha * Ve.y,
		z: Ve.z,
	});

	// 4. Section 3.1: orthonormal basis (with special case for Vh.z == 1)
	const lensq = Vh.x * Vh.x + Vh.y * Vh.y;
	const T1 =
		lensq > 0
			? { x: -Vh.y / Math.sqrt(lensq), y: Vh.x / Math.sqrt(lensq), z: 0 }
			: { x: 1, y: 0, z: 0 };
	const T2 = Vector3.cross(Vh, T1);

	// 5. Section 3.4: parameterization of the projected area
	const r = Math.sqrt(xi.x);
	const phi = 2.0 * Math.PI * xi.y;
	const t1 = r * Math.cos(phi);
	let t2 = r * Math.sin(phi);
	const s = 0.5 * (1.0 + Vh.z);
	t2 = (1.0 - s) * Math.sqrt(1.0 - t1 * t1) + s * t2;

	// 6. Section 3.5: reproduction of the sampled horizon-clipped normal
	const Nh = Vector3.add(
		Vector3.scale(T1, t1),
		Vector3.add(
			Vector3.scale(T2, t2),
			Vector3.scale(Vh, Math.sqrt(Math.max(0.0, 1.0 - t1 * t1 - t2 * t2)))
		)
	);

	// 7. Section 3.6: transforming the normal back to the ellipsoid configuration
	const Ne = Vector3.normalize({
		x: alpha * Nh.x,
		y: alpha * Nh.y,
		z: Math.max(0.0, Nh.z),
	});

	// 8. Local to world space
	const worldH = {
		x: tangent.x * Ne.x + bitangent.x * Ne.y + N.x * Ne.z,
		y: tangent.y * Ne.x + bitangent.y * Ne.y + N.y * Ne.z,
		z: tangent.z * Ne.x + bitangent.z * Ne.y + N.z * Ne.z,
	};

	return Vector3.normalize(worldH);
}
