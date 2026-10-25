import type { IVector3 } from "../maths/types";

const MIN_NORMAL = 2 ** -1022;

/** @internal Spatial query representation; consumers should use index ray queries. */
export interface SpatialRayDirection extends IVector3 {
	sourceX: number;
	sourceY: number;
	sourceZ: number;
	scale: number;
	scaledLength: number;
}

/**
 * Validates a source direction and returns its largest absolute component.
 * @throws Error when a component is non-finite or all components are zero.
 * @internal Spatial query math; consumers should use spatial index ray queries.
 */
export function resolveRayDirectionScale(direction: IVector3, label: string): number {
	const scale = Math.max(Math.abs(direction.x), Math.abs(direction.y), Math.abs(direction.z));
	if (!Number.isFinite(scale)) {
		throw new Error(`${label} direction must have finite components`);
	}
	if (scale === 0) {
		throw new Error(`${label} direction must be non-zero`);
	}
	return scale;
}

/**
 * Normalizes a finite direction while retaining source data for underflowed axes.
 * @param direction - Ray direction, accepted at any finite nonzero scale.
 * @param label - Calling query name included in validation errors.
 * @returns A unit direction with source-scale data for robust slab distances.
 * @throws Error when a component is non-finite or all components are zero.
 * @internal Spatial query math; consumers should use spatial index ray queries.
 */
export function normalizeRayDirection(direction: IVector3, label: string): SpatialRayDirection {
	const scale = resolveRayDirectionScale(direction, label);
	const x = direction.x / scale;
	const y = direction.y / scale;
	const z = direction.z / scale;
	const scaledLength = Math.hypot(x, y, z);
	return {
		x: x / scaledLength,
		y: y / scaledLength,
		z: z / scaledLength,
		sourceX: direction.x,
		sourceY: direction.y,
		sourceZ: direction.z,
		scale,
		scaledLength,
	};
}

/**
 * Intersects a normalized ray with scalar AABB bounds, including boundary contact.
 * @returns The entry distance in world units, or null when the ray misses.
 * @internal Spatial query math; consumers should use spatial index ray queries.
 */
export function intersectRayAABB(
	origin: IVector3,
	direction: SpatialRayDirection,
	maxDistance: number,
	minX: number,
	minY: number,
	minZ: number,
	maxX: number,
	maxY: number,
	maxZ: number
): number | null {
	let tMin = 0;
	let tMax = maxDistance;
	for (let axis = 0; axis < 3; axis++) {
		const originValue = axis === 0 ? origin.x : axis === 1 ? origin.y : origin.z;
		const directionValue = axis === 0 ? direction.x : axis === 1 ? direction.y : direction.z;
		const minValue = axis === 0 ? minX : axis === 1 ? minY : minZ;
		const maxValue = axis === 0 ? maxX : axis === 1 ? maxY : maxZ;
		let t0: number;
		let t1: number;
		if (Math.abs(directionValue) >= MIN_NORMAL) {
			const invDirection = 1 / directionValue;
			t0 = (minValue - originValue) * invDirection;
			t1 = (maxValue - originValue) * invDirection;
		} else {
			const sourceValue = axis === 0 ? direction.sourceX :
				axis === 1 ? direction.sourceY : direction.sourceZ;
			if (sourceValue === 0) {
				if (originValue < minValue || originValue > maxValue) return null;
				continue;
			}
			// Normalization can erase a nonzero axis or round a subnormal one heavily.
			t0 = resolveSubnormalSlabDistance(minValue - originValue, sourceValue, direction);
			t1 = resolveSubnormalSlabDistance(maxValue - originValue, sourceValue, direction);
		}
		if (t0 > t1) {
			const tmp = t0;
			t0 = t1;
			t1 = tmp;
		}
		tMin = Math.max(tMin, t0);
		tMax = Math.min(tMax, t1);
		if (tMax < tMin) return null;
	}
	return tMin;
}

function resolveSubnormalSlabDistance(
	delta: number,
	sourceValue: number,
	direction: SpatialRayDirection
): number {
	if (delta === 0) return 0;
	const quotient = delta / sourceValue;
	// Reorder only when the quotient loses range; the subnormal-axis condition
	// ensures multiplying delta by scale cannot lose range in these cases.
	if (Math.abs(quotient) < MIN_NORMAL ||
		(!Number.isFinite(quotient) && direction.scale < 1)) {
		return ((delta * direction.scale) / sourceValue) * direction.scaledLength;
	}
	return (quotient * direction.scale) * direction.scaledLength;
}
