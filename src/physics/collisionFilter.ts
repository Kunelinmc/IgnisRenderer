import type { PhysicsCollisionFilterDescriptor } from "./types";

export const DEFAULT_COLLISION_LAYER = "default";
export const DEFAULT_COLLISION_FILTER = 0x0001ffff;

export interface DecodedCollisionFilter {
	group: number;
	filter: number;
}

export function sanitizeCollisionFilter(filter: number): number {
	if (!Number.isFinite(filter)) return DEFAULT_COLLISION_FILTER;
	return Math.floor(filter) >>> 0;
}

export function encodeCollisionFilter(group: number, filter: number): number {
	return ((((group & 0xffff) << 16) | (filter & 0xffff)) >>> 0);
}

export function decodeCollisionFilter(filter: number): DecodedCollisionFilter {
	const sanitized = sanitizeCollisionFilter(filter);
	const lowBits = sanitized & 0xffff;
	const highBits = (sanitized >>> 16) & 0xffff;
	if (highBits === 0) {
		return {
			group: lowBits,
			filter: lowBits,
		};
	}
	return {
		group: highBits,
		filter: lowBits,
	};
}

export function collisionFilterToInteractionGroups(filter: number): number {
	const decoded = decodeCollisionFilter(filter);
	return encodeCollisionFilter(decoded.group, decoded.filter);
}

export function cloneCollisionFilterDescriptor(
	filter: PhysicsCollisionFilterDescriptor | undefined
): PhysicsCollisionFilterDescriptor | undefined {
	if (!filter) return undefined;
	return {
		groups: filter.groups ? [...filter.groups] : undefined,
		collidesWith:
			filter.collidesWith === "all" || filter.collidesWith === undefined ?
				filter.collidesWith
			:	[...filter.collidesWith],
	};
}
