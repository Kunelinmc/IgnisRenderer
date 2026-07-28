import type { IVector3 } from "../maths/types";
import type { DecalBlendMode } from "./Decal";

export const DECAL_SUPPORTED_RECEIVER_LAYER_MASK = 0x7ff;

export interface DecalLocalPosition {
	x: number;
	y: number;
	z: number;
}

/**
 * Returns the receiver bits guaranteed by every decal backend.
 *
 * @internal Owned by the decal preparation and backend runtimes.
 */
export function resolveDecalReceiverLayerMask(mask: number): number {
	return (mask >>> 0) & DECAL_SUPPORTED_RECEIVER_LAYER_MASK;
}

/**
 * Resolves the box-edge coverage for normalized projector coordinates.
 *
 * @internal Owned by cross-backend decal evaluation.
 */
export function resolveDecalEdgeCoverage(
	localPosition: DecalLocalPosition,
	edgeFade: number
): number {
	const distanceToEdge = Math.min(
		0.5 - Math.abs(localPosition.x),
		0.5 - Math.abs(localPosition.y),
		0.5 - Math.abs(localPosition.z)
	);
	if (distanceToEdge < 0) {
		return 0;
	}
	const fade = clamp01(edgeFade);
	return fade > 0 ? clamp01(distanceToEdge / fade) : 1;
}

/**
 * Resolves decal coverage after material alpha and alpha-mask rejection.
 *
 * @internal Owned by cross-backend decal evaluation.
 */
export function resolveDecalCoverage(
	decalOpacity: number,
	materialAlpha: number,
	baseTextureAlpha: number,
	edgeCoverage: number,
	alphaMasked: boolean,
	alphaCutoff: number
): number {
	const sourceAlpha = clamp01(materialAlpha) * clamp01(baseTextureAlpha);
	if (alphaMasked && sourceAlpha < clamp01(alphaCutoff)) {
		return 0;
	}
	return clamp01(decalOpacity) * sourceAlpha * clamp01(edgeCoverage);
}

/**
 * Blends one scalar decal channel using the public decal blend contract.
 *
 * @internal Owned by cross-backend decal evaluation.
 */
export function blendDecalScalar(
	receiver: number,
	source: number,
	mode: DecalBlendMode,
	coverage: number
): number {
	const weight = clamp01(coverage);
	if (mode === "disabled" || weight <= 0) {
		return receiver;
	}
	if (mode === "multiply") {
		return receiver + (receiver * source - receiver) * weight;
	}
	if (mode === "add") {
		return receiver + source * weight;
	}
	return receiver + (source - receiver) * weight;
}

/**
 * Blends a three-component decal channel into caller-owned storage.
 *
 * @internal Owned by cross-backend decal evaluation.
 */
export function blendDecalVector3(
	receiver: IVector3,
	source: IVector3,
	mode: DecalBlendMode,
	coverage: number,
	out: IVector3
): void {
	const weight = clamp01(coverage);
	if (mode === "disabled" || weight <= 0) {
		out.x = receiver.x;
		out.y = receiver.y;
		out.z = receiver.z;
		return;
	}
	if (mode === "multiply") {
		out.x = receiver.x + (receiver.x * source.x - receiver.x) * weight;
		out.y = receiver.y + (receiver.y * source.y - receiver.y) * weight;
		out.z = receiver.z + (receiver.z * source.z - receiver.z) * weight;
		return;
	}
	if (mode === "add") {
		out.x = receiver.x + source.x * weight;
		out.y = receiver.y + source.y * weight;
		out.z = receiver.z + source.z * weight;
		return;
	}
	out.x = receiver.x + (source.x - receiver.x) * weight;
	out.y = receiver.y + (source.y - receiver.y) * weight;
	out.z = receiver.z + (source.z - receiver.z) * weight;
}

/**
 * Blends a direction channel and normalizes the result.
 *
 * @internal Owned by cross-backend decal evaluation.
 */
export function blendDecalDirection(
	receiver: IVector3,
	source: IVector3,
	mode: DecalBlendMode,
	coverage: number,
	out: IVector3
): void {
	const weight = clamp01(coverage);
	if (
		mode === "disabled" ||
		mode === "multiply" ||
		mode === "add" ||
		weight <= 0
	) {
		out.x = receiver.x;
		out.y = receiver.y;
		out.z = receiver.z;
		return;
	}
	const x = receiver.x + (source.x - receiver.x) * weight;
	const y = receiver.y + (source.y - receiver.y) * weight;
	const z = receiver.z + (source.z - receiver.z) * weight;
	const inverseLength = 1 / (Math.hypot(x, y, z) || 1);
	out.x = x * inverseLength;
	out.y = y * inverseLength;
	out.z = z * inverseLength;
}

function clamp01(value: number): number {
	return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}
