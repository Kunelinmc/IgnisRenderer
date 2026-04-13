import { AlphaMode, type Material } from "./Material";

const TRANSMISSION_EPSILON = 1e-4;
const TRANSMISSION_ALPHA_FLOOR = 0.12;

function clamp01(value: number): number {
	if (!Number.isFinite(value)) return 0;
	if (value <= 0) return 0;
	if (value >= 1) return 1;
	return value;
}

/**
 * Resolves scalar transmission factor from a material when available.
 * Non-PBR materials are treated as non-transmissive.
 */
export function getMaterialTransmissionFactor(material: Material): number {
	const candidate = (material as Material & { transmissionFactor?: number })
		.transmissionFactor;
	return clamp01(candidate ?? 0);
}

/**
 * Returns true when the material requires transmission composition.
 */
export function materialUsesTransmission(material: Material): boolean {
	return getMaterialTransmissionFactor(material) > TRANSMISSION_EPSILON;
}

/**
 * Determines whether a material should be rendered in transparent pass.
 * Transmission is treated as transparent even when alphaMode is OPAQUE.
 */
export function isMaterialTransparentPass(material: Material): boolean {
	const alphaMode = material.alphaMode ?? AlphaMode.Opaque;
	return alphaMode === AlphaMode.Blend || materialUsesTransmission(material);
}

/**
 * Resolves a compositing alpha for transmission materials under standard
 * forward blending so transmissive surfaces do not behave as fully opaque.
 */
export function resolveTransmissionCompositeAlpha(
	baseAlpha: number,
	transmission: number,
	fresnelHint = TRANSMISSION_ALPHA_FLOOR
): number {
	const clampedBaseAlpha = clamp01(baseAlpha);
	const clampedTransmission = clamp01(transmission);
	if (clampedTransmission <= TRANSMISSION_EPSILON) {
		return clampedBaseAlpha;
	}

	const floorAlpha = Math.max(
		TRANSMISSION_ALPHA_FLOOR,
		clamp01(fresnelHint)
	);
	const mixedAlpha =
		clampedBaseAlpha * (1 - clampedTransmission) +
		floorAlpha * clampedTransmission;
	return clamp01(Math.max(floorAlpha, mixedAlpha));
}
