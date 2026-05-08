import { AlphaMode, ShadingModel, type Material } from "./Material";

const TRANSMISSION_EPSILON = 1e-4;
const TRANSMISSION_ALPHA_FLOOR = 0.12;
const MIN_ATTENUATION_COLOR = 1e-4;

/**
 * Linear RGB direct-light transmittance used by transparent shadow maps.
 */
export interface ShadowTransmittance {
	r: number;
	g: number;
	b: number;
}

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

/**
 * Resolves linear RGB direct-light transmittance for transparent shadow maps.
 *
 * PBR materials use a Beer-Lambert volume term with normal-incidence Fresnel
 * interface loss. Alpha-blended non-transmissive materials keep the legacy
 * opacity-weighted color-filter behavior as a compatibility fallback.
 */
export function resolveMaterialShadowTransmittance(
	material: Material
): ShadowTransmittance {
	const alphaMode = material.alphaMode ?? AlphaMode.Opaque;
	const opacity = clamp01(material.opacity ?? 1);
	const isPBR =
		material.shading === ShadingModel.PBR || material.type === "PBR";
	const baseColor = isPBR ?
		resolvePBRLinearBaseColor(material)
	:	resolveFallbackLinearBaseColor(material);
	const transmission = getMaterialTransmissionFactor(material);

	if (isPBR && transmission > TRANSMISSION_EPSILON) {
		const volume = resolvePBRVolumeAttenuation(material);
		const fresnelTransmittance = resolveNormalIncidenceTransmittance(
			(material as Material & { ior?: number }).ior
		);
		const alphaFilter =
			alphaMode === AlphaMode.Blend ?
				resolveOpacityColorFilter(baseColor, opacity)
			:	baseColor;
		return clampTransmittance({
			r: transmission * fresnelTransmittance * alphaFilter.r * volume.r,
			g: transmission * fresnelTransmittance * alphaFilter.g * volume.g,
			b: transmission * fresnelTransmittance * alphaFilter.b * volume.b,
		});
	}

	if (alphaMode === AlphaMode.Blend || opacity < 1) {
		return resolveOpacityColorFilter(baseColor, opacity);
	}

	return { r: 0, g: 0, b: 0 };
}

function resolvePBRLinearBaseColor(material: Material): ShadowTransmittance {
	const albedo = (material as Material & {
		albedo?: { r?: number; g?: number; b?: number };
	}).albedo;
	return {
		r: clamp01((albedo?.r ?? 255) / 255),
		g: clamp01((albedo?.g ?? 255) / 255),
		b: clamp01((albedo?.b ?? 255) / 255),
	};
}

function resolveFallbackLinearBaseColor(material: Material): ShadowTransmittance {
	const color = material as Material & {
		diffuse?: { r?: number; g?: number; b?: number };
		color?: { r?: number; g?: number; b?: number };
	};
	const diffuse = color.diffuse ?? color.color ?? { r: 255, g: 255, b: 255 };
	return {
		r: clamp01((diffuse.r ?? 255) / 255),
		g: clamp01((diffuse.g ?? 255) / 255),
		b: clamp01((diffuse.b ?? 255) / 255),
	};
}

function resolvePBRVolumeAttenuation(material: Material): ShadowTransmittance {
	const pbr = material as Material & {
		thicknessFactor?: number;
		attenuationDistance?: number;
		attenuationColor?: { r?: number; g?: number; b?: number };
	};
	const thickness = Math.max(0, pbr.thicknessFactor ?? 0);
	const distance = pbr.attenuationDistance ?? Infinity;
	if (thickness <= 0 || !Number.isFinite(distance) || distance <= 0) {
		return { r: 1, g: 1, b: 1 };
	}

	const color = pbr.attenuationColor ?? { r: 255, g: 255, b: 255 };
	const exponent = thickness / distance;
	return {
		r: Math.pow(
			Math.max(MIN_ATTENUATION_COLOR, clamp01((color.r ?? 255) / 255)),
			exponent
		),
		g: Math.pow(
			Math.max(MIN_ATTENUATION_COLOR, clamp01((color.g ?? 255) / 255)),
			exponent
		),
		b: Math.pow(
			Math.max(MIN_ATTENUATION_COLOR, clamp01((color.b ?? 255) / 255)),
			exponent
		),
	};
}

function resolveNormalIncidenceTransmittance(ior: number | undefined): number {
	const safeIor = Math.max(1, Number.isFinite(ior) ? ior : 1.5);
	const f0 = Math.pow((safeIor - 1) / (safeIor + 1), 2);
	return clamp01(1 - f0);
}

function resolveOpacityColorFilter(
	color: ShadowTransmittance,
	opacity: number
): ShadowTransmittance {
	const clampedOpacity = clamp01(opacity);
	const passthrough = 1 - clampedOpacity;
	return clampTransmittance({
		r: color.r * clampedOpacity + passthrough,
		g: color.g * clampedOpacity + passthrough,
		b: color.b * clampedOpacity + passthrough,
	});
}

function clampTransmittance(
	value: ShadowTransmittance
): ShadowTransmittance {
	return {
		r: clamp01(value.r),
		g: clamp01(value.g),
		b: clamp01(value.b),
	};
}
