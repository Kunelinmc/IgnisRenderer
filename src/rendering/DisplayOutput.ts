export type DisplayOutputMode = "sdr" | "auto" | "hdr";
export type DisplayDynamicRange = "sdr" | "hdr";
export type DisplayColorSpace = "srgb" | "display-p3";

export type DisplayOutputFallbackReason =
	| "backend-unsupported"
	| "display-not-hdr-capable"
	| "canvas-tone-mapping-unsupported"
	| "hdr-context-configuration-failed";

export interface DisplayOutputOptions {
	readonly mode?: DisplayOutputMode;
	readonly exposure?: number;
	readonly hdrHeadroom?: number;
}

export interface ResolvedDisplayOutputOptions {
	readonly mode: DisplayOutputMode;
	readonly exposure: number;
	readonly hdrHeadroom: number;
}

export interface DisplayOutputState {
	readonly requested: ResolvedDisplayOutputOptions;
	readonly activeDynamicRange: DisplayDynamicRange;
	readonly colorSpace: DisplayColorSpace;
	readonly fallbackReason?: DisplayOutputFallbackReason;
}

export const DEFAULT_DISPLAY_OUTPUT_OPTIONS: ResolvedDisplayOutputOptions =
	Object.freeze({
		mode: "sdr",
		exposure: 1,
		hdrHeadroom: 4,
	});

/**
 * Resolves and validates renderer display-output options.
 *
 * @internal Owned by the renderer/backend display-output contract.
 */
export function resolveDisplayOutputOptions(
	options: DisplayOutputOptions = {},
	base: ResolvedDisplayOutputOptions = DEFAULT_DISPLAY_OUTPUT_OPTIONS,
): ResolvedDisplayOutputOptions {
	const mode = options.mode ?? base.mode;
	const exposure = options.exposure ?? base.exposure;
	const hdrHeadroom = options.hdrHeadroom ?? base.hdrHeadroom;
	if (mode !== "sdr" && mode !== "auto" && mode !== "hdr") {
		throw new RangeError(
			'DisplayOutputOptions.mode must be "sdr", "auto", or "hdr".',
		);
	}
	assertFiniteRange("exposure", exposure, 0, 64);
	assertFiniteRange("hdrHeadroom", hdrHeadroom, 1, 16);
	return Object.freeze({ mode, exposure, hdrHeadroom });
}

/**
 * Creates the stable SDR state used by non-HDR backends and WebGPU fallback.
 *
 * @internal Owned by the renderer/backend display-output contract.
 */
export function createSDRDisplayOutputState(
	requested: ResolvedDisplayOutputOptions,
	fallbackReason?: DisplayOutputFallbackReason,
): DisplayOutputState {
	return Object.freeze({
		requested,
		activeDynamicRange: "sdr",
		colorSpace: "srgb",
		...(fallbackReason ? { fallbackReason } : {}),
	});
}

/**
 * Resolves display output for a backend that only supports SDR presentation.
 *
 * @internal Owned by WebGL and Software backend fallback implementations.
 */
export function resolveSDROnlyDisplayOutput(
	options: DisplayOutputOptions,
	base: ResolvedDisplayOutputOptions = DEFAULT_DISPLAY_OUTPUT_OPTIONS,
): DisplayOutputState {
	const requested = resolveDisplayOutputOptions(options, base);
	return createSDRDisplayOutputState(
		requested,
		requested.mode === "hdr" ? "backend-unsupported" : undefined,
	);
}

/**
 * Compares observable display-output states.
 *
 * @internal Owned by backend event emission.
 */
export function displayOutputStatesEqual(
	left: DisplayOutputState,
	right: DisplayOutputState,
): boolean {
	return left.activeDynamicRange === right.activeDynamicRange &&
		left.colorSpace === right.colorSpace &&
		left.fallbackReason === right.fallbackReason &&
		left.requested.mode === right.requested.mode &&
		left.requested.exposure === right.requested.exposure &&
		left.requested.hdrHeadroom === right.requested.hdrHeadroom;
}

/**
 * Applies the reference hue-preserving HDR shoulder used by WebGPU shaders.
 *
 * @internal Used for numerical contract tests and non-shader validation.
 */
export function applyHDRSoftShoulder(
	color: readonly [number, number, number],
	exposure: number,
	hdrHeadroom: number,
): [number, number, number] {
	const exposed = color.map((value) => Math.max(0, value * exposure)) as
		[number, number, number];
	const peak = Math.max(exposed[0], exposed[1], exposed[2]);
	if (peak <= 1) return exposed;
	if (hdrHeadroom <= 1.0001) {
		return exposed.map((value) => Math.min(value, 1)) as
			[number, number, number];
	}
	const mappedPeak = 1 + (hdrHeadroom - 1) *
		(1 - Math.exp(-(peak - 1) / (hdrHeadroom - 1)));
	const scale = mappedPeak / Math.max(peak, 1e-6);
	return exposed.map((value) => value * scale) as
		[number, number, number];
}

/**
 * Converts D65 linear-sRGB values to D65 linear Display-P3.
 *
 * @internal Used for numerical contract tests and non-shader validation.
 */
export function linearSrgbToDisplayP3(
	color: readonly [number, number, number],
): [number, number, number] {
	const [red, green, blue] = color;
	return [
		0.82259287 * red + 0.17753395 * green,
		0.03319951 * red + 0.9667835 * green,
		0.01708535 * red + 0.07239572 * green + 0.91030148 * blue,
	];
}

function assertFiniteRange(
	name: "exposure" | "hdrHeadroom",
	value: number,
	minimum: number,
	maximum: number,
): void {
	if (!Number.isFinite(value) || value < minimum || value > maximum) {
		throw new RangeError(
			`DisplayOutputOptions.${name} must be a finite number in ` +
			`[${minimum}, ${maximum}].`,
		);
	}
}
