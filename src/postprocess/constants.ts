/** Shared CPU-side constants owned by post-process effects. */
export const FXAA_EDGE_THRESHOLD_MIN = 0.03125;
export const FXAA_EDGE_THRESHOLD_MULTIPLIER = 0.166;
export const FXAA_SUBPIX_QUALITY = 0.75;
export const FXAA_QUALITY: readonly number[] = [
	1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0,
];

export const VOLUMETRIC_SIGMA_T_SCALE = 0.02;
