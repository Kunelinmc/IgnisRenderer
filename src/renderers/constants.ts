/**
 * Shared post-process constants used across rendering backends.
 */
export const DEFAULT_GAMMA = 2.2;
export const MIN_GAMMA = 0.01;
export const MAX_GAMMA = 8.0;

export const FXAA_EDGE_THRESHOLD_MIN = 0.03125;
export const FXAA_EDGE_THRESHOLD_MULTIPLIER = 0.166;
export const FXAA_SUBPIX_QUALITY = 0.75;
export const FXAA_QUALITY: readonly number[] = [
	1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0,
];

export const POST_PROCESS_NOISE_REFERENCE_WIDTH = 1920;
export const MAX_EXPOSURE = 8;
export const VOLUMETRIC_SIGMA_T_SCALE = 0.02;
export const TAA_JITTER_SEQUENCE_LENGTH = 16;

/**
 * Shared post-process execution order across rendering backends.
 */
export const POST_PROCESS_STAGES = [
	"ssao",
	"ssgi",
	"taa",
	"ssr",
	"volumetric",
	"fog",
	"motion-blur",
	"dof",
	"bloom",
	"tonemap",
	"color-filter",
	"fxaa",
	"interaction-outline",
	"gamma",
] as const;

/**
 * Shared particle billboard quad vertices [x, y, u, v].
 */
export const PARTICLE_QUAD_VERTICES = new Float32Array([
	-0.5,
	-0.5,
	0,
	1,
	0.5,
	-0.5,
	1,
	1,
	0.5,
	0.5,
	1,
	0,
	-0.5,
	-0.5,
	0,
	1,
	0.5,
	0.5,
	1,
	0,
	-0.5,
	0.5,
	0,
	0,
]);
