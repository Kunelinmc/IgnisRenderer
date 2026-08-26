/**
 * Shared renderer constants used across rendering backends.
 */
export const MAX_DIRECTIONAL_LIGHTS = 4;
export const MAX_POINT_LIGHTS = 16;
export const MAX_SPOT_LIGHTS = 8;
export const MAX_AREA_LIGHTS = 8;
export const MAX_CLUSTER_LIGHTS_PER_FRAGMENT = 512;
export const MAX_REFLECTION_PROBES = 8;
export const MAX_LOCAL_LIGHT_PROBES = 8;
export const MAX_VOLUMETRIC_LIGHTS = 65000;

/** Logical frames retained before inactive packet animation resources are released. */
export const ANIMATION_RESOURCE_RELEASE_DELAY_FRAMES = 60;

export const DEFAULT_GAMMA = 2.2;
export const MIN_GAMMA = 0.01;
export const MAX_GAMMA = 8.0;

export const POST_PROCESS_NOISE_REFERENCE_WIDTH = 1920;
export const MAX_EXPOSURE = 8;
export const TAA_JITTER_SEQUENCE_LENGTH = 16;

/**
 * Shared post-process execution order across rendering backends.
 */
export const POST_PROCESS_STAGES = [
	"postprocess",
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
