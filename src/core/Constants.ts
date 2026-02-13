/**
 * Shared constants for the core rendering pipeline.
 */

/**
 * Core mathematical and basic rendering constants.
 */
export class CoreConstants {
	static readonly EPSILON = 1e-6;
	static readonly OPAQUE_ALPHA = 255;
	static readonly WIREFRAME_DEPTH_BIAS = 0.1;
	static readonly MAX_CHANNEL_VALUE = 255;
}

/**
 * Rendering pipeline and buffer related constants.
 */
export class RenderConstants {
	static readonly REFLECTION_BUFFER_ALPHA = 255;
	static readonly MIN_CLIP_W = 1e-6;
	static readonly MIN_NDC_DEPTH = -1.0;
	static readonly MAX_NDC_DEPTH = 1.0;
}

/**
 * Post-processing effects constants (FXAA, Gamma, etc.).
 */
export class PostProcessConstants {
	static readonly FXAA_EDGE_THRESHOLD_MIN = 8;
	static readonly FXAA_EDGE_THRESHOLD_MULTIPLIER = 0.125;
	static readonly NOISE_REFERENCE_WIDTH = 1920;
	static readonly MIN_GAMMA = 0.01;
	static readonly MAX_GAMMA = 8.0;
	static readonly DEFAULT_GAMMA = 1.8;
	static readonly MAX_EXPOSURE = 8;
}

/**
 * Volumetric lighting and atmospheric scattering constants.
 */
export class VolumetricConstants {
	static readonly SIGMA_T_SCALE = 0.02;
	static readonly MIN_RAY_DISTANCE = 0.1;
	static readonly MIN_RAY_DIR_Z = 1e-6;
	static readonly MIN_ADAPTIVE_SAMPLE_COUNT = 8;
	static readonly MIN_DOWN_SAMPLE = 1;
	static readonly MAX_DOWN_SAMPLE = 8;
	static readonly MIN_SAMPLES = 1;
	static readonly MAX_SAMPLES = 256;
	static readonly DEFAULT_DOWN_SAMPLE = 4;
	static readonly DEFAULT_SAMPLES = 32;
	static readonly MIN_SHADOW_SAMPLE_INTERVAL = 1;
	static readonly MAX_SHADOW_SAMPLE_INTERVAL = 32;
	static readonly MIN_BILATERAL_DEPTH_SIGMA = 1e-4;
	static readonly MAX_WEIGHT = 10;
	static readonly DEFAULT_WEIGHT = 4.0;
	static readonly MAX_AIR_DENSITY = 10;
	static readonly TRANSMITTANCE_EARLY_EXIT = 0.001;
	static readonly GRID_SAMPLE_JITTER_STRENGTH = 0.75;
	static readonly SCENE_BOUNDS_FADE_START_MULTIPLIER = 1.05;
	static readonly SCENE_BOUNDS_FADE_END_MULTIPLIER = 1.8;
	static readonly SCENE_DEPTH_LIMIT_MULTIPLIER = 1.6;
	static readonly MIN_SCENE_BOUNDS_RADIUS = 1.0;
}

/**
 * Physically-based rendering (PBR) lighting constants.
 */
export class LightingConstants {
	static readonly PBR_MIN_NDOTV = 0.001;
	static readonly PBR_DENOM_EPSILON = 0.0001;
	static readonly PBR_SPEC_FALLBACK = 0.02;
	static readonly GGX_EPSILON = 1e-6;
}

/**
 * Shadow mapping specific constants.
 */
export class ShadowConstants {
	static readonly MIN_CLIP_W = RenderConstants.MIN_CLIP_W;
	static readonly MIN_NDC_DEPTH = RenderConstants.MIN_NDC_DEPTH;
	static readonly MAX_NDC_DEPTH = RenderConstants.MAX_NDC_DEPTH;

	static readonly CLIP_PLANE_MIN_W = 0;
	static readonly CLIP_PLANE_LEFT = 1;
	static readonly CLIP_PLANE_RIGHT = 2;
	static readonly CLIP_PLANE_BOTTOM = 3;
	static readonly CLIP_PLANE_TOP = 4;
	static readonly CLIP_PLANE_NEAR = 5;
	static readonly CLIP_PLANE_FAR = 6;
	static readonly CLIP_PLANE_COUNT = 7;
	static readonly CLIP_EPSILON = 1e-12;

	static readonly MIN_SHADOW_NEAR = 0.01;
	static readonly MIN_SHADOW_FAR = 0.02;
	static readonly SHADOW_NEAR_FAR_GAP = 0.01;
}
