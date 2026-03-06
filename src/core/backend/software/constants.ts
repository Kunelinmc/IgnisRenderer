/**
 * Constants used by the software rendering backend.
 */

/**
 * Core mathematical and rasterization constants.
 */
export class CoreConstants {
	static readonly EPSILON = 1e-6;
	static readonly OPAQUE_ALPHA = 255;
	static readonly WIREFRAME_DEPTH_BIAS = 0.1;
	static readonly MAX_CHANNEL_VALUE = 255;
}

/**
 * Rendering pipeline and buffer constants.
 */
export class RenderConstants {
	static readonly REFLECTION_BUFFER_ALPHA = 255;
	static readonly REFLECTION_TRANSPARENT_THRESHOLD = 0.99;
}

/**
 * Post-processing constants (FXAA, gamma, exposure).
 */
export class PostProcessConstants {
	static readonly FXAA_EDGE_THRESHOLD_MIN = 0.03125;
	static readonly FXAA_EDGE_THRESHOLD_MULTIPLIER = 0.166;
	static readonly FXAA_SUBPIX_QUALITY = 0.75;
	static readonly FXAA_ITERATIONS = 12;
	static readonly FXAA_QUALITY: number[] = [
		1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0,
	];
	static readonly NOISE_REFERENCE_WIDTH = 1920;
	static readonly MIN_GAMMA = 0.01;
	static readonly MAX_GAMMA = 8.0;
	static readonly DEFAULT_GAMMA = 2.2;
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
	static readonly TEMPORAL_ACCUMULATION_FACTOR = 0.95;
}

/**
 * Screen Space Ambient Occlusion (SSAO) constants.
 */
export class SSAOConstants {
	static readonly DEFAULT_SAMPLES = 16;
	static readonly DEFAULT_RADIUS = 8.0;
	static readonly DEFAULT_BIAS = 0.1;
	static readonly DEFAULT_INTENSITY = 1.0;
	static readonly NOISE_SIZE = 4;
}
