import type { ShadowBackendCapabilities } from "../../pipeline/ShadowMetadata";

export const WEBGL_SHADOW_ATLAS_COLUMNS = 4;
export const WEBGL_SHADOW_ATLAS_ROWS = 3;

export const TAA_HISTORY_WEIGHT_RANGE: [number, number] = [0, 0.99];
export const TAA_DEPTH_THRESHOLD_RANGE: [number, number] = [1e-4, 1];
export const TAA_MOTION_FACTOR_RANGE: [number, number] = [0, 512];
export const TAA_VARIANCE_GAMMA_RANGE: [number, number] = [0, 8];
export const TAA_SHARPEN_RANGE: [number, number] = [0, 2];

export const MOTION_BLUR_SHUTTER_SCALE_RANGE: [number, number] = [0, 2];
export const MOTION_BLUR_MAX_SAMPLES_RANGE: [number, number] = [4, 64];
export const MOTION_BLUR_VELOCITY_CLAMP_RANGE: [number, number] = [
	0.005,
	0.25,
];
export const MOTION_BLUR_DEPTH_REJECT_RANGE: [number, number] = [
	0.0001,
	0.25,
];
export const MOTION_BLUR_CENTER_WEIGHT_RANGE: [number, number] = [0, 4];

export const DOF_NEAR_FAR_STRENGTH_RANGE: [number, number] = [0, 2];
export const DOF_MAX_BLUR_RADIUS_RANGE: [number, number] = [0, 32];
export const DOF_DEPTH_CURVE_RANGE: [number, number] = [0.25, 4];
export const DOF_HIGHLIGHT_GAIN_RANGE: [number, number] = [0, 3];
export const DOF_CHROMATIC_ABERRATION_RANGE: [number, number] = [0, 2];

export const WEBGL_TEXTURE_UNIT_PARTICLE_SHADOW_VOLUME = 14;
export const WEBGL_TEXTURE_UNIT_SHADOW_TRANSMITTANCE = 16;
export const WEBGL_PARTICLE_SHADOW_VOLUME_GRID_WIDTH = 64;
export const WEBGL_PARTICLE_SHADOW_VOLUME_GRID_HEIGHT = 64;
export const WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH = 32;
export const WEBGL_PARTICLE_SHADOW_VOLUME_MAX_SLICES = 4;
export const WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS = 8;

export const WEBGL_REFLECTION_PROBE_CAMERA_WORLD_POSITION_SCRATCH = {
	x: 0,
	y: 0,
	z: 0,
};

export const IDENTITY_MATRIX4_COLUMN_MAJOR = new Float32Array([
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1,
]);

export const SH_COEFFICIENT_COUNT = 16;

export const WEBGL_SHADOW_CAPABILITIES: ShadowBackendCapabilities = {
	backendKey: "webgl",
	supportsSingleMap: true,
	supportsDirectionalCSM: true,
	supportsSpotCSM: false,
	supportsPointCSM: false,
	maxCsmDirectionalLights: 1,
	maxDynamicShadowCost: 24,
};
