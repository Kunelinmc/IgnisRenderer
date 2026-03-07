/**
 * Constants shared by pipeline-level lighting and shadow logic.
 */

export class LightingConstants {
	static readonly PBR_MIN_NDOTV = 0.001;
	static readonly PBR_DENOM_EPSILON = 0.0001;
	static readonly PBR_SPEC_FALLBACK = 0.02;
	static readonly PBR_AMBIENT_FALLBACK_LINEAR = 0.05;
	/**
	 * Baked equirect probes are projected from radiance SH.
	 * The software/WebGPU ambient paths treat SH irradiance as ambient brightness,
	 * so we normalize baked probe coefficients to avoid PI-fold over-brightening.
	 */
	static readonly BAKED_LIGHT_PROBE_SH_SCALE = 1 / Math.PI;
	static readonly GGX_EPSILON = 1e-6;
}

export class ShadowConstants {
	static readonly MIN_CLIP_W = 1e-6;
	static readonly MIN_NDC_DEPTH = -1.0;
	static readonly MAX_NDC_DEPTH = 1.0;

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
