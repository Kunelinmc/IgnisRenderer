/**
 * Constants shared by pipeline-level lighting and ambient SH logic.
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
