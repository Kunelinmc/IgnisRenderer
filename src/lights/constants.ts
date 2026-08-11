/**
 * Shadow camera near/far guardrails shared by light implementations.
 */
export const MIN_SHADOW_NEAR = 0.01;
export const MIN_SHADOW_FAR = 0.02;
export const SHADOW_NEAR_FAR_GAP = 0.01;

/**
 * Constants shared by lighting evaluation and ambient SH handling.
 */
export const PBR_MIN_NDOTV = 0.001;
export const PBR_DENOM_EPSILON = 0.0001;
export const PBR_SPEC_FALLBACK = 0.02;
export const GGX_EPSILON = 1e-6;
