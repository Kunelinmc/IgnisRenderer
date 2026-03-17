/**
 * Shadow camera near/far guardrails shared by light implementations.
 */
export class ShadowConstants {
	static readonly MIN_SHADOW_NEAR = 0.01;
	static readonly MIN_SHADOW_FAR = 0.02;
	static readonly SHADOW_NEAR_FAR_GAP = 0.01;
}
