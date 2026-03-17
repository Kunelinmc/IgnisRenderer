/**
 * Clip-space and frustum constants for software shadow rasterization/sampling.
 */
export class SoftwareShadowConstants {
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
}
