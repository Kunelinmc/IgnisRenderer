/** @internal Shared clip-space constants for Software shadow rasterization. */
export class SoftwareShadowConstants {
	public static readonly MIN_CLIP_W = 1e-6;
	public static readonly MIN_NDC_DEPTH = -1.0;
	public static readonly MAX_NDC_DEPTH = 1.0;
	public static readonly CLIP_PLANE_MIN_W = 0;
	public static readonly CLIP_PLANE_LEFT = 1;
	public static readonly CLIP_PLANE_RIGHT = 2;
	public static readonly CLIP_PLANE_BOTTOM = 3;
	public static readonly CLIP_PLANE_TOP = 4;
	public static readonly CLIP_PLANE_NEAR = 5;
	public static readonly CLIP_PLANE_FAR = 6;
	public static readonly CLIP_PLANE_COUNT = 7;
	public static readonly CLIP_EPSILON = 1e-12;
}
