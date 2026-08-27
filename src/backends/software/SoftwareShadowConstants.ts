/** @internal Minimum positive homogeneous clip-space `w` for Software shadows. */
export const MIN_CLIP_W = 1e-6;
/** @internal Minimum normalized-device-coordinate depth for Software shadows. */
export const MIN_NDC_DEPTH = -1.0;
/** @internal Maximum normalized-device-coordinate depth for Software shadows. */
export const MAX_NDC_DEPTH = 1.0;
/** @internal Clip plane index for the minimum homogeneous `w` plane. */
export const CLIP_PLANE_MIN_W = 0;
/** @internal Clip plane index for the left clip plane. */
export const CLIP_PLANE_LEFT = 1;
/** @internal Clip plane index for the right clip plane. */
export const CLIP_PLANE_RIGHT = 2;
/** @internal Clip plane index for the bottom clip plane. */
export const CLIP_PLANE_BOTTOM = 3;
/** @internal Clip plane index for the top clip plane. */
export const CLIP_PLANE_TOP = 4;
/** @internal Clip plane index for the near clip plane. */
export const CLIP_PLANE_NEAR = 5;
/** @internal Clip plane index for the far clip plane. */
export const CLIP_PLANE_FAR = 6;
/** @internal Number of Software shadow clip planes. */
export const CLIP_PLANE_COUNT = 7;
/** @internal Tolerance for Software shadow clip-edge intersections. */
export const CLIP_EPSILON = 1e-12;
