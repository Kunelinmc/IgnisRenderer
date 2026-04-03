/**
 * Software backend rasterization/runtime constants.
 */
export const CoreConstants = Object.freeze({
	EPSILON: 1e-6,
	OPAQUE_ALPHA: 255,
	WIREFRAME_DEPTH_BIAS: 0.1,
	MAX_CHANNEL_VALUE: 255,
});

export const RenderConstants = Object.freeze({
	REFLECTION_BUFFER_ALPHA: 255,
	REFLECTION_TRANSPARENT_THRESHOLD: 0.99,
});
