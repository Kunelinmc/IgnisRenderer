export interface WebGLVertexTextureUnitLayout {
	readonly animationPayload: number;
	readonly morphPosition: number;
	readonly morphNormal: number;
	readonly maxFragmentUnits: number;
	readonly maxVertexUnits: number;
	readonly maxCombinedUnits: number;
}

/** @internal Reserves disjoint high texture units for WebGL vertex payloads. */
export function createWebGLVertexTextureUnitLayout(
	gl: WebGL2RenderingContext,
): WebGLVertexTextureUnitLayout {
	const maxFragmentUnits = resolveLimit(gl, gl.MAX_TEXTURE_IMAGE_UNITS, 16);
	const maxVertexUnits = resolveLimit(gl, gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS, 0);
	const maxCombinedUnits = resolveLimit(
		gl,
		gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS,
		maxFragmentUnits + maxVertexUnits,
	);
	return {
		animationPayload: maxCombinedUnits - 1,
		morphPosition: maxCombinedUnits - 2,
		morphNormal: maxCombinedUnits - 3,
		maxFragmentUnits,
		maxVertexUnits,
		maxCombinedUnits,
	};
}

export function supportsWebGLVertexTextureCount(
	layout: WebGLVertexTextureUnitLayout,
	required: number,
): boolean {
	if (required <= 0) return true;
	const lowestReserved = layout.maxCombinedUnits - required;
	return (
		layout.maxVertexUnits >= required &&
		lowestReserved >= layout.maxFragmentUnits
	);
}

function resolveLimit(
	gl: WebGL2RenderingContext,
	parameter: number,
	fallback: number,
): number {
	const value = gl.getParameter?.(parameter);
	return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : fallback;
}
