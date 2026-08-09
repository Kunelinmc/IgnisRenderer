import type { RGB } from "../../foundation/Color";
import type { ShadowCastingLight } from "../../lights";
import type { IVector3 } from "../../maths/types";
import type { Matrix4 } from "../../maths/Matrix4";
import type { ParticleShadowVolumeGrid } from "../../pipeline/ParticleShadowVolume";

/** @internal CPU storage owned by the Software shadow runtime. */
export interface SoftwareShadowRenderTarget {
	size: number;
	depthBuffer: Float32Array;
	transmissionBuffer: Float32Array;
	particleVolume: ParticleShadowVolumeGrid;
}

/** @internal Per-light Software shadow targets for one backend runtime. */
export type SoftwareShadowRuntimeMap = Map<
	ShadowCastingLight,
	SoftwareShadowRenderTarget[]
>;

/** @internal Sampling entrypoint exposed to Software render passes. */
export type SoftwareShadowSampler = (
	light: ShadowCastingLight,
	worldPoint: IVector3,
	normal?: IVector3 | null,
) => RGB;

/** @internal Camera data used for Software cascaded-shadow selection. */
export interface SoftwareShadowSamplerCamera {
	readonly viewMatrix?: Matrix4 | null;
	readonly position?: IVector3;
	readonly getWorldPosition?: (target?: IVector3) => IVector3;
	readonly getWorldDirection?: (
		localDirection: IVector3,
		target?: IVector3,
	) => IVector3;
}

export type SoftwareShadowSamplerFactory = (
	camera: SoftwareShadowSamplerCamera,
) => SoftwareShadowSampler;
