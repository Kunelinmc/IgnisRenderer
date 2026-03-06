import type { ShadowCastingLight } from "../../../../lights"

export const SOFTWARE_SHADOW_RUNTIME_KEY = "software-shadow-runtime"

export interface SoftwareShadowRenderTarget {
	size: number
	depthBuffer: Float32Array
	transmissionBuffer: Float32Array
}

export type SoftwareShadowRuntimeMap = Map<
	ShadowCastingLight,
	SoftwareShadowRenderTarget
>
