import type { LogicalGBufferBridge } from "../../postprocess";

/** @internal Canonical WebGL logical G-buffer metadata. */
export const WEBGL_POST_PROCESS_GBUFFER_METADATA = {
	normalSpace: "world",
	depthEncoding: "hardware",
	motionEncoding: "ndc-delta",
} as const satisfies Pick<
	LogicalGBufferBridge,
	"normalSpace" | "depthEncoding" | "motionEncoding"
>;
