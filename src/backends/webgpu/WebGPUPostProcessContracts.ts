import type {
	IBindingGroup,
	IComputePipeline,
	ISampler,
} from "../types";
import type { IWebGPUComputeFacade } from "./ComputeFacade";
import type { WebGPUFrameTargets } from "./WebGPUFrameTargetContracts";
import type { WebGPUDenoiser } from "./WebGPUDenoiser";
import type { WebGPUHiZBuilder } from "./WebGPUHiZBuilder";
import type { DisplayOutputState } from "../../rendering/DisplayOutput";
import type { LogicalGBufferBridge } from "../../postprocess";

/** @internal Canonical WebGPU logical G-buffer metadata. */
export const WEBGPU_POST_PROCESS_GBUFFER_METADATA = {
	normalSpace: "view",
	depthEncoding: "linear-view-z",
	motionEncoding: "ndc-delta",
} as const satisfies Pick<
	LogicalGBufferBridge,
	"normalSpace" | "depthEncoding" | "motionEncoding"
>;

/** @internal Narrow device-lifetime services available to WebGPU passes. */
export interface WebGPUPostProcessServices {
	readonly compute: IWebGPUComputeFacade;
	readonly frameBindGroupLayout: GPUBindGroupLayout | null;
	readonly sampler: ISampler | null;
	getDisplayOutputState(): DisplayOutputState;
	getDenoiser(): WebGPUDenoiser;
	getHiZBuilder(): WebGPUHiZBuilder;
	warn(key: string, message: string): void;
	ensureCommonResources(): Promise<void>;
	getCachedBindGroup(
		key: string,
		pipeline: IComputePipeline,
		entries: Array<{ binding: number; resource: unknown }>,
		label: string
	): IBindingGroup;
	invalidateBindingsByPrefix(prefix: string): void;
	destroyManagedResource(resource: unknown, description?: string): void;
	destroyBindingGroup(group: IBindingGroup | null): void;
}

/** @internal Read-only WebGPU frame target view for pass implementations. */
export type WebGPUPostProcessFrameTargets = Readonly<
	Pick<WebGPUFrameTargets, "postPing" | "postPong">
> & {
	readonly sampleCount: number;
};
