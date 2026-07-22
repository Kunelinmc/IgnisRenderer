import type {
	IBindingGroup,
	IComputePipeline,
	IRenderTexture,
	ISampler,
} from "../types";
import type { IWebGPUComputeFacade } from "./ComputeFacade";
import type { PostProcessCopyHelper } from "./postprocess/PostProcessCopyHelper";
import type { WebGPUHiZBuilder } from "./WebGPUHiZBuilder";

/** @internal Narrow device-lifetime services available to WebGPU passes. */
export interface WebGPUPostProcessServices {
	readonly compute: IWebGPUComputeFacade;
	readonly frameBindGroupLayout: GPUBindGroupLayout | null;
	readonly sampler: ISampler | null;
	getHiZBuilder(): WebGPUHiZBuilder;
	getCopyHelper(): PostProcessCopyHelper;
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

/** @internal WebGPU frame target set exposed to pass-owned implementations. */
export interface WebGPUFrameTargets {
	sceneColor: IRenderTexture;
	sceneColorMain: IRenderTexture;
	postPing?: IRenderTexture | null;
	postPong?: IRenderTexture | null;
	gAlbedoAlpha?: IRenderTexture | null;
	gNormalRoughMetal?: IRenderTexture | null;
	gEmissiveOcclusion?: IRenderTexture | null;
	gMotionDepth?: IRenderTexture | null;
	gSpecular?: IRenderTexture | null;
	gCoatSheen?: IRenderTexture | null;
	gSheenReflectance?: IRenderTexture | null;
	gMaterialExt0?: IRenderTexture | null;
	gMaterialExt1?: IRenderTexture | null;
	gMaterialExt2?: IRenderTexture | null;
	gMaterialExt3?: IRenderTexture | null;
	depth: IRenderTexture;
	oitAccum?: IRenderTexture | null;
	oitReveal?: IRenderTexture | null;
	oitSceneColorCopy?: IRenderTexture | null;
	transmissionSceneColorCopy?: IRenderTexture | null;
	transmissionLighting?: IRenderTexture | null;
	gTransmissionSurface0?: IRenderTexture | null;
	gTransmissionSurface1?: IRenderTexture | null;
	gTransmissionSurface2?: IRenderTexture | null;
	transmissionDepth?: IRenderTexture | null;
	planarReflectionMask?: IRenderTexture | null;
	hiZ?: IRenderTexture | null;
}

/** @internal Read-only WebGPU frame target view for pass implementations. */
export type WebGPUPostProcessFrameTargets = Readonly<
	Pick<WebGPUFrameTargets, "postPing" | "postPong">
>;
