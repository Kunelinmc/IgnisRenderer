import type { FrameContext } from "../../../pipeline/types";
import type { ICommandEncoder } from "../../ICommandEncoder";
import type {
	IBindingGroup,
	IComputePipeline,
	ISampler,
} from "../../types";
import type { IWebGPUComputeFacade } from "../ComputeFacade";
import type { WebGPUPostProcessFrameTargets } from "../WebGPUPostProcessContracts";
import type { WebGPULightingState } from "../types";
import type { WebGPUHiZPostProcessHelper } from "./HiZPostProcessHelper";
import type { PostProcessCopyHelper } from "./PostProcessCopyHelper";

interface WebGPUPostProcessExecuteBaseRequest<TPassId extends string> {
	passId: TPassId;
	encoder: ICommandEncoder;
	targets: WebGPUPostProcessFrameTargets;
	frameContext: FrameContext;
	options: unknown;
}

export interface WebGPUPostProcessSSAOExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest<"ssao"> {
	passId: "ssao";
}

export interface WebGPUPostProcessSSGIExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest<"ssgi"> {
	passId: "ssgi";
}

export interface WebGPUPostProcessTAAExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest<"taa"> {
	passId: "taa";
	historyValid: boolean;
}

export interface WebGPUPostProcessSSRExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest<"ssr"> {
	passId: "ssr";
	historyValid: boolean;
	frameBinding: IBindingGroup;
}

export interface WebGPUPostProcessVolumetricExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest<"volumetric"> {
	passId: "volumetric";
	historyValid: boolean;
	frameBinding: IBindingGroup;
	lightingState: WebGPULightingState | null;
}

export interface WebGPUPostProcessFogExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest<"fog"> {
	passId: "fog";
}

export interface WebGPUPostProcessMotionBlurExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest<"motion-blur"> {
	passId: "motion-blur";
}

export interface WebGPUPostProcessDOFExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest<"dof"> {
	passId: "dof";
}

export interface WebGPUPostProcessBloomExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest<"bloom"> {
	passId: "bloom";
}

export interface WebGPUPostProcessColorFilterExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest<"color-filter"> {
	passId: "color-filter";
}

export interface WebGPUPostProcessFXAAExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest<"fxaa"> {
	passId: "fxaa";
}

export interface WebGPUPostProcessTonemapExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest<"tonemap"> {
	passId: "tonemap";
}

export interface WebGPUCustomPostProcessExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest<string> {
	passId: string;
}

export type WebGPUPostProcessExecuteRequest =
	| WebGPUPostProcessSSAOExecuteRequest
	| WebGPUPostProcessSSGIExecuteRequest
	| WebGPUPostProcessTAAExecuteRequest
	| WebGPUPostProcessSSRExecuteRequest
	| WebGPUPostProcessVolumetricExecuteRequest
	| WebGPUPostProcessFogExecuteRequest
	| WebGPUPostProcessMotionBlurExecuteRequest
	| WebGPUPostProcessDOFExecuteRequest
	| WebGPUPostProcessBloomExecuteRequest
	| WebGPUPostProcessColorFilterExecuteRequest
	| WebGPUPostProcessFXAAExecuteRequest
	| WebGPUPostProcessTonemapExecuteRequest;

export type WebGPUPostProcessPassId = WebGPUPostProcessExecuteRequest["passId"];

export interface WebGPUPostProcessExecuteResult {
	ran: boolean;
	historyUpdated?: boolean;
}

export type WebGPUPostProcessRuntimeExecuteRequest =
	| WebGPUPostProcessExecuteRequest
	| WebGPUCustomPostProcessExecuteRequest;

export interface WebGPUPostProcessRuntimeContext {
	readonly compute: IWebGPUComputeFacade;
	readonly frameBindGroupLayout: GPUBindGroupLayout | null;
	readonly sampler: ISampler | null;
	warn(key: string, message: string): void;
	ensureCommonResources(): Promise<void>;
	/**
	 * Returns the shared Hi-Z helper for depth-aware runtime passes.
	 *
	 * @returns Helper owned by the current WebGPU post-process runtime.
	 */
	getHiZHelper(): WebGPUHiZPostProcessHelper;
	/**
	 * Returns the shared copy helper for ordered post-process texture copies.
	 *
	 * @returns Helper owned by the current WebGPU post-process runtime.
	 */
	getCopyHelper(): PostProcessCopyHelper;
	getCachedBindGroup(
		key: string,
		pipeline: IComputePipeline,
		entries: Array<{ binding: number; resource: unknown }>,
		label: string
	): IBindingGroup;
	invalidateBindingsByPrefix(prefix: string): void;
	destroyBindingGroup(group: IBindingGroup | null): void;
}

export interface WebGPUPostProcessRuntimePass<
	TRequest extends WebGPUPostProcessRuntimeExecuteRequest =
		WebGPUPostProcessRuntimeExecuteRequest,
> {
	readonly id: string;
	readonly warmupHints?: readonly string[];
	warmup?(
		hint: string,
		context: WebGPUPostProcessRuntimeContext
	): Promise<boolean | void> | boolean | void;
	execute(
		request: TRequest,
		context: WebGPUPostProcessRuntimeContext
	): Promise<WebGPUPostProcessExecuteResult | void | null> |
		WebGPUPostProcessExecuteResult |
		void |
		null;
	invalidateBindings?(context: WebGPUPostProcessRuntimeContext): void;
	onShaderRuntimeChanged?(context: WebGPUPostProcessRuntimeContext): void;
}
