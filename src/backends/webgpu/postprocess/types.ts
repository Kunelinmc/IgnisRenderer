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
import type { WebGPUHiZBuilder } from "../WebGPUHiZBuilder";
import type { PostProcessCopyHelper } from "./PostProcessCopyHelper";

interface WebGPUPostProcessExecuteBaseRequest<TPassId extends string> {
	passId: TPassId;
	encoder: ICommandEncoder;
	targets: WebGPUPostProcessFrameTargets;
	frameContext: FrameContext;
	options: unknown;
}

export interface WebGPUPostProcessExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest<string> {
	historyValid?: boolean;
	frameBinding?: IBindingGroup;
	lightingState?: WebGPULightingState | null;
	[key: string]: unknown;
}

export type WebGPUPostProcessPassId = string;

export interface WebGPUCustomPostProcessExecuteRequest
	extends WebGPUPostProcessExecuteRequest {}

export interface WebGPUPostProcessExecuteResult {
	ran: boolean;
	historyUpdated?: boolean;
}

export type WebGPUPostProcessRuntimeExecuteRequest = WebGPUPostProcessExecuteRequest;

export interface WebGPUPostProcessRuntimeContext {
	readonly compute: IWebGPUComputeFacade;
	readonly frameBindGroupLayout: GPUBindGroupLayout | null;
	readonly sampler: ISampler | null;
	warn(key: string, message: string): void;
	ensureCommonResources(): Promise<void>;
	/**
	 * Returns the shared frame-graph Hi-Z builder for depth-aware runtime passes.
	 *
	 * @returns Builder owned by the WebGPU frame graph runtime.
	 */
	getHiZBuilder(): WebGPUHiZBuilder;
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
