import type {
	FrameContext,
	FramePass,
} from "../../pipeline/types";
import type {
	LogicalGBufferBridge,
	PostProcessPassExecutionContextRequest,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "../../postprocess";
import type {
	WarmupPhaseCounters,
	WarmupPlan,
} from "../../pipeline/WarmupPlanner";
import type { WebGPUBackend } from "../WebGPUBackend";
import type {
	WebGPUPreparedFrameResources,
	WebGPURenderResources,
} from "./WebGPURenderResources";
import type { WebGPUSceneTargetMode } from "./WebGPUScenePassDescriptors";
import { WebGPUFrameGraphRuntime } from "./rendergraph/WebGPUFrameGraphRuntime";

/**
 * Backend-facing compatibility facade for the WebGPU internal frame graph.
 */
export class WebGPUFrameExecutor {
	private readonly _runtime: WebGPUFrameGraphRuntime;

	public constructor(
		backend: WebGPUBackend,
		resources: WebGPURenderResources
	) {
		this._runtime = new WebGPUFrameGraphRuntime(backend, resources);
	}

	public beginFrame(context: FrameContext): void {
		this._runtime.beginFrame(context);
	}

	public prepareFrameResources(
		context: FrameContext
	): WebGPUPreparedFrameResources | null {
		return this._runtime.prepareFrameResources(context);
	}

	public getPreparedFrameResources(): WebGPUPreparedFrameResources | null {
		return this._runtime.getPreparedFrameResources();
	}

	public createPostProcessResource(
		desc: PostProcessResourceDescriptor
	): PostProcessResourceHandle {
		return this._runtime.createPostProcessResource(desc);
	}

	public destroyPostProcessResource(
		handle: PostProcessResourceHandle
	): void {
		this._runtime.destroyPostProcessResource(handle);
	}

	public createGBufferBridge(context: FrameContext): LogicalGBufferBridge {
		return this._runtime.createGBufferBridge(context);
	}

	public getSceneTargetModeForFrame(): WebGPUSceneTargetMode {
		return this._runtime.getSceneTargetModeForFrame();
	}

	public getPassExecutionContext(
		request: PostProcessPassExecutionContextRequest
	): unknown {
		return this._runtime.getPassExecutionContext(request);
	}

	public completePostProcessPass(
		request: PostProcessPassRequest,
		result: PostProcessPassResult
	): void {
		this._runtime.completePostProcessPass(request, result);
	}

	public invalidateFrameTargets(): void {
		this._runtime.invalidateFrameTargets();
	}

	public invalidatePostProcessBindings(): void {
		this._runtime.invalidatePostProcessBindings();
	}

	public onShaderRuntimeChanged(): void {
		this._runtime.onShaderRuntimeChanged();
	}

	public warmup(
		context: FrameContext,
		plan: WarmupPlan
	): Promise<WarmupPhaseCounters> {
		return this._runtime.warmup(context, plan);
	}

	public executePass(
		pass: FramePass,
		context: FrameContext
	): Promise<void> {
		return this._runtime.executePass(pass, context);
	}

	public endFrame(): Promise<void> {
		return this._runtime.endFrame();
	}

	public abortFrame(): void {
		this._runtime.abortFrame();
	}

	public destroy(): void {
		this._runtime.destroy();
	}

	public getFrameGraphDebugState(): unknown {
		return this._runtime.getDebugState();
	}
}
