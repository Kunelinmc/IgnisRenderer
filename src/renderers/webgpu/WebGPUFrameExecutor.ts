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
import type {
	NormalizedOcclusionCullingOptions,
	OcclusionVisibilityProvider,
} from "../../pipeline/OcclusionCulling";
import type { WebGPUFrameHost } from "./rendergraph/WebGPUFrameHost";
import type { WebGPUPostProcessSessionPort } from "./WebGPUPostProcessExecutor";
import type { WarmupOptions } from "../IRenderBackend";
import type { RenderTargetReadbackOptions } from "../CustomRenderTargets";
import type { TextureReadbackResult } from "../IComputeRuntime";
import type {
	WebGPUPreparedFrameResources,
	WebGPURenderResources,
} from "./WebGPURenderResources";
import type { WebGPUSceneTargetMode } from "./WebGPUScenePassDescriptors";
import { WebGPUFrameOrchestrator } from "./rendergraph/WebGPUFrameOrchestrator";
import {
	SINGLE_SAMPLE_WEBGPU_MSAA_CONTEXT,
	type WebGPUMSAAContext,
} from "./WebGPUMSAAController";

/**
 * Backend-facing compatibility facade for the WebGPU internal frame graph.
 */
export class WebGPUFrameExecutor {
	private readonly _orchestrator: WebGPUFrameOrchestrator;

	public constructor(
		backend: WebGPUFrameHost,
		resources: WebGPURenderResources,
		msaa: WebGPUMSAAContext = SINGLE_SAMPLE_WEBGPU_MSAA_CONTEXT
	) {
		this._orchestrator = new WebGPUFrameOrchestrator(backend, resources, msaa, {
			enableEarlyZPrepass: backend.enableEarlyZPrepass,
			enableDeferredLighting: backend.enableDeferredLighting,
			frameGraphValidationMode: backend.frameGraphValidationMode,
		});
	}

	public beginFrame(context: FrameContext): void {
		this._orchestrator.beginFrame(context);
	}

	public prepareFrameResources(
		context: FrameContext
	): WebGPUPreparedFrameResources | null {
		return this._orchestrator.prepareFrameResources(context);
	}

	public getPreparedFrameResources(): WebGPUPreparedFrameResources | null {
		return this._orchestrator.getPreparedFrameResources();
	}

	public createPostProcessResource(
		desc: PostProcessResourceDescriptor
	): PostProcessResourceHandle {
		return this._orchestrator.createPostProcessResource(desc);
	}

	public destroyPostProcessResource(
		handle: PostProcessResourceHandle
	): void {
		this._orchestrator.destroyPostProcessResource(handle);
	}

	public createGBufferBridge(context: FrameContext): LogicalGBufferBridge {
		return this._orchestrator.createGBufferBridge(context);
	}

	public getSceneTargetModeForFrame(): WebGPUSceneTargetMode {
		return this._orchestrator.getSceneTargetModeForFrame();
	}

	public getOcclusionVisibilityProvider(
		options: NormalizedOcclusionCullingOptions
	): OcclusionVisibilityProvider {
		return this._orchestrator.getOcclusionVisibilityProvider(options);
	}

	public resetOcclusionCulling(): void {
		this._orchestrator.resetOcclusionCulling();
	}

	public getPassExecutionContext(
		request: PostProcessPassExecutionContextRequest
	): unknown {
		return this._orchestrator.getPassExecutionContext(request);
	}

	public completePostProcessPass(
		request: PostProcessPassRequest,
		result: PostProcessPassResult
	): void {
		this._orchestrator.completePostProcessPass(request, result);
	}

	public invalidateFrameTargets(): void {
		this._orchestrator.invalidateFrameTargets();
	}

	public invalidatePostProcessBindings(): void {
		this._orchestrator.invalidatePostProcessBindings();
	}

	public onShaderRuntimeChanged(): void {
		this._orchestrator.onShaderRuntimeChanged();
	}

	public warmup(
		context: FrameContext,
		plan: WarmupPlan,
		options: WarmupOptions = {}
	): Promise<WarmupPhaseCounters> {
		return this._orchestrator.warmup(context, plan, options);
	}

	public executePass(
		pass: FramePass,
		context: FrameContext
	): Promise<void> {
		return this._orchestrator.executePass(pass, context);
	}

	public endFrame(): Promise<void> {
		return this._orchestrator.endFrame();
	}

	public abortFrame(): void {
		this._orchestrator.abortFrame();
	}

	public createPostProcessSessionPort(): WebGPUPostProcessSessionPort {
		return {
			createGBufferBridge: (context) => this.createGBufferBridge(context),
			getPassExecutionContext: (request) => this.getPassExecutionContext(request),
			completePass: (request, result) => this.completePostProcessPass(request, result),
			invalidateResourceBindings: () => this.invalidatePostProcessBindings(),
		};
	}

	public readRenderTargetColor(
		id: string,
		attachmentIndex?: number,
		options?: RenderTargetReadbackOptions
	): Promise<TextureReadbackResult> {
		return this._orchestrator.readRenderTargetColor(id, attachmentIndex, options);
	}

	public destroy(): void {
		this._orchestrator.destroy();
	}

	public getFrameGraphDebugState(): unknown {
		return this._orchestrator.getDebugState();
	}
}
