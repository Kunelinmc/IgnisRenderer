import type { FrameContext } from "../../../pipeline/types";
import type { WarmupOptions } from "../../IRenderBackend";
import type { IRenderTexture } from "../../types";
import type {
	LogicalGBufferBridge,
	PostProcessPassCompletion,
	PostProcessPassExecutionContextRequest,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "../../../postprocess";
import type {
	BackendPostProcessRuntime,
	PostProcessExecutionPlan,
	PostProcessRenderGraphFrame,
} from "../../../postprocess/BackendPostProcessRuntime";
import type { PostProcessPlan } from "../../../postprocess/PostProcessPlanner";
import type { PostProcessColorDomain } from "../../../postprocess/PostProcessPass";
import type { WarmupPhaseCounters, WarmupPlan } from "../../../pipeline/WarmupPlanner";
import { toShaderCompileError } from "../../../pipeline/WarmupPlanner";
import { createWarmupYieldController } from "../../../pipeline/WarmupScheduler";
import type { ShaderCompileError } from "../../../shaders/runtime";
import type { RenderGraphResourceDescriptor } from "../../../rendergraph/types";
import type { WebGPUPostProcessRuntime } from "../WebGPUPostProcessRuntime";
import type { WebGPUPostProcessSessionPort } from "../WebGPUPostProcessExecutor";

import type { WebGPUFrameGraphCompiler } from "./WebGPUFrameGraphCompiler";
import type {
	WebGPUFrameGraphContribution,
	WebGPUFrameGraphModule,
	WebGPUFrameModuleAnalysisInput,
	WebGPUFrameModuleConfigurationInput,
	WebGPUFrameModulePlanningInput,
	WebGPUFrameModuleStateStore,
} from "./WebGPUFrameGraphModule";
import type {
	WebGPUFrameModuleConfigurationContribution,
} from "./WebGPUFrameConfigurationContribution";
import { WEBGPU_POST_PROCESS_FEATURE_ANALYSIS } from "./WebGPUFrameModuleStateKeys";
import { analyzeWebGPUPostProcessFeatures } from "./WebGPUFrameFeatureAnalyzer";
import type { WebGPUFrameGraphRecordingContext } from "./WebGPUFrameGraphRecordingContext";
import { WEBGPU_FRAME_GRAPH_RESOURCES } from "./WebGPUFrameGraphResourceCatalog";
import type { WebGPUFrameSession } from "./WebGPUFrameSession";
import type { WebGPUPostProcessBridge } from "./WebGPUPostProcessBridge";
import { createWebGPUPostProcessGraphComposition } from "./WebGPUPostProcessGraphAdapter";
import { getWebGPUPostProcessSharedResourceDescriptor } from "./WebGPUPostProcessSharedResourceCatalog";
import type {
	WebGPUCompiledFrameGraphStage,
} from "./types";

interface WebGPUPostProcessPresentationPort {
	present(source: IRenderTexture, session: WebGPUFrameSession): Promise<void>;
	warmup(): Promise<void>;
}

/** @internal Owns WebGPU post-process graph execution and session integration. */
export class WebGPUPostProcessFrameModule implements WebGPUFrameGraphModule {
	public readonly id = "post-process";
	public readonly executors = {
		"post-process-pass": async () => {
			throw new Error(
				"WebGPU post-process pass nodes require the stage transaction coordinator.",
			);
		},
	};

	private _graphFrame: PostProcessRenderGraphFrame | null = null;
	private _graphComposition: ReturnType<
		typeof createWebGPUPostProcessGraphComposition
	> | null = null;
	private _outputColor: string = WEBGPU_FRAME_GRAPH_RESOURCES.frameColor;
	private _outputColorDomain: PostProcessColorDomain = "scene-linear-hdr";

	public constructor(
		private readonly _runtime: WebGPUPostProcessRuntime,
		private readonly _backendRuntime: BackendPostProcessRuntime,
		private readonly _bridge: WebGPUPostProcessBridge,
		private readonly _presentation: WebGPUPostProcessPresentationPort,
		private readonly _recording: WebGPUFrameGraphRecordingContext,
	) {}

	public get graphFrame(): PostProcessRenderGraphFrame | null {
		return this._graphFrame;
	}

	public get outputColor(): string {
		return this._outputColor;
	}

	public set outputColor(value: string) {
		this._outputColor = value;
	}

	public get outputColorDomain(): PostProcessColorDomain {
		return this._outputColorDomain;
	}

	public set outputColorDomain(value: PostProcessColorDomain) {
		this._outputColorDomain = value;
	}

	public beginFrame(): void {
		this._graphFrame = null;
		this._graphComposition = null;
		this._outputColor = WEBGPU_FRAME_GRAPH_RESOURCES.frameColor;
		this._outputColorDomain = "scene-linear-hdr";
		this._bridge.clearPendingFrameState();
	}

	public analyze(
		input: WebGPUFrameModuleAnalysisInput,
		state: WebGPUFrameModuleStateStore,
	): void {
		state.set(
			WEBGPU_POST_PROCESS_FEATURE_ANALYSIS,
			analyzeWebGPUPostProcessFeatures(input.postProcessPasses),
		);
	}

	public contributeConfiguration(
		input: WebGPUFrameModuleConfigurationInput,
	): WebGPUFrameModuleConfigurationContribution {
		const analysis = input.state.require(WEBGPU_POST_PROCESS_FEATURE_ANALYSIS);
		return (builder) => builder.setPostProcess(analysis);
	}

	public describeFrame(context: FrameContext) {
		return this._backendRuntime.describeFrame(context);
	}

	public buildGraphFrame(
		context: FrameContext,
		declarations: ReturnType<BackendPostProcessRuntime["describeFrame"]>,
	): PostProcessRenderGraphFrame {
		this._graphComposition = null;
		this._graphFrame = this._backendRuntime.buildRenderGraphFrame(context, declarations);
		return this._graphFrame;
	}

	public planStage(
		input: WebGPUFrameModulePlanningInput,
	): readonly WebGPUFrameGraphContribution[] {
		if (input.pass.stage !== "postprocess" || input.finalization === true) return [];
		if (!this._graphFrame || this._graphFrame.graph.passes.length === 0) {
			return [];
		}
		const composition = this._getGraphComposition();
		this._outputColor = composition.outputColor;
		this._outputColorDomain = this._graphFrame.graph.outputColorDomain;
		return [{
			order: 100,
			composition: {
				namespace: "postprocess",
				definition: composition.definition,
				inputs: composition.inputs,
			},
		}];
	}

	public getImportResources(): readonly RenderGraphResourceDescriptor[] {
		if (!this._graphFrame || this._graphFrame.graph.passes.length === 0) return [];
		return this._getGraphComposition().importResources;
	}

	public createResource(desc: PostProcessResourceDescriptor): PostProcessResourceHandle {
		return this._bridge.createResource(desc);
	}

	public destroyResource(handle: PostProcessResourceHandle): void {
		this._bridge.destroyResource(handle);
	}

	public createGBufferBridge(context: FrameContext): LogicalGBufferBridge {
		return this._bridge.createGBufferBridge(context);
	}

	public createPassExecutionContext(request: PostProcessPassExecutionContextRequest): unknown {
		return this._bridge.createPassExecutionContext(request);
	}

	public createSessionPort(): WebGPUPostProcessSessionPort {
		return {
			createGBufferBridge: (context) => this.createGBufferBridge(context),
			createPassExecutionContext: (request) => this.createPassExecutionContext(request),
			completePass: (request, result) => this.completePass(request, result),
			isGraphResourceAvailable: (resourceId) => this.isSharedResourceAvailable(resourceId),
			invalidateResourceBindings: () => this.invalidateFrameResources(),
		};
	}

	public isSharedResourceAvailable(resourceId: string): boolean {
		return (
			getWebGPUPostProcessSharedResourceDescriptor(resourceId)?.isAllocated(
				this._recording.getFrameTargets(),
			) ?? false
		);
	}

	public completePass(
		request: PostProcessPassRequest,
		result: PostProcessPassResult,
	): PostProcessPassCompletion {
		return this._bridge.completePass(request, result);
	}

	public async executeStage(
		compiled: WebGPUCompiledFrameGraphStage | undefined,
		compiler: Pick<WebGPUFrameGraphCompiler, "recordSkippedNode">,
		recordExecutedNode: (nodeId: string) => void,
	): Promise<void> {
		const nodes = (compiled?.nodes ?? []).filter((node) => !!node.postProcess);
		if (!this._graphFrame || nodes.length === 0) return;
		const plan: PostProcessExecutionPlan = {
			graph: this._graphFrame.graph,
			outputColor: this._outputColor,
			nodes: nodes.map((node) => ({ ...node.postProcess!, nodeId: node.id })),
		};
		const frame = await this._backendRuntime.beginGraphFrame(plan);
		if (!frame) return;
		let executedColorDomain = plan.graph.initialColorDomain;
		try {
			for (const node of plan.nodes) {
				const result = await this._backendRuntime.executeGraphPass(frame, node.passId);
				recordExecutedNode(node.nodeId);
				if (result.ran === false && node.plannedOutputColor) {
					compiler.recordSkippedNode(
						node.nodeId,
						node.plannedOutputColor,
						this._backendRuntime.resolveGraphColor(frame, node.plannedOutputColor),
					);
				}
				if (result.ran !== false) {
					const plannedPass = plan.graph.passes.find((pass) => pass.id === node.passId);
					if (plannedPass?.pass.colorContract?.input === executedColorDomain) {
						executedColorDomain = plannedPass.pass.colorContract.output;
					}
				}
			}
			this._outputColorDomain = executedColorDomain;
			await this._backendRuntime.endGraphFrame(frame);
		} catch (error) {
			await this._backendRuntime.abortFrame(error);
			throw error;
		}
	}

	public async warmup(
		context: FrameContext,
		plan: WarmupPlan,
		options: WarmupOptions = {},
		postProcessPlan?: PostProcessPlan,
	): Promise<WarmupPhaseCounters> {
		let total = 1;
		let compiled = 0;
		let failed = 0;
		const errors: ShaderCompileError[] = [];
		const yieldController = createWarmupYieldController(options);
		try {
			await this._presentation.warmup();
			compiled++;
		} catch (error) {
			failed++;
			errors.push(toShaderCompileError(error, "webgpu", "WebGPUPresentWarmup"));
		}
		await yieldController.yieldIfNeeded();

		const warmupGraph =
			postProcessPlan ??
			(plan.includePostProcess ? this._backendRuntime.planWarmup(context) : null);
		const warmedImplementations = new Set<string>();
		for (const passId of plan.postProcessPasses) {
			if (warmedImplementations.has(passId)) continue;
			const compiledPass = warmupGraph?.passes.find((pass) => pass.id === passId);
			if (typeof compiledPass?.implementation?.warmup !== "function") continue;
			warmedImplementations.add(passId);
			total++;
			try {
				const warmupContext = this._bridge.getPassWarmupExecutionContext(
					compiledPass.id,
					compiledPass.declaration,
				);
				await compiledPass.implementation.warmup(warmupContext, {
					frameContext: context,
					postProcess: context.postProcess,
					backend: "webgpu",
					context: warmupContext,
					options: compiledPass.options,
				});
				compiled++;
			} catch (error) {
				failed++;
				errors.push(toShaderCompileError(error, "webgpu", `WebGPUPostWarmup:${passId}`));
			}
			await yieldController.yieldIfNeeded();
		}
		return {
			phase: "webgpu-frame",
			total,
			compiled,
			skipped: Math.max(0, total - compiled - failed),
			failed,
			errors,
		};
	}

	public finalizeRecording(session: WebGPUFrameSession): void {
		const targets = this._recording.getFrameTargets();
		const target = session.configuration?.mrtSupported
			? session.motionHistoryWriteTarget
			: null;
		const source = target ? targets?.gMotionDepth : null;
		if (!session.encoder || !source || !target) return;
		session.encoder.copyTextureToTexture?.(
			{ texture: source },
			{ texture: target },
			{
				width: this._recording.getTargetWidth(),
				height: this._recording.getTargetHeight(),
				depthOrArrayLayers: 1,
			},
		);
	}

	public getDebugState() {
		return this._backendRuntime.getDebugState();
	}

	public invalidateFrameResources(): void {
		this._runtime.invalidateBindings();
		this._bridge.clearPendingFrameState();
	}

	public onDisplayOutputChanged(): void {
		this._runtime.invalidateBindings();
	}

	public onShaderRuntimeChanged(): void {
		this._runtime.onShaderRuntimeChanged();
		this._backendRuntime.invalidateImplementations();
	}

	public destroy(): void {
		this._runtime.destroy();
	}

	private _getGraphComposition(): ReturnType<
		typeof createWebGPUPostProcessGraphComposition
	> {
		if (!this._graphComposition) {
			this._graphComposition = createWebGPUPostProcessGraphComposition(this._graphFrame!);
		}
		return this._graphComposition;
	}
}
