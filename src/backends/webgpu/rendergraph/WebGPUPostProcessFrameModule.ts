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
	PlannedPostProcessPass,
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
import type { WebGPUPostProcessRuntime } from "../WebGPUPostProcessRuntime";
import type { WebGPUPostProcessSessionPort } from "../WebGPUPostProcessExecutor";

import type { WebGPUFrameGraphCompiler } from "./WebGPUFrameGraphCompiler";
import type {
	WebGPUFrameGraphContribution,
	WebGPUFrameGraphModule,
	WebGPUFrameModulePlanningInput,
} from "./WebGPUFrameGraphModule";
import {
	defineWebGPUFrameMessage,
	type WebGPUFrameMessageHandler,
} from "./WebGPUFrameMessage";
import {
	WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE,
	WEBGPU_FRAME_LOGICAL_RESOURCES,
	WEBGPU_POST_PROCESS_PASSES_MESSAGE,
} from "./WebGPUFrameMessages";
import { WEBGPU_TRANSPARENCY_FEATURE_ANALYSIS } from "./WebGPUTransparencyRuntime";
import type { WebGPUFrameExecutionContext } from "./WebGPUFrameExecutionContext";
import { WEBGPU_FRAME_GRAPH_RESOURCES } from "./WebGPUFrameGraphResourceCatalog";
import type {
	WebGPURecordingFrameSession as WebGPUFrameSession,
} from "./WebGPUFrameSession";
import type { WebGPUPostProcessBridge } from "./WebGPUPostProcessBridge";
import { createWebGPUPostProcessGraphComposition } from "./WebGPUPostProcessGraphAdapter";
import {
	getWebGPUPostProcessSharedResourceDescriptor,
	type WebGPUPostProcessAllocationGroup,
} from "./WebGPUPostProcessSharedResourceCatalog";
import type {
	WebGPUCompiledFrameGraphStage,
} from "./types";

interface WebGPUPostProcessPresentationPort {
	present(source: IRenderTexture, session: WebGPUFrameSession): Promise<void>;
	warmup(): Promise<void>;
}

export interface WebGPUPostProcessFeatureAnalysis {
	readonly postProcessPasses: readonly PlannedPostProcessPass[];
	readonly needsPostProcessTargets: boolean;
	readonly needsPostProcessGBuffer: boolean;
	readonly needsPlanarReflectionMask: boolean;
	readonly needsTransmissionTargets: boolean;
	readonly needsHiZTarget: boolean;
}

export const WEBGPU_POST_PROCESS_FEATURE_ANALYSIS =
	defineWebGPUFrameMessage<WebGPUPostProcessFeatureAnalysis>({
		id: "webgpu:post-process-analysis",
		ownerId: "post-process",
		phase: "analysis",
	});

export function analyzeWebGPUPostProcessFeatures(
	passes: readonly PlannedPostProcessPass[],
): WebGPUPostProcessFeatureAnalysis {
	const groups = new Set<WebGPUPostProcessAllocationGroup>();
	for (const pass of passes) {
		for (const resource of pass.declaration.shared ?? []) {
			const descriptor = getWebGPUPostProcessSharedResourceDescriptor(resource.id);
			if (descriptor && (resource.optional !== true || descriptor.allocateWhenOptional)) {
				groups.add(descriptor.allocationGroup);
			}
		}
	}
	return {
		postProcessPasses: passes,
		needsPostProcessTargets: passes.length > 0,
		needsPostProcessGBuffer: passes.some(
			(pass) => (pass.declaration.gBuffer?.length ?? 0) > 0,
		),
		needsPlanarReflectionMask: groups.has("planar-reflection-mask"),
		needsTransmissionTargets: groups.has("transmission"),
		needsHiZTarget: groups.has("hiz"),
	};
}

/** @internal Owns WebGPU post-process graph execution and session integration. */
export class WebGPUPostProcessFrameModule implements WebGPUFrameGraphModule {
	public readonly id = "post-process";
	public readonly messageHandlers: readonly WebGPUFrameMessageHandler[] = [{
		id: "analyze",
		moduleId: this.id,
		phase: "analysis",
		inputs: [{ descriptor: WEBGPU_POST_PROCESS_PASSES_MESSAGE }],
		outputs: [WEBGPU_POST_PROCESS_FEATURE_ANALYSIS],
		run: (messages, publisher) => publisher.publish(
			WEBGPU_POST_PROCESS_FEATURE_ANALYSIS,
			analyzeWebGPUPostProcessFeatures(
				messages.get(WEBGPU_POST_PROCESS_PASSES_MESSAGE),
			),
		),
	}, {
		id: "configure",
		moduleId: this.id,
		phase: "configuration",
		inputs: [
			{ descriptor: WEBGPU_POST_PROCESS_FEATURE_ANALYSIS },
			{ descriptor: WEBGPU_TRANSPARENCY_FEATURE_ANALYSIS },
		],
		outputs: [WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE],
		run: (messages, publisher) => {
			const analysis = messages.get(WEBGPU_POST_PROCESS_FEATURE_ANALYSIS);
			const transparency = messages.get(WEBGPU_TRANSPARENCY_FEATURE_ANALYSIS);
			const needsTransmissionTargets = analysis.needsTransmissionTargets &&
				transparency.transmissionPackets.length > 0;
			publisher.publish(WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE, {
				source: this.id,
				targetClass: analysis.needsPostProcessGBuffer || analysis.needsHiZTarget
					? "mrt"
					: analysis.needsPostProcessTargets ? "color" : "single",
				resources: [
					...(analysis.needsPostProcessTargets
						? [{ id: WEBGPU_FRAME_LOGICAL_RESOURCES.postProcessTargets }]
						: []),
					...(needsTransmissionTargets
						? [{ id: WEBGPU_FRAME_LOGICAL_RESOURCES.transmissionTargets }]
						: []),
					...(analysis.needsPlanarReflectionMask
						? [{ id: WEBGPU_FRAME_LOGICAL_RESOURCES.planarReflectionMask }]
						: []),
					...(analysis.needsHiZTarget
						? [{ id: WEBGPU_FRAME_LOGICAL_RESOURCES.hiZTarget }]
						: []),
				],
				needsHiZBuild: analysis.needsHiZTarget,
			});
		},
	}];
	public readonly executors = {
		"post-process-pass": async () => {
			throw new Error(
				"WebGPU post-process pass nodes require the stage transaction coordinator.",
			);
		},
	};

	private _graphFrame: PostProcessRenderGraphFrame | null = null;
	private _declarations: ReturnType<BackendPostProcessRuntime["describeFrame"]> | null = null;
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
	) {}

	public get graphFrame(): PostProcessRenderGraphFrame | null {
		return this._graphFrame;
	}

	public beginFrame(): void {
		this._declarations = null;
		this._graphFrame = null;
		this._graphComposition = null;
		this._outputColor = WEBGPU_FRAME_GRAPH_RESOURCES.frameColor;
		this._outputColorDomain = "scene-linear-hdr";
		this._bridge.clearPendingFrameState();
	}

	public createAnalysisSeeds(context: FrameContext) {
		this._declarations = this._backendRuntime.describeFrame(context);
		return [{
			descriptor: WEBGPU_POST_PROCESS_PASSES_MESSAGE,
			value: this._declarations.passes,
		}];
	}

	public sealFrame(context: FrameContext) {
		if (!this._declarations) {
			throw new Error("WebGPU post-process declarations are unavailable.");
		}
		this._graphComposition = null;
		this._graphFrame = this._backendRuntime.buildRenderGraphFrame(
			context,
			this._declarations,
		);
		return this._graphFrame.graph.frameRequirements;
	}

	public activateFrame(context: WebGPUFrameExecutionContext): void {
		this._bridge.bindFrame(context);
	}

	public closeFrame(): void {
		this._bridge.unbindFrame();
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
			lane: "postprocess",
			finalOutput: {
				resource: composition.outputColor,
				colorDomain: this._graphFrame.graph.outputColorDomain,
			},
			imports: composition.importResources,
			composition: {
				namespace: "postprocess",
				definition: composition.definition,
				inputs: composition.inputs,
			},
		}];
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
				this._bridge.getFrameTargets(),
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

	public async executeComposedStage(
		compiled: WebGPUCompiledFrameGraphStage | undefined,
		compiler: Pick<WebGPUFrameGraphCompiler, "recordSkippedNode">,
		recordExecutedNode: (nodeId: string) => void,
	): Promise<boolean> {
		if (compiled?.pass.stage !== "postprocess") return false;
		await this.executeStage(compiled, compiler, recordExecutedNode);
		return true;
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
		const targets = session.targets.frameTargets;
		const target = session.configuration?.mrtSupported
			? this._bridge.motionHistoryWriteTarget
			: null;
		const source = target ? targets.gMotionDepth : null;
		const encoder = session.commands.encoder;
		if (!encoder || !source || !target) return;
		encoder.copyTextureToTexture?.(
			{ texture: source },
			{ texture: target },
			{
				width: session.targets.width,
				height: session.targets.height,
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
