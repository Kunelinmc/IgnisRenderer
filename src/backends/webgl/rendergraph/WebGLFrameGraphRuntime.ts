import type {
	FrameContext,
	FramePass,
} from "../../../pipeline/types";
import { Logger } from "../../../foundation/Logger";
import type {
	BackendPostProcessRuntime,
	PostProcessExecutionPlan,
	PostProcessRenderGraphFrame,
} from "../../../postprocess/BackendPostProcessRuntime";
import type { PostProcessDeclarationPlan } from "../../../postprocess/PostProcessPlanner";
import type { FramePreparationRequirements } from "../../../pipeline/FrameRequirements";
import type {
	RenderGraphDiagnostic,
	RenderGraphResourceDescriptor,
} from "../../../rendergraph/types";
import { WebGLFrameGraphCompiler } from "./WebGLFrameGraphCompiler";
import { WebGLFrameGraphPlanner } from "./WebGLFrameGraphPlanner";
import { createWebGLPostProcessGraphComposition } from "./WebGLPostProcessGraphAdapter";
import {
	WebGLFrameNodeExecutorRegistry,
	type WebGLFrameNodeServices,
} from "./WebGLFrameNodeExecutorRegistry";
import type {
	WebGLCompiledFrameGraphStage,
	WebGLFrameGraphDebugState,
	WebGLFrameGraphDiagnostic,
	WebGLFrameGraphFramePlan,
	WebGLFrameGraphNode,
	WebGLFrameGraphPlannerState,
	WebGLFrameGraphResourceCatalogSnapshot,
	WebGLFrameGraphStagePlan,
} from "./types";

const WEBGL_MATERIAL_GBUFFER_SEMANTICS = new Set([
	"albedo",
	"roughness",
	"metallic",
	"specular",
]);

export interface WebGLFrameExecutionFacade extends WebGLFrameNodeServices {
	beginFrame(context: FrameContext, materialGBufferRequested: boolean): void;
	beginTemporalFrame(
		context: FrameContext,
		frameRequirements: FramePreparationRequirements,
	): void;
	finishFrame(): void;
	abortFrame(): void;
	isOITActive(): boolean;
	hasPresentedInFrame(): boolean;
	collectFrameGraphResources(): readonly string[];
	collectFrameGraphResourceCatalog(
		includeShadowResources?: boolean,
	): WebGLFrameGraphResourceCatalogSnapshot;
	hasCustomRenderPass(pass: FramePass, context: FrameContext): boolean;
	executeCustomRenderPass(pass: FramePass, context: FrameContext): Promise<void>;
}

/**
 * Orchestrates WebGL backend-private frame graph planning and node execution.
 */
export class WebGLFrameGraphRuntime {
	private readonly _executor: WebGLFrameExecutionFacade;
	private readonly _postProcessRuntime: BackendPostProcessRuntime;
	private readonly _planner = new WebGLFrameGraphPlanner();
	private readonly _compiler = new WebGLFrameGraphCompiler();
	private readonly _nodeState = {
		earlyZPacketIds: new Set<string>() as ReadonlySet<string>,
	};
	private readonly _nodeExecutors: WebGLFrameNodeExecutorRegistry;
	private _active = false;
	private _lastPlannedGraphNodes: WebGLFrameGraphNode[] = [];
	private _lastCompiledGraphStages: WebGLCompiledFrameGraphStage[] = [];
	private _lastExecutedGraphNodeIds: string[] = [];
	private _runtimeDiagnostics: WebGLFrameGraphDiagnostic[] = [];
	private _wholeFrameGraphCompiled = false;
	private _postProcessGraphFrame: PostProcessRenderGraphFrame | null = null;
	private _postProcessOutputColor = "frame:scene-color";

	public constructor(
		executor: WebGLFrameExecutionFacade,
		postProcessRuntime: BackendPostProcessRuntime
	) {
		this._executor = executor;
		this._postProcessRuntime = postProcessRuntime;
		this._nodeExecutors = WebGLFrameNodeExecutorRegistry.fromServices(
			executor,
			postProcessRuntime,
			this._nodeState,
		);
	}

	/**
	 * Begins a WebGL frame and executes synthetic setup nodes.
	 *
	 * @internal Called by `WebGLBackend.beginFrame`.
	 * @param context Active renderer frame context.
	 * @returns Nothing.
	 * @sideEffects Prepares WebGL frame targets and clears scene attachments.
	 */
	public beginFrame(context: FrameContext): void {
		this._active = true;
		this._nodeState.earlyZPacketIds = new Set<string>();
		this._lastPlannedGraphNodes = [];
		this._lastCompiledGraphStages = [];
		this._lastExecutedGraphNodeIds = [];
		this._runtimeDiagnostics = [];
		this._wholeFrameGraphCompiled = false;
		this._postProcessGraphFrame = null;
		this._postProcessOutputColor = "frame:scene-color";
		const postProcessDeclarations =
			this._postProcessRuntime.describeFrame(context);
		this._executor.beginFrame(
			context,
			this._requiresMaterialGBuffer(postProcessDeclarations),
		);
		this._postProcessGraphFrame =
			this._postProcessRuntime.buildRenderGraphFrame(
				context,
				postProcessDeclarations,
			);
		this._executor.beginTemporalFrame(
			context,
			this._postProcessGraphFrame.graph.frameRequirements,
		);
		if (context.framePlan) {
			try {
				this._compileWholeFrameGraph(context);
				this._executeCompiledStage("webgl-begin-frame", context);
			} catch (error) {
				this._compiler.abort(error);
				this._executor.abortFrame();
				this._active = false;
				throw error;
			}
			return;
		}
		this._compiler.beginFrame(this._executor.collectFrameGraphResources());
		const pass = this._createSyntheticPass("webgl-begin-frame");
		const plan: WebGLFrameGraphStagePlan = {
			pass,
			nodes: this._planner.planBeginFrame(context, this._createPlannerState(context)),
		};
		this._compileAndExecute(plan, context);
	}

	/**
	 * Executes one renderer-level backend pass through WebGL graph nodes.
	 *
	 * @internal Called by `WebGLBackend.executePass`.
	 * @param pass Renderer frame pass.
	 * @param context Active renderer frame context.
	 * @returns Promise that resolves after all planned nodes complete.
	 * @sideEffects Mutates WebGL state and backend-owned render targets.
	 */
	public executePass(
		pass: FramePass,
		context: FrameContext
	): void | Promise<void> {
		if (
			typeof this._executor.hasCustomRenderPass === "function" &&
			this._executor.hasCustomRenderPass(pass, context)
		) {
			if (!this._wholeFrameGraphCompiled) {
				this._compiler.recordOpaqueStage(
					pass.stage,
					`Custom render pass "${pass.stage}" executes outside the logical graph.`,
				);
			}
			const result = this._executor.executeCustomRenderPass(pass, context);
			return result.then(() => this._recordCompiledStageExecution(pass.stage));
		}
		if (this._wholeFrameGraphCompiled) {
			const compiled = this._findCompiledStage(pass.stage);
			if (pass.stage === "postprocess") {
				return this._executePostProcessStage(compiled, context);
			}
			if (!compiled || compiled.nodes.length <= 0) {
				this._warnUnsupportedPass(pass);
				return;
			}
			return this._executeCompiledStage(pass.stage, context);
		}
		const plan = this._planner.planStage(
			pass,
			context,
			this._createPlannerState(context)
		);
		if (plan.nodes.length <= 0) {
			this._warnUnsupportedPass(pass);
			return;
		}
		return this._compileAndExecute(plan, context);
	}

	/**
	 * Executes the synthetic present node and clears active frame state.
	 *
	 * @internal Called by `WebGLBackend.endFrame`.
	 * @param context Last active frame context.
	 * @returns Promise that resolves after present node execution.
	 * @sideEffects May present to canvas and clears executor frame state.
	 */
	public endFrame(context: FrameContext): void | Promise<void> {
		const result = this._wholeFrameGraphCompiled
			? this._executeCompiledStage("webgl-present", context)
			: this._compileAndExecute({
				pass: this._createSyntheticPass("webgl-present"),
				nodes: this._planner.planPresent(),
			}, context);
		const finish = () => {
			this._executor.finishFrame();
			this._compiler.seal();
			this._active = false;
			this._nodeState.earlyZPacketIds = new Set<string>();
		};
		if (result && typeof (result as Promise<void>).then === "function") {
			return (result as Promise<void>).then(finish);
		}
		finish();
	}

	/**
	 * Aborts active graph state after a failed WebGL frame.
	 *
	 * @internal Called by `WebGLBackend.abortFrame`.
	 * @returns Nothing.
	 * @sideEffects Clears executor frame state without presenting.
	 */
	public abortFrame(error?: unknown): void {
		this._executor.abortFrame();
		this._compiler.abort(error);
		this._active = false;
		this._nodeState.earlyZPacketIds = new Set<string>();
	}

	public getDebugState(): WebGLFrameGraphDebugState {
		return {
			active: this._active,
			oitActive: this._executor.isOITActive(),
			hasPresentedInFrame: this._executor.hasPresentedInFrame(),
			lastPlannedNodeIds: this._lastPlannedGraphNodes.map((node) => node.id),
			lastExecutedNodeIds: this._lastExecutedGraphNodeIds.slice(),
			compiledStages: this._lastCompiledGraphStages.slice(),
			compiledGraph: this._compiler.getCompiledFrame()?.graph ?? null,
			graphResources: this._compiler.getResourceDebugState(),
			graphBarriers: this._compiler.getBarriers(),
			graphDiagnostics: [
				...this._compiler.getDiagnostics(),
				...this._runtimeDiagnostics,
			],
			graphAnalysis: this._compiler.getGraphAnalysis(),
			postProcess: this._postProcessRuntime.getDebugState?.() ?? {
				lastAttempt: null,
				lastSuccessful: null,
			},
		};
	}

	/** @internal Completes graph analysis after all backend transaction commits. */
	public commitGraphAnalysis(): void {
		this._compiler.commit();
	}

	/** @internal Aborts graph analysis without changing native frame ownership. */
	public abortGraphAnalysis(error?: unknown): void {
		this._compiler.abort(error);
	}

	/** @internal Records a backend pass that bypasses logical resource analysis. */
	public recordOpaqueGraphStage(stage: string, message: string): void {
		if (this._wholeFrameGraphCompiled) return;
		this._compiler.recordOpaqueStage(stage, message);
	}

	private _compileWholeFrameGraph(context: FrameContext): void {
		const includeShadowResources = context.framePlan?.backendPasses.some(
			(pass) => pass.enabled && pass.stage === "shadow",
		) === true;
		const catalog = this._executor.collectFrameGraphResourceCatalog(
			includeShadowResources,
		);
		const stages: WebGLFrameGraphStagePlan[] = [];
		const postProcessImportResources: RenderGraphResourceDescriptor[] = [];
		const shadowDiagnostics: RenderGraphDiagnostic[] = [];
		const setupPass = this._createSyntheticPass("webgl-begin-frame");
		stages.push({
			pass: setupPass,
			nodes: this._planner.planBeginFrame(context, this._createPlannerState(context)),
		});

		let lastStage = setupPass.stage;
		let hasOpaqueStage = false;
		for (const pass of context.framePlan?.backendPasses ?? []) {
			if (!pass.enabled) continue;
			let stagePlan: WebGLFrameGraphStagePlan;
			const custom =
				typeof this._executor.hasCustomRenderPass === "function" &&
				this._executor.hasCustomRenderPass(pass, context);
			if (pass.stage === "particle-sim" || custom) {
				const reason = custom ? "custom render target" : "particle simulation";
				stagePlan = {
					pass,
					nodes: [{
						id: `${pass.stage}:opaque-external:frame`,
						stage: pass.stage,
						kind: "opaque-external",
						label: `WebGLOpaque:${pass.stage}`,
						domain: "cpu",
						retention: "always",
						opaque: true,
					}],
				};
				hasOpaqueStage = true;
				shadowDiagnostics.push({
					phase: "compile",
					enforcement: "shadow",
					severity: "warning",
					code: "opaque-stage-effects",
					stage: pass.stage,
					message: `WebGL ${reason} stage "${pass.stage}" has undeclared resource effects.`,
				});
			} else if (pass.stage === "postprocess") {
				const frame = this._postProcessGraphFrame;
				if (!frame) {
					throw new Error(
						"WebGL post-process graph frame was not prepared before compilation.",
					);
				}
				this._postProcessGraphFrame = frame;
				if (frame.graph.passes.length > 0) {
					const composition = createWebGLPostProcessGraphComposition(frame);
					postProcessImportResources.push(...composition.importResources);
					this._postProcessOutputColor = composition.outputColor;
					stagePlan = {
						pass,
						nodes: [],
						composition: {
							namespace: "postprocess",
							definition: composition.definition,
							inputs: composition.inputs,
						},
					};
				} else {
					stagePlan = { pass, nodes: [] };
				}
			} else {
				stagePlan = this._planner.planStage(
					pass,
					context,
					this._createPlannerState(context),
				);
				if (stagePlan.nodes.length <= 0) this._warnUnsupportedPass(pass);
			}
			stages.push(stagePlan);
			if (stagePlan.nodes.length > 0 || stagePlan.composition) lastStage = pass.stage;
		}

		const presentPass: FramePass = {
			stage: "webgl-present",
			executor: "backend",
			enabled: true,
			dependsOn: [lastStage],
		};
		stages.push({
			pass: presentPass,
			nodes: this._planner.planPresent(this._postProcessOutputColor),
		});
		const framePlan: WebGLFrameGraphFramePlan = {
			resources: [...catalog.resources, ...postProcessImportResources],
			bindings: catalog.bindings,
			stages,
			exports: [{ name: "presented-color", resource: "canvas:color" }],
			completeness: hasOpaqueStage ? "opaque" : "complete",
			shadowDiagnostics,
		};
		const compiled = this._compiler.compileFrame(framePlan);
		this._handleWholeFrameGraphDiagnostics(compiled.graph.diagnostics);
		this._lastCompiledGraphStages = compiled.stages.slice();
		this._lastPlannedGraphNodes = compiled.stages.flatMap((stage) => [...stage.nodes]);
		this._wholeFrameGraphCompiled = true;
	}

	private _findCompiledStage(stage: string): WebGLCompiledFrameGraphStage | undefined {
		return this._compiler.getCompiledStages().find(
			(compiled) => compiled.pass.stage === stage,
		);
	}

	private _executeCompiledStage(
		stage: string,
		context: FrameContext,
	): void | Promise<void> {
		const compiled = this._findCompiledStage(stage);
		let chain: Promise<void> | null = null;
		for (const node of compiled?.nodes ?? []) {
			if (chain) {
				chain = chain.then(() => this._executeGraphNode(node, context));
			} else {
				const result = this._executeGraphNode(node, context);
				if (result && typeof (result as Promise<void>).then === "function") {
					chain = result as Promise<void>;
				}
			}
		}
		return chain ?? undefined;
	}

	private async _executePostProcessStage(
		compiled: WebGLCompiledFrameGraphStage | undefined,
		_context: FrameContext,
	): Promise<void> {
		const graphFrame = this._postProcessGraphFrame;
		const nodes = (compiled?.nodes ?? []).filter((node) => !!node.postProcess);
		if (!graphFrame || nodes.length === 0) return;
		const plan: PostProcessExecutionPlan = {
			graph: graphFrame.graph,
			outputColor: this._postProcessOutputColor,
			nodes: nodes.map((node) => ({
				...node.postProcess!,
				nodeId: node.id,
			})),
		};
		const frame = await this._postProcessRuntime.beginGraphFrame(plan);
		if (!frame) return;
		try {
			for (const node of plan.nodes) {
				const result = await this._postProcessRuntime.executeGraphPass(frame, node.passId);
				this._lastExecutedGraphNodeIds.push(node.nodeId);
				if (result.ran === false && node.plannedOutputColor) {
					this._compiler.recordSkippedNode(
						node.nodeId,
						node.plannedOutputColor,
						this._postProcessRuntime.resolveGraphColor(frame, node.plannedOutputColor),
					);
				}
			}
			await this._postProcessRuntime.endGraphFrame(frame);
		} catch (error) {
			await this._postProcessRuntime.abortFrame(error);
			throw error;
		}
	}

	private _recordCompiledStageExecution(stage: string): void {
		if (!this._wholeFrameGraphCompiled) return;
		for (const node of this._findCompiledStage(stage)?.nodes ?? []) {
			this._lastExecutedGraphNodeIds.push(node.id);
		}
	}

	private _warnUnsupportedPass(pass: FramePass): void {
		const key = `webgl-frame-graph-stage-unsupported-${pass.stage}`;
		Logger.warn(
			`[${key}] WebGL frame graph has no nodes for pass "${pass.stage}"; skipping.`,
			{ scope: "WebGLFrameGraphRuntime", onceKey: key },
		);
	}

	private _handleWholeFrameGraphDiagnostics(
		diagnostics: readonly RenderGraphDiagnostic[],
	): void {
		const errors = diagnostics.filter((diagnostic) =>
			diagnostic.enforcement === "enforced" && diagnostic.severity === "error",
		);
		if (errors.length <= 0) return;
		throw new Error(
			`WebGL internal whole-frame graph validation failed: ` +
				errors.map((diagnostic) => diagnostic.message).join(" "),
		);
	}

	private _compileAndExecute(
		plan: WebGLFrameGraphStagePlan,
		context: FrameContext
	): void | Promise<void> {
		this._lastPlannedGraphNodes.push(...plan.nodes);
		const compiled = this._compiler.compileStage(plan);
		this._lastCompiledGraphStages.push(compiled);
		this._handleGraphDiagnostics(compiled);
		let chain: Promise<void> | null = null;
		for (const node of compiled.nodes) {
			if (chain) {
				chain = chain.then(() => this._executeGraphNode(node, context));
			} else {
				const result = this._executeGraphNode(node, context);
				if (result && typeof (result as Promise<void>).then === "function") {
					chain = result as Promise<void>;
				}
			}
		}
		return chain ?? undefined;
	}

	private _executeGraphNode(
		node: WebGLFrameGraphNode,
		context: FrameContext
	): void | Promise<void> {
		const result = this._nodeExecutors.execute(node, context);
		this._lastExecutedGraphNodeIds.push(node.id);
		return result;
	}

	private _handleGraphDiagnostics(
		compiled: WebGLCompiledFrameGraphStage
	): void {
		const errors = compiled.diagnostics.filter(
			(diagnostic) => diagnostic.severity === "error"
		);
		if (errors.length <= 0) {
			return;
		}
		throw new Error(
			`WebGL internal frame graph validation failed for stage ` +
				`"${compiled.pass.stage}": ` +
				errors.map((diagnostic) => diagnostic.message).join(" ")
		);
	}

	private _createPlannerState(context: FrameContext): WebGLFrameGraphPlannerState {
		return {
			oitActive: this._executor.isOITActive(),
			hasParticleSystems: (context.scene.particleSystems?.length ?? 0) > 0,
			hasEnvironmentBackground:
				!this._isIncrementalPartial(context) &&
				context.features.enableEnvironment &&
				context.scene.environment.backgroundEnabled &&
				!!context.scene.environment.backgroundTexture,
		};
	}

	private _requiresMaterialGBuffer(
		declarations: PostProcessDeclarationPlan,
	): boolean {
		return declarations.passes.some((pass) =>
			pass.declaration.gBuffer?.some((entry) =>
				WEBGL_MATERIAL_GBUFFER_SEMANTICS.has(entry.semantic),
			) === true,
		);
	}

	private _isIncrementalPartial(context: FrameContext): boolean {
		const incremental = context.incremental;
		return (
			incremental.enabled &&
			!incremental.forceFullFrame &&
			incremental.dirtyRects.length > 0
		);
	}

	private _createSyntheticPass(stage: string): FramePass {
		return {
			stage,
			executor: "backend",
			enabled: true,
			dependsOn: [],
		};
	}
}
