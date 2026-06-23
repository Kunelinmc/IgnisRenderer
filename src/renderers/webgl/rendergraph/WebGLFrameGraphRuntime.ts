import type {
	FrameContext,
	FramePass,
} from "../../../pipeline/types";
import { Logger } from "../../../foundation/Logger";
import type { BackendPostProcessRuntime } from "../../../postprocess/BackendPostProcessRuntime";
import type { WebGLFrameExecutor } from "../WebGLFrameExecutor";
import { WebGLFrameGraphCompiler } from "./WebGLFrameGraphCompiler";
import { WebGLFrameGraphPlanner } from "./WebGLFrameGraphPlanner";
import type {
	WebGLCompiledFrameGraphStage,
	WebGLFrameGraphDebugState,
	WebGLFrameGraphDiagnostic,
	WebGLFrameGraphNode,
	WebGLFrameGraphNodeKind,
	WebGLFrameGraphPlannerState,
	WebGLFrameGraphStagePlan,
} from "./types";

type WebGLGraphNodeExecutor = (
	node: WebGLFrameGraphNode,
	context: FrameContext
) => void | Promise<void>;

/**
 * Orchestrates WebGL backend-private frame graph planning and node execution.
 */
export class WebGLFrameGraphRuntime {
	private readonly _executor: WebGLFrameExecutor;
	private readonly _postProcessRuntime: BackendPostProcessRuntime;
	private readonly _planner = new WebGLFrameGraphPlanner();
	private readonly _compiler = new WebGLFrameGraphCompiler();
	private readonly _nodeExecutors: Map<
		WebGLFrameGraphNodeKind,
		WebGLGraphNodeExecutor
	>;
	private _active = false;
	private _earlyZPacketIds: ReadonlySet<string> = new Set<string>();
	private _lastPlannedGraphNodes: WebGLFrameGraphNode[] = [];
	private _lastCompiledGraphStages: WebGLCompiledFrameGraphStage[] = [];
	private _lastExecutedGraphNodeIds: string[] = [];
	private _runtimeDiagnostics: WebGLFrameGraphDiagnostic[] = [];

	public constructor(
		executor: WebGLFrameExecutor,
		postProcessRuntime: BackendPostProcessRuntime
	) {
		this._executor = executor;
		this._postProcessRuntime = postProcessRuntime;
		this._nodeExecutors = this._createNodeExecutors();
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
		this._earlyZPacketIds = new Set<string>();
		this._lastPlannedGraphNodes = [];
		this._lastCompiledGraphStages = [];
		this._lastExecutedGraphNodeIds = [];
		this._runtimeDiagnostics = [];
		this._executor.beginFrame(context);
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
		const plan = this._planner.planStage(
			pass,
			context,
			this._createPlannerState(context)
		);
		if (plan.nodes.length <= 0) {
			const key = `webgl-frame-graph-stage-unsupported-${pass.stage}`;
			Logger.warn(
				`[${key}] WebGL frame graph has no nodes for pass "${pass.stage}"; skipping.`,
				{ scope: "WebGLFrameGraphRuntime", onceKey: key }
			);
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
		const pass = this._createSyntheticPass("webgl-present");
		const plan: WebGLFrameGraphStagePlan = {
			pass,
			nodes: this._planner.planPresent(),
		};
		const result = this._compileAndExecute(plan, context);
		const finish = () => {
			this._executor.finishFrame();
			this._active = false;
			this._earlyZPacketIds = new Set<string>();
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
	public abortFrame(): void {
		this._executor.abortFrame();
		this._active = false;
		this._earlyZPacketIds = new Set<string>();
	}

	public getDebugState(): WebGLFrameGraphDebugState {
		return {
			active: this._active,
			oitActive: this._executor.isOITActive(),
			hasPresentedInFrame: this._executor.hasPresentedInFrame(),
			lastPlannedNodeIds: this._lastPlannedGraphNodes.map((node) => node.id),
			lastExecutedNodeIds: this._lastExecutedGraphNodeIds.slice(),
			compiledStages: this._lastCompiledGraphStages.slice(),
			graphResources: this._compiler.getResourceDebugState(),
			graphBarriers: this._compiler.getBarriers(),
			graphDiagnostics: [
				...this._compiler.getDiagnostics(),
				...this._runtimeDiagnostics,
			],
		};
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
		const executor = this._nodeExecutors.get(node.kind);
		if (!executor) {
			const diagnostic: WebGLFrameGraphDiagnostic = {
				severity: "error",
				nodeId: node.id,
				resource: node.kind,
				code: "missing-node-executor",
				message:
					`WebGL frame graph node kind "${node.kind}" has no executor.`,
			};
			this._runtimeDiagnostics.push(diagnostic);
			throw new Error(diagnostic.message);
		}
		const result = executor(node, context);
		this._lastExecutedGraphNodeIds.push(node.id);
		return result;
	}

	private _createNodeExecutors(): Map<
		WebGLFrameGraphNodeKind,
		WebGLGraphNodeExecutor
	> {
		return new Map([
			["scene-clear", (_node, context) => this._executor.clearFrameTargets(context)],
			["environment", (_node, context) => this._executor.renderEnvironmentNode(context)],
			["shadow", (_node, context) => this._executor.renderShadowNode(context)],
			[
				"opaque-depth-prepass",
				(_node, context) => {
					this._earlyZPacketIds =
						this._executor.renderOpaqueDepthPrepass(context);
				},
			],
			[
				"opaque-scene",
				(_node, context) =>
					this._executor.renderOpaqueScene(context, this._earlyZPacketIds),
			],
			[
				"transparent-legacy",
				(node, context) => {
					if (node.scope === "transparent" || node.scope === "particles") {
						this._executor.renderOITLegacyTransparent(context);
						return;
					}
					this._executor.renderTransparentLegacy(context);
				},
			],
			[
				"oit-clear",
				(node, context) => {
					if (node.scope === "particles") {
						this._executor.prepareOITParticles();
						return;
					}
					this._executor.prepareOITTransparent(context);
				},
			],
			[
				"oit-accum",
				(node, context) => {
					if (node.scope === "particles") {
						this._executor.renderOITParticleAccum(context);
						return;
					}
					this._executor.renderOITTransparentAccum(context);
				},
			],
			[
				"oit-reveal",
				(node, context) => {
					if (node.scope === "particles") {
						this._executor.renderOITParticleReveal(context);
						return;
					}
					this._executor.renderOITTransparentReveal(context);
				},
			],
			["oit-resolve", (_node, context) => this._executor.resolveOIT(context)],
			[
				"particles",
				(node, context) => {
					if (node.scope === "particles") {
						this._executor.renderOITAdditiveParticles(context);
						return;
					}
					this._executor.renderParticlesLegacy(context);
				},
			],
			[
				"postprocess",
				(_node, context) => this._postProcessRuntime.execute(context),
			],
			["present", () => this._executor.presentFrame()],
		]);
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
