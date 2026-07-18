import type { FrameContext } from "../pipeline/types";
import { RenderGraphCompiler } from "../rendergraph/RenderGraphCompiler";
import type { CompiledRenderGraph } from "../rendergraph/types";
import {
	PostProcessGraphCompiler,
	type CompiledPostProcessGraph,
} from "./PostProcessGraphCompiler";
import { PostProcessResourcePool } from "./PostProcessResourcePool";
import {
	PostProcessRenderGraphAdapter,
	type PostProcessRenderGraphNodePayload,
} from "./PostProcessRenderGraphAdapter";
import type { IRenderBackend } from "../renderers/IRenderBackend";
import type {
	IPostProcessExecutor,
	LogicalGBufferBridge,
	LogicalGBufferSemantic,
	PostProcessFrameRequest,
	PostProcessPassExecutionContextRequest,
	PostProcessPassCompletion,
	PostProcessGraphFrameBinding,
	PostProcessPassRequest,
	PostProcessPassImplementation,
	PostProcessPassResult,
} from "./types";
import type { PostProcessPass } from "./PostProcessPass";

export interface BackendPostProcessRuntimeOptions {
	readonly executor: IPostProcessExecutor;
	readonly backend: IRenderBackend;
	readonly warn?: (key: string, message: string) => void;
}

interface PendingBackendPostProcessFrame {
	readonly frameRequest: PostProcessFrameRequest;
	readonly executedPassIds: string[];
	readonly attemptedPassIds: Set<string>;
	readonly graph: CompiledPostProcessGraph;
	readonly token: object;
	readonly logicalGraph: CompiledRenderGraph<PostProcessRenderGraphNodePayload>;
	readonly resolvedColorAliases: Map<string, string>;
	readonly skippedPassIds: string[];
	status: "prepared" | "executing" | "ended";
	readonly binding: PostProcessGraphFrameBinding | null;
}

/** @internal Fully compiled post-process frame, before pool allocation. */
export interface CompiledPostProcessFrame {
	readonly graph: CompiledPostProcessGraph;
	readonly logicalGraph: CompiledRenderGraph<PostProcessRenderGraphNodePayload>;
	readonly outputColor: string;
}

/** @internal Sanitized outcome of one completed post-process graph transaction. */
export interface PostProcessGraphExecutionResult {
	readonly outputColor: string;
	readonly resolvedOutputColor: string;
	readonly executedPassIds: readonly string[];
	readonly skippedPassIds: readonly string[];
	readonly preservesOutsideDirtyTiles: boolean;
}

/** @internal Internal diagnostic snapshots without native resource handles. */
export interface PostProcessGraphDebugState {
	readonly lastAttempt: PostProcessGraphExecutionResult | null;
	readonly lastSuccessful: PostProcessGraphExecutionResult | null;
}

/** @internal Opaque prepared post-process graph frame owned by one backend runtime. */
export interface PreparedPostProcessFrame {
	readonly graph: CompiledPostProcessGraph;
	/** @internal Opaque identity that prevents cross-frame reuse. */
	readonly token: object;
	readonly compiled: CompiledPostProcessFrame;
}

/**
 * Executes backend-owned post-process graph lifecycle for one render backend.
 */
export class BackendPostProcessRuntime {
	private readonly _executor: IPostProcessExecutor;
	private readonly _backend: IRenderBackend;
	private readonly _warn: (key: string, message: string) => void;
	private readonly _compiler = new PostProcessGraphCompiler();
	private readonly _renderGraphCompiler = new RenderGraphCompiler();
	private readonly _renderGraphAdapter = new PostProcessRenderGraphAdapter();
	private readonly _resources = new PostProcessResourcePool();
	private readonly _observedPasses = new Set<PostProcessPass>();
	private readonly _implementations = new Map<PostProcessPass, PostProcessPassImplementation>();
	private _pendingFrame: PendingBackendPostProcessFrame | null = null;
	private _completedFramePreservesOutsideDirtyTiles = true;
	private _lastRenderGraph: CompiledRenderGraph<PostProcessRenderGraphNodePayload> | null = null;
	private _lastAttempt: PostProcessGraphExecutionResult | null = null;
	private _lastSuccessful: PostProcessGraphExecutionResult | null = null;

	constructor(options: BackendPostProcessRuntimeOptions) {
		this._executor = options.executor;
		this._backend = options.backend;
		this._warn = options.warn ?? (() => {});
	}

	private _resolveImplementation(pass: PostProcessPass): PostProcessPassImplementation | null {
		let impl = this._implementations.get(pass);
		if (!impl) {
			const factory = pass.getImplementationFactory(this._executor.backend);
			if (factory) {
				impl = typeof factory === "function" ? factory(this._backend) : (factory as any);
				this._implementations.set(pass, impl);
			}
		}
		return impl ?? null;
	}

	private _compileExistingGraph(graph: CompiledPostProcessGraph): CompiledPostProcessFrame {
		const subgraph = this._renderGraphAdapter.build(graph);
		const logicalGraph = this._renderGraphCompiler.compile(subgraph);
		this._lastRenderGraph = logicalGraph;
		const errors = logicalGraph.diagnostics.filter(
			(diagnostic) => diagnostic.severity === "error",
		);
		if (errors.length > 0) {
			throw new Error(errors.map((diagnostic) => diagnostic.message).join(" "));
		}
		return Object.freeze({ graph, logicalGraph, outputColor: subgraph.outputColor });
	}

	/**
	 * Compiles executable post-process graph metadata for the current frame.
	 *
	 * @internal Owned by render backends.
	 * @param context Active renderer frame context.
	 * @returns Compiled graph filtered by runtime G-buffer availability.
	 * @sideEffects May emit diagnostics through the configured warning sink.
	 */
	public compileGraph(context: FrameContext): CompiledPostProcessGraph {
		const graph = this._compiler.compile({
			postProcess: context.postProcess,
			backend: this._executor.backend,
			frameContext: context,
			gBuffer: this._executor.createGBufferBridge(context),
			warn: this._warn,
			resolveImplementation: (pass) => this._resolveImplementation(pass),
			isGraphResourceAvailable: (resourceId) =>
				this._executor.isGraphResourceAvailable?.(resourceId) ?? true,
		});
		this._observePasses(graph);
		return graph;
	}

	/** @internal Alias used by post-process render graph runners. */
	public compileFrame(context: FrameContext): CompiledPostProcessGraph {
		return this.compileGraph(context);
	}

	/** @internal Compiles the executable logical subgraph without allocating resources. */
	public compileRenderGraphFrame(context: FrameContext): CompiledPostProcessFrame {
		const graph = this.compileFrame(context);
		const subgraph = this._renderGraphAdapter.build(graph);
		const logicalGraph = this._renderGraphCompiler.compile(subgraph);
		this._lastRenderGraph = logicalGraph;
		const errors = logicalGraph.diagnostics.filter(
			(diagnostic) => diagnostic.severity === "error",
		);
		if (errors.length > 0) {
			throw new Error(errors.map((diagnostic) => diagnostic.message).join(" "));
		}
		return Object.freeze({ graph, logicalGraph, outputColor: subgraph.outputColor });
	}

	/**
	 * Compiles post-process metadata for backend warmup without live targets.
	 *
	 * @internal Owned by backend warmup implementations.
	 * @param context Warmup frame context.
	 * @returns Compiled warmup graph using synthetic logical G-buffer metadata.
	 * @sideEffects None.
	 */
	public compileWarmupGraph(context: FrameContext): CompiledPostProcessGraph {
		const graph = this._compiler.compile({
			postProcess: context.postProcess,
			backend: this._executor.backend,
			frameContext: context,
			gBuffer: this._createWarmupGBuffer(context),
			warn: () => {},
			resolveImplementation: (pass) => this._resolveImplementation(pass),
		});
		this._observePasses(graph);
		return graph;
	}

	/**
	 * Executes the backend-owned `"postprocess"` pass for one frame.
	 *
	 * @internal Called from `IRenderBackend.executePass()`.
	 * @param context Active renderer frame context.
	 * @returns Promise that resolves after executor pass hooks complete.
	 * @sideEffects Allocates resources, mutates backend frame targets, and records
	 * pending history writes without committing them.
	 */
	public async execute(context: FrameContext): Promise<void> {
		this._completedFramePreservesOutsideDirtyTiles = true;
		if (
			context.incremental.enabled &&
			!context.incremental.forceFullFrame &&
			context.incremental.dirtyRects.length === 0
		) {
			return;
		}
		{
			const activePasses = new Set(
				context.postProcess.getEnabledPasses().map((resolved) => resolved.pass),
			);
			for (const [pass, implementation] of this._implementations.entries()) {
				if (!activePasses.has(pass)) {
					implementation.destroy?.();
					this._implementations.delete(pass);
				}
			}
			const compiled = this.compileRenderGraphFrame(context);
			if (compiled.graph.passes.length <= 0) return;
			const frame = await this.beginGraphFrame(compiled);
			if (!frame) return;
			try {
				for (const node of compiled.logicalGraph.nodes) {
					await this.executeGraphPass(frame, node.payload.passId);
				}
				await this.endGraphFrame(frame);
			} catch (error) {
				await this.abortFrame(error);
				throw error;
			}
			return;
		}

		/* Legacy monolithic loop retained only as migration reference.
		if (this._pendingFrame) {
			await this.abortFrame();
		}

		// Clean up unregistered implementations from cache
		const activePasses = new Set(context.postProcess.getEnabledPasses().map((r) => r.pass));
		for (const [pass, impl] of this._implementations.entries()) {
			if (!activePasses.has(pass)) {
				impl.destroy?.();
				this._implementations.delete(pass);
			}
		}

		const graph = this.compileFrame(context);
		if (graph.passes.length <= 0) {
			return;
		}
		const subgraph = this._renderGraphAdapter.build(graph);
		const compiled = this._renderGraphCompiler.compile(subgraph);
		this._lastRenderGraph = compiled;
		const graphErrors = compiled.diagnostics.filter(
			(diagnostic) => diagnostic.severity === "error"
		);
		if (graphErrors.length > 0) {
			throw new Error(graphErrors.map((diagnostic) => diagnostic.message).join(" "));
		}
		const resources = this._resources.prepare({
			executor: this._executor,
			graph,
			reset: context.incremental.temporalHistoryReset,
		});
		if (resources.transientsChanged) {
			this._executor.invalidateResourceBindings?.();
		}
		const frameRequest: PostProcessFrameRequest = {
			frameContext: context,
			postProcess: context.postProcess,
			gBuffer: graph.gBuffer,
			histories: resources.histories,
			transients: resources.transients,
		};
		const executedPassIds: string[] = [];
		const binding = this._executor.createGraphBinding?.(frameRequest) ?? null;
		this._pendingFrame = {
			frameRequest,
			executedPassIds,
			attemptedPassIds: new Set(),
			graph,
			token: {},
		};

		try {
			await this._executor.beginFrame?.(frameRequest);
			for (const resolved of graph.passes) {
				const passRequest: PostProcessPassRequest = {
					...frameRequest,
					pass: resolved.pass,
					passId: resolved.id,
					implementation: resolved.implementation,
					options: resolved.options,
					startPassId: graph.startPassId,
				};
				const executionContext =
					resolved.implementation?.execute ?
						this._executor.getPassExecutionContext?.({
							...passRequest,
							implementation: resolved.implementation,
						} satisfies PostProcessPassExecutionContextRequest)
					:	undefined;
				const result = await resolved.pass.execute(
					passRequest,
					executionContext,
					this._executor
				);
				await this._executor.completePass?.(passRequest, result ?? {});
				if (result?.ran === false) {
					continue;
				}
				if (result?.preservesOutsideDirtyTiles !== true) {
					this._completedFramePreservesOutsideDirtyTiles = false;
				}
				executedPassIds.push(resolved.id);
				if (result?.updatedHistoryIds) {
					this._resources.markUpdatedMany(result.updatedHistoryIds);
				} else if (result?.historyUpdated) {
					this._resources.markUpdatedMany(
						resolved.historyIds.filter((id) => id !== "motion")
					);
				}
			}
			await this._executor.endFrame?.({
				...frameRequest,
				executedPassIds,
			});
		} catch (error) {
			await this.abortFrame(error);
			throw error;
		}
		*/
	}

	/** @internal Prepares pool-backed resources and opens one graph transaction. */
	public async beginGraphFrame(
		compiled: CompiledPostProcessFrame | CompiledPostProcessGraph,
	): Promise<PreparedPostProcessFrame | null> {
		const frameCompiled =
			"logicalGraph" in compiled ? compiled : this._compileExistingGraph(compiled);
		const graph = frameCompiled.graph;
		if (graph.passes.length <= 0) return null;
		if (this._pendingFrame) await this.abortFrame();
		const resources = this._resources.prepare({
			executor: this._executor,
			graph,
			reset: graph.frameContext.incremental.temporalHistoryReset,
		});
		if (resources.transientsChanged) this._executor.invalidateResourceBindings?.();
		const frameRequest: PostProcessFrameRequest = {
			frameContext: graph.frameContext,
			postProcess: graph.postProcess,
			gBuffer: graph.gBuffer,
			histories: resources.histories,
			transients: resources.transients,
		};
		const binding = this._executor.createGraphBinding?.(frameRequest);
		const token = {};
		this._pendingFrame = {
			frameRequest,
			executedPassIds: [],
			attemptedPassIds: new Set(),
			graph,
			token,
			logicalGraph: frameCompiled.logicalGraph,
			resolvedColorAliases: new Map([["frame:scene-color", "frame:scene-color"]]),
			skippedPassIds: [],
			status: "prepared",
			binding,
		};
		if (!binding) await this._executor.beginFrame?.(frameRequest);
		return { graph, token, compiled: frameCompiled };
	}

	/** @internal Executes exactly one pass belonging to an active graph frame. */
	public async executeGraphPass(
		frame: PreparedPostProcessFrame,
		passId: string,
	): Promise<PostProcessPassResult> {
		const pending = this._pendingFrame;
		if (!pending || pending.graph !== frame.graph || pending.token !== frame.token) {
			throw new Error("Post-process graph frame is not active.");
		}
		if (pending.status === "ended") {
			throw new Error("Post-process graph frame has already ended.");
		}
		const resolved = frame.graph.passes.find((pass) => pass.id === passId);
		if (!resolved) throw new Error(`Post-process graph has no pass "${passId}".`);
		if (pending.attemptedPassIds.has(resolved.id)) {
			throw new Error(`Post-process graph pass "${passId}" already executed.`);
		}
		const node = pending.logicalGraph.nodes.find(
			(candidate) => candidate.payload.passId === passId,
		);
		if (!node) throw new Error(`Post-process graph has no node for pass "${passId}".`);
		const expectedNode = pending.logicalGraph.nodes[pending.attemptedPassIds.size];
		if (expectedNode?.id !== node.id) {
			throw new Error(`Post-process graph pass "${passId}" executed out of order.`);
		}
		pending.attemptedPassIds.add(resolved.id);
		pending.status = "executing";
		const request: PostProcessPassRequest = {
			...pending.frameRequest,
			pass: resolved.pass,
			passId: resolved.id,
			implementation: resolved.implementation,
			options: resolved.options,
			startPassId: frame.graph.startPassId,
		};
		const executionContext = resolved.implementation?.execute
			? this._executor.getPassExecutionContext?.({
					...request,
					implementation: resolved.implementation,
				} satisfies PostProcessPassExecutionContextRequest)
			: undefined;
		await pending.binding?.beginPass?.(request);
		const result =
			(await resolved.pass.execute(request, executionContext, this._executor)) ?? {};
		this._validateHistoryUpdates(resolved, result);
		const completion = pending.binding
			? await pending.binding.completePass?.(request, result)
			: await this._executor.completePass?.(request, result);
		if (result.ran === false) {
			pending.skippedPassIds.push(resolved.id);
			if (node.payload.plannedOutputColor && node.payload.inputColor) {
				pending.resolvedColorAliases.set(
					node.payload.plannedOutputColor,
					this._resolveColorAlias(pending, node.payload.inputColor),
				);
			}
			return result;
		}
		if (
			node.payload.outputValidation === "strict" &&
			node.payload.color.output === "new-version" &&
			completion &&
			completion.published !== true
		) {
			throw new Error(
				`Post-process pass "${passId}" did not publish its assigned color output.`,
			);
		}
		if (node.payload.plannedOutputColor) {
			pending.resolvedColorAliases.set(
				node.payload.plannedOutputColor,
				node.payload.plannedOutputColor,
			);
		}
		if (result.preservesOutsideDirtyTiles !== true) {
			this._completedFramePreservesOutsideDirtyTiles = false;
		}
		pending.executedPassIds.push(resolved.id);
		if (result.updatedHistoryIds) {
			this._resources.markUpdatedMany(result.updatedHistoryIds);
		} else if (result.historyUpdated) {
			this._resources.markUpdatedMany(resolved.historyIds.filter((id) => id !== "motion"));
		}
		return result;
	}

	/** @internal Completes post-process hooks without committing temporal history. */
	public async endGraphFrame(
		frame: PreparedPostProcessFrame,
	): Promise<PostProcessGraphExecutionResult> {
		const pending = this._pendingFrame;
		if (!pending || pending.graph !== frame.graph || pending.token !== frame.token) {
			throw new Error("Post-process graph frame is not active.");
		}
		if (pending.status === "ended") {
			throw new Error("Post-process graph frame has already ended.");
		}
		const pendingResult = this._createExecutionResult(pending, frame.compiled.outputColor);
		if (pending.binding) {
			await pending.binding.endFrame?.(pendingResult.resolvedOutputColor);
		} else {
			await this._executor.endFrame?.({
				...pending.frameRequest,
				executedPassIds: pending.executedPassIds,
			});
		}
		pending.status = "ended";
		const result = pendingResult;
		this._lastAttempt = result;
		return result;
	}

	/** @internal Returns the latest logical post-process graph compilation. */
	public getLastRenderGraph(): CompiledRenderGraph<PostProcessRenderGraphNodePayload> | null {
		return this._lastRenderGraph;
	}

	/** @internal Returns sanitized post-process graph attempt snapshots. */
	public getDebugState(): PostProcessGraphDebugState {
		return Object.freeze({
			lastAttempt: this._lastAttempt,
			lastSuccessful: this._lastSuccessful,
		});
	}

	/**
	 * Reports whether the completed post-process chain proved local preservation.
	 *
	 * @internal Used by backend frame-coverage reporting only.
	 */
	public get completedFramePreservesOutsideDirtyTiles(): boolean {
		return this._completedFramePreservesOutsideDirtyTiles;
	}

	/**
	 * Commits history writes after the backend frame has succeeded.
	 *
	 * @internal Called from backend `endFrame()` after all frame work succeeds.
	 * @returns Nothing.
	 * @sideEffects Swaps pending history resources and clears pending frame state.
	 */
	public commitFrame(): void {
		if (!this._pendingFrame) {
			return;
		}
		if (this._pendingFrame.status !== "ended") {
			throw new Error("Post-process graph frame must end before history can commit.");
		}
		this._resources.commitFrame();
		this._lastSuccessful = this._lastAttempt;
		this._pendingFrame = null;
	}

	/**
	 * Aborts pending post-process state after a failed backend frame.
	 *
	 * @internal Called from backend `abortFrame()`.
	 * @param error Optional failure reason forwarded to executor abort hooks.
	 * @returns Promise that resolves after executor abort hooks complete.
	 * @sideEffects Clears pending history writes without committing them.
	 */
	public async abortFrame(error?: unknown): Promise<void> {
		const pending = this._pendingFrame;
		this._pendingFrame = null;
		this._resources.abortFrame();
		if (!pending) {
			return;
		}
		this._lastAttempt = this._createExecutionResult(
			pending,
			pending.graph.passes.length > 0
				? (pending.logicalGraph.nodes.at(-1)?.payload.plannedOutputColor ??
						"frame:scene-color")
				: "frame:scene-color",
		);
		if (pending.binding) {
			await pending.binding.abortFrame?.(error);
		} else {
			await this._executor.abortFrame?.({
				...pending.frameRequest,
				executedPassIds: pending.executedPassIds,
				error,
			});
		}
	}

	private _resolveColorAlias(pending: PendingBackendPostProcessFrame, color: string): string {
		let resolved = color;
		const visited = new Set<string>();
		while (pending.resolvedColorAliases.has(resolved) && !visited.has(resolved)) {
			visited.add(resolved);
			const next = pending.resolvedColorAliases.get(resolved)!;
			if (next === resolved) break;
			resolved = next;
		}
		return resolved;
	}

	private _createExecutionResult(
		pending: PendingBackendPostProcessFrame,
		outputColor: string,
	): PostProcessGraphExecutionResult {
		return Object.freeze({
			outputColor,
			resolvedOutputColor: this._resolveColorAlias(pending, outputColor),
			executedPassIds: Object.freeze(pending.executedPassIds.slice()),
			skippedPassIds: Object.freeze(pending.skippedPassIds.slice()),
			preservesOutsideDirtyTiles: this._completedFramePreservesOutsideDirtyTiles,
		});
	}

	private _validateHistoryUpdates(
		resolved: CompiledPostProcessGraph["passes"][number],
		result: PostProcessPassResult,
	): void {
		if (!result.updatedHistoryIds) return;
		const undeclared = result.updatedHistoryIds.find((id) => !resolved.historyIds.includes(id));
		if (undeclared) {
			throw new Error(
				`Post-process pass "${resolved.id}" updated undeclared history "${undeclared}".`,
			);
		}
	}

	/**
	 * Invalidates frame-size dependent resources and pass implementations.
	 *
	 * @internal Called from backend resize and target recreation paths.
	 * @returns Nothing.
	 * @sideEffects Destroys active post-process resources and invalidates passes.
	 */
	public invalidateFrameSized(): void {
		this._resources.invalidateFrameSized(this._executor);
		for (const impl of this._implementations.values()) {
			impl.invalidate?.();
		}
		for (const pass of this._observedPasses) {
			pass.invalidate();
		}
	}

	/**
	 * Destroys all runtime-owned resources and pass implementations.
	 *
	 * @internal Called from backend device/context reset and destroy paths.
	 * @returns Nothing.
	 * @sideEffects Releases resources and clears pending frame state.
	 */
	public destroy(): void {
		this._resources.destroy(this._executor);
		for (const impl of this._implementations.values()) {
			impl.destroy?.();
		}
		this._implementations.clear();
		for (const pass of this._observedPasses) {
			pass.destroy();
		}
		this._observedPasses.clear();
		this._pendingFrame = null;
	}

	private _observePasses(graph: CompiledPostProcessGraph): void {
		for (const resolved of graph.passes) {
			this._observedPasses.add(resolved.pass);
		}
	}

	private _createWarmupGBuffer(context: FrameContext): LogicalGBufferBridge {
		const width = Math.max(1, context.attachments?.width ?? 1);
		const height = Math.max(1, context.attachments?.height ?? 1);
		const semantics: readonly LogicalGBufferSemantic[] = [
			"color",
			"depth",
			"normal",
			"motion",
			"world-position",
			"albedo",
			"roughness",
			"metallic",
			"specular",
			"transmission",
			"emissive",
			"occlusion",
		];
		const channels: LogicalGBufferBridge["channels"] = {};
		for (const semantic of semantics) {
			channels[semantic] = {
				semantic,
				width,
				height,
				handle: {
					backend: this._executor.backend,
					resource: null,
				},
			};
		}
		return {
			width,
			height,
			normalSpace: "world",
			depthEncoding: "linear-view-z",
			motionEncoding: "ndc-delta",
			channels,
			worldPosition: {
				source: "derived",
				available: true,
			},
		};
	}
}
