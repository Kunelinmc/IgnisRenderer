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
}

/** @internal Opaque prepared post-process graph frame owned by one backend runtime. */
export interface PreparedPostProcessFrame {
	readonly graph: CompiledPostProcessGraph;
	/** @internal Opaque identity that prevents cross-frame reuse. */
	readonly token: object;
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

	public constructor(options: BackendPostProcessRuntimeOptions) {
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
		});
		this._observePasses(graph);
		return graph;
	}

	/** @internal Alias used by post-process render graph runners. */
	public compileFrame(context: FrameContext): CompiledPostProcessGraph {
		return this.compileGraph(context);
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
				context.postProcess.getEnabledPasses().map((resolved) => resolved.pass)
			);
			for (const [pass, implementation] of this._implementations.entries()) {
				if (!activePasses.has(pass)) {
					implementation.destroy?.();
					this._implementations.delete(pass);
				}
			}
			const graph = this.compileFrame(context);
			if (graph.passes.length <= 0) return;
			const compiled = this._renderGraphCompiler.compile(
				this._renderGraphAdapter.build(graph)
			);
			this._lastRenderGraph = compiled;
			const errors = compiled.diagnostics.filter(
				(diagnostic) => diagnostic.severity === "error"
			);
			if (errors.length > 0) {
				throw new Error(errors.map((diagnostic) => diagnostic.message).join(" "));
			}
			const frame = await this.beginGraphFrame(graph);
			if (!frame) return;
			try {
				for (const node of compiled.nodes) {
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
		graph: CompiledPostProcessGraph
	): Promise<PreparedPostProcessFrame | null> {
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
		const token = {};
		this._pendingFrame = {
			frameRequest,
			executedPassIds: [],
			attemptedPassIds: new Set(),
			graph,
			token,
		};
		await this._executor.beginFrame?.(frameRequest);
		return { graph, token };
	}

	/** @internal Executes exactly one pass belonging to an active graph frame. */
	public async executeGraphPass(
		frame: PreparedPostProcessFrame,
		passId: string
	): Promise<PostProcessPassResult> {
		const pending = this._pendingFrame;
		if (!pending || pending.graph !== frame.graph || pending.token !== frame.token) {
			throw new Error("Post-process graph frame is not active.");
		}
		const resolved = frame.graph.passes.find((pass) => pass.id === passId);
		if (!resolved) throw new Error(`Post-process graph has no pass "${passId}".`);
		if (pending.attemptedPassIds.has(resolved.id)) {
			throw new Error(`Post-process graph pass "${passId}" already executed.`);
		}
		pending.attemptedPassIds.add(resolved.id);
		const request: PostProcessPassRequest = {
			...pending.frameRequest,
			pass: resolved.pass,
			passId: resolved.id,
			implementation: resolved.implementation,
			options: resolved.options,
			startPassId: frame.graph.startPassId,
		};
		const executionContext = resolved.implementation?.execute ?
			this._executor.getPassExecutionContext?.({
				...request,
				implementation: resolved.implementation,
			} satisfies PostProcessPassExecutionContextRequest)
		: undefined;
		const result = (await resolved.pass.execute(request, executionContext, this._executor)) ?? {};
		await this._executor.completePass?.(request, result);
		if (result.ran === false) return result;
		if (result.preservesOutsideDirtyTiles !== true) {
			this._completedFramePreservesOutsideDirtyTiles = false;
		}
		pending.executedPassIds.push(resolved.id);
		if (result.updatedHistoryIds) {
			const undeclared = result.updatedHistoryIds.find(
				(id) => !resolved.historyIds.includes(id)
			);
			if (undeclared) {
				throw new Error(
					`Post-process pass "${resolved.id}" updated undeclared history "${undeclared}".`
				);
			}
			this._resources.markUpdatedMany(result.updatedHistoryIds);
		} else if (result.historyUpdated) {
			this._resources.markUpdatedMany(
				resolved.historyIds.filter((id) => id !== "motion")
			);
		}
		return result;
	}

	/** @internal Completes post-process hooks without committing temporal history. */
	public async endGraphFrame(frame: PreparedPostProcessFrame): Promise<void> {
		const pending = this._pendingFrame;
		if (!pending || pending.graph !== frame.graph || pending.token !== frame.token) {
			throw new Error("Post-process graph frame is not active.");
		}
		await this._executor.endFrame?.({
			...pending.frameRequest,
			executedPassIds: pending.executedPassIds,
		});
	}

	/** @internal Returns the latest logical post-process graph compilation. */
	public getLastRenderGraph(): CompiledRenderGraph<PostProcessRenderGraphNodePayload> | null {
		return this._lastRenderGraph;
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
		this._resources.commitFrame();
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
		await this._executor.abortFrame?.({
			...pending.frameRequest,
			executedPassIds: pending.executedPassIds,
			error,
		});
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
