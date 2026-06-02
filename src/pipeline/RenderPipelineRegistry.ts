import {
	type FrameContext,
	type FramePass,
	type FramePassStage,
	type PreparedScene,
	type RendererFramePlan,
	type ResolvedFeatureState,
	type TransientStore,
} from "./types";
import {
	RendererStageGraph,
	type RendererStageDefinition,
	type RendererStageEnableContext,
} from "./RendererStageGraph";
import {
	getDefaultIncrementalRegistry,
	type IncrementalDirtyReasonDescriptor,
	type IncrementalFrameContext,
	type IncrementalRegistry,
	type PostProcessIncrementalMetadata,
	type RenderDirtyReason,
} from "./incremental";
import type {
	PostProcessPass,
	ResolvedPostProcessState,
} from "../postprocess";

export type RenderPipelineStageKind =
	| "renderer"
	| "backend-pass"
	| "shared-pass";

export interface RenderPipelineStageRunContext {
	frame: PreparedScene;
	features: ResolvedFeatureState;
	postProcess: ResolvedPostProcessState;
	transient: TransientStore;
	frameContext?: FrameContext;
	incremental?: IncrementalFrameContext;
}

export type RenderPipelineStagePredicate = (
	context: RenderPipelineStageRunContext
) => boolean;

export interface RenderPipelineStageIncrementalOptions {
	order?: number;
}

export interface RenderPipelineStageRegistration {
	id: FramePassStage;
	kind: RenderPipelineStageKind;
	dependsOn?: readonly FramePassStage[];
	enabled?: (context: RendererStageEnableContext) => boolean;
	shouldRun?: RenderPipelineStagePredicate;
	incremental?: RenderPipelineStageIncrementalOptions;
}

export interface RenderPipelineRegistryOptions {
	stages?: readonly RenderPipelineStageRegistration[];
	incrementalRegistry?: IncrementalRegistry;
}

export interface RenderPipelineFramePlanOptions {
	stageOrder: readonly RendererStageDefinition[];
	frame: PreparedScene;
	features: ResolvedFeatureState;
	postProcess: ResolvedPostProcessState;
	transient: TransientStore;
	incremental?: IncrementalFrameContext;
	frameContext?: FrameContext;
	incrementalStartStageIndex?: number;
}

interface RegisteredPipelineStage extends RenderPipelineStageRegistration {
	dependsOn: readonly FramePassStage[];
	builtIn: boolean;
}

function isFramePassKind(kind: RenderPipelineStageKind): boolean {
	return kind === "backend-pass" || kind === "shared-pass";
}

export class RenderPipelineRegistry {
	private _stageGraph: RendererStageGraph;
	private _stages = new Map<FramePassStage, RegisteredPipelineStage>();
	private _incrementalRegistry: IncrementalRegistry;

	constructor(options: RenderPipelineRegistryOptions = {}) {
		this._incrementalRegistry =
			options.incrementalRegistry ?? getDefaultIncrementalRegistry();
		this._stageGraph = new RendererStageGraph();
		for (const stage of options.stages ?? []) {
			this._registerPipelineStage(stage, true);
		}
	}

	public get incremental(): IncrementalRegistry {
		return this._incrementalRegistry;
	}

	/**
	 * Registers a custom dirty reason in the shared incremental registry.
	 *
	 * @param descriptor Dirty reason behavior used by incremental planning.
	 * @returns Allocated dirty reason mask.
	 * @sideEffects Mutates the incremental registry.
	 */
	public registerDirtyReason(
		descriptor: IncrementalDirtyReasonDescriptor
	): number {
		return this._incrementalRegistry.registerDirtyReason(descriptor);
	}

	public unregisterDirtyReason(id: RenderDirtyReason): void {
		this._incrementalRegistry.unregisterDirtyReason(id);
	}

	/**
	 * Registers a renderer pipeline stage or backend/shared frame pass stage.
	 *
	 * @param stage Stage metadata used for ordering, enablement, pass planning,
	 * and incremental first-pass ordering.
	 * @returns Nothing.
	 * @constraints Custom `renderer` stages are planning-only unless `Renderer`
	 * owns an internal executor for the id.
	 * @sideEffects Mutates the stage graph and, for pass stages, incremental
	 * pass ordering metadata.
	 */
	public registerPipelineStage(stage: RenderPipelineStageRegistration): void {
		this._registerPipelineStage(stage, false);
	}

	public unregisterPipelineStage(id: FramePassStage): void {
		const stage = this._stages.get(id);
		if (!stage) {
			return;
		}
		if (stage.builtIn) {
			throw new Error(`Cannot unregister built-in pipeline stage "${id}".`);
		}
		this._stages.delete(id);
		this._stageGraph.removeDependency("sync-out", id);
		this._stageGraph.unregisterStage(id);
		if (isFramePassKind(stage.kind)) {
			this._incrementalRegistry.unregisterFramePass(id);
		}
	}

	public getExecutionOrder(
		context: RendererStageEnableContext,
		warn: (key: string, message: string) => void
	): RendererStageDefinition[] {
		return this._stageGraph.getExecutionOrder(context, warn);
	}

	/**
	 * Registers post-process incremental metadata from a logical pass.
	 *
	 * @param pass Logical post-process pass whose `builtIn` flag controls
	 * incremental metadata ownership.
	 * @returns Nothing.
	 * @sideEffects Mutates the incremental post-process metadata registry.
	 */
	public registerPostProcessPass(pass: PostProcessPass): void {
		this._incrementalRegistry.registerPostProcessPass(pass);
	}

	public unregisterPostProcessPass(id: string): void {
		this._incrementalRegistry.unregisterPostProcessPass(id);
	}

	public getStageKind(stageId: FramePassStage): RenderPipelineStageKind | null {
		return this._stages.get(stageId)?.kind ?? null;
	}

	public isFramePassStage(stageId: FramePassStage): boolean {
		const stage = this._stages.get(stageId);
		return stage ? isFramePassKind(stage.kind) : false;
	}

	public createFramePass(stageId: FramePassStage): FramePass {
		const stage = this._stages.get(stageId);
		if (!stage || !isFramePassKind(stage.kind)) {
			throw new Error(`Pipeline stage "${stageId}" is not a frame pass.`);
		}
		return {
			stage: stage.id,
			executor: stage.kind === "shared-pass" ? "shared" : "backend",
			enabled: true,
			dependsOn: stage.dependsOn.slice(),
		};
	}

	/**
	 * Builds the renderer-owned backend pass plan for one frame.
	 *
	 * @param options Current stage order, frame state, and incremental context
	 * used to resolve backend/shared pass enablement.
	 * @returns Immutable-by-convention frame plan consumed by the renderer and
	 * backend validators.
	 * @constraints `stageOrder` must come from this registry's stage graph for
	 * the same frame.
	 * @sideEffects None.
	 */
	public createFramePlan(
		options: RenderPipelineFramePlanOptions
	): RendererFramePlan {
		const stageIndexById = new Map<string, number>();
		for (let index = 0; index < options.stageOrder.length; index++) {
			stageIndexById.set(options.stageOrder[index].id, index);
		}

		const backendPasses: FramePass[] = [];
		for (const stage of options.stageOrder) {
			const stageId = stage.id as FramePassStage;
			if (!this.isFramePassStage(stageId)) {
				continue;
			}
			const pass = this.createFramePass(stageId);
			const stageIndex =
				stageIndexById.get(stage.id) ?? Number.MAX_SAFE_INTEGER;
			const skippedByIncremental =
				options.incrementalStartStageIndex !== undefined &&
				options.incrementalStartStageIndex >= 0 &&
				stageIndex < options.incrementalStartStageIndex;
			const enabled =
				!skippedByIncremental &&
				this.shouldRunFramePass(stageId, {
					frame: options.frame,
					features: options.features,
					postProcess: options.postProcess,
					transient: options.transient,
					frameContext: options.frameContext,
					incremental: options.incremental,
				});
			backendPasses.push({
				...pass,
				enabled,
			});
		}

		return {
			stageOrder: options.stageOrder.map((stage) => ({
				id: stage.id,
				kind:
					this._stages.get(stage.id as FramePassStage)?.kind ??
					"renderer",
				dependsOn: Array.from(stage.dependsOn ?? []),
			})),
			backendPasses,
		};
	}

	public shouldRunFramePass(
		stageId: FramePassStage,
		context: RenderPipelineStageRunContext
	): boolean {
		const incremental = context.incremental;
		if (
			incremental?.enabled &&
			!incremental.forceFullFrame &&
			incremental.dirtyRects.length === 0
		) {
			return false;
		}
		const stage = this._stages.get(stageId);
		if (!stage || !isFramePassKind(stage.kind)) {
			return false;
		}
		return stage.shouldRun ? stage.shouldRun(context) : true;
	}

	private _registerPipelineStage(
		stage: RenderPipelineStageRegistration,
		builtIn: boolean
	): void {
		if (!stage.id) {
			throw new Error("Renderer pipeline stage id is required.");
		}
		const current = this._stages.get(stage.id);
		if (current && !current.builtIn) {
			throw new Error(`Pipeline stage "${stage.id}" is already registered.`);
		}
		if (current?.builtIn && !builtIn) {
			throw new Error(`Cannot register built-in pipeline stage "${stage.id}".`);
		}
		const dependsOn = Array.from(stage.dependsOn ?? []);
		const registered: RegisteredPipelineStage = {
			...stage,
			dependsOn,
			builtIn,
		};
		this._stages.set(stage.id, registered);
		this._stageGraph.registerStage({
			id: stage.id,
			dependsOn,
			enabled: stage.enabled,
		});
		if (
			!builtIn &&
			isFramePassKind(stage.kind) &&
			this._stageGraph.hasStage("sync-out")
		) {
			this._stageGraph.addDependency("sync-out", stage.id);
		}
		if (isFramePassKind(stage.kind)) {
			this._incrementalRegistry.registerFramePass(
				{
					id: stage.id,
					order: stage.incremental?.order,
				},
				builtIn
			);
		}
	}
}

export type { PostProcessIncrementalMetadata };
