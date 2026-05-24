import {
	FRAME_PASS_DEPENDENCIES,
	type FrameContext,
	type FramePass,
	type PreparedScene,
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
} from "./PostProcessController";

export interface RenderPipelinePassRunContext {
	frame: PreparedScene;
	features: ResolvedFeatureState;
	postProcess: ResolvedPostProcessState;
	transient: TransientStore;
	frameContext?: FrameContext;
	incremental?: IncrementalFrameContext;
}

export type RenderPipelinePassPredicate = (
	context: RenderPipelinePassRunContext
) => boolean;

export interface RenderPipelineBackendPassIncrementalOptions {
	order?: number;
}

export interface RenderPipelineBackendPassRegistration {
	id: FramePass["stage"];
	dependsOn?: readonly string[];
	executor?: FramePass["executor"];
	enabled?: (context: RendererStageEnableContext) => boolean;
	shouldRun?: RenderPipelinePassPredicate;
	incremental?: RenderPipelineBackendPassIncrementalOptions;
}

export interface RenderPipelineRegistryOptions {
	stages?: readonly RendererStageDefinition[];
	backendPasses?: readonly RenderPipelineBackendPassRegistration[];
	incrementalRegistry?: IncrementalRegistry;
}

interface RegisteredBackendPass extends RenderPipelineBackendPassRegistration {
	dependsOn: readonly string[];
	builtIn: boolean;
}

export class RenderPipelineRegistry {
	private _stageGraph: RendererStageGraph;
	private _backendPasses = new Map<FramePass["stage"], RegisteredBackendPass>();
	private _incrementalRegistry: IncrementalRegistry;

	constructor(options: RenderPipelineRegistryOptions = {}) {
		this._incrementalRegistry =
			options.incrementalRegistry ?? getDefaultIncrementalRegistry();
		this._stageGraph = new RendererStageGraph(
			Array.from(options.stages ?? [])
		);
		for (const pass of options.backendPasses ?? []) {
			this._registerBackendPass(pass, true);
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

	public registerStage(stage: RendererStageDefinition): void {
		this._stageGraph.registerStage(stage);
	}

	public unregisterStage(id: string): void {
		this._stageGraph.unregisterStage(id);
	}

	public setStages(stages: readonly RendererStageDefinition[]): void {
		this._stageGraph.setStages(Array.from(stages));
	}

	public getExecutionOrder(
		context: RendererStageEnableContext,
		warn: (key: string, message: string) => void
	): RendererStageDefinition[] {
		return this._stageGraph.getExecutionOrder(context, warn);
	}

	/**
	 * Registers a custom backend or shared frame pass.
	 *
	 * @param pass Pass metadata used by Renderer for ordering, enablement,
	 * executor selection, and incremental first-pass ordering.
	 * @sideEffects Mutates the renderer stage graph and incremental pass order.
	 */
	public registerBackendPass(
		pass: RenderPipelineBackendPassRegistration
	): void {
		this._registerBackendPass(pass, false);
	}

	public unregisterBackendPass(id: FramePass["stage"]): void {
		const pass = this._backendPasses.get(id);
		if (!pass) {
			return;
		}
		if (pass.builtIn) {
			throw new Error(`Cannot unregister built-in backend pass "${id}".`);
		}
		this._backendPasses.delete(id);
		this._stageGraph.removeDependency("sync-out", id);
		this._stageGraph.unregisterStage(id);
		FRAME_PASS_DEPENDENCIES.delete(id);
		this._incrementalRegistry.unregisterFramePass(id);
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

	public isBackendPassStage(stageId: FramePass["stage"]): boolean {
		return this._backendPasses.has(stageId);
	}

	public createBackendPass(
		stageId: FramePass["stage"],
		passExecutors?: Partial<Record<FramePass["stage"], FramePass["executor"]>>
	): FramePass {
		const pass = this._backendPasses.get(stageId);
		return {
			stage: stageId,
			executor: passExecutors?.[stageId] ?? pass?.executor ?? "backend",
			enabled: true,
		};
	}

	public shouldRunBackendPass(
		stageId: FramePass["stage"],
		context: RenderPipelinePassRunContext
	): boolean {
		const incremental = context.incremental;
		if (
			incremental?.enabled &&
			!incremental.forceFullFrame &&
			incremental.dirtyRects.length === 0
		) {
			return false;
		}
		const pass = this._backendPasses.get(stageId);
		if (!pass) {
			return false;
		}
		return pass.shouldRun ? pass.shouldRun(context) : true;
	}

	private _registerBackendPass(
		pass: RenderPipelineBackendPassRegistration,
		builtIn: boolean
	): void {
		if (!pass.id) {
			throw new Error("Renderer backend pass id is required.");
		}
		const current = this._backendPasses.get(pass.id);
		if (current && !current.builtIn) {
			throw new Error(`Backend pass "${pass.id}" is already registered.`);
		}
		if (current?.builtIn && !builtIn) {
			throw new Error(`Cannot register built-in backend pass "${pass.id}".`);
		}
		const dependsOn = Array.from(pass.dependsOn ?? []);
		const registered: RegisteredBackendPass = {
			...pass,
			dependsOn,
			builtIn,
		};
		this._backendPasses.set(pass.id, registered);
		this._stageGraph.registerStage({
			id: pass.id,
			dependsOn,
			enabled: pass.enabled,
		});
		if (!builtIn && this._stageGraph.hasStage("sync-out")) {
			this._stageGraph.addDependency("sync-out", pass.id);
		}
		if (!builtIn) {
			FRAME_PASS_DEPENDENCIES.set(pass.id, dependsOn);
		}
		this._incrementalRegistry.registerFramePass(
			{
				id: pass.id,
				order: pass.incremental?.order,
			},
			builtIn
		);
	}
}

export type { PostProcessIncrementalMetadata };
