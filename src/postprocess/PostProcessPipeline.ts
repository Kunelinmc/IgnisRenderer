import {
	getEnabledCustomPostProcessPassIds,
	hasEnabledCustomPostProcessPass,
	isFogPostProcessEnabled,
	POST_PROCESS_PASS_IDS,
	type ResolvedPostProcessState,
} from "../pipeline/PostProcessController";
import type { FrameContext } from "../pipeline/types";
import { INTERACTION_TRANSIENT_STATE_KEY } from "../pipeline/types";
import { PostProcessGraph } from "./PostProcessGraph";
import { PostProcessHistoryManager } from "./PostProcessHistoryManager";
import type {
	IPostProcessExecutor,
	LogicalGBufferSemantic,
	PostProcessHistoryDescriptor,
	PostProcessPassDescriptor,
	PostProcessPipelineExecuteRequest,
	PostProcessPipelineExecuteResult,
} from "./types";

const KNOWN_BACKENDS = ["software", "webgpu", "webgl"] as const;

const DEFAULT_HISTORY_USAGE = ["sampled", "storage", "render-target"] as const;

function allBackendsImplementation(): PostProcessPassDescriptor["implementations"] {
	return {
		software: {},
		webgpu: {},
		webgl: {},
	};
}

function enabled(id: string): (state: ResolvedPostProcessState) => boolean {
	return (state) => state.enabled[id] === true;
}

const BUILTIN_POST_PROCESS_PASSES: readonly PostProcessPassDescriptor[] = [
	{
		id: "ssao",
		dependsOn: [],
		requirements: { gBuffer: ["depth", "normal"] },
		isEnabled: enabled("ssao"),
		implementations: allBackendsImplementation(),
	},
	{
		id: "ssgi",
		dependsOn: ["ssao"],
		requirements: { gBuffer: ["depth", "normal"] },
		isEnabled: enabled("ssgi"),
		implementations: allBackendsImplementation(),
	},
	{
		id: "taa",
		dependsOn: ["ssgi", "ssao"],
		requirements: { gBuffer: ["motion"] },
		history: [
			{ id: "taa", usage: DEFAULT_HISTORY_USAGE },
			{ id: "motion", usage: ["sampled", "copy-dst", "render-target"] },
		],
		isEnabled: enabled("taa"),
		implementations: allBackendsImplementation(),
	},
	{
		id: "ssr",
		dependsOn: ["taa"],
		requirements: { gBuffer: ["depth", "normal", "motion"] },
		history: [
			{ id: "ssr", usage: DEFAULT_HISTORY_USAGE },
			{ id: "motion", usage: ["sampled", "copy-dst", "render-target"] },
		],
		isEnabled: enabled("ssr"),
		implementations: allBackendsImplementation(),
	},
	{
		id: "volumetric",
		dependsOn: ["ssr"],
		requirements: { gBuffer: ["depth", "motion"] },
		history: [
			{ id: "volumetric", usage: DEFAULT_HISTORY_USAGE },
			{ id: "volumetric-reservoir", usage: DEFAULT_HISTORY_USAGE },
			{ id: "motion", usage: ["sampled", "copy-dst", "render-target"] },
		],
		isEnabled: enabled("volumetric"),
		implementations: allBackendsImplementation(),
	},
	{
		id: "fog",
		dependsOn: ["volumetric"],
		requirements: { gBuffer: ["depth"] },
		isEnabled: isFogPostProcessEnabled,
		implementations: allBackendsImplementation(),
	},
	{
		id: "motion-blur",
		dependsOn: ["fog"],
		requirements: { gBuffer: ["motion"] },
		isEnabled: enabled("motion-blur"),
		implementations: allBackendsImplementation(),
	},
	{
		id: "dof",
		dependsOn: ["motion-blur"],
		requirements: { gBuffer: ["depth"] },
		isEnabled: enabled("dof"),
		implementations: allBackendsImplementation(),
	},
	{
		id: "bloom",
		dependsOn: ["dof"],
		isEnabled: enabled("bloom"),
		implementations: allBackendsImplementation(),
	},
	{
		id: "tonemap",
		dependsOn: ["bloom"],
		isEnabled: enabled("tonemap"),
		implementations: allBackendsImplementation(),
	},
	{
		id: "color-filter",
		dependsOn: ["tonemap"],
		isEnabled: enabled("color-filter"),
		implementations: allBackendsImplementation(),
	},
	{
		id: "fxaa",
		dependsOn: ["color-filter"],
		isEnabled: enabled("fxaa"),
		implementations: allBackendsImplementation(),
	},
	{
		id: "interaction-outline",
		dependsOn: ["fxaa"],
		isEnabled: (state) => state.enabled["interaction-outline"],
		implementations: allBackendsImplementation(),
	},
	{
		id: "gamma",
		dependsOn: ["interaction-outline", "tonemap"],
		isEnabled: (state) =>
			state.enabled.gamma || hasEnabledCustomPostProcessPass(state),
		implementations: allBackendsImplementation(),
	},
];

/**
 * Executes logical post-process passes across backends.
 */
export class PostProcessPipeline {
	private _graph = new PostProcessGraph<PostProcessPassDescriptor>(
		BUILTIN_POST_PROCESS_PASSES,
		POST_PROCESS_PASS_IDS
	);
	private _history = new PostProcessHistoryManager();

	/**
	 * Registers a custom logical post-process pass.
	 *
	 * @param pass Descriptor with backend implementation metadata.
	 * @returns Nothing.
	 * @throws If the id is empty, built-in, or already registered.
	 * @sideEffects Mutates the logical post-process graph.
	 */
	public registerPass(pass: PostProcessPassDescriptor): void {
		this._graph.register(pass);
	}

	/**
	 * Unregisters a custom logical post-process pass.
	 *
	 * @param id Custom pass id.
	 * @returns Nothing.
	 * @throws If `id` is a built-in pass id.
	 * @sideEffects Mutates the logical post-process graph.
	 */
	public unregisterPass(id: string): void {
		this._graph.unregister(id);
	}

	/**
	 * Destroys pipeline-owned history resources.
	 *
	 * @param executor Executor that owns concrete resources.
	 * @returns Nothing.
	 * @sideEffects Destroys all active history handles.
	 */
	public destroy(executor: IPostProcessExecutor): void {
		this._history.destroy(executor);
	}

	/**
	 * Resolves the enabled logical pass order for one backend.
	 *
	 * @param postProcess Resolved post-process state.
	 * @param executor Active backend executor.
	 * @param warn Diagnostic sink.
	 * @returns Enabled descriptors in dependency order.
	 * @sideEffects Emits graph diagnostics through `warn`.
	 */
	public getExecutionOrder(
		postProcess: ResolvedPostProcessState,
		executor: IPostProcessExecutor,
		warn: (key: string, message: string) => void = () => {}
	): PostProcessPassDescriptor[] {
		return this._graph.getExecutionOrder(
			(pass) => this._isPassEnabled(pass, postProcess, executor),
			warn
		);
	}

	/**
	 * Executes the post-process pipeline for one frame.
	 *
	 * @param request Frame context, executor, G-buffer bridge, and diagnostics.
	 * @returns Executed pass ids and incremental start metadata.
	 * @sideEffects Allocates or destroys histories and dispatches backend passes.
	 */
	public async execute(
		request: PostProcessPipelineExecuteRequest
	): Promise<PostProcessPipelineExecuteResult> {
		const warn = request.warn ?? (() => {});
		const { frameContext, executor, gBuffer } = request;
		const postProcess = frameContext.postProcess;
		const orderedPasses = this.getExecutionOrder(postProcess, executor, warn);
		const startPassId = request.startPassId ?? this._resolveIncrementalStartPass(
			frameContext,
			orderedPasses
		);
		const passes = this._sliceFromStartPass(orderedPasses, startPassId);
		const historyDescriptors = this._collectHistoryDescriptors(passes);
		const histories = this._history.prepare({
			executor,
			descriptors: historyDescriptors,
			width: Math.max(1, gBuffer.width),
			height: Math.max(1, gBuffer.height),
			reset: frameContext.incremental.temporalHistoryReset,
			signature: this._createHistorySignature(frameContext),
		});
		const frameRequest = {
			frameContext,
			postProcess,
			gBuffer,
			histories,
		};
		const executedPassIds: string[] = [];

		await executor.beginFrame?.(frameRequest);
		for (const pass of passes) {
			if (!this._requirementsSatisfied(pass, gBuffer)) {
				warn(
					`postprocess-requirement-missing-${pass.id}`,
					`Post-process pass "${pass.id}" is missing required G-buffer channels; skipping it`
				);
				continue;
			}
			const implementation = pass.implementations[executor.backend];
			if (!implementation) {
				continue;
			}
			const result = await executor.executePass(pass.id, {
				...frameRequest,
				pass,
				passId: pass.id,
				implementation,
				options: postProcess.options[pass.id],
				startPassId,
			});
			if (result?.ran === false) {
				continue;
			}
			executedPassIds.push(pass.id);
			if (result?.updatedHistoryIds) {
				this._history.markUpdatedMany(result.updatedHistoryIds);
			} else if (result?.historyUpdated) {
				this._history.markUpdatedMany(
					(pass.history ?? [])
						.map((history) => history.id)
						.filter((id) => id !== "motion")
				);
			}
		}
		if (historyDescriptors.some((history) => history.id === "motion")) {
			this._history.markUpdated("motion");
		}
		await executor.endFrame?.({
			...frameRequest,
			executedPassIds,
		});
		this._history.endFrame();

		return {
			executedPassIds,
			firstStage: "postprocess",
			startPassId,
		};
	}

	private _isPassEnabled(
		pass: PostProcessPassDescriptor,
		postProcess: ResolvedPostProcessState,
		executor: IPostProcessExecutor
	): boolean {
		const enabledByState =
			pass.isEnabled ? pass.isEnabled(postProcess) : postProcess.enabled[pass.id];
		if (!enabledByState) {
			return false;
		}
		if (POST_PROCESS_PASS_IDS.includes(pass.id as any)) {
			return executor.capabilities[pass.id] === true;
		}
		return !!pass.implementations[executor.backend];
	}

	private _requirementsSatisfied(
		pass: PostProcessPassDescriptor,
		gBuffer: PostProcessPipelineExecuteRequest["gBuffer"]
	): boolean {
		for (const semantic of pass.requirements?.gBuffer ?? []) {
			if (!gBuffer.channels[semantic]) {
				if (semantic === "world-position" && gBuffer.worldPosition.available) {
					continue;
				}
				return false;
			}
		}
		return true;
	}

	private _collectHistoryDescriptors(
		passes: readonly PostProcessPassDescriptor[]
	): PostProcessHistoryDescriptor[] {
		const descriptors = new Map<string, PostProcessHistoryDescriptor>();
		for (const pass of passes) {
			for (const history of pass.history ?? []) {
				descriptors.set(history.id, history);
			}
		}
		return Array.from(descriptors.values());
	}

	private _sliceFromStartPass(
		passes: readonly PostProcessPassDescriptor[],
		startPassId: string | null
	): PostProcessPassDescriptor[] {
		if (!startPassId) {
			return Array.from(passes);
		}
		const index = passes.findIndex((pass) => pass.id === startPassId);
		if (index < 0) {
			return Array.from(passes);
		}
		return passes.slice(index);
	}

	private _resolveIncrementalStartPass(
		frameContext: FrameContext,
		passes: readonly PostProcessPassDescriptor[]
	): string | null {
		if (
			!frameContext.incremental.enabled ||
			frameContext.incremental.forceFullFrame ||
			frameContext.incremental.firstPass !== "postprocess"
		) {
			return null;
		}
		const startPassId = frameContext.incremental.postProcessStartPass ?? null;
		if (!startPassId || !passes.some((pass) => pass.id === startPassId)) {
			return null;
		}
		return startPassId;
	}

	private _createHistorySignature(context: FrameContext): string {
		const camera = context.camera;
		const postProcess = context.postProcess;
		const interaction = context.transient.get(INTERACTION_TRANSIENT_STATE_KEY);
		const customIds = getEnabledCustomPostProcessPassIds(postProcess).join(",");
		return [
			context.attachments.width,
			context.attachments.height,
			camera.type,
			camera.fov,
			camera.aspectRatio,
			camera.near,
			camera.far,
			`ssao:${postProcess.enabled.ssao ? 1 : 0}`,
			`ssgi:${postProcess.enabled.ssgi ? 1 : 0}`,
			`taa:${postProcess.enabled.taa ? 1 : 0}`,
			`ssr:${postProcess.enabled.ssr ? 1 : 0}`,
			`vol:${postProcess.enabled.volumetric ? 1 : 0}`,
			`fog:${isFogPostProcessEnabled(postProcess) ? 1 : 0}`,
			`mblur:${postProcess.enabled["motion-blur"] ? 1 : 0}`,
			`dof:${postProcess.enabled.dof ? 1 : 0}`,
			`bloom:${postProcess.enabled.bloom ? 1 : 0}`,
			`tonemap:${postProcess.enabled.tonemap ? 1 : 0}`,
			`color:${postProcess.enabled["color-filter"] ? 1 : 0}`,
			`fxaa:${postProcess.enabled.fxaa ? 1 : 0}`,
			`outline:${
				postProcess.enabled["interaction-outline"] &&
				((interaction as { selectedEntityIds?: unknown[] } | undefined)
					?.selectedEntityIds?.length ?? 0) > 0 ?
					1
				:	0
			}`,
			`gamma:${postProcess.enabled.gamma ? 1 : 0}`,
			`custom:${customIds}`,
		].join("|");
	}
}

/**
 * Returns whether a stage id names the renderer post-process stage or a logical pass.
 *
 * @param stage Stage or pass id.
 * @returns `true` for `postprocess` or a built-in logical post-process pass.
 * @sideEffects None.
 */
export function isPostProcessPassStage(stage: string): boolean {
	return stage === "postprocess" || POST_PROCESS_PASS_IDS.includes(stage as any);
}

/**
 * Returns the built-in logical post-process descriptors.
 *
 * @returns Read-only built-in descriptor list.
 * @sideEffects None.
 */
export function getBuiltinPostProcessPasses(): readonly PostProcessPassDescriptor[] {
	return BUILTIN_POST_PROCESS_PASSES;
}

/**
 * Returns required G-buffer channels for a logical pass.
 *
 * @param pass Logical pass descriptor.
 * @returns Required semantic channels, or an empty list.
 * @sideEffects None.
 */
export function getPostProcessRequirementChannels(
	pass: PostProcessPassDescriptor
): readonly LogicalGBufferSemantic[] {
	return pass.requirements?.gBuffer ?? [];
}
