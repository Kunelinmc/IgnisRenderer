import {
	getEnabledCustomPostProcessPassIds,
	hasEnabledCustomPostProcessPass,
	isFogPostProcessEnabled,
	POST_PROCESS_PASS_IDS,
	type ResolvedPostProcessState,
} from "../pipeline/PostProcessController";
import type { FrameContext } from "../pipeline/types";
import { INTERACTION_TRANSIENT_STATE_KEY } from "../pipeline/types";
import { PostProcessHistoryManager } from "./PostProcessHistoryManager";
import {
	DEFAULT_POST_PROCESS_PLACEMENT,
	getBuiltinPostProcessOrder,
	getCustomPostProcessPlacementOrder,
	isPostProcessPlacement,
} from "./ordering";
import { FAST_APPROXIMATE_ANTI_ALIASING_PASS } from "./passes/FastApproximateAntiAliasingPass";
import { SCREEN_SPACE_AMBIENT_OCCLUSION_PASS } from "./passes/ScreenSpaceAmbientOcclusionPass";
import { SCREEN_SPACE_GLOBAL_ILLUMINATION_PASS } from "./passes/ScreenSpaceGlobalIlluminationPass";
import { SCREEN_SPACE_REFLECTIONS_PASS } from "./passes/ScreenSpaceReflectionsPass";
import { TEMPORAL_ANTI_ALIASING_PASS } from "./passes/TemporalAntiAliasingPass";
import type {
	IPostProcessExecutor,
	LogicalGBufferSemantic,
	PostProcessHistoryDescriptor,
	PostProcessHistoryResolveRequest,
	PostProcessPassDescriptor,
	PostProcessPipelineExecuteRequest,
	PostProcessPipelineExecuteResult,
} from "./types";

const KNOWN_BACKENDS = ["software", "webgpu", "webgl"] as const;
const POST_PROCESS_PASS_ID_SET = new Set<string>(POST_PROCESS_PASS_IDS);

const DEFAULT_HISTORY_USAGE = ["sampled", "storage", "render-target"] as const;
const CUSTOM_ORDER_SCALE = 0.001;
const CUSTOM_ORDER_LIMIT = 999;

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
	SCREEN_SPACE_AMBIENT_OCCLUSION_PASS,
	SCREEN_SPACE_GLOBAL_ILLUMINATION_PASS,
	TEMPORAL_ANTI_ALIASING_PASS,
	SCREEN_SPACE_REFLECTIONS_PASS,
	{
		id: "volumetric",
		placement: "atmosphere",
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
		placement: "atmosphere",
		requirements: { gBuffer: ["depth"] },
		isEnabled: isFogPostProcessEnabled,
		implementations: allBackendsImplementation(),
	},
	{
		id: "motion-blur",
		placement: "camera",
		requirements: { gBuffer: ["depth", "motion"] },
		isEnabled: enabled("motion-blur"),
		implementations: allBackendsImplementation(),
	},
	{
		id: "dof",
		placement: "camera",
		requirements: { gBuffer: ["depth"] },
		isEnabled: enabled("dof"),
		implementations: allBackendsImplementation(),
	},
	{
		id: "bloom",
		placement: "hdr",
		isEnabled: enabled("bloom"),
		implementations: allBackendsImplementation(),
	},
	{
		id: "tonemap",
		placement: "hdr",
		isEnabled: enabled("tonemap"),
		implementations: allBackendsImplementation(),
	},
	{
		id: "color-filter",
		placement: "ldr",
		isEnabled: enabled("color-filter"),
		implementations: allBackendsImplementation(),
	},
	FAST_APPROXIMATE_ANTI_ALIASING_PASS,
	{
		id: "interaction-outline",
		placement: "overlay",
		isEnabled: (state) => state.enabled["interaction-outline"],
		implementations: allBackendsImplementation(),
	},
	{
		id: "gamma",
		placement: "present",
		isEnabled: (state) =>
			state.enabled.gamma || hasEnabledCustomPostProcessPass(state),
		implementations: allBackendsImplementation(),
	},
];

/**
 * Executes logical post-process passes across backends.
 */
export class PostProcessPipeline {
	private _passes = new Map<string, PostProcessPassDescriptor>(
		BUILTIN_POST_PROCESS_PASSES.map((pass) => [pass.id, pass])
	);
	private _customInsertionOrder = new Map<string, number>();
	private _nextCustomInsertionOrder = 0;
	private _history = new PostProcessHistoryManager();

	/**
	 * Registers a custom logical post-process pass.
	 *
	 * @param pass Descriptor with backend implementation metadata.
	 * @returns Nothing.
	 * @throws If the id is empty, built-in, or already registered.
	 * @sideEffects Mutates the logical post-process pass registry.
	 */
	public registerPass(pass: PostProcessPassDescriptor): void {
		this._assertCanRegisterPass(pass);
		this._passes.set(pass.id, pass);
		this._customInsertionOrder.set(
			pass.id,
			this._nextCustomInsertionOrder++
		);
	}

	/**
	 * Unregisters a custom logical post-process pass.
	 *
	 * @param id Custom pass id.
	 * @returns Nothing.
	 * @throws If `id` is a built-in pass id.
	 * @sideEffects Mutates the logical post-process pass registry.
	 */
	public unregisterPass(id: string): void {
		if (POST_PROCESS_PASS_ID_SET.has(id)) {
			throw new Error(`Cannot unregister built-in post-process pass "${id}".`);
		}
		this._passes.delete(id);
		this._customInsertionOrder.delete(id);
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
	 * @returns Enabled descriptors in placement order.
	 * @sideEffects None.
	 */
	public getExecutionOrder(
		postProcess: ResolvedPostProcessState,
		executor: IPostProcessExecutor,
		warn: (key: string, message: string) => void = () => {}
	): PostProcessPassDescriptor[] {
		void warn;
		const enabled = Array.from(this._passes.values()).filter((pass) =>
			this._isPassEnabled(pass, postProcess, executor)
		);
		enabled.sort((left, right) => {
			const leftOrder = this._getPassSortOrder(left);
			const rightOrder = this._getPassSortOrder(right);
			if (leftOrder !== rightOrder) {
				return leftOrder - rightOrder;
			}
			return (
				(this._customInsertionOrder.get(left.id) ?? 0) -
				(this._customInsertionOrder.get(right.id) ?? 0)
			);
		});
		return enabled;
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
		const historyResolveRequest: PostProcessHistoryResolveRequest = {
			frameContext,
			postProcess,
			backend: executor.backend,
			gBuffer,
			width: Math.max(1, gBuffer.width),
			height: Math.max(1, gBuffer.height),
		};
		const historyDescriptors = this._collectHistoryDescriptors(
			passes,
			historyResolveRequest
		);
		const histories = this._history.prepare({
			executor,
			descriptors: historyDescriptors,
			width: historyResolveRequest.width,
			height: historyResolveRequest.height,
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
			const passRequest = {
				...frameRequest,
				pass,
				passId: pass.id,
				implementation,
				options: postProcess.options[pass.id],
				startPassId,
			};
			const result =
				typeof implementation.execute === "function" ?
					await implementation.execute(
						passRequest,
						executor.getPassExecutionContext?.(pass.id, passRequest)
					)
				:	await executor.executePass(pass.id, passRequest);
			if (result?.ran === false) {
				if (!POST_PROCESS_PASS_ID_SET.has(pass.id)) {
					warn(
						`postprocess-custom-pass-skipped-${executor.backend}-${pass.id}`,
						`Post-process custom pass "${pass.id}" returned ran=false on backend "${executor.backend}" and was skipped`
					);
				}
				continue;
			}
			executedPassIds.push(pass.id);
			if (result?.updatedHistoryIds) {
				this._history.markUpdatedMany(result.updatedHistoryIds);
			} else if (result?.historyUpdated) {
				this._history.markUpdatedMany(
					this._resolvePassHistoryDescriptors(pass, historyResolveRequest)
						.map((history) => history.id)
						.filter((id) => id !== "motion")
				);
			}
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

	private _assertCanRegisterPass(pass: PostProcessPassDescriptor): void {
		if (!pass.id) {
			throw new Error("Post-process pass id is required.");
		}
		if (POST_PROCESS_PASS_ID_SET.has(pass.id)) {
			throw new Error(
				`Cannot register built-in post-process pass "${pass.id}".`
			);
		}
		if (this._passes.has(pass.id)) {
			throw new Error(
				`Post-process pass "${pass.id}" is already registered.`
			);
		}
	}

	private _getPassSortOrder(pass: PostProcessPassDescriptor): number {
		const builtInOrder = getBuiltinPostProcessOrder(pass.id);
		if (builtInOrder) {
			return builtInOrder.order;
		}
		const placement =
			isPostProcessPlacement(pass.placement) ?
				pass.placement
			:	DEFAULT_POST_PROCESS_PLACEMENT;
		const localOrder =
			typeof pass.order === "number" && Number.isFinite(pass.order) ?
				Math.max(
					-CUSTOM_ORDER_LIMIT,
					Math.min(CUSTOM_ORDER_LIMIT, pass.order)
				)
			:	0;
		return (
			getCustomPostProcessPlacementOrder(placement) +
			localOrder * CUSTOM_ORDER_SCALE
		);
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
		passes: readonly PostProcessPassDescriptor[],
		request: PostProcessHistoryResolveRequest
	): PostProcessHistoryDescriptor[] {
		const descriptors = new Map<string, PostProcessHistoryDescriptor>();
		for (const pass of passes) {
			for (const history of this._resolvePassHistoryDescriptors(pass, request)) {
				descriptors.set(history.id, history);
			}
		}
		return Array.from(descriptors.values());
	}

	private _resolvePassHistoryDescriptors(
		pass: PostProcessPassDescriptor,
		request: PostProcessHistoryResolveRequest
	): readonly PostProcessHistoryDescriptor[] {
		return pass.resolveHistory?.(request) ?? pass.history ?? [];
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
