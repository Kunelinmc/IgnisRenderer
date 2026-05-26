import type { FrameContext } from "../pipeline/types";
import { PostProcessHistoryManager } from "./PostProcessHistoryManager";
import { PostProcessTransientManager } from "./PostProcessTransientManager";
import {
	DEFAULT_POST_PROCESS_PLACEMENT,
	getBuiltinPostProcessOrder,
	getCustomPostProcessPlacementOrder,
	isPostProcessPlacement,
} from "./ordering";
import type {
	LogicalGBufferSemantic,
	IPostProcessExecutor,
	PostProcessHistoryDescriptor,
	PostProcessHistoryResolveRequest,
	PostProcessBackendKind,
	PostProcessPassExecutionContextRequest,
	PostProcessPassRequirements,
	PostProcessPipelineExecuteRequest,
	PostProcessPipelineExecuteResult,
	PostProcessTransientDescriptor,
} from "./types";
import type {
	PostProcessPass,
	PostProcessPassRegistrySnapshot,
	PostProcessPassResolveRequest,
	ResolvedPostProcessPass,
} from "./PostProcessPass";

const CUSTOM_ORDER_SCALE = 0.001;
const CUSTOM_ORDER_LIMIT = 999;

export interface PostProcessExecutionOrderContext {
	readonly backend?: PostProcessBackendKind;
	readonly frameContext?: FrameContext;
}

/**
 * Resolves the logical post-process passes that should be considered for
 * execution in the current frame.
 *
 * @param postProcess Per-frame post-process snapshot.
 * @param context Optional backend and frame context used by frame-conditional
 * passes.
 * @returns Enabled executable passes in deterministic placement order.
 * @sideEffects None.
 */
export function resolvePostProcessExecutionOrder(
	postProcess: PostProcessPassRegistrySnapshot,
	context: PostProcessExecutionOrderContext = {}
): ResolvedPostProcessPass[] {
	const enabled = Array.from(postProcess.getEnabledPasses()).filter((resolved) =>
		resolved.pass.shouldExecute(
			createPostProcessResolveRequest(postProcess, resolved, context)
		)
	);
	enabled.sort(comparePostProcessPassOrder);
	return enabled;
}

/**
 * Returns whether the logical post-process pipeline has work for a frame.
 *
 * @param postProcess Per-frame post-process snapshot.
 * @param context Optional backend and frame context used by frame-conditional
 * passes.
 * @returns `true` when at least one enabled pass should execute.
 * @sideEffects None.
 */
export function hasPostProcessExecutionPasses(
	postProcess: PostProcessPassRegistrySnapshot,
	context: PostProcessExecutionOrderContext = {}
): boolean {
	for (const resolved of postProcess.getEnabledPasses()) {
		if (
			resolved.pass.shouldExecute(
				createPostProcessResolveRequest(postProcess, resolved, context)
			)
		) {
			return true;
		}
	}
	return false;
}

/**
 * Executes logical post-process passes across backends.
 */
export class PostProcessPipeline {
	private _history = new PostProcessHistoryManager();
	private _transients = new PostProcessTransientManager();

	/**
	 * Destroys pipeline-owned history resources.
	 *
	 * @param executor Executor that owns concrete resources.
	 * @returns Nothing.
	 * @sideEffects Destroys all active history handles.
	 */
	public destroy(executor: IPostProcessExecutor): void {
		this._history.destroy(executor);
		this._transients.destroy(executor);
	}

	/**
	 * Resolves the enabled logical pass order for one backend.
	 *
	 * @param postProcess Per-frame post-process snapshot.
	 * @param executor Active backend executor.
	 * @param warn Diagnostic sink.
	 * @param frameContext Optional frame context for frame-conditional passes.
	 * @returns Enabled passes in placement order.
	 * @sideEffects None.
	 */
	public getExecutionOrder(
		postProcess: PostProcessPassRegistrySnapshot,
		executor: IPostProcessExecutor,
		warn: (key: string, message: string) => void = () => {},
		frameContext?: FrameContext
	): ResolvedPostProcessPass[] {
		void warn;
		return resolvePostProcessExecutionOrder(postProcess, {
			backend: executor.backend,
			frameContext,
		});
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
		const orderedPasses = this.getExecutionOrder(
			postProcess,
			executor,
			warn,
			frameContext
		);
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
		const eligiblePasses = this._filterPassesByRequirements(
			passes,
			historyResolveRequest,
			gBuffer,
			warn
		);
		const historyDescriptors = this._collectHistoryDescriptors(
			eligiblePasses,
			historyResolveRequest,
			warn
		);
		const transientDescriptors = this._collectTransientDescriptors(
			eligiblePasses,
			historyResolveRequest,
			warn
		);
		const histories = this._history.prepare({
			executor,
			descriptors: historyDescriptors,
			width: historyResolveRequest.width,
			height: historyResolveRequest.height,
			reset: frameContext.incremental.temporalHistoryReset,
			signature: this._createHistorySignature(
				frameContext,
				eligiblePasses,
				historyResolveRequest
			),
		});
		const transientResult = this._transients.prepare({
			executor,
			descriptors: transientDescriptors,
			width: historyResolveRequest.width,
			height: historyResolveRequest.height,
		});
		if (transientResult.changed) {
			executor.invalidateResourceBindings?.();
		}
		const frameRequest = {
			frameContext,
			postProcess,
			gBuffer,
			histories,
			transients: transientResult.slots,
		};
		const executedPassIds: string[] = [];

		await executor.beginFrame?.(frameRequest);
		for (const resolved of eligiblePasses) {
			const pass = resolved.pass;
			const implementation = pass.getImplementation(executor.backend);
			const passRequest = {
				...frameRequest,
				pass,
				passId: pass.id,
				implementation,
				options: resolved.options,
				startPassId,
			};
			const executionContext =
				implementation?.execute ?
					executor.getPassExecutionContext?.({
						...passRequest,
						implementation,
					} satisfies PostProcessPassExecutionContextRequest)
				:	undefined;
			const result = await pass.execute(
				passRequest,
				executionContext,
				executor
			);
			if (result?.ran === false) {
				continue;
			}
			executedPassIds.push(pass.id);
			if (result?.updatedHistoryIds) {
				this._history.markUpdatedMany(result.updatedHistoryIds);
			} else if (result?.historyUpdated) {
				this._history.markUpdatedMany(
					this._resolvePassHistoryDescriptors(resolved, historyResolveRequest)
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

	private _requirementsSatisfied(
		requirements: PostProcessPassRequirements,
		gBuffer: PostProcessPipelineExecuteRequest["gBuffer"]
	): boolean {
		for (const semantic of requirements.gBuffer ?? []) {
			if (!gBuffer.channels[semantic]) {
				if (semantic === "world-position" && gBuffer.worldPosition.available) {
					continue;
				}
				return false;
			}
		}
		return true;
	}

	private _filterPassesByRequirements(
		passes: readonly ResolvedPostProcessPass[],
		request: PostProcessHistoryResolveRequest,
		gBuffer: PostProcessPipelineExecuteRequest["gBuffer"],
		warn: (key: string, message: string) => void
	): ResolvedPostProcessPass[] {
		const eligible: ResolvedPostProcessPass[] = [];
		for (const resolved of passes) {
			const pass = resolved.pass;
			const resolveRequest = this._createResolveRequest(resolved, request);
			if (!this._requirementsSatisfied(pass.getRequirements(resolveRequest), gBuffer)) {
				warn(
					`postprocess-requirement-missing-${pass.id}`,
					`Post-process pass "${pass.id}" is missing required G-buffer channels; skipping it`
				);
				continue;
			}
			eligible.push(resolved);
		}
		return eligible;
	}

	private _collectHistoryDescriptors(
		passes: readonly ResolvedPostProcessPass[],
		request: PostProcessHistoryResolveRequest,
		warn: (key: string, message: string) => void
	): PostProcessHistoryDescriptor[] {
		const descriptors = new Map<string, PostProcessHistoryDescriptor>();
		const descriptorKeys = new Map<string, string>();
		for (const pass of passes) {
			for (const history of this._resolvePassHistoryDescriptors(pass, request)) {
				const key = this._createHistoryDescriptorKey(history);
				const currentKey = descriptorKeys.get(history.id);
				if (currentKey && currentKey !== key) {
					warn(
						`postprocess-history-conflict-${history.id}`,
						`Post-process history "${history.id}" was requested with incompatible descriptors; keeping the first descriptor`
					);
					continue;
				}
				descriptorKeys.set(history.id, key);
				descriptors.set(history.id, history);
			}
		}
		return Array.from(descriptors.values());
	}

	private _resolvePassHistoryDescriptors(
		resolved: ResolvedPostProcessPass,
		request: PostProcessHistoryResolveRequest
	): readonly PostProcessHistoryDescriptor[] {
		return resolved.pass.getHistoryDescriptors(
			this._createResolveRequest(resolved, request)
		);
	}

	private _collectTransientDescriptors(
		passes: readonly ResolvedPostProcessPass[],
		request: PostProcessHistoryResolveRequest,
		warn: (key: string, message: string) => void
	): PostProcessTransientDescriptor[] {
		const descriptors = new Map<string, PostProcessTransientDescriptor>();
		const descriptorKeys = new Map<string, string>();
		for (const pass of passes) {
			for (const transient of this._resolvePassTransientDescriptors(pass, request)) {
				const key = this._createTransientDescriptorKey(transient);
				const currentKey = descriptorKeys.get(transient.id);
				if (currentKey && currentKey !== key) {
					warn(
						`postprocess-transient-conflict-${transient.id}`,
						`Post-process transient "${transient.id}" was requested with incompatible descriptors; keeping the first descriptor`
					);
					continue;
				}
				descriptorKeys.set(transient.id, key);
				descriptors.set(transient.id, transient);
			}
		}
		return Array.from(descriptors.values());
	}

	private _resolvePassTransientDescriptors(
		resolved: ResolvedPostProcessPass,
		request: PostProcessHistoryResolveRequest
	): readonly PostProcessTransientDescriptor[] {
		return resolved.pass.getTransientResourceDescriptors(
			this._createResolveRequest(resolved, request)
		);
	}

	private _sliceFromStartPass(
		passes: readonly ResolvedPostProcessPass[],
		startPassId: string | null
	): ResolvedPostProcessPass[] {
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
		passes: readonly ResolvedPostProcessPass[]
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

	private _createHistorySignature(
		context: FrameContext,
		passes: readonly ResolvedPostProcessPass[],
		request: PostProcessHistoryResolveRequest
	): string {
		const camera = context.camera;
		return [
			context.attachments.width,
			context.attachments.height,
			camera.type,
			camera.fov,
			camera.aspectRatio,
			camera.near,
			camera.far,
			...passes.map((resolved) =>
				`${resolved.id}:${resolved.pass.getHistorySignature(
					this._createResolveRequest(resolved, request)
				)}`
			),
		].join("|");
	}

	private _createResolveRequest<TOptions>(
		resolved: ResolvedPostProcessPass<TOptions>,
		request: PostProcessHistoryResolveRequest
	): PostProcessPassResolveRequest<TOptions> {
		return {
			frameContext: request.frameContext,
			postProcess: request.postProcess,
			backend: request.backend,
			gBuffer: request.gBuffer,
			width: request.width,
			height: request.height,
			options: resolved.options,
		};
	}

	private _createHistoryDescriptorKey(
		descriptor: PostProcessHistoryDescriptor
	): string {
		return [
			descriptor.widthScale ?? 1,
			descriptor.heightScale ?? 1,
			descriptor.format ?? "rgba16float",
			[...(descriptor.usage ?? ["sampled", "storage", "render-target"])]
				.sort()
				.join(","),
		].join("|");
	}

	private _createTransientDescriptorKey(
		descriptor: PostProcessTransientDescriptor
	): string {
		return [
			descriptor.widthScale ?? 1,
			descriptor.heightScale ?? 1,
			descriptor.format ?? "rgba16float",
			descriptor.mipMode ?? "single",
			[...(descriptor.usage ?? ["sampled", "storage", "render-target"])]
				.sort()
				.join(","),
		].join("|");
	}
}

function comparePostProcessPassOrder(
	left: ResolvedPostProcessPass,
	right: ResolvedPostProcessPass
): number {
	const leftOrder = getPostProcessPassSortOrder(left.pass);
	const rightOrder = getPostProcessPassSortOrder(right.pass);
	if (leftOrder !== rightOrder) {
		return leftOrder - rightOrder;
	}
	return left.id.localeCompare(right.id);
}

function getPostProcessPassSortOrder(pass: PostProcessPass): number {
	if (
		pass.builtIn &&
		typeof pass.order === "number" &&
		Number.isFinite(pass.order)
	) {
		return pass.order;
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

function createPostProcessResolveRequest<TOptions>(
	postProcess: PostProcessPassRegistrySnapshot,
	resolved: ResolvedPostProcessPass<TOptions>,
	context: PostProcessExecutionOrderContext
): PostProcessPassResolveRequest<TOptions> {
	return {
		frameContext: context.frameContext,
		postProcess,
		backend: context.backend,
		options: resolved.options,
	};
}

/**
 * Returns whether a stage id names the renderer post-process stage or a logical pass.
 *
 * @param stage Stage or pass id.
 * @returns `true` for `postprocess` or a built-in logical post-process pass.
 * @sideEffects None.
 */
export function isPostProcessPassStage(stage: string): boolean {
	return stage === "postprocess" || getBuiltinPostProcessOrder(stage) !== null;
}

/**
 * Returns required G-buffer channels for a logical pass.
 *
 * @param pass Logical pass instance.
 * @returns Required semantic channels, or an empty list.
 * @sideEffects None.
 */
export function getPostProcessRequirementChannels(
	pass: PostProcessPass
): readonly LogicalGBufferSemantic[] {
	return pass.getRequirements({}).gBuffer ?? [];
}
