import type { FrameContext } from "../pipeline/types";
import type { RenderBackendType } from "../renderers/IRenderBackend";
import {
	DEFAULT_POST_PROCESS_PLACEMENT,
	getBuiltinPostProcessOrder,
	getCustomPostProcessPlacementOrder,
	isPostProcessPlacement,
} from "./ordering";
import { createPostProcessScaledResourceDescriptorKey } from "./resourceDescriptors";
import type {
	LogicalGBufferBridge,
	LogicalGBufferSemantic,
	PostProcessHistoryDescriptor,
	PostProcessHistoryResolveRequest,
	PostProcessPassImplementation,
	PostProcessPassRequirements,
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
	readonly backend?: RenderBackendType;
	readonly frameContext?: FrameContext;
}

export interface PostProcessGraphCompileRequest {
	readonly postProcess: PostProcessPassRegistrySnapshot;
	readonly backend: RenderBackendType;
	readonly frameContext: FrameContext;
	readonly gBuffer: LogicalGBufferBridge;
	readonly startPassId?: string | null;
	readonly warn?: (key: string, message: string) => void;
	readonly resolveImplementation?: (
		pass: PostProcessPass
	) => PostProcessPassImplementation | null;
}

export interface CompiledPostProcessPass<TOptions = unknown>
	extends ResolvedPostProcessPass<TOptions> {
	readonly implementation: PostProcessPassImplementation | null;
	readonly historyIds: readonly string[];
}

export interface CompiledPostProcessGraph {
	readonly backend: RenderBackendType;
	readonly postProcess: PostProcessPassRegistrySnapshot;
	readonly frameContext: FrameContext;
	readonly gBuffer: LogicalGBufferBridge;
	readonly width: number;
	readonly height: number;
	readonly orderedPasses: readonly ResolvedPostProcessPass[];
	readonly passes: readonly CompiledPostProcessPass[];
	readonly startPassId: string | null;
	readonly historyDescriptors: readonly PostProcessHistoryDescriptor[];
	readonly transientDescriptors: readonly PostProcessTransientDescriptor[];
	readonly signature: string;
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
 * Compiles backend-agnostic post-process graph metadata for one backend frame.
 */
export class PostProcessGraphCompiler {
	/**
	 * Compiles logical post-process pass order, eligibility, resource descriptors,
	 * and history signature for one backend frame.
	 *
	 * @internal Owned by backend post-process runtimes; applications should
	 * register passes through `renderer.postProcess`.
	 * @param request Frame graph compilation request.
	 * @returns Compiled graph metadata consumed by backend runtime execution.
	 * @sideEffects Calls the supplied diagnostic callback for skipped or
	 * conflicting resources.
	 */
	public compile(request: PostProcessGraphCompileRequest): CompiledPostProcessGraph {
		const warn = request.warn ?? (() => {});
		const orderedPasses = resolvePostProcessExecutionOrder(request.postProcess, {
			backend: request.backend,
			frameContext: request.frameContext,
		});
		const startPassId =
			request.startPassId ??
			this._resolveIncrementalStartPass(request.frameContext, orderedPasses);
		const passes = this._sliceFromStartPass(orderedPasses, startPassId);
		const width = Math.max(1, request.gBuffer.width);
		const height = Math.max(1, request.gBuffer.height);
		const historyResolveRequest: PostProcessHistoryResolveRequest = {
			frameContext: request.frameContext,
			postProcess: request.postProcess,
			backend: request.backend,
			gBuffer: request.gBuffer,
			width,
			height,
		};
		const eligiblePasses = this._filterPassesByRequirements(
			passes,
			historyResolveRequest,
			request.gBuffer,
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
		const historyIdsByPass = new Map<string, readonly string[]>();
		for (const resolved of eligiblePasses) {
			historyIdsByPass.set(
				resolved.id,
				this._resolvePassHistoryDescriptors(
					resolved,
					historyResolveRequest
				).map((history) => history.id)
			);
		}
		const compiledPasses = eligiblePasses.map((resolved) => ({
			...resolved,
			implementation: request.resolveImplementation?.(resolved.pass) ?? null,
			historyIds: historyIdsByPass.get(resolved.id) ?? [],
		}));

		return {
			backend: request.backend,
			postProcess: request.postProcess,
			frameContext: request.frameContext,
			gBuffer: request.gBuffer,
			width,
			height,
			orderedPasses,
			passes: compiledPasses,
			startPassId,
			historyDescriptors,
			transientDescriptors,
			signature: this._createHistorySignature(
				request.frameContext,
				eligiblePasses,
				historyResolveRequest
			),
		};
	}

	private _requirementsSatisfied(
		requirements: PostProcessPassRequirements,
		gBuffer: LogicalGBufferBridge
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
		gBuffer: LogicalGBufferBridge,
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
		const incremental = frameContext.incremental;
		if (
			!incremental?.enabled ||
			incremental.forceFullFrame ||
			incremental.firstPass !== "postprocess"
		) {
			return null;
		}
		const startPassId = incremental.postProcessStartPass ?? null;
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
		const camera = context.viewCamera;
		return [
			context.attachments?.width ?? 1,
			context.attachments?.height ?? 1,
			camera?.type,
			camera?.fov,
			camera?.aspectRatio,
			camera?.near,
			camera?.far,
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
		return createPostProcessScaledResourceDescriptorKey(descriptor);
	}

	private _createTransientDescriptorKey(
		descriptor: PostProcessTransientDescriptor
	): string {
		return createPostProcessScaledResourceDescriptorKey(descriptor, {
			includeMipMode: true,
		});
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
 * Returns whether a stage id names the renderer post-process stage or a
 * renderer-default built-in logical pass.
 *
 * @param stage Stage or pass id.
 * @returns `true` for `postprocess` or a default built-in logical pass.
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
