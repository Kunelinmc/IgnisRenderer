import type { RenderBackendType } from "../backends/IRenderBackend";
import type { FrameContext } from "../pipeline/types";
import {
	DEFAULT_POST_PROCESS_PLACEMENT,
	getBuiltinPostProcessOrder,
	getCustomPostProcessPlacementOrder,
	isPostProcessPlacement,
} from "./ordering";
import type {
	PostProcessPass,
	PostProcessPassRegistrySnapshot,
	PostProcessPassResolveRequest,
	ResolvedPostProcessPass,
} from "./PostProcessPass";
import { createPostProcessScaledResourceDescriptorKey } from "./resourceDescriptors";
import type {
	LogicalGBufferBridge,
	PostProcessExecutionDeclaration,
	PostProcessExecutionResourceUse,
	PostProcessHistoryDescriptor,
	PostProcessPassImplementation,
	PostProcessTransientDescriptor,
} from "./types";

const CUSTOM_ORDER_SCALE = 0.001;
const CUSTOM_ORDER_LIMIT = 999;
const VALID_ACCESS = new Set(["read", "write", "read-write"]);
const VALID_USAGE = new Set([
	"sampled",
	"color-attachment",
	"depth-attachment",
	"storage",
	"copy-source",
	"copy-target",
	"present",
	"cpu-read",
	"cpu-write",
]);

export interface PostProcessExecutionOrderContext {
	readonly backend?: RenderBackendType;
	readonly frameContext?: FrameContext;
}

export interface PostProcessPlanRequest {
	readonly postProcess: PostProcessPassRegistrySnapshot;
	readonly backend: RenderBackendType;
	readonly frameContext: FrameContext;
	readonly gBuffer: LogicalGBufferBridge;
	readonly startPassId?: string | null;
	readonly warn?: (key: string, message: string) => void;
	readonly resolveImplementation: (
		pass: PostProcessPass
	) => PostProcessPassImplementation | null;
	readonly isSharedResourceAvailable?: (resourceId: string) => boolean;
}

export interface PlannedPostProcessPass<TOptions = unknown>
	extends ResolvedPostProcessPass<TOptions> {
	readonly implementation: PostProcessPassImplementation<unknown, TOptions>;
	readonly declaration: PostProcessExecutionDeclaration;
	readonly historyIds: readonly string[];
	readonly transientIds: readonly string[];
}

export interface PostProcessPlan {
	readonly backend: RenderBackendType;
	readonly postProcess: PostProcessPassRegistrySnapshot;
	readonly frameContext: FrameContext;
	readonly gBuffer: LogicalGBufferBridge;
	readonly width: number;
	readonly height: number;
	readonly orderedPasses: readonly ResolvedPostProcessPass[];
	readonly passes: readonly PlannedPostProcessPass[];
	readonly startPassId: string | null;
	readonly historyDescriptors: readonly PostProcessHistoryDescriptor[];
	readonly transientDescriptors: readonly PostProcessTransientDescriptor[];
	readonly signature: string;
}

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

export function hasPostProcessExecutionPasses(
	postProcess: PostProcessPassRegistrySnapshot,
	context: PostProcessExecutionOrderContext = {}
): boolean {
	return resolvePostProcessExecutionOrder(postProcess, context).length > 0;
}

/** @internal Resolves one immutable backend post-process plan. */
export class PostProcessPlanner {
	public plan(request: PostProcessPlanRequest): PostProcessPlan {
		const warn = request.warn ?? (() => {});
		const orderedPasses = resolvePostProcessExecutionOrder(request.postProcess, {
			backend: request.backend,
			frameContext: request.frameContext,
		});
		const startPassId = request.startPassId ??
			this._resolveIncrementalStartPass(request.frameContext, orderedPasses);
		const slicedPasses = this._sliceFromStartPass(orderedPasses, startPassId);
		const width = Math.max(1, request.gBuffer.width);
		const height = Math.max(1, request.gBuffer.height);
		const passes: PlannedPostProcessPass[] = [];

		for (const resolved of slicedPasses) {
			const implementation = request.resolveImplementation(resolved.pass);
			if (!implementation) {
				warn(
					`postprocess-implementation-missing-${resolved.id}`,
					`Post-process pass "${resolved.id}" has no ${request.backend} implementation; skipping it`,
				);
				continue;
			}
			const resolveRequest = this._createResolveRequest(
				resolved,
				request,
				width,
				height,
			);
			const declaration = deepFreezeDeclaration(
				implementation.describeExecution(resolveRequest),
			);
			this._validateDeclaration(request.backend, resolved.id, declaration);
			if (!this._isEligible(resolved.id, declaration, request, warn)) {
				continue;
			}
			passes.push(Object.freeze({
				...resolved,
				implementation,
				declaration,
				historyIds: Object.freeze(
					(declaration.histories ?? []).map((entry) => entry.descriptor.id),
				),
				transientIds: Object.freeze(
					(declaration.transients ?? []).map((entry) => entry.descriptor.id),
				),
			}));
		}

		const historyDescriptors = this._collectHistoryDescriptors(request.backend, passes);
		const transientDescriptors = this._collectTransientDescriptors(request.backend, passes);
		return Object.freeze({
			backend: request.backend,
			postProcess: request.postProcess,
			frameContext: request.frameContext,
			gBuffer: request.gBuffer,
			width,
			height,
			orderedPasses: Object.freeze(orderedPasses),
			passes: Object.freeze(passes),
			startPassId,
			historyDescriptors: Object.freeze(historyDescriptors),
			transientDescriptors: Object.freeze(transientDescriptors),
			signature: this._createSignature(request.frameContext, passes),
		});
	}

	private _validateDeclaration(
		backend: RenderBackendType,
		passId: string,
		declaration: PostProcessExecutionDeclaration
	): void {
		const violations: string[] = [];
		const color = declaration?.color;
		if (!color || !["none", "read", "read-write"].includes(color.access)) {
			violations.push("color access is invalid");
		}
		if (!color || !["preserve", "new-version"].includes(color.output)) {
			violations.push("color output is invalid");
		}
		if (color?.output === "preserve" && color.access === "read") {
			violations.push("preserved color cannot be read-only");
		}
		if (color?.output === "new-version" && color.access === "read-write") {
			violations.push("new color version cannot use read-write access");
		}
		this._validateUnique(
			declaration.gBuffer?.map((entry) => entry.semantic) ?? [],
			"G-buffer semantic",
			violations,
		);
		this._validateUnique(
			declaration.histories?.map((entry) => entry.descriptor.id) ?? [],
			"history",
			violations,
		);
		this._validateUnique(
			declaration.transients?.map((entry) => entry.descriptor.id) ?? [],
			"transient",
			violations,
		);
		this._validateUnique(
			declaration.shared?.map((entry) => entry.id) ?? [],
			"shared resource",
			violations,
		);
		for (const entry of declaration.gBuffer ?? []) {
			this._validateUses([entry], `G-buffer "${entry.semantic}"`, violations);
		}
		for (const entry of declaration.histories ?? []) {
			if (!entry.descriptor.id) violations.push("history id is required");
			if (entry.read.length === 0 || entry.write.length === 0) {
				violations.push(
					`history "${entry.descriptor.id}" requires read and write uses`,
				);
			}
			this._validateUses(entry.read, `history "${entry.descriptor.id}" read`, violations);
			this._validateUses(entry.write, `history "${entry.descriptor.id}" write`, violations);
		}
		for (const entry of declaration.transients ?? []) {
			if (!entry.descriptor.id) violations.push("transient id is required");
			if (entry.uses.length === 0) {
				violations.push(`transient "${entry.descriptor.id}" requires uses`);
			}
			this._validateUses(entry.uses, `transient "${entry.descriptor.id}"`, violations);
		}
		for (const entry of declaration.shared ?? []) {
			if (!entry.id) violations.push("shared resource id is required");
			this._validateUses([entry], `shared resource "${entry.id}"`, violations);
		}
		if (violations.length > 0) {
			throw new Error(
				`Post-process pass "${passId}" has invalid ${backend} execution declaration: ` +
				violations.join("; "),
			);
		}
	}

	private _validateUses(
		uses: readonly PostProcessExecutionResourceUse[],
		label: string,
		violations: string[]
	): void {
		for (const use of uses) {
			if (!VALID_ACCESS.has(use.access)) violations.push(`${label} has invalid access`);
			if (!VALID_USAGE.has(use.usage)) violations.push(`${label} has invalid usage`);
		}
	}

	private _validateUnique(
		ids: readonly string[],
		label: string,
		violations: string[]
	): void {
		const seen = new Set<string>();
		for (const id of ids) {
			if (seen.has(id)) violations.push(`${label} "${id}" is duplicated`);
			seen.add(id);
		}
	}

	private _isEligible(
		passId: string,
		declaration: PostProcessExecutionDeclaration,
		request: PostProcessPlanRequest,
		warn: (key: string, message: string) => void
	): boolean {
		for (const entry of declaration.gBuffer ?? []) {
			if (entry.optional === true) continue;
			const available = entry.semantic === "world-position" ?
				request.gBuffer.worldPosition.available || !!request.gBuffer.channels[entry.semantic]
				: !!request.gBuffer.channels[entry.semantic];
			if (!available) {
				warn(
					`postprocess-requirement-missing-${passId}`,
					`Post-process pass "${passId}" is missing required G-buffer channel "${entry.semantic}"; skipping it`,
				);
				return false;
			}
		}
		for (const entry of declaration.shared ?? []) {
			if (
				entry.optional !== true &&
				request.isSharedResourceAvailable?.(entry.id) === false
			) {
				warn(
					`postprocess-backend-shared-unavailable-${passId}`,
					`Post-process pass "${passId}" requires unavailable shared resource "${entry.id}"; skipping it`,
				);
				return false;
			}
		}
		return true;
	}

	private _collectHistoryDescriptors(
		backend: RenderBackendType,
		passes: readonly PlannedPostProcessPass[]
	): PostProcessHistoryDescriptor[] {
		return this._collectDescriptors(
			passes.flatMap((pass) =>
				(pass.declaration.histories ?? []).map((entry) => ({
					passId: pass.id,
					descriptor: entry.descriptor,
				}))
			),
			"history",
			backend,
			(descriptor) => createPostProcessScaledResourceDescriptorKey(descriptor),
		);
	}

	private _collectTransientDescriptors(
		backend: RenderBackendType,
		passes: readonly PlannedPostProcessPass[]
	): PostProcessTransientDescriptor[] {
		return this._collectDescriptors(
			passes.flatMap((pass) =>
				(pass.declaration.transients ?? []).map((entry) => ({
					passId: pass.id,
					descriptor: entry.descriptor,
				}))
			),
			"transient",
			backend,
			(descriptor) => createPostProcessScaledResourceDescriptorKey(descriptor, {
				includeMipMode: true,
			}),
		);
	}

	private _collectDescriptors<T extends PostProcessHistoryDescriptor>(
		entries: readonly { readonly passId: string; readonly descriptor: T }[],
		kind: "history" | "transient",
		backend: RenderBackendType,
		getKey: (descriptor: T) => string
	): T[] {
		const descriptors = new Map<string, T>();
		const keys = new Map<string, string>();
		for (const entry of entries) {
			const id = entry.descriptor.id;
			const key = getKey(entry.descriptor);
			const previousKey = keys.get(id);
			if (previousKey !== undefined && previousKey !== key) {
				throw new Error(
					`Post-process ${backend} pass "${entry.passId}" has incompatible ` +
					`${kind} descriptor "${id}".`,
				);
			}
			keys.set(id, key);
			descriptors.set(id, entry.descriptor);
		}
		return Array.from(descriptors.values());
	}

	private _sliceFromStartPass(
		passes: readonly ResolvedPostProcessPass[],
		startPassId: string | null
	): ResolvedPostProcessPass[] {
		if (!startPassId) return Array.from(passes);
		const index = passes.findIndex((pass) => pass.id === startPassId);
		return index < 0 ? Array.from(passes) : passes.slice(index);
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
		) return null;
		const id = incremental.postProcessStartPass ?? null;
		return id && passes.some((pass) => pass.id === id) ? id : null;
	}

	private _createResolveRequest<TOptions>(
		resolved: ResolvedPostProcessPass<TOptions>,
		request: PostProcessPlanRequest,
		width: number,
		height: number
	): PostProcessPassResolveRequest<TOptions> {
		return {
			frameContext: request.frameContext,
			postProcess: request.postProcess,
			backend: request.backend,
			gBuffer: request.gBuffer,
			width,
			height,
			options: resolved.options,
		};
	}

	private _createSignature(
		context: FrameContext,
		passes: readonly PlannedPostProcessPass[]
	): string {
		return [
			context.attachments?.width ?? 1,
			context.attachments?.height ?? 1,
			...passes.map((pass) =>
				`${pass.id}:${stableSerialize({
					options: pass.options,
					declaration: pass.declaration,
					historyDescriptors: (pass.declaration.histories ?? [])
						.map((entry) => entry.descriptor),
				})}`
			),
		].join("|");
	}
}

function comparePostProcessPassOrder(
	left: ResolvedPostProcessPass,
	right: ResolvedPostProcessPass
): number {
	const difference = getPostProcessPassSortOrder(left.pass) -
		getPostProcessPassSortOrder(right.pass);
	return difference !== 0 ? difference : left.id.localeCompare(right.id);
}

function getPostProcessPassSortOrder(pass: PostProcessPass): number {
	const order = pass.schedule.order;
	if (pass.builtIn && typeof order === "number" && Number.isFinite(order)) {
		return order;
	}
	const placement = isPostProcessPlacement(pass.schedule.placement) ?
		pass.schedule.placement : DEFAULT_POST_PROCESS_PLACEMENT;
	const localOrder = typeof order === "number" && Number.isFinite(order) ?
		Math.max(-CUSTOM_ORDER_LIMIT, Math.min(CUSTOM_ORDER_LIMIT, order)) : 0;
	return getCustomPostProcessPlacementOrder(placement) +
		localOrder * CUSTOM_ORDER_SCALE;
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

function deepFreezeDeclaration(
	declaration: PostProcessExecutionDeclaration
): PostProcessExecutionDeclaration {
	if (!declaration || typeof declaration !== "object") return declaration;
	for (const value of Object.values(declaration)) deepFreezeValue(value);
	return Object.freeze(declaration);
}

function deepFreezeValue(value: unknown): void {
	if (!value || typeof value !== "object" || Object.isFrozen(value)) return;
	for (const child of Object.values(value)) deepFreezeValue(child);
	Object.freeze(value);
}

function stableSerialize(value: unknown): string {
	if (value === null || typeof value !== "object") return JSON.stringify(value);
	if (Array.isArray(value)) {
		return `[${value.map((entry) => stableSerialize(entry)).join(",")}]`;
	}
	const record = value as Record<string, unknown>;
	return `{${Object.keys(record).sort().map((key) =>
		`${JSON.stringify(key)}:${stableSerialize(record[key])}`
	).join(",")}}`;
}

export function isPostProcessPassStage(stage: string): boolean {
	return stage === "postprocess" || getBuiltinPostProcessOrder(stage) !== null;
}
