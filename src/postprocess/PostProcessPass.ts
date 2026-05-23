import { EventEmitter } from "../core/EventEmitter";
import type { FeatureWarning, FrameContext } from "../pipeline/types";
import type { PostProcessIncrementalMetadata } from "../pipeline/incremental";
import type { PostProcessPlacement } from "./ordering";
import type {
	IPostProcessExecutor,
	LogicalGBufferBridge,
	PostProcessBackendKind,
	PostProcessHistoryDescriptor,
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassRequirements,
	PostProcessPassResult,
} from "./types";

export const POST_PROCESS_PASS_IDS = [
	"ssao",
	"ssgi",
	"taa",
	"ssr",
	"volumetric",
	"fog",
	"motion-blur",
	"dof",
	"bloom",
	"tonemap",
	"color-filter",
	"fxaa",
	"interaction-outline",
	"gamma",
] as const;

export type PostProcessPassId = (typeof POST_PROCESS_PASS_IDS)[number];

const POST_PROCESS_PASS_ID_SET = new Set<string>(POST_PROCESS_PASS_IDS);

const POST_PROCESS_WARNING_LABELS: Record<PostProcessPassId, string> = {
	ssao: "SSAO",
	ssgi: "SSGI",
	taa: "TAA",
	ssr: "SSR",
	volumetric: "volumetric effects",
	fog: "fog",
	"motion-blur": "motion blur",
	dof: "depth of field",
	bloom: "bloom",
	tonemap: "tone mapping",
	"color-filter": "color filter",
	fxaa: "FXAA",
	"interaction-outline": "interaction outline",
	gamma: "gamma correction",
};

export type PostProcessCapabilities = {
	[K in PostProcessPassId]: boolean;
};

export const DEFAULT_POST_PROCESS_CAPABILITIES: PostProcessCapabilities = {
	ssao: false,
	ssgi: false,
	taa: false,
	ssr: false,
	volumetric: false,
	fog: false,
	"motion-blur": false,
	dof: false,
	bloom: false,
	tonemap: false,
	"color-filter": false,
	fxaa: false,
	"interaction-outline": false,
	gamma: false,
};

export interface PostProcessPassConfig<TRawOptions = unknown> {
	readonly id: string;
	readonly placement?: PostProcessPlacement;
	readonly order?: number;
	readonly enabled?: boolean;
	readonly options?: Partial<TRawOptions>;
	readonly incremental?: PostProcessIncrementalMetadata;
	readonly implementations?: Partial<
		Record<PostProcessBackendKind, PostProcessPassImplementation>
	>;
}

export interface PostProcessPassChange {
	readonly passId: string;
	readonly reason: "enabled" | "options" | "reset" | "lifecycle";
}

export interface PostProcessPassResolveRequest<TOptions = unknown> {
	readonly frameContext?: FrameContext;
	readonly postProcess?: PostProcessPassRegistrySnapshot;
	readonly backend?: PostProcessBackendKind;
	readonly gBuffer?: LogicalGBufferBridge;
	readonly width?: number;
	readonly height?: number;
	readonly options?: TOptions;
}

export interface PostProcessPassWarmupRequest<TOptions = unknown> {
	readonly frameContext: FrameContext;
	readonly postProcess: PostProcessPassRegistrySnapshot;
	readonly backend: PostProcessBackendKind;
	readonly context: unknown;
	readonly options: TOptions;
}

export interface ResolvedPostProcessPass<TOptions = unknown> {
	readonly id: string;
	readonly pass: PostProcessPass<unknown, TOptions>;
	readonly options: TOptions;
}

/**
 * Stateful logical post-process pass.
 */
export abstract class PostProcessPass<
	TRawOptions = unknown,
	TOptions = TRawOptions,
> extends EventEmitter<{ change: [PostProcessPassChange] }> {
	public readonly id: string;
	public readonly placement?: PostProcessPlacement;
	public readonly order?: number;
	public readonly incremental?: PostProcessIncrementalMetadata;
	private readonly _implementations: Partial<
		Record<PostProcessBackendKind, PostProcessPassImplementation>
	>;
	private readonly _initialOptions: Partial<TRawOptions>;
	private _enabled: boolean;
	private _options: Partial<TRawOptions>;

	protected constructor(config: PostProcessPassConfig<TRawOptions>) {
		super();
		if (!config.id) {
			throw new Error("Post-process pass id is required.");
		}
		this.id = config.id;
		this.placement = config.placement;
		this.order = config.order;
		this.incremental = config.incremental;
		this._enabled = config.enabled === true;
		this._initialOptions = clonePlainOptions(config.options);
		this._options = clonePlainOptions(config.options);
		this._implementations = config.implementations ?? {};
	}

	public get enabled(): boolean {
		return this._enabled;
	}

	public enable(options?: Partial<TRawOptions>): this {
		if (options) {
			this._options = mergePlainOptions(this._options, options);
		}
		const changed = !this._enabled || !!options;
		this._enabled = true;
		if (changed) {
			this._emitChange(options ? "options" : "enabled");
		}
		return this;
	}

	public disable(): this {
		return this.setEnabled(false);
	}

	public setEnabled(enabled: boolean): this {
		if (this._enabled === enabled) {
			return this;
		}
		this._enabled = enabled;
		this._emitChange("enabled");
		return this;
	}

	public setOptions(options: Partial<TRawOptions>): this {
		this._options = mergePlainOptions(this._options, options);
		this._emitChange("options");
		return this;
	}

	public resetOptions(): this {
		this._options = clonePlainOptions(this._initialOptions);
		this._emitChange("reset");
		return this;
	}

	public getRawOptions(): Readonly<Partial<TRawOptions>> {
		return clonePlainOptions(this._options);
	}

	public normalizeOptions(_request: PostProcessPassResolveRequest): TOptions {
		return clonePlainOptions(this._options) as TOptions;
	}

	public isEnabled(_request: PostProcessPassResolveRequest): boolean {
		return this._enabled;
	}

	public getRequirements(
		_request: PostProcessPassResolveRequest<TOptions>
	): PostProcessPassRequirements {
		return {};
	}

	public getHistoryDescriptors(
		_request: PostProcessPassResolveRequest<TOptions>
	): readonly PostProcessHistoryDescriptor[] {
		return [];
	}

	public getHistorySignature(
		request: PostProcessPassResolveRequest<TOptions>
	): string {
		return stableSerialize(request.options ?? this._options);
	}

	public getImplementation(
		backend: PostProcessBackendKind
	): PostProcessPassImplementation | null {
		return this._implementations[backend] ?? null;
	}

	public supportsBackend(backend: PostProcessBackendKind): boolean {
		if (this.getImplementation(backend) !== null) {
			return true;
		}
		return backend !== "software" && backend !== "webgpu" && backend !== "webgl";
	}

	public async warmup(
		request: PostProcessPassWarmupRequest<TOptions>
	): Promise<void> {
		const implementation = this.getImplementation(request.backend);
		await implementation?.warmup?.(request.context, request);
	}

	public execute(
		request: PostProcessPassRequest<TOptions>,
		context: unknown,
		executor: IPostProcessExecutor
	): PostProcessPassResult | Promise<PostProcessPassResult> {
		const implementation = this.getImplementation(executor.backend);
		if (implementation?.execute) {
			return implementation.execute(request, context);
		}
		return executor.executePass(this.id, request);
	}

	public invalidate(backend?: PostProcessBackendKind): void {
		for (const [kind, implementation] of Object.entries(this._implementations)) {
			if (backend && kind !== backend) {
				continue;
			}
			implementation?.invalidate?.();
		}
		this._emitChange("lifecycle");
	}

	public destroy(backend?: PostProcessBackendKind): void {
		for (const [kind, implementation] of Object.entries(this._implementations)) {
			if (backend && kind !== backend) {
				continue;
			}
			implementation?.destroy?.();
		}
		this._emitChange("lifecycle");
	}

	private _emitChange(reason: PostProcessPassChange["reason"]): void {
		this.emit("change", { passId: this.id, reason });
	}
}

export interface PostProcessPassRegistryChange {
	readonly passId: string;
	readonly reason: PostProcessPassChange["reason"] | "register" | "unregister";
}

/**
 * Registry exposed as `renderer.postProcess`.
 */
export class PostProcessPassRegistry extends EventEmitter<{
	change: [PostProcessPassRegistryChange];
}> {
	private _passes = new Map<string, PostProcessPass>();
	private _passChangeListeners = new Map<
		string,
		(change: PostProcessPassChange) => void
	>();

	public registerPass(pass: PostProcessPass): this {
		if (!(pass instanceof PostProcessPass)) {
			throw new Error("renderer.postProcess.registerPass requires a PostProcessPass.");
		}
		if (this._passes.has(pass.id)) {
			throw new Error(`Post-process pass "${pass.id}" is already registered.`);
		}
		this._passes.set(pass.id, pass);
		const listener = (change: PostProcessPassChange): void => {
			this.emit("change", {
				passId: change.passId,
				reason: change.reason,
			});
		};
		pass.on("change", listener);
		this._passChangeListeners.set(pass.id, listener);
		this.emit("change", { passId: pass.id, reason: "register" });
		return this;
	}

	public unregisterPass(id: string): this {
		const pass = this._passes.get(id);
		if (!pass) {
			return this;
		}
		const listener = this._passChangeListeners.get(id);
		if (listener) {
			pass.off("change", listener);
			this._passChangeListeners.delete(id);
		}
		this._passes.delete(id);
		pass.destroy();
		this.emit("change", { passId: id, reason: "unregister" });
		return this;
	}

	public getPass<TPass extends PostProcessPass = PostProcessPass>(
		id: string
	): TPass | null {
		return (this._passes.get(id) as TPass | undefined) ?? null;
	}

	public getPasses(): readonly PostProcessPass[] {
		return Array.from(this._passes.values());
	}

	/**
	 * Invalidates backend resources owned by registered pass implementations.
	 *
	 * @param backend Optional backend kind to invalidate.
	 * @returns This registry.
	 * @sideEffects May clear implementation-owned backend caches.
	 */
	public invalidatePasses(backend?: PostProcessBackendKind): this {
		for (const pass of this._passes.values()) {
			pass.invalidate(backend);
		}
		return this;
	}

	/**
	 * Destroys backend resources owned by registered pass implementations.
	 *
	 * @param backend Optional backend kind to destroy.
	 * @returns This registry.
	 * @sideEffects Releases implementation-owned backend resources.
	 */
	public destroyPasses(backend?: PostProcessBackendKind): this {
		for (const pass of this._passes.values()) {
			pass.destroy(backend);
		}
		return this;
	}

	public createSnapshot(
		capabilities: PostProcessCapabilities,
		backendType: string
	): PostProcessPassRegistrySnapshot {
		return new PostProcessPassRegistrySnapshot(
			this.getPasses(),
			capabilities,
			backendType
		);
	}
}

/**
 * Immutable per-frame post-process view consumed by the renderer pipeline.
 */
export class PostProcessPassRegistrySnapshot {
	private _passes = new Map<string, ResolvedPostProcessPass>();
	private _warnings: FeatureWarning[] = [];

	constructor(
		passes: readonly PostProcessPass[],
		capabilities: PostProcessCapabilities,
		backendType: string,
		resolvedPasses?: readonly ResolvedPostProcessPass[],
		warnings?: readonly FeatureWarning[]
	) {
		if (resolvedPasses) {
			for (const pass of resolvedPasses) {
				this._passes.set(pass.id, pass);
			}
			this._warnings = warnings?.slice() ?? [];
			return;
		}
		for (const pass of passes) {
			const backend = backendType as PostProcessBackendKind;
			const options = pass.normalizeOptions({
				backend,
				postProcess: this,
			});
			if (!pass.isEnabled({ backend, postProcess: this, options })) {
				continue;
			}
			if (!this._isCapabilitySupported(pass, capabilities)) {
				this._warnings.push({
					key: `${backendType}-postprocess-unsupported-${pass.id}`,
					message:
						`${backendType} backend does not support ` +
						`${getPostProcessWarningLabel(pass.id)} post-processing; disabling it`,
				});
				continue;
			}
			if (!pass.supportsBackend(backend)) {
				continue;
			}
			this._passes.set(pass.id, {
				id: pass.id,
				pass: pass as PostProcessPass<unknown, unknown>,
				options,
			});
		}
	}

	public getPass<TOptions = unknown>(
		id: string
	): ResolvedPostProcessPass<TOptions> | null {
		return (this._passes.get(id) as ResolvedPostProcessPass<TOptions> | undefined) ??
			null;
	}

	public isEnabled(id: string): boolean {
		return this._passes.has(id);
	}

	public getOptions<TOptions = unknown>(id: string): TOptions | null {
		return (this._passes.get(id)?.options as TOptions | undefined) ?? null;
	}

	public getEnabledPasses(): readonly ResolvedPostProcessPass[] {
		return Array.from(this._passes.values());
	}

	public hasEnabledCustomPass(): boolean {
		for (const id of this._passes.keys()) {
			if (!isBuiltInPostProcessPassId(id)) {
				return true;
			}
		}
		return false;
	}

	public getWarnings(): readonly FeatureWarning[] {
		return this._warnings.slice();
	}

	public withPassDisabled(id: string): PostProcessPassRegistrySnapshot {
		return new PostProcessPassRegistrySnapshot(
			[],
			DEFAULT_POST_PROCESS_CAPABILITIES,
			"snapshot",
			this.getEnabledPasses().filter((pass) => pass.id !== id),
			this._warnings
		);
	}

	private _isCapabilitySupported(
		pass: PostProcessPass,
		capabilities: PostProcessCapabilities
	): boolean {
		if (!isBuiltInPostProcessPassId(pass.id)) {
			return true;
		}
		return capabilities[pass.id] === true;
	}
}

export function isBuiltInPostProcessPassId(
	id: string
): id is PostProcessPassId {
	return POST_PROCESS_PASS_ID_SET.has(id);
}

export function getPostProcessWarningLabel(id: string): string {
	if (isBuiltInPostProcessPassId(id)) {
		return POST_PROCESS_WARNING_LABELS[id];
	}
	return id;
}

export function getEnabledCustomPostProcessPassIds(
	postProcess: PostProcessPassRegistrySnapshot
): string[] {
	return postProcess
		.getEnabledPasses()
		.map((pass) => pass.id)
		.filter((id) => !isBuiltInPostProcessPassId(id));
}

export function hasEnabledCustomPostProcessPass(
	postProcess: PostProcessPassRegistrySnapshot
): boolean {
	return postProcess.hasEnabledCustomPass();
}

function clonePlainOptions<TOptions>(options?: Partial<TOptions>): Partial<TOptions> {
	return { ...((options ?? {}) as object) } as Partial<TOptions>;
}

function mergePlainOptions<TOptions>(
	base: Partial<TOptions>,
	override: Partial<TOptions>
): Partial<TOptions> {
	return {
		...(base as object),
		...(override as object),
	} as Partial<TOptions>;
}

function stableSerialize(value: unknown): string {
	if (value === null || typeof value !== "object") {
		return JSON.stringify(value);
	}
	if (Array.isArray(value)) {
		return `[${value.map((entry) => stableSerialize(entry)).join(",")}]`;
	}
	const record = value as Record<string, unknown>;
	const keys = Object.keys(record).sort();
	return `{${keys
		.map((key) => `${JSON.stringify(key)}:${stableSerialize(record[key])}`)
		.join(",")}}`;
}
