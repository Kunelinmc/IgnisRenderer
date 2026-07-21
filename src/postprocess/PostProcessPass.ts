import { EventEmitter } from "../core/EventEmitter";
import type { FeatureWarning, FrameContext } from "../pipeline/types";
import type { PostProcessIncrementalMetadata } from "../pipeline/incremental";
import type { PostProcessPlacement } from "./ordering";
import type {
	IRenderBackend,
	RenderBackendType,
} from "../backends/IRenderBackend";
import type {
	LogicalGBufferBridge,
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassResult,
} from "./types";

export type PostProcessPassId = string;

export type PostProcessPassImplementationFactory = (
	backend: IRenderBackend
) => PostProcessPassImplementation;

export interface PostProcessSchedule {
	readonly placement?: PostProcessPlacement;
	readonly order?: number;
	readonly incremental?: PostProcessIncrementalMetadata;
}

export interface PostProcessPassConfig<TRawOptions = unknown> {
	readonly id: string;
	/**
	 * Marks renderer-default built-in passes so registry consumers can separate
	 * them from manually registered passes.
	 *
	 * Omit this value for engine-provided or user-defined passes that must be
	 * registered explicitly.
	 */
	readonly builtIn?: boolean;
	/**
	 * Human-readable pass name used in unsupported-pass diagnostics.
	 *
	 * Omit this value to use `id` as the diagnostic label.
	 */
	readonly warningLabel?: string;
	readonly schedule?: PostProcessSchedule;
	readonly enabled?: boolean;
	readonly options?: Partial<TRawOptions>;
	readonly implementations?: Partial<
		Record<RenderBackendType, PostProcessPassImplementationFactory>
	>;
}

export interface PostProcessPassChange {
	readonly passId: string;
	readonly builtIn: boolean;
	readonly reason: "enabled" | "options" | "reset" | "lifecycle";
}

export interface PostProcessPassResolveRequest<TOptions = unknown> {
	readonly frameContext?: FrameContext;
	readonly postProcess?: PostProcessPassRegistrySnapshot;
	readonly backend?: RenderBackendType;
	readonly gBuffer?: LogicalGBufferBridge;
	readonly width?: number;
	readonly height?: number;
	readonly options?: TOptions;
}

export interface PostProcessPassWarmupRequest<TOptions = unknown> {
	readonly frameContext: FrameContext;
	readonly postProcess: PostProcessPassRegistrySnapshot;
	readonly backend: RenderBackendType;
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
	/**
	 * Whether this pass is a renderer-default built-in pass.
	 *
	 * The value is resolved during construction and has no side effects.
	 */
	public readonly builtIn: boolean;
	/**
	 * Human-readable pass name used by diagnostics.
	 *
	 * The value is resolved during construction and has no side effects.
	 */
	public readonly warningLabel: string;
	public readonly schedule: Readonly<PostProcessSchedule>;
	private readonly _implementations: Partial<
		Record<RenderBackendType, PostProcessPassImplementationFactory>
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
		this.builtIn = config.builtIn === true;
		this.warningLabel = config.warningLabel ?? config.id;
		this.schedule = Object.freeze({ ...config.schedule });
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

	/**
	 * Returns whether an enabled pass should participate in the logical
	 * post-process pipeline for the current frame.
	 *
	 * @param request Resolved pass request with backend, normalized options, and
	 * optional frame context.
	 * @returns `true` when the pass should be ordered and considered for
	 * execution.
	 * @remarks Snapshot enablement and backend implementation filtering happen
	 * before this method. Implementations must keep this predicate
	 * deterministic for the supplied request and must not allocate backend
	 * resources.
	 * @sideEffects None.
	 */
	public shouldExecute(_request: PostProcessPassResolveRequest<TOptions>): boolean {
		return true;
	}

	private readonly _cachedTestImplementations = new Map<string, PostProcessPassImplementation>();

	public getImplementationFactory(
		backend: RenderBackendType
	): PostProcessPassImplementationFactory | null {
		return this._implementations[backend] ?? null;
	}

	public getImplementation(
		backend: RenderBackendType
	): PostProcessPassImplementation | null {
		const cached = this._cachedTestImplementations.get(backend);
		if (cached) {
			return cached;
		}
		const factory = this.getImplementationFactory(backend);
		if (!factory) {
			return null;
		}
		const mockSession: any = {
			type: backend,
			profile: {
				id: backend,
				capabilities: {},
				frameScheduling: "on-demand",
				shadow: {},
				lighting: {},
			},
			extensions: {
				getBackendExtension: () => null,
				requireBackendExtension: () => { throw new Error("Mock"); },
			},
		};
		const instance = factory(mockSession);
		this._cachedTestImplementations.set(backend, instance);
		return instance;
	}

	public supportsBackend(backend: RenderBackendType): boolean {
		if (this.getImplementationFactory(backend) !== null) {
			return true;
		}
		return backend !== "software" && backend !== "webgpu" && backend !== "webgl";
	}

	public async warmup(
		request: PostProcessPassWarmupRequest<TOptions>,
		implementation?: PostProcessPassImplementation
	): Promise<void> {
		await implementation?.warmup?.(request.context, request);
	}

	public execute(
		request: PostProcessPassRequest<TOptions>,
		context: unknown
	): PostProcessPassResult | Promise<PostProcessPassResult> {
		const implementation = request.implementation;
		if (!implementation) {
			throw new Error(
				`Post-process pass "${this.id}" has no executable implementation.`,
			);
		}
		return implementation.execute(request, context);
	}

	public invalidate(backend?: RenderBackendType): void {
		this._emitChange("lifecycle");
		if (backend) {
			const impl = this._cachedTestImplementations.get(backend);
			impl?.invalidate?.();
		} else {
			for (const impl of this._cachedTestImplementations.values()) {
				impl.invalidate?.();
			}
		}
	}

	public destroy(backend?: RenderBackendType): void {
		this._emitChange("lifecycle");
		if (backend) {
			const impl = this._cachedTestImplementations.get(backend);
			impl?.destroy?.();
		} else {
			for (const impl of this._cachedTestImplementations.values()) {
				impl.destroy?.();
			}
			this._cachedTestImplementations.clear();
		}
	}

	private _emitChange(reason: PostProcessPassChange["reason"]): void {
		this.emit("change", {
			passId: this.id,
			builtIn: this.builtIn,
			reason,
		});
	}
}

export interface PostProcessPassRegistryChange {
	readonly passId: string;
	readonly builtIn: boolean;
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
				builtIn: change.builtIn,
				reason: change.reason,
			});
		};
		pass.on("change", listener);
		this._passChangeListeners.set(pass.id, listener);
		this.emit("change", {
			passId: pass.id,
			builtIn: pass.builtIn,
			reason: "register",
		});
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
		this.emit("change", {
			passId: id,
			builtIn: pass.builtIn,
			reason: "unregister",
		});
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
	public invalidatePasses(backend?: RenderBackendType): this {
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
	public destroyPasses(backend?: RenderBackendType): this {
		for (const pass of this._passes.values()) {
			pass.destroy(backend);
		}
		return this;
	}

	public createSnapshot(
		backendType: RenderBackendType
	): PostProcessPassRegistrySnapshot {
		return new PostProcessPassRegistrySnapshot(this.getPasses(), backendType);
	}
}

/**
 * Immutable per-frame post-process view consumed by the renderer pipeline.
 */
export class PostProcessPassRegistrySnapshot {
	private _passes = new Map<string, ResolvedPostProcessPass>();
	private _warnings: FeatureWarning[] = [];
	private readonly _backendType: RenderBackendType;

	constructor(
		passes: readonly PostProcessPass[],
		backendType: RenderBackendType,
		resolvedPasses?: readonly ResolvedPostProcessPass[],
		warnings?: readonly FeatureWarning[]
	) {
		this._backendType = backendType;
		if (resolvedPasses) {
			for (const pass of resolvedPasses) {
				this._passes.set(pass.id, pass);
			}
			this._warnings = warnings?.slice() ?? [];
			return;
		}
		for (const pass of passes) {
			const backend = backendType;
			const options = pass.normalizeOptions({
				backend,
				postProcess: this,
			});
			if (!pass.isEnabled({ backend, postProcess: this, options })) {
				continue;
			}
			if (!pass.supportsBackend(backend)) {
				if (pass.builtIn) {
					this._warnings.push({
						key: `${backendType}-postprocess-unsupported-${pass.id}`,
						message:
							`${backendType} backend does not support ` +
							`${pass.warningLabel} post-processing; disabling it`,
					});
				}
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
		for (const pass of this._passes.values()) {
			if (!pass.pass.builtIn) {
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
			this._backendType,
			this.getEnabledPasses().filter((pass) => pass.id !== id),
			this._warnings
		);
	}
}

export function getEnabledCustomPostProcessPassIds(
	postProcess: PostProcessPassRegistrySnapshot
): string[] {
	return postProcess
		.getEnabledPasses()
		.filter((pass) => !pass.pass.builtIn)
		.map((pass) => pass.id);
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
