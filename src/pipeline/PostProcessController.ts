import type {
	BloomOptions,
	ColorFilterOptions,
	DOFOptions,
	FeatureWarning,
	FogOptions,
	MotionBlurOptions,
	SSAOOptions,
	SSGIOptions,
	SSROptions,
	TAAOptions,
	VolumetricOptions,
} from "./types";
import {
	DEFAULT_BLOOM_OPTIONS,
	DEFAULT_COLOR_FILTER_OPTIONS,
	DEFAULT_DOF_OPTIONS,
	DEFAULT_FOG_OPTIONS,
	DEFAULT_MOTION_BLUR_OPTIONS,
	DEFAULT_SSAO_OPTIONS,
	DEFAULT_SSGI_OPTIONS,
	DEFAULT_SSR_OPTIONS,
	DEFAULT_TAA_OPTIONS,
	DEFAULT_VOLUMETRIC_OPTIONS,
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

export interface PostProcessOptionsMap {
	ssao: SSAOOptions;
	ssgi: SSGIOptions;
	taa: TAAOptions;
	ssr: SSROptions;
	volumetric: VolumetricOptions;
	fog: FogOptions;
	"motion-blur": MotionBlurOptions;
	dof: DOFOptions;
	bloom: BloomOptions;
	tonemap: Record<string, never>;
	"color-filter": ColorFilterOptions;
	fxaa: Record<string, never>;
	"interaction-outline": Record<string, never>;
	gamma: Record<string, never>;
}

export type PostProcessCapabilities = {
	[K in PostProcessPassId]: boolean;
};

export interface PostProcessPassRequest<TOptions = Record<string, never>> {
	enabled?: boolean;
	options?: TOptions;
}

export type PostProcessRequest = {
	[K in PostProcessPassId]?: PostProcessPassRequest<PostProcessOptionsMap[K]>;
} & Record<string, PostProcessPassRequest<any> | undefined>;

type MutablePostProcessRequest = Record<
	string,
	PostProcessPassRequest<any> | undefined
>;

export type ResolvedPostProcessOptionsMap = {
	[K in PostProcessPassId]: PostProcessOptionsMap[K];
} & Record<string, unknown>;

export interface ResolvedPostProcessState {
	enabled: Record<PostProcessPassId, boolean> & Record<string, boolean>;
	options: ResolvedPostProcessOptionsMap;
	warnings: FeatureWarning[];
}

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

const DEFAULT_POST_PROCESS_REQUEST: PostProcessRequest = {
	tonemap: { enabled: true },
	"interaction-outline": { enabled: true },
	gamma: { enabled: true },
};

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

export interface PostProcessCustomPassDescriptor {
	readonly id: string;
}

export interface PostProcessPassRegistry<
	TBackendPass extends PostProcessCustomPassDescriptor =
		PostProcessCustomPassDescriptor,
> {
	registerPass(pass: TBackendPass): void;
	unregisterPass(id: string): void;
}

export interface PostProcessControllerOptions<
	TBackendPass extends PostProcessCustomPassDescriptor =
		PostProcessCustomPassDescriptor,
> {
	passRegistry?: PostProcessPassRegistry<TBackendPass> | null;
	onChange?: (() => void) | null;
}

/**
 * Stores public post-process requests for `Renderer`.
 */
export class PostProcessController<
	TBackendPass extends PostProcessCustomPassDescriptor =
		PostProcessCustomPassDescriptor,
> {
	private _request: PostProcessRequest;
	private _customPasses = new Map<string, TBackendPass>();
	private _passRegistry: PostProcessPassRegistry<TBackendPass> | null;
	private _onChange: (() => void) | null;

	constructor(
		initial?: PostProcessRequest,
		options: PostProcessControllerOptions<TBackendPass> = {}
	) {
		this._request = clonePostProcessRequest(initial ?? {});
		this._passRegistry = options.passRegistry ?? null;
		this._onChange = options.onChange ?? null;
	}

	/**
	 * Enables a built-in or registered custom post-process pass and optionally
	 * merges pass options.
	 *
	 * @param id Built-in pass id or registered custom pass id.
	 * @param options Optional pass-specific options merged with existing options.
	 * @returns This controller for call chaining.
	 * @sideEffects Mutates the renderer post-process request state.
	 */
	public enable<TPassId extends string>(
		id: TPassId,
		options?: TPassId extends PostProcessPassId
			? PostProcessOptionsMap[TPassId]
			: Record<string, unknown>
	): this {
		this._assertKnownPassId(id);
		const mutable = this._request as MutablePostProcessRequest;
		const current = (mutable[id] ?? {}) as PostProcessPassRequest<any>;
		mutable[id] = {
			...current,
			enabled: true,
			options: mergeOptions(current.options, options as Record<string, unknown>),
		};
		this._notifyChanged();
		return this;
	}

	/**
	 * Disables a built-in or registered custom post-process pass.
	 *
	 * @param id Built-in pass id or registered custom pass id.
	 * @returns This controller for call chaining.
	 * @sideEffects Mutates the renderer post-process request state.
	 */
	public disable(id: string): this {
		this._assertKnownPassId(id);
		const mutable = this._request as MutablePostProcessRequest;
		const current = (mutable[id] ?? {}) as PostProcessPassRequest<any>;
		mutable[id] = {
			...current,
			enabled: false,
		};
		this._notifyChanged();
		return this;
	}

	/**
	 * Updates pass options without changing whether the pass is enabled.
	 *
	 * @param id Built-in pass id or registered custom pass id.
	 * @param options Pass-specific options merged with existing options.
	 * @returns This controller for call chaining.
	 * @sideEffects Mutates the renderer post-process request state.
	 */
	public setOptions<TPassId extends string>(
		id: TPassId,
		options: TPassId extends PostProcessPassId
			? PostProcessOptionsMap[TPassId]
			: Record<string, unknown>
	): this {
		this._assertKnownPassId(id);
		const mutable = this._request as MutablePostProcessRequest;
		const current = (mutable[id] ?? {}) as PostProcessPassRequest<any>;
		mutable[id] = {
			...current,
			options: mergeOptions(current.options, options as Record<string, unknown>),
		};
		this._notifyChanged();
		return this;
	}

	/**
	 * Resets one pass or the full post-process request to defaults.
	 *
	 * @param id Optional built-in or registered custom pass id. Omit to reset
	 * all pass requests.
	 * @returns This controller for call chaining.
	 * @sideEffects Mutates the renderer post-process request state.
	 */
	public reset(id?: string): this {
		if (!id) {
			this._request = {};
			this._notifyChanged();
			return this;
		}
		this._assertKnownPassId(id);
		const mutable = this._request as MutablePostProcessRequest;
		delete mutable[id];
		this._notifyChanged();
		return this;
	}

	/**
	 * Returns a cloned snapshot of the current post-process request state.
	 *
	 * @returns Post-process request snapshot.
	 * @sideEffects None.
	 */
	public getState(): PostProcessRequest {
		return clonePostProcessRequest(this._request);
	}

	/**
	 * Registers a custom backend post-process pass and makes its id available
	 * to `enable`, `disable`, `setOptions`, and `reset`.
	 *
	 * @param pass Backend-specific pass descriptor with a unique custom id.
	 * @returns This controller for call chaining.
	 * @sideEffects Mutates controller registration state and forwards the pass
	 * to the connected backend registry when available.
	 */
	public registerPass(pass: TBackendPass): this {
		this._assertCanRegisterCustomPass(pass);
		if (!this._passRegistry) {
			throw new Error(
				"Post-process backend pass registry is not available for this renderer."
			);
		}
		this._passRegistry.registerPass(pass);
		this._customPasses.set(pass.id, pass);
		this._notifyChanged();
		return this;
	}

	/**
	 * Unregisters a custom backend post-process pass.
	 *
	 * @param id Custom pass id to remove.
	 * @returns This controller for call chaining.
	 * @sideEffects Mutates controller registration state, removes any stored
	 * request for the pass, and forwards unregister to the connected backend
	 * registry when available.
	 */
	public unregisterPass(id: string): this {
		if (!this._customPasses.has(id)) {
			return this;
		}
		if (!this._passRegistry) {
			throw new Error(
				"Post-process backend pass registry is not available for this renderer."
			);
		}
		this._passRegistry.unregisterPass(id);
		this._customPasses.delete(id);
		const mutable = this._request as MutablePostProcessRequest;
		delete mutable[id];
		this._notifyChanged();
		return this;
	}

	private _assertKnownPassId(id: string): void {
		if (POST_PROCESS_PASS_ID_SET.has(id) || this._customPasses.has(id)) {
			return;
		}
		throw new Error(`Unknown post-process pass "${id}".`);
	}

	private _assertCanRegisterCustomPass(pass: TBackendPass): void {
		if (!pass.id) {
			throw new Error("Custom post-process pass id is required.");
		}
		if (POST_PROCESS_PASS_ID_SET.has(pass.id)) {
			throw new Error(
				`Cannot register built-in post-process pass "${pass.id}" as a custom pass.`
			);
		}
		if (this._customPasses.has(pass.id)) {
			throw new Error(
				`Custom post-process pass "${pass.id}" is already registered.`
			);
		}
	}

	private _notifyChanged(): void {
		this._onChange?.();
	}
}

export function resolvePostProcessState(
	request: PostProcessRequest,
	capabilities: PostProcessCapabilities,
	backendType: string
): ResolvedPostProcessState {
	const mergedRequest = mergePostProcessRequest(
		DEFAULT_POST_PROCESS_REQUEST,
		request
	);
	const warnings: FeatureWarning[] = [];
	const enabled = {} as Record<PostProcessPassId, boolean> &
		Record<string, boolean>;

	for (const id of POST_PROCESS_PASS_IDS) {
		const passRequest = mergedRequest[id];
		const explicitRequest = request[id];
		const requested = passRequest?.enabled === true;
		const explicitlyEnabled = explicitRequest?.enabled === true;
		const supported = capabilities[id] === true;
		enabled[id] = requested && supported;
		if (explicitlyEnabled && !supported) {
			warnings.push({
				key: `${backendType}-postprocess-unsupported-${id}`,
				message:
					`${backendType} backend does not support ` +
					`${POST_PROCESS_WARNING_LABELS[id]} post-processing; disabling it`,
			});
		}
	}
	for (const id of Object.keys(mergedRequest)) {
		if (POST_PROCESS_PASS_ID_SET.has(id)) {
			continue;
		}
		enabled[id] = mergedRequest[id]?.enabled === true;
	}

	return {
		enabled,
		options: resolvePostProcessOptions(mergedRequest),
		warnings,
	};
}

export function isFogPostProcessEnabled(
	postProcess: ResolvedPostProcessState
): boolean {
	return (
		postProcess.enabled.fog &&
		(postProcess.options.fog.application ?? "postprocess") !== "scene"
	);
}

export function getEnabledCustomPostProcessPassIds(
	postProcess: ResolvedPostProcessState
): string[] {
	const ids: string[] = [];
	for (const id of Object.keys(postProcess.enabled)) {
		if (POST_PROCESS_PASS_ID_SET.has(id)) {
			continue;
		}
		if (postProcess.enabled[id]) {
			ids.push(id);
		}
	}
	return ids;
}

export function hasEnabledCustomPostProcessPass(
	postProcess: ResolvedPostProcessState
): boolean {
	for (const id of Object.keys(postProcess.enabled)) {
		if (POST_PROCESS_PASS_ID_SET.has(id)) {
			continue;
		}
		if (postProcess.enabled[id]) {
			return true;
		}
	}
	return false;
}

export function clonePostProcessRequest(
	request: PostProcessRequest
): PostProcessRequest {
	const clone: PostProcessRequest = {};
	const mutable = clone as MutablePostProcessRequest;
	for (const id of Object.keys(request)) {
		const pass = request[id];
		if (!pass) continue;
		mutable[id] = clonePostProcessPassRequest(
			pass as PostProcessPassRequest<any>
		);
	}
	return clone;
}

function clonePostProcessPassRequest<TOptions>(
	pass: PostProcessPassRequest<TOptions>
): PostProcessPassRequest<TOptions> {
	return {
		enabled: pass.enabled,
		options: cloneOptions(pass.options),
	};
}

function mergePostProcessRequest(
	base: PostProcessRequest,
	override: PostProcessRequest
): PostProcessRequest {
	const merged = clonePostProcessRequest(base);
	const mutable = merged as MutablePostProcessRequest;
	for (const id of Object.keys(override)) {
		const pass = override[id];
		if (!pass) continue;
		const current = (mutable[id] ?? {}) as PostProcessPassRequest<any>;
		mutable[id] = {
			...current,
			...pass,
			options: mergeOptions(current.options, pass.options),
		};
	}
	return merged;
}

function resolvePostProcessOptions(
	request: PostProcessRequest
): ResolvedPostProcessOptionsMap {
	const options = {
		ssao: mergeOptions(DEFAULT_SSAO_OPTIONS, request.ssao?.options),
		ssgi: mergeOptions(DEFAULT_SSGI_OPTIONS, request.ssgi?.options),
		taa: mergeOptions(DEFAULT_TAA_OPTIONS, request.taa?.options),
		ssr: mergeOptions(DEFAULT_SSR_OPTIONS, request.ssr?.options),
		volumetric: mergeOptions(
			DEFAULT_VOLUMETRIC_OPTIONS,
			request.volumetric?.options
		),
		fog: mergeOptions(DEFAULT_FOG_OPTIONS, request.fog?.options),
		"motion-blur": mergeOptions(
			DEFAULT_MOTION_BLUR_OPTIONS,
			request["motion-blur"]?.options
		),
		dof: mergeOptions(DEFAULT_DOF_OPTIONS, request.dof?.options),
		bloom: mergeOptions(DEFAULT_BLOOM_OPTIONS, request.bloom?.options),
		tonemap: {},
		"color-filter": mergeOptions(
			DEFAULT_COLOR_FILTER_OPTIONS,
			request["color-filter"]?.options
		),
		fxaa: {},
		"interaction-outline": {},
		gamma: {},
	} as ResolvedPostProcessOptionsMap;
	for (const id of Object.keys(request)) {
		if (POST_PROCESS_PASS_ID_SET.has(id)) {
			continue;
		}
		options[id] = cloneOptions(request[id]?.options) ?? {};
	}
	return options;
}

function mergeOptions<TOptions>(
	base?: TOptions,
	override?: TOptions
): TOptions {
	return {
		...((base ?? {}) as object),
		...((override ?? {}) as object),
	} as TOptions;
}

function cloneOptions<TOptions>(options?: TOptions): TOptions | undefined {
	if (!options) {
		return undefined;
	}
	return { ...(options as object) } as TOptions;
}
