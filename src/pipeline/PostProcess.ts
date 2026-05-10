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
};

type MutablePostProcessRequest = Partial<
	Record<PostProcessPassId, PostProcessPassRequest<any>>
>;

export type ResolvedPostProcessOptionsMap = {
	[K in PostProcessPassId]: PostProcessOptionsMap[K];
};

export interface ResolvedPostProcessState {
	enabled: Record<PostProcessPassId, boolean>;
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

/**
 * Stores public post-process requests for `Renderer`.
 */
export class PostProcessController {
	private _request: PostProcessRequest;

	constructor(initial?: PostProcessRequest) {
		this._request = clonePostProcessRequest(initial ?? {});
	}

	/**
	 * Enables a built-in post-process pass and optionally merges pass options.
	 *
	 * @param id Built-in post-process pass id.
	 * @param options Optional pass-specific options merged with existing options.
	 * @returns This controller for call chaining.
	 * @sideEffects Mutates the renderer post-process request state.
	 */
	public enable<TPassId extends PostProcessPassId>(
		id: TPassId,
		options?: PostProcessOptionsMap[TPassId]
	): this {
		const mutable = this._request as MutablePostProcessRequest;
		const current = (mutable[id] ?? {}) as PostProcessPassRequest<any>;
		mutable[id] = {
			...current,
			enabled: true,
			options: mergeOptions(current.options, options),
		};
		return this;
	}

	/**
	 * Disables a built-in post-process pass.
	 *
	 * @param id Built-in post-process pass id.
	 * @returns This controller for call chaining.
	 * @sideEffects Mutates the renderer post-process request state.
	 */
	public disable(id: PostProcessPassId): this {
		const mutable = this._request as MutablePostProcessRequest;
		const current = (mutable[id] ?? {}) as PostProcessPassRequest<any>;
		mutable[id] = {
			...current,
			enabled: false,
		};
		return this;
	}

	/**
	 * Updates pass options without changing whether the pass is enabled.
	 *
	 * @param id Built-in post-process pass id.
	 * @param options Pass-specific options merged with existing options.
	 * @returns This controller for call chaining.
	 * @sideEffects Mutates the renderer post-process request state.
	 */
	public setOptions<TPassId extends PostProcessPassId>(
		id: TPassId,
		options: PostProcessOptionsMap[TPassId]
	): this {
		const mutable = this._request as MutablePostProcessRequest;
		const current = (mutable[id] ?? {}) as PostProcessPassRequest<any>;
		mutable[id] = {
			...current,
			options: mergeOptions(current.options, options),
		};
		return this;
	}

	/**
	 * Resets one pass or the full post-process request to defaults.
	 *
	 * @param id Optional built-in pass id. Omit to reset all passes.
	 * @returns This controller for call chaining.
	 * @sideEffects Mutates the renderer post-process request state.
	 */
	public reset(id?: PostProcessPassId): this {
		if (!id) {
			this._request = {};
			return this;
		}
		const mutable = this._request as MutablePostProcessRequest;
		delete mutable[id];
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
	const enabled = {} as Record<PostProcessPassId, boolean>;

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

export function clonePostProcessRequest(
	request: PostProcessRequest
): PostProcessRequest {
	const clone: PostProcessRequest = {};
	const mutable = clone as MutablePostProcessRequest;
	for (const id of POST_PROCESS_PASS_IDS) {
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
	for (const id of POST_PROCESS_PASS_IDS) {
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
	return {
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
	};
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
