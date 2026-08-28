import { LightType } from "../../lights";
import type { PreparedShadowLight } from "../../lights/shadows/ShadowFramePlan";
import type { FrameContext } from "../../pipeline/types";

export type WebGPUPagedShadowFeedbackMode = "conservative" | "screen-feedback";

/** @internal Fully resolved device-lifetime paged-shadow experiment settings. */
export interface WebGPUPagedShadowExperimentConfig {
	readonly enabled: boolean;
	readonly pageSize: number;
	readonly pageGridSize: number;
	readonly physicalPageCount: number;
	readonly maxPagesPerFrame: number;
	readonly cacheFrames: number;
	readonly feedbackMode: WebGPUPagedShadowFeedbackMode;
}

/** @internal Runtime settings consumed only by the WebGPU paged technique. */
export interface WebGPUPagedShadowSettings {
	readonly pageSize: number;
	readonly pageGridSize: number;
	readonly physicalPageCount: number;
	readonly maxPagesPerFrame: number;
	readonly cacheFrames: number;
	readonly feedbackMode: WebGPUPagedShadowFeedbackMode;
}

/** @internal One main-view light selected by the WebGPU experiment. */
export interface WebGPUPagedShadowFrameState {
	readonly prepared: PreparedShadowLight;
	readonly settings: Readonly<WebGPUPagedShadowSettings>;
}

/** @internal Disabled default used by the WebGPU composition root. */
export const DEFAULT_WEBGPU_PAGED_SHADOW_EXPERIMENT:
	Readonly<WebGPUPagedShadowExperimentConfig> = Object.freeze({
		enabled: false,
		pageSize: 128,
		pageGridSize: 128,
		physicalPageCount: 2048,
		maxPagesPerFrame: 256,
		cacheFrames: 120,
		feedbackMode: "conservative",
	});

/**
 * Selects the main-view directional light owned by the private experiment.
 *
 * @internal `WebGPUShadowRuntime` owns one instance for its device lifetime.
 */
export class WebGPUPagedShadowExperiment {
	private readonly _enabled: boolean;
	private readonly _settings: Readonly<WebGPUPagedShadowSettings>;

	public constructor(
		config: Readonly<WebGPUPagedShadowExperimentConfig> =
			DEFAULT_WEBGPU_PAGED_SHADOW_EXPERIMENT,
	) {
		this._enabled = config.enabled;
		this._settings = Object.freeze({
			pageSize: config.pageSize,
			pageGridSize: config.pageGridSize,
			physicalPageCount: config.physicalPageCount,
			maxPagesPerFrame: config.maxPagesPerFrame,
			cacheFrames: config.cacheFrames,
			feedbackMode: config.feedbackMode,
		});
	}

	public resolve(context: FrameContext): WebGPUPagedShadowFrameState | null {
		if (
			!this._enabled ||
			!context.features.enableShadows ||
			!context.shadowPlan.hasRasterWork
		) {
			return null;
		}

		const prepared = context.shadowPlan.lights.find((candidate) =>
			candidate.light.type === LightType.Directional &&
			candidate.effectiveFilterMode === "pcf" &&
			candidate.slices.length >= 1 &&
			candidate.slices.length <= 4,
		);
		if (!prepared) return null;

		return Object.freeze({
			prepared,
			settings: this._settings,
		});
	}
}
