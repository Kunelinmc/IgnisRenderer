import type { PreparedFramePacketSet } from "../../../pipeline/FramePacketContributorRegistry";

import type {
	WebGPUDeferredFeatureAnalysis,
	WebGPUFrameFeatureAnalysis,
	WebGPUPostProcessFeatureAnalysis,
	WebGPUReflectionFeatureAnalysis,
	WebGPUTransparencyAnalysis,
	WebGPUVisibilityFeatureAnalysis,
} from "./WebGPUFrameFeatureAnalyzer";

/** @internal Applies one feature-owned result to frame configuration input. */
export type WebGPUFrameModuleConfigurationContribution = (
	builder: WebGPUFrameConfigurationBuilder,
) => void;

/** @internal Aggregates feature-owned analysis without rescanning frame work. */
export class WebGPUFrameConfigurationBuilder {
	private _deferred: WebGPUDeferredFeatureAnalysis | null = null;
	private _transparency: {
		readonly analysis: WebGPUTransparencyAnalysis;
		readonly oitRequested: boolean;
	} | null = null;
	private _reflection: WebGPUReflectionFeatureAnalysis | null = null;
	private _visibility: WebGPUVisibilityFeatureAnalysis | null = null;
	private _postProcess: WebGPUPostProcessFeatureAnalysis | null = null;

	constructor(private readonly _framePackets: PreparedFramePacketSet) {}

	public setDeferred(analysis: WebGPUDeferredFeatureAnalysis): void {
		this._deferred = this._setOnce("deferred", this._deferred, analysis);
	}

	public setTransparency(analysis: WebGPUTransparencyAnalysis, oitRequested: boolean): void {
		this._transparency = this._setOnce("transparency", this._transparency, {
			analysis,
			oitRequested,
		});
	}

	public setReflection(analysis: WebGPUReflectionFeatureAnalysis): void {
		this._reflection = this._setOnce("reflection", this._reflection, analysis);
	}

	public setVisibility(analysis: WebGPUVisibilityFeatureAnalysis): void {
		this._visibility = this._setOnce("visibility", this._visibility, analysis);
	}

	public setPostProcess(analysis: WebGPUPostProcessFeatureAnalysis): void {
		this._postProcess = this._setOnce("post-process", this._postProcess, analysis);
	}

	public build(): WebGPUFrameFeatureAnalysis {
		const deferred = this._require("deferred", this._deferred);
		const transparency = this._require("transparency", this._transparency);
		const reflection = this._require("reflection", this._reflection);
		const visibility = this._require("visibility", this._visibility);
		const postProcess = this._require("post-process", this._postProcess);
		return {
			framePackets: this._framePackets,
			postProcessPasses: postProcess.postProcessPasses,
			hasDeferredLightingWork: deferred.hasDeferredLightingWork,
			deferredGBufferLayout: deferred.deferredGBufferLayout,
			oitRequested: transparency.oitRequested,
			hasOITWork: transparency.analysis.hasOITContributors,
			transparency: transparency.analysis,
			needsPostProcessTargets: postProcess.needsPostProcessTargets,
			needsPostProcessGBuffer: postProcess.needsPostProcessGBuffer,
			needsPlanarReflection: reflection.needsPlanarReflection,
			needsPlanarReflectionMask:
				reflection.needsPlanarReflection || postProcess.needsPlanarReflectionMask,
			needsTransmissionTargets:
				postProcess.needsTransmissionTargets &&
				transparency.analysis.transmissionPackets.length > 0,
			needsOcclusionTargets: visibility.needsOcclusionTargets,
			needsHiZTarget: visibility.needsOcclusionTargets || postProcess.needsHiZTarget,
		};
	}

	private _setOnce<TValue>(name: string, current: TValue | null, value: TValue): TValue {
		if (current !== null) {
			throw new Error(
				`WebGPU frame configuration received duplicate "${name}" contributions.`,
			);
		}
		return value;
	}

	private _require<TValue>(name: string, value: TValue | null): TValue {
		if (value === null) {
			throw new Error(`WebGPU frame configuration requires a "${name}" contribution.`);
		}
		return value;
	}
}
