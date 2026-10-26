import type { WarmupPhaseCounters } from "../../pipeline/WarmupPlanner";

/** @internal Feature-owned warmup contract aggregated by the WebGPU runtime composition. */
export interface WebGPUFeatureWarmupContributor<TRequest> {
	warmup(request: TRequest): Promise<WarmupPhaseCounters>;
}
