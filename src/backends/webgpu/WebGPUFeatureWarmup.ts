import type { WarmupPhaseCounters } from "../../pipeline/WarmupPlanner";

/** @internal Feature-owned warmup contract aggregated by the WebGPU coordinator. */
export interface WebGPUFeatureWarmupContributor<TRequest> {
	warmup(request: TRequest): Promise<WarmupPhaseCounters>;
}
