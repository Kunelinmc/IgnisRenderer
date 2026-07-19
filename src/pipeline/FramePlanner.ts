import {
	type FramePass,
	type FrameContext,
	type PreparedScene,
	type RendererFramePlan,
	type ResolvedFeatureState,
	type TransientStore,
	createTransientStore,
} from "./types";
import { type ResolvedPostProcessState } from "../postprocess";
import type { IncrementalFrameContext } from "./incremental";
import type { RendererStageDefinition } from "./RendererStageGraph";
import { RenderPipelineRegistry } from "./RenderPipelineRegistry";
import { createDefaultPipelineStages } from "./defaultPipeline";
import type {
	BackendCapabilities,
	RenderBackendType,
} from "../backends/IRenderBackend";

export interface FramePlannerBuildOptions {
	registry?: RenderPipelineRegistry;
	stageOrder?: readonly RendererStageDefinition[];
	transient?: TransientStore;
	backendType?: RenderBackendType;
	backendCapabilities?: BackendCapabilities;
	incremental?: IncrementalFrameContext;
	frameContext?: FrameContext;
	incrementalStartStageIndex?: number;
}

export class FramePlanner {
	/**
	 * Builds the backend pass list for a frame using the default registry.
	 *
	 * @param frame Prepared scene snapshot.
	 * @param features Resolved renderer feature flags.
	 * @param postProcess Resolved post-process snapshot.
	 * @returns Renderer-owned backend pass entries in execution order.
	 * @sideEffects None.
	 */
	public static build(
		frame: PreparedScene,
		features: ResolvedFeatureState,
		postProcess: ResolvedPostProcessState
	): FramePass[] {
		return Array.from(
			this.buildFramePlan(frame, features, postProcess).backendPasses
		);
	}

	/**
	 * Builds the full renderer-owned frame plan.
	 *
	 * @param frame Prepared scene snapshot.
	 * @param features Resolved renderer feature flags.
	 * @param postProcess Resolved post-process snapshot.
	 * @param options Optional registry, stage order, transient, and incremental
	 * context overrides.
	 * @returns Stage order plus backend/shared pass entries for this frame.
	 * @sideEffects None.
	 */
	public static buildFramePlan(
		frame: PreparedScene,
		features: ResolvedFeatureState,
		postProcess: ResolvedPostProcessState,
		options: FramePlannerBuildOptions = {}
	): RendererFramePlan {
		const registry =
			options.registry ??
			new RenderPipelineRegistry({
				stages: createDefaultPipelineStages(),
			});
		const stageOrder =
			options.stageOrder ??
			registry.getExecutionOrder(
				{
					hasActiveAnimations: frame.hasActiveAnimations,
					hasParticleSystems: (frame.particleSystems?.length ?? 0) > 0,
				},
				() => {}
			);
		return registry.createFramePlan({
			stageOrder,
			frame,
			features,
			postProcess,
			transient: options.transient ?? createTransientStore(),
			backendType: options.backendType,
			backendCapabilities: options.backendCapabilities,
			incremental: options.incremental,
			frameContext: options.frameContext,
			incrementalStartStageIndex: options.incrementalStartStageIndex,
		});
	}
}
