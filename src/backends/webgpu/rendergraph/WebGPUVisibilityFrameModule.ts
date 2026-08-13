import { Logger } from "../../../foundation/Logger";
import {
	normalizeOcclusionCullingOptions,
	type NormalizedOcclusionCullingOptions,
	type OcclusionVisibilityProvider,
} from "../../../pipeline/OcclusionCulling";
import type { FrameContext } from "../../../pipeline/types";
import type { WebGPUHiZBuilder } from "../WebGPUHiZBuilder";
import type { WebGPUOcclusionCullingRuntime } from "../WebGPUOcclusionCullingRuntime";

import type {
	WebGPUFrameGraphModule,
	WebGPUFrameGraphContribution,
	WebGPUFrameModulePlanningInput,
} from "./WebGPUFrameGraphModule";
import {
	defineWebGPUFrameMessage,
	type WebGPUFrameMessageHandler,
} from "./WebGPUFrameMessage";
import {
	WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE,
	WEBGPU_FRAME_LOGICAL_RESOURCES,
	WEBGPU_FRAME_CONTEXT_MESSAGE,
} from "./WebGPUFrameMessages";
import {
	createWebGPUFrameGraphNode,
	readWebGPUFrameGraphResource,
	writeWebGPUFrameGraphResource,
} from "./WebGPUFrameGraphDsl";
import type { WebGPUFrameGraphRecordingContext } from "./WebGPUFrameGraphRecordingContext";
import type { WebGPUFrameSession } from "./WebGPUFrameSession";

export interface WebGPUVisibilityFeatureAnalysis {
	readonly needsOcclusionTargets: boolean;
}

export const WEBGPU_VISIBILITY_FEATURE_ANALYSIS =
	defineWebGPUFrameMessage<WebGPUVisibilityFeatureAnalysis>({
		id: "webgpu:visibility-analysis",
		ownerId: "visibility",
		phase: "analysis",
	});

export function analyzeWebGPUVisibilityFeatures(
	context: FrameContext,
): WebGPUVisibilityFeatureAnalysis {
	return {
		needsOcclusionTargets:
			context.features.enableOcclusionCulling === true &&
			(context.scene.occlusion?.eligibleCandidateCount ?? 0) > 0,
	};
}

/** @internal Owns shared Hi-Z and occlusion graph execution. */
export class WebGPUVisibilityFrameModule implements WebGPUFrameGraphModule {
	public readonly id = "visibility";
	public readonly messageHandlers: readonly WebGPUFrameMessageHandler[] = [{
		id: "analyze",
		moduleId: this.id,
		phase: "analysis",
		inputs: [{ descriptor: WEBGPU_FRAME_CONTEXT_MESSAGE }],
		outputs: [WEBGPU_VISIBILITY_FEATURE_ANALYSIS],
		run: (messages, publisher) => publisher.publish(
			WEBGPU_VISIBILITY_FEATURE_ANALYSIS,
			analyzeWebGPUVisibilityFeatures(messages.get(WEBGPU_FRAME_CONTEXT_MESSAGE)),
		),
	}, {
		id: "configure",
		moduleId: this.id,
		phase: "configuration",
		inputs: [{ descriptor: WEBGPU_VISIBILITY_FEATURE_ANALYSIS }],
		outputs: [WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE],
		run: (messages, publisher) => {
			const analysis = messages.get(WEBGPU_VISIBILITY_FEATURE_ANALYSIS);
			publisher.publish(WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE, {
				source: this.id,
				targetClass: analysis.needsOcclusionTargets ? "mrt" : "single",
				resources: analysis.needsOcclusionTargets
					? [{ id: WEBGPU_FRAME_LOGICAL_RESOURCES.hiZTarget }]
					: [],
				needsHiZBuild: analysis.needsOcclusionTargets,
				needsOcclusionTest: analysis.needsOcclusionTargets,
			});
		},
	}];
	public readonly executors = {
		"hiz-build": async (_node: unknown, session: WebGPUFrameSession) => {
			await this._buildHiZ(session);
		},
		"occlusion-test": async (_node: unknown, session: WebGPUFrameSession) => {
			await this._recordOcclusion(session);
		},
	};
	public constructor(
		private readonly _builder: WebGPUHiZBuilder,
		private readonly _runtime: WebGPUOcclusionCullingRuntime,
		private readonly _recording: WebGPUFrameGraphRecordingContext,
	) {}

	public planStage(
		input: WebGPUFrameModulePlanningInput,
	): readonly WebGPUFrameGraphContribution[] {
		if (
			input.pass.stage !== "main-opaque" ||
			!input.state.needsHiZBuild ||
			input.state.sceneTargetMode === "single"
		) return [];
		const nodes = [createWebGPUFrameGraphNode(
			input.pass,
			"hiz-build",
			"WebGPUHiZBuild",
			{
				reads: [readWebGPUFrameGraphResource(
					"gbuffer:motion-depth",
					"texture-binding",
				)],
				writes: [writeWebGPUFrameGraphResource(
					"frame:hiz",
					"storage-binding",
				)],
			},
		)];
		if (input.state.needsOcclusionTest) {
			nodes.push(createWebGPUFrameGraphNode(
				input.pass,
				"occlusion-test",
				"WebGPUOcclusionTest",
				{
					reads: [readWebGPUFrameGraphResource(
						"frame:hiz",
						"texture-binding",
					)],
					writes: [writeWebGPUFrameGraphResource(
						"occlusion:results",
						"storage-binding",
					)],
				},
			));
		}
		return nodes.length > 0 ? [{ lane: "visibility", before: ["shadow"], nodes }] : [];
	}

	public beginFrame(context: FrameContext): void {
		this._runtime.beginFrame(context);
	}

	public getVisibilityProvider(
		options?: NormalizedOcclusionCullingOptions,
	): OcclusionVisibilityProvider {
		return this._runtime.getVisibilityProvider(options);
	}

	public reset(): void {
		this._runtime.resetVisibility();
	}

	public afterSubmit(): void {
		this._runtime.scheduleQueuedReadbacks();
	}

	public onShaderRuntimeChanged(): void {
		this._builder.invalidateShaderResources();
		this._runtime.onShaderRuntimeChanged();
	}

	public destroy(): void {
		this._runtime.destroy();
		this._builder.destroy();
	}

	private async _buildHiZ(session: WebGPUFrameSession): Promise<void> {
		const targets = this._recording.getFrameTargets();
		if (!session.encoder || !targets?.gMotionDepth || !targets.hiZ) {
			session.hiZStatus = "unavailable";
			return;
		}
		try {
			await this._builder.build({
				encoder: session.encoder,
				depth: targets.gMotionDepth,
				hiZ: targets.hiZ,
			});
			session.hiZStatus = "ready";
			session.hiZBuildCount++;
		} catch (error) {
			session.hiZStatus = "failed";
			Logger.warn(
				"[webgpu-hiz-build-failed] Shared WebGPU Hi-Z build failed; " +
					`dependent effects will be skipped. ${String(error)}`,
				{
					scope: "WebGPUFrameOrchestrator",
					onceKey: "webgpu-hiz-build-failed",
				},
			);
		}
	}

	private async _recordOcclusion(session: WebGPUFrameSession): Promise<void> {
		const hiZ = this._recording.getFrameTargets()?.hiZ;
		if (!session.encoder || session.hiZStatus !== "ready" || !hiZ) return;
		await this._runtime.recordVisibilityPass({
			context: session.context,
			encoder: session.encoder,
			hiZ,
			options: normalizeOcclusionCullingOptions(
				session.context.features.occlusionCullingOptions,
			),
		});
	}
}
