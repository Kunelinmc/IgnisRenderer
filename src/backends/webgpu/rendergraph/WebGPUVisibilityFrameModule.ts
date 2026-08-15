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
import type {
	WebGPURecordingFrameSession as WebGPUFrameSession,
} from "./WebGPUFrameSession";

export interface WebGPUVisibilityFeatureAnalysis {
	readonly needsOcclusionTargets: boolean;
}

/** @internal Read-only Hi-Z readiness consumed by post-processing. */
export interface WebGPUHiZReadinessPort {
	isHiZReady(): boolean;
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
	private _hiZStatus: "unavailable" | "pending" | "ready" | "failed" =
		"unavailable";
	private _hiZBuildCount = 0;
	public constructor(
		private readonly _builder: WebGPUHiZBuilder,
		private readonly _runtime: WebGPUOcclusionCullingRuntime,
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
		this._hiZStatus = "pending";
		this._runtime.beginFrame(context);
	}

	public isHiZReady(): boolean {
		return this._hiZStatus === "ready";
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
		const targets = session.targets.frameTargets;
		const encoder = session.commands.encoder;
		if (!encoder || !targets?.gMotionDepth || !targets.hiZ) {
			this._hiZStatus = "unavailable";
			return;
		}
		try {
			await this._builder.build({
				encoder,
				depth: targets.gMotionDepth,
				hiZ: targets.hiZ,
			});
			this._hiZStatus = "ready";
			this._hiZBuildCount++;
		} catch (error) {
			this._hiZStatus = "failed";
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
		const hiZ = session.targets.frameTargets.hiZ;
		const encoder = session.commands.encoder;
		if (!encoder || this._hiZStatus !== "ready" || !hiZ) return;
		await this._runtime.recordVisibilityPass({
			context: session.context,
			encoder,
			hiZ,
			options: normalizeOcclusionCullingOptions(
				session.context.features.occlusionCullingOptions,
			),
		});
	}
}
