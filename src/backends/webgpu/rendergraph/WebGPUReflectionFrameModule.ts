import type { WebGPUPlanarReflectionMSAATargets } from "../WebGPUPlanarReflectionPass";
import type { WebGPUPlanarReflectionPass } from "../WebGPUPlanarReflectionPass";

import type { WebGPUFrameHost } from "./WebGPUFrameHost";
import type { FrameContext } from "../../../pipeline/types";
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
import { WEBGPU_FRAME_GRAPH_RESOURCES } from "./WebGPUFrameGraphResourceCatalog";

export interface WebGPUReflectionFeatureAnalysis {
	readonly needsPlanarReflection: boolean;
}

export const WEBGPU_REFLECTION_FEATURE_ANALYSIS =
	defineWebGPUFrameMessage<WebGPUReflectionFeatureAnalysis>({
		id: "webgpu:reflection-analysis",
		ownerId: "reflection",
		phase: "analysis",
	});

export function analyzeWebGPUReflectionFeatures(
	context: FrameContext,
): WebGPUReflectionFeatureAnalysis {
	return {
		needsPlanarReflection:
			context.features.enableReflection && context.scene.reflectivePackets.length > 0,
	};
}

/** @internal Owns planar-reflection graph execution and pass resources. */
export class WebGPUReflectionFrameModule implements WebGPUFrameGraphModule {
	public readonly id = "reflection";
	public readonly planningInputs = [{
		descriptor: WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE,
		required: false,
	}] as const;
	public readonly messageHandlers: readonly WebGPUFrameMessageHandler[] = [{
		id: "analyze",
		moduleId: this.id,
		phase: "analysis",
		inputs: [{ descriptor: WEBGPU_FRAME_CONTEXT_MESSAGE }],
		outputs: [WEBGPU_REFLECTION_FEATURE_ANALYSIS],
		run: (messages, publisher) => publisher.publish(
			WEBGPU_REFLECTION_FEATURE_ANALYSIS,
			analyzeWebGPUReflectionFeatures(messages.get(WEBGPU_FRAME_CONTEXT_MESSAGE)),
		),
	}, {
		id: "configure",
		moduleId: this.id,
		phase: "configuration",
		inputs: [{ descriptor: WEBGPU_REFLECTION_FEATURE_ANALYSIS }],
		outputs: [WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE],
		run: (messages, publisher) => {
			const analysis = messages.get(WEBGPU_REFLECTION_FEATURE_ANALYSIS);
			publisher.publish(WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE, {
				source: this.id,
				targetClass: analysis.needsPlanarReflection ? "color" : "single",
				resources: analysis.needsPlanarReflection
					? [{ id: WEBGPU_FRAME_LOGICAL_RESOURCES.planarReflectionMask }]
					: [],
				needsPlanarReflectionComposite: analysis.needsPlanarReflection,
			});
		},
	}];
	public readonly executors = {
		"planar-reflection-capture": async (_node: unknown, session: WebGPUFrameSession) => {
			await this._capture(session);
		},
		"planar-reflection-composite": async (_node: unknown, session: WebGPUFrameSession) => {
			await this.composite(session);
		},
	};
	public constructor(
		private readonly _host: WebGPUFrameHost,
		private readonly _pass: WebGPUPlanarReflectionPass,
		private readonly _recording: WebGPUFrameGraphRecordingContext,
	) {}

	public planStage(
		input: WebGPUFrameModulePlanningInput,
	): readonly WebGPUFrameGraphContribution[] {
		if (input.pass.stage === "reflection") {
			return [{
				lane: "geometry",
				nodes: [createWebGPUFrameGraphNode(
					input.pass,
					"planar-reflection-capture",
					"WebGPUPlanarReflectionCapture",
					{
						reads: [readWebGPUFrameGraphResource(
							"shadow-atlas",
							"texture-binding",
							true,
						), readWebGPUFrameGraphResource(
							"shadow-transmittance-atlas",
							"texture-binding",
							true,
						)],
						writes: [writeWebGPUFrameGraphResource(
							"planar-reflection:capture",
							"render-attachment",
						)],
					},
				)],
			}];
		}
		const demand = input.messages
			.getAll(WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE)
			.find((candidate) => candidate.source === this.id);
		if (
			input.pass.stage !== "main-opaque" ||
			demand?.needsPlanarReflectionComposite !== true
		) return [];
		const r = WEBGPU_FRAME_GRAPH_RESOURCES;
		return [{
			lane: "composite",
			nodes: [createWebGPUFrameGraphNode(
				input.pass,
				"planar-reflection-composite",
				"WebGPUPlanarReflectionComposite",
				{
					reads: [
						readWebGPUFrameGraphResource(
							r.planarReflectionCapture,
							"texture-binding",
						),
						readWebGPUFrameGraphResource(
							r.frameColor,
							"render-attachment",
							true,
						),
					],
					writes: [
						writeWebGPUFrameGraphResource(
							r.frameColor,
							"render-attachment",
							true,
						),
						writeWebGPUFrameGraphResource(
							r.planarReflectionMask,
							"render-attachment",
							true,
						),
					],
				},
			)],
		}];
	}

	public async composite(session: WebGPUFrameSession): Promise<void> {
		const encoder = session.encoder;
		const targets = this._recording.getFrameTargets();
		if (!encoder || session.configuration?.mrtSupported !== true || !targets) return;
		this._clearMask(session);
		if (!session.resources) {
			throw new Error("WebGPUFrameOrchestrator requires prepared main-frame resources.");
		}
		await this._pass.composite({
			encoder,
			context: session.context,
			frameResources: session.resources,
			frameTargets: targets,
			msaaTargets:
				this._recording.getMSAATargets() as WebGPUPlanarReflectionMSAATargets | null,
			sampleCount: session.configuration.samplePlan.sampleCount,
		});
	}

	public invalidateFrameResources(): void {
		this._pass.destroy();
	}

	public onShaderRuntimeChanged(): void {
		this._pass.destroy();
	}

	public destroy(): void {
		this._pass.destroy();
	}

	private async _capture(session: WebGPUFrameSession): Promise<void> {
		if (!session.encoder || !session.committer) return;
		session.committer.enqueueEncoder("main:before-reflection", session.encoder);
		session.encoder = null;
		await this._pass.capture(session.context, (label, encoder) => {
			session.committer!.enqueueEncoder(label, encoder);
		});
		session.encoder = this._host.createCommandEncoder();
	}

	private _clearMask(session: WebGPUFrameSession): void {
		const mask = this._recording.getFrameTargets()?.planarReflectionMask;
		if (!session.encoder || !mask) return;
		session.encoder.beginRenderPass({
			label: "WebGPUPlanarReflectionMaskClear",
			colorAttachments: [{
				view: mask,
				clearValue: { r: 0, g: 0, b: 0, a: 0 },
				loadOp: "clear",
				storeOp: "store",
			}],
		});
		session.encoder.endRenderPass();
	}
}
