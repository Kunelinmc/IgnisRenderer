import type { WebGPUPlanarReflectionMSAATargets } from "../WebGPUPlanarReflectionPass";
import type { WebGPUPlanarReflectionPass } from "../WebGPUPlanarReflectionPass";

import type { WebGPUFrameHost } from "./WebGPUFrameHost";
import type {
	WebGPUFrameGraphModule,
	WebGPUFrameGraphContribution,
	WebGPUFrameModuleAnalysisInput,
	WebGPUFrameModuleConfigurationInput,
	WebGPUFrameModulePlanningInput,
	WebGPUFrameModuleStateStore,
} from "./WebGPUFrameGraphModule";
import type {
	WebGPUFrameModuleConfigurationContribution,
} from "./WebGPUFrameConfigurationContribution";
import { WEBGPU_REFLECTION_FEATURE_ANALYSIS } from "./WebGPUFrameModuleStateKeys";
import { analyzeWebGPUReflectionFeatures } from "./WebGPUFrameFeatureAnalyzer";
import {
	createWebGPUFrameGraphNode,
	readWebGPUFrameGraphResource,
	writeWebGPUFrameGraphResource,
} from "./WebGPUFrameGraphPlanningUtils";
import type { WebGPUFrameGraphRecordingContext } from "./WebGPUFrameGraphRecordingContext";
import type { WebGPUFrameSession } from "./WebGPUFrameSession";
import { WEBGPU_FRAME_GRAPH_RESOURCES } from "./WebGPUFrameGraphResourceCatalog";

/** @internal Owns planar-reflection graph execution and pass resources. */
export class WebGPUReflectionFrameModule implements WebGPUFrameGraphModule {
	public readonly id = "reflection";
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

	public analyze(
		input: WebGPUFrameModuleAnalysisInput,
		state: WebGPUFrameModuleStateStore,
	): void {
		state.set(
			WEBGPU_REFLECTION_FEATURE_ANALYSIS,
			analyzeWebGPUReflectionFeatures(input.context),
		);
	}

	public contributeConfiguration(
		input: WebGPUFrameModuleConfigurationInput,
	): WebGPUFrameModuleConfigurationContribution {
		const analysis = input.state.require(WEBGPU_REFLECTION_FEATURE_ANALYSIS);
		return (builder) => builder.setReflection(analysis);
	}

	public planStage(
		input: WebGPUFrameModulePlanningInput,
	): readonly WebGPUFrameGraphContribution[] {
		if (input.pass.stage === "reflection") {
			return [{
				order: 100,
				nodes: [createWebGPUFrameGraphNode(
					input.pass,
					"planar-reflection-capture",
					"WebGPUPlanarReflectionCapture",
					{
						reads: [readWebGPUFrameGraphResource(
							"shadow-atlas",
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
		if (
			input.pass.stage !== "main-opaque" ||
			!input.state.needsPlanarReflectionComposite
		) return [];
		const r = WEBGPU_FRAME_GRAPH_RESOURCES;
		return [{
			order: 300,
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
