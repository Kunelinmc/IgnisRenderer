import type { PostProcessColorDomain } from "../../../postprocess/PostProcessPass";
import type { FrameContext } from "../../../pipeline/types";
import type { IRenderTexture } from "../../types";

import type { WebGPUFrameHost } from "./WebGPUFrameHost";
import type { WebGPUFrameGraphRecordingContext } from "./WebGPUFrameGraphRecordingContext";
import type {
	WebGPUFrameGraphContribution,
	WebGPUFrameGraphModule,
	WebGPUFrameModulePlanningInput,
} from "./WebGPUFrameGraphModule";
import {
	createWebGPUFrameGraphNode,
	readWebGPUFrameGraphResource,
	writeWebGPUFrameGraphResource,
} from "./WebGPUFrameGraphPlanningUtils";
import type { WebGPUFrameSession } from "./WebGPUFrameSession";
import { WEBGPU_FRAME_GRAPH_RESOURCES } from "./WebGPUFrameGraphResourceCatalog";
import { WebGPUPresentPass } from "./WebGPUPresentPass";

export interface WebGPUPresentationRuntimeContext {
	readonly recording: Pick<
		WebGPUFrameGraphRecordingContext,
		"getFrameTargets" | "resolveDirtyRects"
	>;
	getOutputColorDomain(): PostProcessColorDomain;
}

/**
 * Owns WebGPU presentation node execution and presentation-pass resources.
 *
 * @internal Owned by `WebGPUFrameOrchestrator`; applications must use
 * `Renderer` display-output APIs.
 */
export class WebGPUPresentationRuntime implements WebGPUFrameGraphModule {
	public readonly id = "presentation";
	public readonly executors = {
		presentation: async (_node: unknown, session: WebGPUFrameSession) => {
			if (session.presented) return;
			const source = this._context.recording.getFrameTargets()?.sceneColor;
			if (source) await this.present(source, session);
		},
	};

	private readonly _pass: WebGPUPresentPass;
	public constructor(
		host: WebGPUFrameHost,
		private readonly _context: WebGPUPresentationRuntimeContext,
	) {
		this._pass = new WebGPUPresentPass(host);
	}

	public beginFrame(_context: FrameContext): void {}

	public planStage(
		input: WebGPUFrameModulePlanningInput,
	): readonly WebGPUFrameGraphContribution[] {
		if (input.finalization !== true || input.state.hasFrameTargets !== true) {
			return [];
		}
		const source =
			input.finalColorResource ?? WEBGPU_FRAME_GRAPH_RESOURCES.frameColor;
		return [{
			order: 100,
			nodes: [createWebGPUFrameGraphNode(
				input.pass,
				"presentation",
				"WebGPUPresentation",
				{
					reads: [readWebGPUFrameGraphResource(source, "texture-binding")],
					writes: [writeWebGPUFrameGraphResource(
						WEBGPU_FRAME_GRAPH_RESOURCES.canvasColor,
						"present",
					)],
				},
			)],
		}];
	}

	public warmup(): Promise<void> {
		return this._pass.warmup();
	}

	public async present(
		source: IRenderTexture,
		session: WebGPUFrameSession,
	): Promise<void> {
		if (!session.encoder || session.presented) return;
		await this._pass.present({
			encoder: session.encoder,
			frameContext: session.context,
			source,
			colorDomain: this._context.getOutputColorDomain(),
			resolveDirtyRects: (context, width, height) =>
				this._context.recording.resolveDirtyRects(context, width, height),
		});
		session.presented = true;
	}

	public invalidateFrameResources(): void {
		this._pass.invalidateBindings();
	}

	public onDisplayOutputChanged(): void {
		this._pass.onShaderRuntimeChanged();
	}

	public onShaderRuntimeChanged(): void {
		this._pass.onShaderRuntimeChanged();
	}

	public destroy(): void {
		this._pass.destroy();
	}
}
