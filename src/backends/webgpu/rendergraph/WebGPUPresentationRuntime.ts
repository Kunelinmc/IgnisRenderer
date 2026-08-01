import type { PostProcessColorDomain } from "../../../postprocess/PostProcessPass";
import type { FrameContext } from "../../../pipeline/types";
import type { IRenderTexture } from "../../types";

import type { WebGPUFrameHost } from "./WebGPUFrameHost";
import type { WebGPUFrameGraphRecordingContext } from "./WebGPUFrameGraphRecordingContext";
import type { WebGPUFrameNodeRuntime } from "./WebGPUFrameNodeRuntimes";
import type { WebGPUFrameSession } from "./WebGPUFrameSession";
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
export class WebGPUPresentationRuntime implements WebGPUFrameNodeRuntime {
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

	public onShaderRuntimeChanged(): void {
		this._pass.onShaderRuntimeChanged();
	}

	public destroy(): void {
		this._pass.destroy();
	}
}
