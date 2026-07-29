import type { FrameContext } from "../../pipeline/types";

import type { WebGLLightState } from "./WebGLLightCollector";

/** Mutable state whose lifetime is exactly one active WebGL frame. */
export class WebGLFrameSession {
	public context: FrameContext | null = null;
	public width = 1;
	public height = 1;
	public lightState: WebGLLightState | null = null;
	public presented = false;

	public begin(context: FrameContext): void {
		this.context = context;
		this.width = Math.max(1, Math.floor(context.attachments.width));
		this.height = Math.max(1, Math.floor(context.attachments.height));
		this.presented = false;
	}

	public finish(): void {
		this.context = null;
		this.lightState = null;
	}

	public abort(): void {
		this.finish();
		this.presented = false;
	}
}
