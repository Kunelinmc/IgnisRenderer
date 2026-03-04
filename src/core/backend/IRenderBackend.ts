import type {
	FrameAttachments,
	FrameContext,
	FramePass,
} from "../pipeline/types";

export interface BackendCapabilities {
	sh: boolean;
	shadows: boolean;
	reflection: boolean;
	skybox: boolean;
	ssao: boolean;
	volumetric: boolean;
}

export interface IRenderBackend {
	readonly type: "software" | "webgpu";
	readonly capabilities: BackendCapabilities;
	init(canvas: HTMLCanvasElement): Promise<void>;
	resize(width: number, height: number): void;
	getAttachments(width: number, height: number): FrameAttachments;
	beginFrame(context: FrameContext): void | Promise<void>;
	executePass(pass: FramePass, context: FrameContext): void | Promise<void>;
	endFrame(): void | Promise<void>;
}
