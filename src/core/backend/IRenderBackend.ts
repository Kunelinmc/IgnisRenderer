import type { Camera } from "../../cameras/Camera";
import type { SceneLight } from "../../lights";
import type {
	FrameAttachments,
	FrameContext,
	FramePass,
} from "../pipeline/types";

export type KnownBackendType = "software" | "webgpu";
export type RenderBackendType = KnownBackendType | (string & {});
export type FrameSchedulingMode = "always" | "on-demand";
export type PassExecutorMap = Partial<
	Record<FramePass["stage"], FramePass["executor"]>
>;

export interface BackendCapabilities {
	sh: boolean;
	shadows: boolean;
	reflection: boolean;
	skybox: boolean;
	ssao: boolean;
	taa: boolean;
	ssr: boolean;
	volumetric: boolean;
}

export interface RendererBackendBridge {
	readonly canvas: HTMLCanvasElement;
	readonly camera: Camera;
	readonly scene: { lights: SceneLight[] };
	readonly features: { enableShadows: boolean };
	warnOnce(key: string, message: string): void;
	pixels?: Uint8ClampedArray | null;
}

export interface IRenderBackend {
	readonly type: RenderBackendType;
	readonly capabilities: BackendCapabilities;
	readonly frameScheduling: FrameSchedulingMode;
	readonly passExecutors?: PassExecutorMap;
	setRenderer?(renderer: RendererBackendBridge): void;
	init(canvas: HTMLCanvasElement): Promise<void>;
	resize(width: number, height: number): void;
	getAttachments(width: number, height: number): FrameAttachments;
	beginFrame(context: FrameContext): void | Promise<void>;
	executeSharedPass?(
		pass: FramePass,
		context: FrameContext
	): void | Promise<void>;
	executePass(pass: FramePass, context: FrameContext): void | Promise<void>;
	endFrame(): void | Promise<void>;
}
