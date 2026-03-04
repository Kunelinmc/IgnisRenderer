import type {
	FramePass,
	PreparedScene,
	ResolvedFeatureState,
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
	beginFrame(
		frame: PreparedScene,
		features: ResolvedFeatureState
	): void | Promise<void>;
	executePass(pass: FramePass, frame: PreparedScene): void | Promise<void>;
	endFrame(): void | Promise<void>;
}
