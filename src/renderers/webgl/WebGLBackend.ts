import type { FrameContext, FramePass } from "../../pipeline/types"
import type {
	IRenderBackend,
	RendererBackendBridge,
} from "../IRenderBackend"

const WEBGL_STUB_ERROR_MESSAGE =
	"WebGLBackend is a stub and is not implemented yet"

export class WebGLBackend implements IRenderBackend {
	public readonly type = "webgl";
	public readonly frameScheduling = "on-demand";
	public readonly passExecutors = {
		"animation-sim": "shared",
		"particle-sim": "backend",
	} as const;
	public readonly capabilities = {
		sh: false,
		shadows: false,
		reflection: false,
		skybox: false,
		ssao: false,
		taa: false,
		ssr: false,
		volumetric: false,
	};

	public setRenderer(_renderer: RendererBackendBridge): void {}

	public async init(_canvas: HTMLCanvasElement): Promise<void> {
		throw new Error(WEBGL_STUB_ERROR_MESSAGE);
	}

	public resize(_width: number, _height: number): void {}

	public getAttachments(
		width: number,
		height: number
	): {
		width: number;
		height: number;
	} {
		return { width, height };
	}

	public beginFrame(_context: FrameContext): void {
		throw new Error(WEBGL_STUB_ERROR_MESSAGE);
	}

	public executePass(_pass: FramePass, _context: FrameContext): void {
		throw new Error(WEBGL_STUB_ERROR_MESSAGE);
	}

	public endFrame(): void {
		throw new Error(WEBGL_STUB_ERROR_MESSAGE);
	}
}
