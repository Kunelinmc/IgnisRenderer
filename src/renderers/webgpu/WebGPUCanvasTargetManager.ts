import {
	TextureUsage,
	type IRenderTexture,
	type TextureDesc,
	type TextureFormat,
} from "../types";
import {
	attachWebGPUTexture,
	createWebGPUTexture,
	getWebGPUTexture,
	type WebGPUTexture,
} from "./WebGPUResourceAccess";

interface InternalTexture extends IRenderTexture {
	_gpuResource: GPUTexture;
	_gpuTexture: GPUTexture;
	_gpuView: GPUTextureView;
	_webgpuTexture: WebGPUTexture;
}

function resolveGPUTextureExtent(
	texture: GPUTexture,
	fallbackWidth: number,
	fallbackHeight: number
): { width: number; height: number } {
	const textureWithExtent = texture as GPUTexture & {
		width?: unknown;
		height?: unknown;
	};
	const width =
		typeof textureWithExtent.width === "number" &&
		Number.isFinite(textureWithExtent.width)
			? textureWithExtent.width
			: fallbackWidth;
	const height =
		typeof textureWithExtent.height === "number" &&
		Number.isFinite(textureWithExtent.height)
			? textureWithExtent.height
			: fallbackHeight;
	return {
		width: Math.max(1, Math.floor(width)),
		height: Math.max(1, Math.floor(height)),
	};
}

export class WebGPUCanvasTargetManager {
	private _depthTexture: IRenderTexture | null = null;
	private _currentCanvasTexture: GPUTexture | null = null;
	private _currentCanvasView: GPUTextureView | null = null;
	private _canvasFormat: TextureFormat | null = null;

	public get depthTexture(): IRenderTexture | null {
		return this._depthTexture;
	}

	public configureContext(
		context: GPUCanvasContext,
		device: GPUDevice,
		format: GPUTextureFormat
	): void {
		this.resetCurrentCanvasTargets();
		this._canvasFormat = format as TextureFormat;
		context.configure({
			device,
			format,
			alphaMode: "premultiplied",
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
		});
	}

	public recreateDepthTexture(
		canvas: HTMLCanvasElement,
		depthFormat: TextureFormat,
		createTexture: (desc: TextureDesc) => IRenderTexture
	): void {
		if (canvas.width <= 0 || canvas.height <= 0) {
			if (this._depthTexture) {
				this._depthTexture.destroy();
				this._depthTexture = null;
			}
			return;
		}
		this._depthTexture?.destroy();
		this._depthTexture = createTexture({
			width: canvas.width,
			height: canvas.height,
			format: depthFormat,
			usage: TextureUsage.RenderAttachment,
			label: "WebGPUCanvasDepth",
		});
	}

	public resetCurrentCanvasTargets(): void {
		this._currentCanvasTexture = null;
		this._currentCanvasView = null;
	}

	public release(): void {
		this._depthTexture?.destroy();
		this._depthTexture = null;
		this.resetCurrentCanvasTargets();
	}

	public getCanvasColorTexture(
		context: GPUCanvasContext,
		canvas: HTMLCanvasElement
	): IRenderTexture {
		const current = this._getCurrentCanvasTexture(context);
		const size = resolveGPUTextureExtent(current.texture, canvas.width, canvas.height);
		const texture: InternalTexture = {
			width: size.width,
			height: size.height,
			requestedFormat: this._canvasFormat ?? undefined,
			format: this._canvasFormat ?? undefined,
			destroy: () => {},
			_gpuResource: current.texture,
			_gpuTexture: current.texture,
			_gpuView: current.view,
			_webgpuTexture: current,
		};
		attachWebGPUTexture(texture, current);
		return texture;
	}

	public getCurrentColorView(context: GPUCanvasContext): GPUTextureView {
		return this._getCurrentCanvasTexture(context).view;
	}

	public getCurrentDepthView(): GPUTextureView {
		if (!this._depthTexture) {
			throw new Error("WebGPU depth texture is not initialized.");
		}
		return getWebGPUTexture(this._depthTexture).view;
	}

	public getCanvasDepthTexture(): IRenderTexture {
		if (!this._depthTexture) {
			throw new Error("Depth texture not initialized (possibly zero dimension canvas)");
		}
		return this._depthTexture;
	}

	private _getCurrentCanvasTexture(context: GPUCanvasContext): WebGPUTexture {
		if (!this._currentCanvasTexture || !this._currentCanvasView) {
			this._currentCanvasTexture = context.getCurrentTexture();
			this._currentCanvasView = this._currentCanvasTexture.createView();
		}
		return createWebGPUTexture(this._currentCanvasTexture, this._currentCanvasView);
	}
}
