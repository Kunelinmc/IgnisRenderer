import type { TextureFormat } from "../../core/TextureFormat";
import {
	TextureUsage,
	type IRenderTexture,
	type TextureDesc,
} from "../types";
import type { WebGPUResourceManager, WebGPUTexture } from "./WebGPUResourceManager";

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
	private _currentFacade: IRenderTexture | null = null;
	private _depthTexture: IRenderTexture | null = null;
	private _currentCanvasTexture: GPUTexture | null = null;
	private _currentCanvasView: GPUTextureView | null = null;
	private _canvasFormat: TextureFormat | null = null;

	public constructor(private readonly _resources: WebGPUResourceManager) {}

	public get depthTexture(): IRenderTexture | null {
		return this._depthTexture;
	}

	public configureContext(
		context: GPUCanvasContext,
		configuration: GPUCanvasConfiguration,
		format: TextureFormat,
	): void {
		this.resetCurrentCanvasTargets();
		this._canvasFormat = format;
		context.configure(configuration);
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
		this._currentFacade?.destroy();
		this._currentFacade = null;
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
		if (this._currentFacade && this._resources.tryGetTexture(this._currentFacade)) {
			return this._currentFacade;
		}
		const current = this._getCurrentCanvasTexture(context);
		const size = resolveGPUTextureExtent(current.texture, canvas.width, canvas.height);
		const texture = this._resources.registerTexture(current, {
			width: size.width,
			height: size.height,
			requestedFormat: this._canvasFormat ?? undefined,
			format: this._canvasFormat ?? undefined,
		});
		this._currentFacade = texture;
		return texture;
	}

	public getCurrentColorView(context: GPUCanvasContext): GPUTextureView {
		return this._getCurrentCanvasTexture(context).view;
	}

	public getCurrentDepthView(): GPUTextureView {
		if (!this._depthTexture) {
			throw new Error("WebGPU depth texture is not initialized.");
		}
		return this._resources.getTexture(this._depthTexture).view;
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
		return { texture: this._currentCanvasTexture, view: this._currentCanvasView };
	}
}
