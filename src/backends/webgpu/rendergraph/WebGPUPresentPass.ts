import type { FrameContext } from "../../../pipeline/types";
import type { ICommandEncoder } from "../../ICommandEncoder";
import {
	AddressMode,
	BufferUsage,
	FilterMode,
	type IBindingGroup,
	type IRenderBuffer,
	type IRenderPipeline,
	type IRenderTexture,
	type ISampler,
	type IShaderModule,
} from "../../types";
import type { WebGPUFrameHost } from "./WebGPUFrameHost";
import { ShaderSource } from "../../../shaders/ShaderSource";
import type { PostProcessColorDomain } from "../../../postprocess/PostProcessPass";

export interface WebGPUPresentDirtyRect {
	readonly x: number;
	readonly y: number;
	readonly width: number;
	readonly height: number;
}

export interface WebGPUPresentRequest {
	readonly encoder: ICommandEncoder;
	readonly frameContext: FrameContext | null;
	readonly source: IRenderTexture;
	readonly colorDomain: PostProcessColorDomain;
	readonly resolveDirtyRects: (
		context: FrameContext | null,
		targetWidth: number,
		targetHeight: number,
	) => readonly WebGPUPresentDirtyRect[];
}

/**
 * Records the WebGPU full-screen present pass and owns its GPU resources.
 */
export class WebGPUPresentPass {
	private readonly _host: WebGPUFrameHost;
	private _shaderModule: IShaderModule | null = null;
	private _pipeline: IRenderPipeline | null = null;
	private _sampler: ISampler | null = null;
	private _binding: IBindingGroup | null = null;
	private _bindingSource: IRenderTexture | null = null;
	private _params: IRenderBuffer | null = null;
	private readonly _paramData = new Float32Array(4);

	constructor(host: WebGPUFrameHost) {
		this._host = host;
	}

	public async warmup(): Promise<void> {
		await this._ensureResources();
	}

	public async present(request: WebGPUPresentRequest): Promise<void> {
		await this._ensureResources();
		if (!this._pipeline || !this._sampler || !this._params) {
			return;
		}
		const display = this._host.displayOutputState;
		this._paramData[0] = display?.requested.exposure ?? 1;
		this._paramData[1] = display?.requested.hdrHeadroom ?? 4;
		this._paramData[2] = display?.activeDynamicRange === "hdr" ? 1 : 0;
		this._paramData[3] = encodeColorDomain(request.colorDomain) +
			(request.frameContext?.presentationAlphaMode === "premultiplied" ? 4 : 0);
		this._host.writeBuffer(this._params, this._paramData);

		if (!this._binding || this._bindingSource !== request.source) {
			this._destroyBindingGroup(this._binding);
			this._binding = this._host.createBindingGroup({
				pipeline: this._pipeline,
				layoutIndex: 0,
				entries: [
					{ binding: 0, resource: request.source },
					{ binding: 1, resource: this._sampler },
					{ binding: 2, resource: this._params },
				],
				label: "WebGPUPresentBinding",
			});
			this._bindingSource = request.source;
		}

		const incrementalPartial =
			request.frameContext?.incremental?.enabled === true &&
			request.frameContext?.incremental?.forceFullFrame === false &&
			(request.frameContext?.incremental?.dirtyRects?.length ?? 0) > 0;
		const canvasTarget = this._host.getCanvasColorTexture();
		// WebGPU canvas swap-chain textures do not provide a cross-frame content
		// preservation guarantee. Composite the whole target instead of relying on
		// a `load` operation for non-dirty tiles.
		const dirtyRects = incrementalPartial
			? [{ x: 0, y: 0, width: canvasTarget.width, height: canvasTarget.height }]
			: request.resolveDirtyRects(
					request.frameContext,
					canvasTarget.width,
					canvasTarget.height,
				);
		request.encoder.beginRenderPass({
			label: "WebGPUPresentPass",
			colorAttachments: [
				{
					clearValue: {
						r: 0,
						g: 0,
						b: 0,
						a: request.frameContext?.presentationAlphaMode === "premultiplied" ? 0 : 1,
					},
					loadOp: "clear",
					storeOp: "store",
				},
			],
		});
		request.encoder.setPipeline(this._pipeline);
		request.encoder.setBindingGroup(0, this._binding);
		for (const rect of dirtyRects) {
			request.encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
			request.encoder.draw(3);
		}
		request.encoder.endRenderPass();
	}

	public invalidateBindings(): void {
		this._destroyBindingGroup(this._binding);
		this._binding = null;
		this._bindingSource = null;
	}

	public onShaderRuntimeChanged(): void {
		this._destroyManagedResource(this._shaderModule);
		this._destroyManagedResource(this._pipeline);
		this._destroyManagedResource(this._sampler);
		this._destroyManagedResource(this._params);
		this._shaderModule = null;
		this._pipeline = null;
		this._sampler = null;
		this._params = null;
		this.invalidateBindings();
	}

	public destroy(): void {
		this.onShaderRuntimeChanged();
	}

	private async _ensureResources(): Promise<void> {
		if (!this._shaderModule) {
			const composite = await ShaderSource.load("webgpu.utility.present");
			this._shaderModule = await this._host.createShaderModule({
				label: "WebGPUPresentShader",
				code: composite.source.code,
				sourceMap: composite.source.sourceMap,
				language: "wgsl",
				stage: "unknown",
				sourceKind: "builtin-present",
			});
		}

		if (!this._pipeline) {
			this._pipeline = await this._host.createPipeline({
				label: "WebGPUPresentPipeline",
				vertex: {
					module: this._shaderModule,
					entryPoint: "vsMain",
				},
				fragment: {
					module: this._shaderModule,
					entryPoint: "fsMain",
					targets: [{ format: this._host.canvasFormat }],
				},
				primitive: {
					topology: "triangle-list" as any,
					cullMode: "none",
					frontFace: "ccw",
				},
			} as any);
		}

		if (!this._sampler) {
			this._sampler = this._host.createSampler({
				label: "WebGPUPresentSampler",
				magFilter: FilterMode.Linear,
				minFilter: FilterMode.Linear,
				mipmapFilter: FilterMode.Linear,
				addressModeU: AddressMode.ClampToEdge,
				addressModeV: AddressMode.ClampToEdge,
			});
		}
		if (!this._params) {
			this._params = this._host.createBuffer({
				label: "WebGPUPresentParams",
				size: 16,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
	}

	private _destroyBindingGroup(group: IBindingGroup | null): void {
		const destroyFn = (group as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(group);
		}
	}

	private _destroyManagedResource(resource: unknown): void {
		const destroyFn = (resource as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(resource);
		}
	}
}

function encodeColorDomain(domain: PostProcessColorDomain): number {
	switch (domain) {
		case "scene-linear-hdr":
			return 0;
		case "display-linear":
			return 1;
		case "display-encoded":
			return 2;
	}
}
