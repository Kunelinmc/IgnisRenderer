import type { FrameContext } from "../../../pipeline/types";
import { DEFAULT_GAMMA } from "../../constants";
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
import type { WebGPUBackend } from "../../WebGPUBackend";
import { ShaderSource } from "../../../shaders/ShaderSource";

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
	readonly applyGamma: boolean;
	readonly resolveDirtyRects: (
		context: FrameContext | null,
		targetWidth: number,
		targetHeight: number
	) => readonly WebGPUPresentDirtyRect[];
}

/**
 * Records the WebGPU full-screen present pass and owns its GPU resources.
 */
export class WebGPUPresentPass {
	private readonly _backend: WebGPUBackend;
	private _shaderModule: IShaderModule | null = null;
	private _pipeline: IRenderPipeline | null = null;
	private _sampler: ISampler | null = null;
	private _paramsBuffer: IRenderBuffer | null = null;
	private _binding: IBindingGroup | null = null;
	private _bindingSource: IRenderTexture | null = null;

	public constructor(backend: WebGPUBackend) {
		this._backend = backend;
	}

	public async warmup(): Promise<void> {
		await this._ensureResources();
	}

	public async present(request: WebGPUPresentRequest): Promise<void> {
		await this._ensureResources();
		if (!this._pipeline || !this._sampler || !this._paramsBuffer) {
			return;
		}

		this._backend.writeBuffer(
			this._paramsBuffer,
			new Float32Array([DEFAULT_GAMMA, request.applyGamma ? 1 : 0, 0, 0])
		);

		if (!this._binding || this._bindingSource !== request.source) {
			this._destroyBindingGroup(this._binding);
			this._binding = this._backend.createBindingGroup({
				pipeline: this._pipeline,
				layoutIndex: 0,
				entries: [
					{ binding: 0, resource: request.source },
					{ binding: 1, resource: this._sampler },
					{ binding: 2, resource: this._paramsBuffer },
				],
				label: "WebGPUPresentBinding",
			});
			this._bindingSource = request.source;
		}

		request.encoder.beginRenderPass({
			label: "WebGPUPresentPass",
			colorAttachments: [
				{
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: "clear",
					storeOp: "store",
				},
			],
		});
		request.encoder.setPipeline(this._pipeline);
		request.encoder.setBindingGroup(0, this._binding);
		const canvasTarget = this._backend.getCanvasColorTexture();
		const dirtyRects = request.resolveDirtyRects(
			request.frameContext,
			canvasTarget.width,
			canvasTarget.height
		);
		for (const rect of dirtyRects) {
			request.encoder.setScissorRect?.(
				rect.x,
				rect.y,
				rect.width,
				rect.height
			);
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
		this._shaderModule = null;
		this._pipeline = null;
		this._sampler = null;
		this.invalidateBindings();
	}

	public destroy(): void {
		this.onShaderRuntimeChanged();
		this._destroyManagedResource(this._paramsBuffer);
		this._paramsBuffer = null;
	}

	private async _ensureResources(): Promise<void> {
		if (!this._shaderModule) {
			const composite = await ShaderSource.load("webgpu.utility.present.composite");
			this._shaderModule = await this._backend.createShaderModule({
				label: "WebGPUPresentShader",
				code: composite.code,
				sourceMap: composite.sourceMap,
				language: "wgsl",
				stage: "unknown",
				sourceKind: "builtin-present",
			});
		}

		if (!this._pipeline) {
			this._pipeline = this._backend.createPipeline({
				label: "WebGPUPresentPipeline",
				vertex: {
					module: this._shaderModule,
					entryPoint: "vsMain",
				},
				fragment: {
					module: this._shaderModule,
					entryPoint: "fsMain",
					targets: [{ format: this._backend.canvasFormat as any }],
				},
				primitive: {
					topology: "triangle-list" as any,
					cullMode: "none",
					frontFace: "ccw",
				},
			} as any);
		}

		if (!this._sampler) {
			this._sampler = this._backend.createSampler({
				label: "WebGPUPresentSampler",
				magFilter: FilterMode.Linear,
				minFilter: FilterMode.Linear,
				mipmapFilter: FilterMode.Linear,
				addressModeU: AddressMode.ClampToEdge,
				addressModeV: AddressMode.ClampToEdge,
			});
		}

		if (!this._paramsBuffer) {
			this._paramsBuffer = this._backend.createBuffer({
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
