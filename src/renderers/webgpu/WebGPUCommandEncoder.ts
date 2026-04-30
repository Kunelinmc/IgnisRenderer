import {
	type ComputePassDesc,
	type ICommandBuffer,
	type ICommandEncoder,
	type RenderPassDesc,
} from "../ICommandEncoder";
import type {
	IBindingGroup,
	IComputePipeline,
	IRenderBuffer,
	IRenderPipeline,
	IndexFormat,
} from "../types";
import {
	getWebGPUBindGroup,
	getWebGPUBuffer,
	getWebGPUComputePipeline,
	getWebGPURenderPipeline,
	tryGetWebGPUTexture,
} from "./WebGPUResourceAccess";
import type { WebGPUBackend } from "../WebGPUBackend";

export class WebGPUCommandEncoder implements ICommandEncoder {
	private _encoder: GPUCommandEncoder;
	private _backend: WebGPUBackend;
	private _commandBufferOwnerToken: object;
	private _renderPass: GPURenderPassEncoder | null = null;
	private _computePass: GPUComputePassEncoder | null = null;
	private _renderPassWidth = 1;
	private _renderPassHeight = 1;

	constructor(
		encoder: GPUCommandEncoder,
		backend: WebGPUBackend,
		commandBufferOwnerToken: object
	) {
		this._encoder = encoder;
		this._backend = backend;
		this._commandBufferOwnerToken = commandBufferOwnerToken;
	}

	public beginRenderPass(desc: RenderPassDesc): void {
		const timestampWrites = this._backend.createPassTimestampWrites(
			desc.label ?? "render-pass"
		);
		const passExtent = this._resolveRenderPassExtent(desc);
		this._renderPassWidth = passExtent.width;
		this._renderPassHeight = passExtent.height;
		this._renderPass = this._encoder.beginRenderPass({
			colorAttachments: desc.colorAttachments.map((attachment) => ({
				view:
					tryGetWebGPUTexture(attachment.view)?.view ??
					this._backend.getCurrentColorView(),
				resolveTarget:
					attachment.resolveTarget ? (
						tryGetWebGPUTexture(attachment.resolveTarget)?.view
					) : undefined,
				clearValue:
					attachment.loadOp === "clear" ? attachment.clearValue : undefined,
				loadOp: attachment.loadOp,
				storeOp: attachment.storeOp,
			})),
			depthStencilAttachment:
				desc.depthStencilAttachment ?
					{
						view:
							tryGetWebGPUTexture(desc.depthStencilAttachment.view)?.view ??
							this._backend.getCurrentDepthView(),
						depthClearValue:
							(desc.depthStencilAttachment.depthLoadOp ?? "clear") === "clear" ?
								(desc.depthStencilAttachment.depthClearValue ?? 1)
							: undefined,
						depthLoadOp: desc.depthStencilAttachment.depthLoadOp ?? "clear",
						depthStoreOp: desc.depthStencilAttachment.depthStoreOp ?? "store",
					}
				: undefined,
			label: desc.label,
			timestampWrites,
		});
	}

	public beginComputePass(desc?: ComputePassDesc): void {
		const timestampWrites = this._backend.createPassTimestampWrites(
			desc?.label ?? "compute-pass"
		);
		this._computePass = this._encoder.beginComputePass({
			label: desc?.label,
			timestampWrites,
		});
	}

	public setComputePipeline(pipeline: IComputePipeline): void {
		this._computePass?.setPipeline(getWebGPUComputePipeline(pipeline));
	}

	public dispatchWorkgroups(x: number, y: number = 1, z: number = 1): void {
		this._computePass?.dispatchWorkgroups(x, y, z);
	}

	public endComputePass(): void {
		this._computePass?.end();
		this._computePass = null;
	}

	public setPipeline(pipeline: IRenderPipeline): void {
		this._renderPass?.setPipeline(getWebGPURenderPipeline(pipeline));
	}

	public setBindingGroup(index: number, group: IBindingGroup): void {
		const groupResource = getWebGPUBindGroup(group);
		if (this._renderPass) {
			this._renderPass.setBindGroup(index, groupResource);
		} else if (this._computePass) {
			this._computePass.setBindGroup(index, groupResource);
		}
	}

	public setVertexBuffer(slot: number, buffer: IRenderBuffer): void {
		this._renderPass?.setVertexBuffer(slot, getWebGPUBuffer(buffer));
	}

	public setIndexBuffer(buffer: IRenderBuffer, format: IndexFormat): void {
		this._renderPass?.setIndexBuffer(getWebGPUBuffer(buffer), format);
	}

	public drawIndexed(
		indexCount: number,
		instanceCount: number = 1,
		firstIndex: number = 0,
		baseVertex: number = 0,
		firstInstance: number = 0
	): void {
		this._renderPass?.drawIndexed(
			indexCount,
			instanceCount,
			firstIndex,
			baseVertex,
			firstInstance
		);
	}

	public draw(
		vertexCount: number,
		instanceCount: number = 1,
		firstVertex: number = 0,
		firstInstance: number = 0
	): void {
		this._renderPass?.draw(
			vertexCount,
			instanceCount,
			firstVertex,
			firstInstance
		);
	}

	public setScissorRect(
		x: number,
		y: number,
		width: number,
		height: number
	): void {
		const resolvedWidth = Math.max(0, Math.floor(width));
		const resolvedHeight = Math.max(0, Math.floor(height));
		if (resolvedWidth <= 0 || resolvedHeight <= 0) {
			return;
		}
		const resolvedX = Math.max(0, Math.floor(x));
		const resolvedY = Math.max(0, Math.floor(y));
		const maxWidth = Math.max(1, Math.floor(this._renderPassWidth));
		const maxHeight = Math.max(1, Math.floor(this._renderPassHeight));
		if (resolvedX >= maxWidth || resolvedY >= maxHeight) {
			return;
		}
		const maxX = Math.min(maxWidth, resolvedX + resolvedWidth);
		const maxY = Math.min(maxHeight, resolvedY + resolvedHeight);
		const clampedWidth = maxX - resolvedX;
		const clampedHeight = maxY - resolvedY;
		if (clampedWidth <= 0 || clampedHeight <= 0) {
			return;
		}
		this._renderPass?.setScissorRect(
			resolvedX,
			resolvedY,
			clampedWidth,
			clampedHeight
		);
	}

	public endRenderPass(): void {
		this._renderPass?.end();
		this._renderPass = null;
		this._renderPassWidth = 1;
		this._renderPassHeight = 1;
	}

	public finish(): ICommandBuffer {
		if (this._renderPass || this._computePass) throw Error("Pass still active");
		return {
			_backendCommandBuffer: this._encoder.finish(),
			_ownerToken: this._commandBufferOwnerToken,
			_submitted: false,
		};
	}

	public getNativeWebGPUCommandEncoder(): GPUCommandEncoder {
		return this._encoder;
	}

	private _resolveRenderPassExtent(
		desc: RenderPassDesc
	): { width: number; height: number } {
		const colorTarget =
			desc.colorAttachments.find((attachment) => attachment.view)?.view ?? null;
		const depthTarget = desc.depthStencilAttachment?.view ?? null;
		const target = colorTarget ?? depthTarget;
		if (target) {
			return {
				width: Math.max(1, Math.floor(target.width)),
				height: Math.max(1, Math.floor(target.height)),
			};
		}
		const canvasTarget = this._backend.getCanvasColorTexture();
		return {
			width: Math.max(1, Math.floor(canvasTarget.width)),
			height: Math.max(1, Math.floor(canvasTarget.height)),
		};
	}
}
