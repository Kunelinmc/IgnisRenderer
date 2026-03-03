import {
	IRenderBuffer,
	IRenderPipeline,
	IComputePipeline,
	IBindingGroup,
	IndexFormat,
} from "./types";

/**
 * ICommandEncoder: Generic interface for recording draw commands.
 */
export interface ICommandEncoder {
	/** Begin a rendering pass targeting a specific framebuffer/texture */
	beginRenderPass(desc: RenderPassDesc): void;
	/** Begin a compute pass */
	beginComputePass(desc?: ComputePassDesc): void;

	/** Set the active pipeline (state machine configuration) */
	setPipeline(pipeline: IRenderPipeline): void;

	/** Bind a set of resources (uniforms, textures) to a specific slot */
	setBindingGroup(index: number, group: IBindingGroup): void;

	/** Set vertex buffer for a specific slot */
	setVertexBuffer(slot: number, buffer: IRenderBuffer): void;

	/** Set index buffer */
	setIndexBuffer(buffer: IRenderBuffer, format: IndexFormat): void;

	/** Standard indexed draw call */
	drawIndexed(
		indexCount: number,
		instanceCount?: number,
		firstIndex?: number
	): void;

	/** Standard non-indexed draw call */
	draw(vertexCount: number, instanceCount?: number, firstVertex?: number): void;

	/** End the current render pass */
	endRenderPass(): void;

	/** Set pipeline for compute */
	setComputePipeline(pipeline: IComputePipeline): void;

	/** Dispatch compute workgroups */
	dispatchWorkgroups(x: number, y?: number, z?: number): void;

	/** End the current compute pass */
	endComputePass(): void;

	/** Finish recording and return a command buffer object */
	finish(): any;
}

export interface RenderPassDesc {
	colorAttachments: ColorAttachment[];
	depthStencilAttachment?: DepthStencilAttachment;
	label?: string;
}

export interface ColorAttachment {
	view: any; // RenderTargetView
	clearValue?: { r: number; g: number; b: number; a: number };
	loadOp: "clear" | "load";
	storeOp: "store" | "discard";
}

export interface DepthStencilAttachment {
	view: any;
	depthClearValue?: number;
	depthLoadOp?: "clear" | "load";
	depthStoreOp?: "store" | "discard";
}

export interface ComputePassDesc {
	label?: string;
}
