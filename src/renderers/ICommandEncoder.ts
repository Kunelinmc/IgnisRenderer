import type {
	IRenderBuffer,
	IRenderTexture,
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
		firstIndex?: number,
		baseVertex?: number,
		firstInstance?: number
	): void;

	/** Standard non-indexed draw call */
	draw(
		vertexCount: number,
		instanceCount?: number,
		firstVertex?: number,
		firstInstance?: number
	): void;

	/** Indirect non-indexed draw call (optional by backend support) */
	drawIndirect?(buffer: IRenderBuffer, offset?: number): void;

	/** Indirect indexed draw call (optional by backend support) */
	drawIndexedIndirect?(buffer: IRenderBuffer, offset?: number): void;

	/** Optional render-pass scissor rect for partial updates */
	setScissorRect?(
		x: number,
		y: number,
		width: number,
		height: number
	): void;

	/**
	 * Copies one texture region into another texture within this command stream.
	 *
	 * @param source Source texture and optional region origin/aspect.
	 * @param destination Destination texture and optional region origin/aspect.
	 * @param copySize Region size to copy.
	 * @constraints No render or compute pass may be active. Source and
	 * destination textures must be owned by the encoder backend and created
	 * with compatible copy usage flags.
	 * @sideEffects Records a texture-copy command in encoder order.
	 */
	copyTextureToTexture?(
		source: TextureCopyView,
		destination: TextureCopyView,
		copySize: TextureCopySize
	): void;

	/** End the current render pass */
	endRenderPass(): void;

	/** Set pipeline for compute */
	setComputePipeline(pipeline: IComputePipeline): void;

	/** Dispatch compute workgroups */
	dispatchWorkgroups(x: number, y?: number, z?: number): void;

	/** End the current compute pass */
	endComputePass(): void;

	/** Finish recording and return a command buffer object */
	finish(): ICommandBuffer;
}

export interface ICommandBuffer {
	readonly _backendCommandBuffer?: unknown;
	_ownerToken?: unknown;
	_submitted?: boolean;
}

export type TextureCopyAspect = "all" | "depth-only" | "stencil-only";

export interface TextureCopyOrigin {
	readonly x?: number;
	readonly y?: number;
	readonly z?: number;
}

export interface TextureCopyView {
	readonly texture: IRenderTexture;
	readonly origin?: TextureCopyOrigin;
	readonly aspect?: TextureCopyAspect;
}

export interface TextureCopySize {
	readonly width: number;
	readonly height: number;
	readonly depthOrArrayLayers?: number;
}

export interface RenderPassDesc {
	colorAttachments: ColorAttachment[];
	depthStencilAttachment?: DepthStencilAttachment;
	label?: string;
}

export interface ColorAttachment {
	view?: IRenderTexture; // Optional, defaults to canvas if null/undefined
	resolveTarget?: IRenderTexture;
	clearValue?: { r: number; g: number; b: number; a: number };
	loadOp: "clear" | "load";
	storeOp: "store" | "discard";
}

export interface DepthStencilAttachment {
	view?: IRenderTexture;
	depthClearValue?: number;
	depthLoadOp?: "clear" | "load";
	depthStoreOp?: "store" | "discard";
}

export interface ComputePassDesc {
	label?: string;
}
