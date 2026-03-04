import {
	BufferDesc,
	IRenderBuffer,
	TextureDesc,
	IRenderTexture,
	PipelineDesc,
	IRenderPipeline,
	BindingGroupDesc,
	IBindingGroup,
	ShaderModuleDesc,
	IShaderModule,
	SamplerDesc,
	ISampler,
	ComputePipelineDesc,
	IComputePipeline,
	TextureDataLayout,
} from "./types";
import { ICommandEncoder } from "./ICommandEncoder";

/**
 * IDevice: Professional Rendering Hardware Abstraction Layer
 * Acts as a factory for resources and an orchestrator for execution.
 */
export interface IDevice {
	/** Initialize the underlying graphics context (WebGPU, GL, etc.) */
	init(): Promise<void>;

	/** Resource Factory Methods */
	createBuffer(desc: BufferDesc): IRenderBuffer;
	createTexture(desc: TextureDesc): IRenderTexture;
	createSampler(desc: SamplerDesc): ISampler;
	createShaderModule(desc: ShaderModuleDesc): Promise<IShaderModule>;
	createPipeline(desc: PipelineDesc): IRenderPipeline;
	createComputePipeline(desc: ComputePipelineDesc): IComputePipeline;
	createBindingGroup(desc: BindingGroupDesc): IBindingGroup;

	/** Command Recording */
	createCommandEncoder(): ICommandEncoder;

	/** Get the current backbuffer/canvas textures */
	getCanvasColorTexture(): IRenderTexture;
	getCanvasDepthTexture(): IRenderTexture;

	/** Execution & Submission */
	writeBuffer(buffer: IRenderBuffer, data: ArrayBuffer, offset?: number): void;
	writeTexture(
		texture: IRenderTexture,
		data: BufferSource,
		desc: TextureDataLayout,
		size: { width: number; height: number; depthOrArrayLayers?: number }
	): void;

	copyTextureToTexture(
		source: { texture: IRenderTexture; origin?: any; aspect?: any },
		destination: { texture: IRenderTexture; origin?: any; aspect?: any },
		copySize: { width: number; height: number; depthOrArrayLayers?: number }
	): void;

	submit(commands: any[]): void; // Simplified 'any' for CommandBuffer

	/** Lifecycle */
	resize(width: number, height: number): void;

	/** Get the hardware/backend type */
	readonly type: "webgpu" | "webgl" | "software";

	/** Get the default texture format for the canvas */
	readonly canvasFormat: string;
}
