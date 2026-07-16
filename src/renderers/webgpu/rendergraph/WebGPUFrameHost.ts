import type { ICommandBuffer, ICommandEncoder } from "../../ICommandEncoder";
import type {
	BindingGroupDesc,
	BufferDesc,
	ComputePipelineDesc,
	IBindingGroup,
	IComputePipeline,
	IRenderBuffer,
	IRenderPipeline,
	IRenderTexture,
	ISampler,
	IShaderModule,
	PipelineDesc,
	SamplerDesc,
	ShaderModuleDesc,
	TextureDesc,
	TextureFormat,
} from "../../types";
import type { IWebGPUComputeFacade } from "../ComputeFacade";
import type { BackendPostProcessRuntime } from "../../../postprocess/BackendPostProcessRuntime";

/**
 * Narrow device-session surface consumed by the WebGPU frame subsystem.
 *
 * @internal Owned by `WebGPUBackend`; applications must use `Renderer`.
 */
export interface WebGPUFrameHost {
	readonly device: GPUDevice | null;
	readonly queue: GPUQueue | null;
	readonly canvasFormat: GPUTextureFormat;
	readonly canvasDepthFormat: TextureFormat;
	readonly computeFacade: IWebGPUComputeFacade;
	readonly postProcessRuntime: BackendPostProcessRuntime;
	readonly enableEarlyZPrepass: boolean;
	readonly enableDeferredLighting: boolean;
	readonly frameGraphValidationMode: "throw" | "warn";
	createBuffer(desc: BufferDesc): IRenderBuffer;
	createTexture(desc: TextureDesc): IRenderTexture;
	createSampler(desc: SamplerDesc): ISampler;
	createShaderModule(desc: ShaderModuleDesc): Promise<IShaderModule>;
	createPipeline(desc: PipelineDesc): Promise<IRenderPipeline>;
	createComputePipeline(desc: ComputePipelineDesc): Promise<IComputePipeline>;
	createBindingGroup(desc: BindingGroupDesc): IBindingGroup;
	createTextureView(
		texture: IRenderTexture,
		desc?: GPUTextureViewDescriptor,
	): GPUTextureView;
	createCommandEncoder(): ICommandEncoder;
	submit(commands: ICommandBuffer[]): void;
	writeBuffer(buffer: IRenderBuffer, data: BufferSource, offset?: number): void;
	getCanvasColorTexture(): IRenderTexture;
	getCanvasDepthTexture(): IRenderTexture;
	assertDeviceOperational(operation: string): void;
}
