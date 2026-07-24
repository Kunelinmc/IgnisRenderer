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
} from "../types";

/**
 * Narrow device-scoped capability consumed by WebGPU resource owners.
 *
 * @internal Owned by `WebGPUBackend`; applications must use `Renderer`.
 */
export interface WebGPUDeviceResourceHost {
	readonly device: GPUDevice | null;
	readonly queue: GPUQueue | null;
	readonly canvasFormat: TextureFormat;
	readonly canvasDepthFormat: TextureFormat;
	readonly shaderRuntime?: {
		revision?: number;
		getMode?: () => "strict" | "warn" | "silent";
		onDidChange?: (listener: () => void) => () => void;
	};
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
	writeBuffer(buffer: IRenderBuffer, data: BufferSource, offset?: number): void;
}
