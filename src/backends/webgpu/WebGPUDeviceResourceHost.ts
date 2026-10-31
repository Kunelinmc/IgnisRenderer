import type { TextureFormat } from "../../core/TextureFormat";
import type {
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
} from "../types";
import type { WebGPUShaderRuntimeView } from "./WebGPUMaterialPipelineResolver";
import type { WebGPUBindingGroupDesc } from "./WebGPUBindingGroupCache";
import type { WebGPUResourceManager } from "./WebGPUResourceManager";

/**
 * Narrow device-scoped capability consumed by WebGPU resource owners.
 *
 * @internal Owned by `WebGPUBackend`; applications must use `Renderer`.
 */
export interface WebGPUDeviceResourceHost {
	readonly resourceManager: WebGPUResourceManager;
	readonly device: GPUDevice | null;
	readonly queue: GPUQueue | null;
	readonly canvasFormat: TextureFormat;
	readonly canvasDepthFormat: TextureFormat;
	readonly shaderRuntime?: {
		revision?: number;
		getMode?: () => "strict" | "warn" | "silent";
		onDidChange?: (listener: () => void) => () => void;
	};
	getShaderDirectiveCacheTag?(): string;
	getShaderRuntimeView(): WebGPUShaderRuntimeView;
	createBuffer(desc: BufferDesc): IRenderBuffer;
	createTexture(desc: TextureDesc): IRenderTexture;
	createSampler(desc: SamplerDesc): ISampler;
	createShaderModule(desc: ShaderModuleDesc): Promise<IShaderModule>;
	createPipeline(desc: PipelineDesc): Promise<IRenderPipeline>;
	createComputePipeline(desc: ComputePipelineDesc): Promise<IComputePipeline>;
	createBindingGroup(desc: WebGPUBindingGroupDesc): IBindingGroup;
	createTextureView(
		texture: IRenderTexture,
		desc?: GPUTextureViewDescriptor,
	): GPUTextureView;
	writeBuffer(buffer: IRenderBuffer, data: BufferSource, offset?: number): void;
}
