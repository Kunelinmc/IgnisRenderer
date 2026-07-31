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
import type { WebGPUDeviceResourceHost } from "../WebGPUDeviceResourceHost";
import type { DisplayOutputState } from "../../../rendering/DisplayOutput";

/**
 * Narrow device-session surface consumed by the WebGPU frame subsystem.
 *
 * @internal Owned by `WebGPUBackend`; applications must use `Renderer`.
 */
export interface WebGPUFrameHost extends WebGPUDeviceResourceHost {
	readonly computeFacade: IWebGPUComputeFacade;
	readonly postProcessRuntime: BackendPostProcessRuntime;
	readonly enableEarlyZPrepass: boolean;
	readonly enableDeferredLighting: boolean;
	readonly frameGraphValidationMode: "throw" | "warn";
	readonly displayOutputState: DisplayOutputState;
	createCommandEncoder(): ICommandEncoder;
	submit(commands: ICommandBuffer[]): void;
	getCanvasColorTexture(): IRenderTexture;
	getCanvasDepthTexture(): IRenderTexture;
	assertDeviceOperational(operation: string): void;
}
