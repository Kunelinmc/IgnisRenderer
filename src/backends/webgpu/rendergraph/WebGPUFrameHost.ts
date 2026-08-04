import type { ICommandBuffer, ICommandEncoder } from "../../ICommandEncoder";
import type { IRenderTexture } from "../../types";
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
	onMainTargetSampleCountRuntimeFallback(): void;
	assertDeviceOperational(operation: string): void;
}
