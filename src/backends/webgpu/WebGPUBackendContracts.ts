import type { ICommandBuffer, ICommandEncoder } from "../ICommandEncoder";
import type { IRenderTexture } from "../types";

export interface WebGPUCommandBufferInternals {
	_backendCommandBuffer: GPUCommandBuffer;
	_ownerToken: object;
	_submitted: boolean;
}

export interface WebGPUTimestampWrites {
	querySet: GPUQuerySet;
	beginningOfPassWriteIndex: number;
	endOfPassWriteIndex: number;
}

export interface WebGPUCommandEncoderHost {
	createPassTimestampWrites(label: string): WebGPUTimestampWrites | undefined;
	getCurrentColorView(): GPUTextureView;
	getCurrentDepthView(): GPUTextureView;
	getCanvasColorTexture(): IRenderTexture;
}

export interface WebGPUCommandSchedulerHost extends WebGPUCommandEncoderHost {
	readonly device: GPUDevice | null;
	readonly queue: GPUQueue | null;
	assertDeviceOperational(operation: string): void;
	runValidationScope<T>(label: string, operation: () => T): T;
	reportNonFatalError(scope: string, error: unknown): void;
	onSubmittedCommandBuffers(): void;
	isFrameActive(): boolean;
}
