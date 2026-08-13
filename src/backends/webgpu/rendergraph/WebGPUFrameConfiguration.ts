import type { WebGPUDeferredGBufferLayout } from "../constants";
import type { WebGPUSceneTargetMode } from "../WebGPUScenePassDescriptors";
import type { WebGPUFrameTargetRequirements } from "./WebGPUFrameTargetManager";

export interface WebGPUFrameCapabilitySnapshot {
	readonly maxColorAttachments: number;
	readonly maxColorAttachmentBytesPerSample: number;
	readonly maxStorageTexturesPerShaderStage: number;
}

export interface WebGPUFrameConfigurationOptions {
	readonly enableDeferredLighting: boolean;
	readonly enableEarlyZPrepass: boolean;
	readonly samplePlan: WebGPUFrameSamplePlan;
	readonly supportsInFrameTextureCopy: boolean;
	readonly forceDeferredFallback?: boolean;
	readonly forceForwardMrt?: boolean;
}

export interface WebGPUFrameSamplePlan {
	readonly requestedSampleCount: number;
	readonly sampleCount: number;
	readonly selectionSignature: string;
	readonly runtimeFallbackActive: boolean;
}

export interface WebGPUFrameDiagnostic {
	readonly code: string;
	readonly message: string;
}

export interface WebGPUFrameConfiguration {
	readonly mrtSupported: boolean;
	readonly deferredSupported: boolean;
	readonly deferredActive: boolean;
	readonly oitActive: boolean;
	readonly transparencyMode: "legacy" | "oit";
	readonly sceneTargetMode: WebGPUSceneTargetMode;
	readonly deferredGBufferLayout: WebGPUDeferredGBufferLayout;
	readonly targetRequirements: WebGPUFrameTargetRequirements | null;
	readonly needsHiZBuild: boolean;
	readonly needsOcclusionTest: boolean;
	readonly enableEarlyZPrepass: boolean;
	readonly samplePlan: WebGPUFrameSamplePlan;
	readonly diagnostics: readonly WebGPUFrameDiagnostic[];
}
