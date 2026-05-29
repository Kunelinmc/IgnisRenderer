import type { FogOptions } from "../pipeline/types";
import type {
	PostProcessPass,
	PostProcessPassRegistrySnapshot,
} from "./PostProcessPass";

export type ResolvedPostProcessState = PostProcessPassRegistrySnapshot;
export type PostProcessCustomPassDescriptor = PostProcessPass;

/**
 * Returns whether fog should execute as a post-process pass for this frame.
 *
 * @param postProcess Per-frame post-process registry snapshot.
 * @returns `true` when `fog` is enabled and configured for post-process mode.
 * @sideEffects None.
 */
export function isFogPostProcessEnabled(
	postProcess: PostProcessPassRegistrySnapshot
): boolean {
	return (
		postProcess.isEnabled("fog") &&
		(postProcess.getOptions<FogOptions>("fog")?.application ?? "postprocess") !==
			"scene"
	);
}

export {
	PostProcessHistoryManager,
	type PostProcessHistoryPrepareRequest,
} from "./PostProcessHistoryManager";
export {
	PostProcessTransientManager,
	type PostProcessTransientPrepareRequest,
	type PostProcessTransientPrepareResult,
} from "./PostProcessTransientManager";
export {
	BUILTIN_POST_PROCESS_ORDER,
	DEFAULT_POST_PROCESS_PLACEMENT,
	POST_PROCESS_PLACEMENTS,
	getBuiltinPostProcessOrder,
	getCustomPostProcessPlacementOrder,
	isPostProcessPlacement,
	type BuiltinPostProcessOrderEntry,
	type PostProcessPlacement,
} from "./ordering";
export {
	PostProcessPipeline,
	getPostProcessRequirementChannels,
	hasPostProcessExecutionPasses,
	isPostProcessPassStage,
	resolvePostProcessExecutionOrder,
	type PostProcessExecutionOrderContext,
} from "./PostProcessPipeline";
export {
	FAST_APPROXIMATE_ANTI_ALIASING_PASS_ID,
	FastApproximateAntiAliasingPass,
	createFXAAKernelParams,
	type FastApproximateAntiAliasingPassConfig,
	type SoftwareFXAAContext,
	type WebGPUFXAAContext,
	type WebGLFXAAContext,
} from "./passes/FastApproximateAntiAliasingPass";
export {
	SCREEN_SPACE_AMBIENT_OCCLUSION_PASS_ID,
	ScreenSpaceAmbientOcclusionPass,
	SoftwareScreenSpaceAmbientOcclusionImplementation,
	WebGPUScreenSpaceAmbientOcclusionImplementation,
	WebGLScreenSpaceAmbientOcclusionImplementation,
	createSSAOKernelParams,
	resolveSSAODownsample,
	resolveSSAOOptions,
	type ResolvedSSAOOptions,
	type ScreenSpaceAmbientOcclusionPassConfig,
	type SoftwareSSAOContext,
	type WebGPUSSAOContext,
	type WebGLSSAOContext,
} from "./passes/ScreenSpaceAmbientOcclusionPass";
export {
	SCREEN_SPACE_GLOBAL_ILLUMINATION_PASS_ID,
	ScreenSpaceGlobalIlluminationPass,
	WebGPUScreenSpaceGlobalIlluminationImplementation,
	createSSGIKernelParams,
	resolveSSGIOptions,
	type ResolvedSSGIOptions,
	type ScreenSpaceGlobalIlluminationPassConfig,
	type WebGPUSSGIContext,
} from "./passes/ScreenSpaceGlobalIlluminationPass";
export {
	SCREEN_SPACE_REFLECTIONS_PASS_ID,
	ScreenSpaceReflectionsPass,
	createSSRTraceParams,
	resolveSSRHistoryDescriptors,
	resolveSSRHistoryValid,
	resolveSSROptions,
	type ResolvedSSROptions,
	type ScreenSpaceReflectionsPassConfig,
	type WebGPUSSRContext,
} from "./passes/ScreenSpaceReflectionsPass";
export {
	TEMPORAL_ANTI_ALIASING_PASS_ID,
	TemporalAntiAliasingPass,
	resolveTAAOptions,
	type ResolvedTAAOptions,
	type TemporalAntiAliasingPassConfig,
} from "./passes/TemporalAntiAliasingPass";
export {
	BLOOM_PASS_ID,
	BloomPass,
	WebGPUBloomImplementation,
	WebGLBloomImplementation,
	type BloomPassConfig,
	type WebGPUBloomContext,
	type WebGLBloomContext,
} from "./passes/BloomPass";
export {
	FOG_PASS_ID,
	FogPass,
	WebGPUFogImplementation,
	WebGLFogImplementation,
	type FogPassConfig,
	type WebGPUFogContext,
	type WebGLFogContext,
} from "./passes/FogPass";
export {
	VOLUMETRIC_LIGHTING_PASS_ID,
	VolumetricLightingPass,
	SoftwareVolumetricLightingImplementation,
	WebGPUVolumetricLightingImplementation,
	type SoftwareVolumetricLightingContext,
	type VolumetricLightingPassConfig,
	type WebGPUVolumetricLightingContext,
} from "./passes/VolumetricLightingPass";
export {
	COLOR_FILTER_PASS_ID,
	ColorFilterPass,
	DEPTH_OF_FIELD_PASS_ID,
	DepthOfFieldPass,
	GAMMA_PASS_ID,
	GammaPass,
	INTERACTION_OUTLINE_PASS_ID,
	InteractionOutlinePass,
	MOTION_BLUR_PASS_ID,
	MotionBlurPass,
	SoftwareColorFilterImplementation,
	SoftwareGammaImplementation,
	SoftwareInteractionOutlineImplementation,
	SoftwareToneMappingImplementation,
	TONE_MAPPING_PASS_ID,
	ToneMappingPass,
	WebGLColorFilterImplementation,
	WebGLDepthOfFieldImplementation,
	WebGLGammaImplementation,
	WebGLInteractionOutlineImplementation,
	WebGLMotionBlurImplementation,
	WebGLToneMappingImplementation,
	WebGPUColorFilterImplementation,
	WebGPUDepthOfFieldImplementation,
	WebGPUGammaImplementation,
	WebGPUInteractionOutlineImplementation,
	WebGPUMotionBlurImplementation,
	WebGPUToneMappingImplementation,
	type ColorFilterPassConfig,
	type DepthOfFieldPassConfig,
	type MotionBlurPassConfig,
	type SoftwareBuiltinPostProcessContext,
	type WebGLColorFilterContext,
	type WebGLDepthOfFieldContext,
	type WebGLGammaContext,
	type WebGLInteractionOutlineContext,
	type WebGLMotionBlurContext,
	type WebGLScreenPostProcessContext,
	type WebGLToneMappingContext,
	type WebGPUColorFilterContext,
	type WebGPUDepthOfFieldContext,
	type WebGPUGammaContext,
	type WebGPUInteractionOutlineContext,
	type WebGPUMotionBlurContext,
	type WebGPURuntimePostProcessContext,
	type WebGPUScreenPostProcessContext,
	type WebGPUToneMappingContext,
} from "./passes/BuiltinFallbackPasses";
export {
	PostProcessPass,
	PostProcessPassRegistry,
	PostProcessPassRegistrySnapshot,
	getEnabledCustomPostProcessPassIds,
	hasEnabledCustomPostProcessPass,
	type PostProcessPassChange,
	type PostProcessPassConfig,
	type PostProcessPassId,
	type PostProcessPassRegistryChange,
	type PostProcessPassResolveRequest,
	type PostProcessPassWarmupRequest,
	type ResolvedPostProcessPass,
} from "./PostProcessPass";
export {
	registerPostProcessBackendAdapter,
	resolvePostProcessBackendAdapter,
	unregisterPostProcessBackendAdapter,
} from "./PostProcessBackendAdapterRegistry";
export type {
	IPostProcessExecutor,
	LogicalGBufferBridge,
	LogicalGBufferChannel,
	LogicalGBufferHandle,
	LogicalGBufferSemantic,
	PostProcessBackendKind,
	PostProcessFrameAbortRequest,
	PostProcessFrameEndRequest,
	PostProcessFrameRequest,
	PostProcessHistoryDescriptor,
	PostProcessHistoryResolveRequest,
	PostProcessHistorySlot,
	PostProcessHistorySlots,
	PostProcessPassExecutionContextRequest,
	PostProcessPassImplementation,
	PostProcessPassImplementationMetadata,
	PostProcessPassRequest,
	PostProcessPassRequirements,
	PostProcessPassResult,
	PostProcessBackendAdapter,
	PostProcessPipelineExecuteRequest,
	PostProcessPipelineExecuteResult,
	PostProcessResourceMipMode,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
	PostProcessTransientDescriptor,
	PostProcessTransientSlot,
	PostProcessTransientSlots,
} from "./types";
