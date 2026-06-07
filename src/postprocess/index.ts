import type { FogOptions } from "./passes/FogPass";
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
	DEFAULT_SSAO_OPTIONS,
	SCREEN_SPACE_AMBIENT_OCCLUSION_PASS_ID,
	ScreenSpaceAmbientOcclusionPass,
	SoftwareScreenSpaceAmbientOcclusionImplementation,
	WebGPUScreenSpaceAmbientOcclusionImplementation,
	WebGLScreenSpaceAmbientOcclusionImplementation,
	createSSAOKernelParams,
	resolveSSAODownsample,
	resolveSSAOOptions,
	type ResolvedSSAOOptions,
	type SSAOOptions,
	type ScreenSpaceAmbientOcclusionPassConfig,
	type SoftwareSSAOContext,
	type WebGPUSSAOContext,
	type WebGLSSAOContext,
} from "./passes/ScreenSpaceAmbientOcclusionPass";
export {
	DEFAULT_SSGI_OPTIONS,
	SCREEN_SPACE_GLOBAL_ILLUMINATION_PASS_ID,
	ScreenSpaceGlobalIlluminationPass,
	WebGPUScreenSpaceGlobalIlluminationImplementation,
	createSSGIKernelParams,
	resolveSSGIOptions,
	type ResolvedSSGIOptions,
	type SSGIOptions,
	type ScreenSpaceGlobalIlluminationPassConfig,
	type WebGPUSSGIContext,
} from "./passes/ScreenSpaceGlobalIlluminationPass";
export {
	DEFAULT_SSR_OPTIONS,
	SCREEN_SPACE_REFLECTIONS_PASS_ID,
	ScreenSpaceReflectionsPass,
	createSSRTraceParams,
	resolveSSRHistoryDescriptors,
	resolveSSRHistoryValid,
	resolveSSROptions,
	type ResolvedSSROptions,
	type SSROptions,
	type ScreenSpaceReflectionsPassConfig,
	type WebGPUSSRContext,
} from "./passes/ScreenSpaceReflectionsPass";
export {
	DEFAULT_SSREFRACTION_OPTIONS,
	SCREEN_SPACE_REFRACTIONS_PASS_ID,
	ScreenSpaceRefractionsPass,
	WebGPUScreenSpaceRefractionsImplementation,
	createSSRefractionTraceParams,
	resolveSSRefractionOptions,
	resolveSSRefractionTransientDescriptors,
	type ResolvedSSRefractionOptions,
	type SSRefractionOptions,
	type ScreenSpaceRefractionsPassConfig,
	type WebGPUSSRefractionContext,
} from "./passes/ScreenSpaceRefractionsPass";
export {
	DEFAULT_TAA_OPTIONS,
	TEMPORAL_ANTI_ALIASING_PASS_ID,
	TemporalAntiAliasingPass,
	resolveTAAOptions,
	type ResolvedTAAOptions,
	type TAAOptions,
	type TemporalAntiAliasingPassConfig,
} from "./passes/TemporalAntiAliasingPass";
export {
	BLOOM_PASS_ID,
	DEFAULT_BLOOM_OPTIONS,
	BloomPass,
	WebGPUBloomImplementation,
	WebGLBloomImplementation,
	type BloomOptions,
	type BloomPassConfig,
	type WebGPUBloomContext,
	type WebGLBloomContext,
} from "./passes/BloomPass";
export {
	DEFAULT_FOG_OPTIONS,
	FOG_PASS_ID,
	FogPass,
	WebGPUFogImplementation,
	WebGLFogImplementation,
	type FogOptions,
	type FogPassConfig,
	type WebGPUFogContext,
	type WebGLFogContext,
} from "./passes/FogPass";
export {
	DEFAULT_VOLUMETRIC_OPTIONS,
	VOLUMETRIC_LIGHTING_PASS_ID,
	VolumetricLightingPass,
	SoftwareVolumetricLightingImplementation,
	WebGPUVolumetricLightingImplementation,
	type SoftwareVolumetricLightingContext,
	type VolumetricOptions,
	type VolumetricLightingPassConfig,
	type WebGPUVolumetricLightingContext,
} from "./passes/VolumetricLightingPass";
export {
	COLOR_FILTER_PASS_ID,
	ColorFilterPass,
	DEFAULT_COLOR_FILTER_OPTIONS,
	DEFAULT_DOF_OPTIONS,
	DEFAULT_MOTION_BLUR_OPTIONS,
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
	type ColorFilterOptions,
	type ColorFilterPassConfig,
	type DOFOptions,
	type DepthOfFieldPassConfig,
	type MotionBlurOptions,
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
