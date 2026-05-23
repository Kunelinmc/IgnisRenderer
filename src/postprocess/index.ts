export {
	PostProcessHistoryManager,
	type PostProcessHistoryPrepareRequest,
} from "./PostProcessHistoryManager";
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
	isPostProcessPassStage,
} from "./PostProcessPipeline";
export {
	FastApproximateAntiAliasingPass,
	createFXAAKernelParams,
	type FastApproximateAntiAliasingPassConfig,
	type SoftwareFXAAContext,
	type WebGPUFXAAContext,
	type WebGLFXAAContext,
} from "./passes/FastApproximateAntiAliasingPass";
export {
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
	ScreenSpaceGlobalIlluminationPass,
	WebGPUScreenSpaceGlobalIlluminationImplementation,
	createSSGIKernelParams,
	resolveSSGIOptions,
	type ResolvedSSGIOptions,
	type ScreenSpaceGlobalIlluminationPassConfig,
	type WebGPUSSGIContext,
} from "./passes/ScreenSpaceGlobalIlluminationPass";
export {
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
	TemporalAntiAliasingPass,
	resolveTAAOptions,
	type ResolvedTAAOptions,
	type TemporalAntiAliasingPassConfig,
} from "./passes/TemporalAntiAliasingPass";
export {
	BloomPass,
	WebGPUBloomImplementation,
	WebGLBloomImplementation,
	type BloomPassConfig,
	type WebGPUBloomContext,
	type WebGLBloomContext,
} from "./passes/BloomPass";
export {
	FogPass,
	WebGPUFogImplementation,
	WebGLFogImplementation,
	type FogPassConfig,
	type WebGPUFogContext,
	type WebGLFogContext,
} from "./passes/FogPass";
export {
	VolumetricLightingPass,
	SoftwareVolumetricLightingImplementation,
	WebGPUVolumetricLightingImplementation,
	type SoftwareVolumetricLightingContext,
	type VolumetricLightingPassConfig,
	type WebGPUVolumetricLightingContext,
} from "./passes/VolumetricLightingPass";
export {
	ColorFilterPass,
	DepthOfFieldPass,
	GammaPass,
	InteractionOutlinePass,
	MotionBlurPass,
	ToneMappingPass,
	type ColorFilterPassConfig,
	type DepthOfFieldPassConfig,
	type MotionBlurPassConfig,
} from "./passes/BuiltinFallbackPasses";
export {
	DEFAULT_POST_PROCESS_CAPABILITIES,
	POST_PROCESS_PASS_IDS,
	PostProcessPass,
	PostProcessPassRegistry,
	PostProcessPassRegistrySnapshot,
	getEnabledCustomPostProcessPassIds,
	getPostProcessWarningLabel,
	hasEnabledCustomPostProcessPass,
	isBuiltInPostProcessPassId,
	type PostProcessCapabilities,
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
	PostProcessFrameEndRequest,
	PostProcessFrameRequest,
	PostProcessHistoryDescriptor,
	PostProcessHistoryResolveRequest,
	PostProcessHistorySlot,
	PostProcessHistorySlots,
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassRequirements,
	PostProcessPassResult,
	PostProcessBackendSupport,
	PostProcessPipelineExecuteRequest,
	PostProcessPipelineExecuteResult,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "./types";
