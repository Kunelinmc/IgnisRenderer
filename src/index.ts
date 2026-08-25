export { Vector2 } from "./maths/Vector2";
export { Vector3 } from "./maths/Vector3";
export { Vector4 } from "./maths/Vector4";
export { Box2 } from "./maths/Box2";
export { Box3 } from "./maths/Box3";
export { Matrix3 } from "./maths/Matrix3";
export { Matrix4 } from "./maths/Matrix4";
export { Quaternion } from "./maths/Quaternion";

export * from "./maths/Common";
export * from "./maths/Noise";
export * from "./maths/types";
export * from "./foundation/Color";
export * from "./foundation/Logger";
export * from "./foundation/Platform";
export { Node } from "./core/Node";
export { Environment } from "./core/Environment";
export { TextureFormat } from "./core/TextureFormat";
export { MeshAsset, MeshInstance, LODMeshInstance, MeshFactory } from "./meshes";
export * from "./decals";
export * from "./csg";
export * from "./spatial";
export type {
	IVertex,
	IPrimitive,
	IPrimitiveGeometry,
	BoundingSphere,
	BoundingBox,
} from "./core/types";
export { Renderer } from "./rendering/Renderer";
export type {
	RenderFrameResult,
	RendererEvents,
	RendererOptions,
} from "./rendering/Renderer";
export {
	DEFAULT_DISPLAY_OUTPUT_OPTIONS,
	type DisplayColorSpace,
	type DisplayDynamicRange,
	type DisplayOutputFallbackReason,
	type DisplayOutputMode,
	type DisplayOutputOptions,
	type DisplayOutputState,
	type ResolvedDisplayOutputOptions,
} from "./rendering/DisplayOutput";
export {
	RenderTargetManager,
	type CustomRenderPassContext,
	type CustomRenderPassResourceFacade,
	type CustomRenderTargetAttachment,
	type CustomRenderTargetExecutionTarget,
	type RenderTargetColorAttachmentDescriptor,
	type RenderTargetCreateDescriptor,
	type RenderTargetCustomPassJobDescriptor,
	type RenderTargetDepthAttachmentDescriptor,
	type RenderTargetDescriptor,
	type RenderTargetHandle,
	type RenderTargetJobCompletion,
	type RenderTargetJobDescriptor,
	type RenderTargetJobReadbackOptions,
	type RenderTargetJobRegistration,
	type RenderTargetJobTicket,
	type RenderTargetReadbackOptions,
	type RenderTargetReadbackOrigin,
	type RenderTargetReadbackResult,
	type RenderTargetSceneViewContent,
	type RenderTargetSceneViewJobDescriptor,
	type RenderTargetSizeDescriptor,
} from "./rendering/CustomRenderTargets";
export type {
	IRenderBackend,
	PresentationAlphaMode,
	RenderBackendType,
	RenderBackendAttachContext,
	RenderBackendDebugInfo,
	RenderBackendDeviceDebugInfo,
	RenderBackendProfile,
	RenderSurface,
	RenderSurfaceSize,
	RenderBackendDeviceLostInfo,
	RendererBackendResourceEvent,
	WarmupOptions,
	WarmupProgress,
	WarmupReport,
	WarmupSchedulingMode,
} from "./backends/IRenderBackend";
export {
	AddressMode,
	BufferUsage,
	FilterMode,
	PrimitiveTopology,
	TextureUsage,
	type BindingGroupDesc,
	type BufferDesc,
	type ColorTargetState,
	type IBindingGroup,
	type IRenderBuffer,
	type IRenderPipeline,
	type IRenderTexture,
	type IShaderModule,
	type ISampler,
	type PipelineDesc,
	type SamplerDesc,
	type ShaderModuleDesc,
	type TextureDataLayout,
	type TextureDesc,
	type VertexAttribute,
	type VertexBufferLayout,
	type VertexFormat,
} from "./backends/types";
export type { ICommandEncoder } from "./backends/ICommandEncoder";
export {
	RENDERER_OCCLUSION_CULLING_EXTENSION_ID,
	RENDERER_OCCLUSION_VISIBILITY_INSERTION_POINT,
	WEBGPU_OCCLUSION_AFTER_DEPTH_INSERTION_POINT,
	createRenderBackendExtensionRegistry,
	resolveOcclusionCullingBackendExtension,
	type RenderBackendExtension,
	type BackendExtensionAvailability,
	type RenderBackendExtensionId,
	type RenderBackendExtensionInsertionPoint,
	type RenderBackendExtensionRegistry,
} from "./backends/BackendExtensions";
export {
	WEBGL_AUXILIARY_RASTER_EXTENSION,
	WEBGL_AUXILIARY_RASTER_EXTENSION_ID,
	type IWebGLAuxiliaryRasterFacade,
	type IWebGLAuxiliaryRasterEncoder,
	type WebGLAuxiliaryRasterAvailabilityOptions,
	type WebGLAuxiliaryRasterContext,
	type WebGLAuxiliaryRasterContextLossPolicy,
	type WebGLAuxiliaryRasterFramePolicy,
	type WebGLAuxiliaryRasterRequest,
	type WebGLAuxiliaryRasterRequirements,
	type WebGLAuxiliaryRasterResourceFacade,
	type WebGLAuxiliaryUniform,
	type WebGLAuxiliaryUniformMatrixType,
	type WebGLAuxiliaryUniformScalarType,
	type WebGLAuxiliaryUniformType,
	type WebGLAuxiliaryUniformVectorType,
} from "./backends/webgl/WebGLAuxiliaryRaster";
export { EventEmitter } from "./core/EventEmitter";
export * from "./workers";
export {
	SoftwareBackend,
	type SoftwareBackendOptions,
} from "./backends/software/SoftwareBackend";
export { WebGPUBackend } from "./backends/webgpu/WebGPUBackend";
export { WebGLBackend } from "./backends/webgl/WebGLBackend";
export {
	type WebGPUPostProcessFrameTargets,
} from "./backends/webgpu/WebGPUPostProcessContracts";
export type { IWebGPUComputeFacade } from "./backends/webgpu/ComputeFacade";
export { resolveWebGPUComputeFacade } from "./backends/webgpu/ComputeFacade";
export {
	ComputeRuntime,
	ComputeKernel,
	type BufferReadbackResult,
	type ComputeRuntimeResourceStats,
	type TextureReadbackResult,
	type ComputeBindingSchemaEntry,
	type ComputeKernelDescriptor,
	type ComputeDispatchOptions,
	type ComputeDispatchTicket,
	type ComputeResolvedBindingSchemaEntry,
	type ComputeResolvedWorkgroupSize,
	type IComputeKernel,
	type IComputeRuntime,
	type ReadBufferOptions,
	type ReadTextureOptions,
} from "./backends/webgpu/ComputeRuntime";
export type {
	FrameContext,
	FramePass,
	FramePassStage,
	RendererFramePlan,
	RendererFramePlanStage,
} from "./pipeline/types";
export type {
	PreparedShadowLight,
	PreparedShadowSlice,
	ShadowStorageTechnique,
	ShadowCasterIntent,
	ShadowDiagnostic,
	ShadowFramePlan,
	ShadowRenderJob,
	ShadowWorkSet,
} from "./lights/shadows/ShadowFramePlan";
export {
	getEnabledCustomPostProcessPassIds,
	hasEnabledCustomPostProcessPass,
} from "./postprocess";
export type {
	BloomOptions,
	ColorFilterOptions,
	DOFOptions,
	FogOptions,
	MotionBlurOptions,
	PostProcessCustomPassDescriptor,
	PostProcessPassId,
	ResolvedPostProcessState,
	SSAOOptions,
	SSGIOptions,
	SSROptions,
	TAAOptions,
	VolumetricOptions,
} from "./postprocess";
export * from "./postprocess";
export {
	DEFAULT_INCREMENTAL_DIRTY_TILE_SIZE,
	DEFAULT_INCREMENTAL_RENDERING_OPTIONS,
	createIncrementalTileCoverage,
	IncrementalFramePlanner,
	RENDER_DIRTY_GROUP,
	RENDER_DIRTY_REASON_MASK,
	doesRenderDirtyReasonInvalidateSceneBounds,
	getDefaultIncrementalRegistry,
	hasAnyDirtyReason,
	registerRenderDirtyReason,
	renderDirtyReasonToMask,
	unregisterRenderDirtyReason,
} from "./pipeline/incremental";
export type {
	BuiltinRenderDirtyReason,
	DirtyRect,
	DirtyTileCoverage,
	IncrementalDirtyReasonDescriptor,
	IncrementalFrameContext,
	IncrementalFramePassDescriptor,
	IncrementalFrameStatus,
	IncrementalFrameStats,
	IncrementalTileCoverage,
	IncrementalTileCoverageMode,
	IncrementalTileRange,
	IncrementalPlan,
	IncrementalPlanInput,
	IncrementalRenderingOptions,
	PostProcessGrade,
	PostProcessIncrementalMetadata,
	RenderDirtyReason,
} from "./pipeline/incremental";
export {
	RenderPipelineRegistry,
	type RenderPipelineStageIncrementalOptions,
	type RenderPipelineStageKind,
	type RenderPipelineStagePredicate,
	type RenderPipelineStageRegistration,
	type RenderPipelineStageRunContext,
	type RenderPipelineFramePlanOptions,
} from "./pipeline/RenderPipelineRegistry";
export type { RendererStageDefinition } from "./pipeline/RendererStageGraph";
export type {
	FramePassRenderSupport,
	FramePassRequirements,
} from "./pipeline/FramePassRequirements";
export type {
	CameraJitterRequirement,
	FramePreparationRequirements,
} from "./pipeline/FrameRequirements";
export {
	IBL_PREFILTER_MAX_MIP_LEVELS,
	IBL_PREFILTER_MAX_SAMPLE_HEIGHT,
	IBL_PREFILTER_MAX_SAMPLE_WIDTH,
	IBLPrefilter,
	prefilterEnvironmentIBL,
} from "./lights/ibl/IBLPrefilter";
export type {
	IBLPrefilterAcceleration,
	IBLPrefilterOptions,
	IBLPrefilterProgress,
	IBLPrefilterServiceOptions,
	PrefilterEnvironmentIBLOptions,
} from "./lights/ibl/IBLPrefilter";
export {
	projectEnvironmentTextureToSH,
	type EnvironmentSHProjectionOptions,
} from "./lights/ibl/EnvironmentSH";
export { Scene } from "./core/Scene";
export { Camera, CameraType } from "./cameras/Camera";
export { OrthographicCamera } from "./cameras/OrthographicCamera";
export { OrbitCamera } from "./cameras/OrbitCamera";
export { FPSCamera } from "./cameras/FPSCamera";
export {
	CameraShakePlugin,
	type CameraShakePluginOptions,
	type CameraShakeImpulse,
} from "./addons/CameraShakePlugin";
export {
	InteractionController,
	type InteractionControllerOptions,
	type InteractionClickEvent,
	type InteractionDragRectState,
	type InteractionEntityEvent,
	type InteractionPointerEventLike,
	type InteractionEvents,
	type InteractionGizmoState,
	type InteractionSelectionMode,
	type InteractionState,
	type InteractionTransformEvent,
	type GizmoMode,
	type GizmoPivot,
	type GizmoSpace,
} from "./interaction/InteractionController";
export type {
	Interactable,
	InteractableComponent,
	InteractionCallback,
	InteractionCallbackContext,
	InteractionEventPhase,
	InteractionPointerState,
} from "./interaction/Interactable";
export { InteractableRegistry } from "./interaction/Interactable";
export {
	SobelNormalMapper,
	type SobelNormalMapperOptions,
	type SobelNormalMapperHeightSource,
} from "./addons/SobelNormalMapper";
export {
	screenToWorldRay,
	type ScreenRayInput,
	type ScreenRay,
} from "./interaction/screenToWorldRay";
export {
	Material,
	BasicMaterial,
	PhongMaterial,
	GouraudMaterial,
	UnlitMaterial,
	PBRMaterial,
	ShaderMaterial,
} from "./materials";
export * from "./animation";
export * from "./particles";
export * from "./physics";
export * from "./lights";
export * from "./lights/shadows";
export {
	Texture,
	type TextureBaseParams,
	type TextureColorSpace,
	type TextureParams,
	type TextureSource,
	type TextureSourceKind,
} from "./core/Texture";
export {
	CubeTexture,
	CubeTextureFace,
	type CubeTextureFaceData,
	type CubeTextureParams,
} from "./core/CubeTexture";
export { CanvasTexture, type CanvasTextureParams } from "./core/CanvasTexture";
export { VideoTexture, type VideoTextureParams } from "./core/VideoTexture";
export { TextureLoader } from "./loaders/TextureLoader";
export { OBJLoader } from "./loaders/OBJLoader";
export { GLTFLoader } from "./loaders/GLTFLoader";
export { HDRLoader } from "./loaders/HDRLoader";
export {
	EXRLoader,
	type EXREnvironmentApplyOptions,
	type EXREnvironmentTarget,
	type EXRLoadEnvironmentOptions,
	type EXRParseOptions,
} from "./loaders/EXRLoader";
export { BVHLoader } from "./loaders/BVHLoader";
export { Loader } from "./loaders/Loader";
export * as experimentalECS from "./ecs";
export type { EntityPrefab } from "./ecs";
