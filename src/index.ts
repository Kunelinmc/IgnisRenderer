export { Vector2 } from "./maths/Vector2";
export { Vector3 } from "./maths/Vector3";
export { Vector4 } from "./maths/Vector4";
export { Box2 } from "./maths/Box2";
export { Box3 } from "./maths/Box3";
export { Matrix3 } from "./maths/Matrix3";
export { Matrix4 } from "./maths/Matrix4";
export { Quaternion } from "./maths/Quaternion";

export * from "./maths/Common";
export * from "./maths/types";
export * from "./foundation/Color";
export * from "./foundation/Logger";
export * from "./foundation/Platform";
export { Node } from "./core/Node";
export { Environment } from "./core/Environment";
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
export { Renderer } from "./renderers/Renderer";
export type { RendererEvents } from "./renderers/Renderer";
export type {
	IRenderBackend,
	RenderBackendDeviceLostInfo,
	RendererBackendResourceEvent,
} from "./renderers/IRenderBackend";
export { EventEmitter } from "./core/EventEmitter";
export * from "./workers";
export { SoftwareBackend } from "./renderers/SoftwareBackend";
export type {
	SoftwareRasterMode,
	SoftwareTileOptions,
	SoftwareBackendOptions,
} from "./renderers/software/types";
export { WebGPUBackend } from "./renderers/WebGPUBackend";
export { WebGLBackend } from "./renderers/WebGLBackend";
export {
	WEBGPU_PRESENT_POST_PROCESS_CONTEXT_METADATA,
	WEBGPU_SCREEN_POST_PROCESS_CONTEXT_METADATA,
	isWebGPUPostProcessContextMetadata,
	type WebGPUPostProcessContextKind,
	type WebGPUPostProcessContextMetadata,
	type WebGPUPostProcessFrameTargets,
	type WebGPUPostProcessHistoryBindingMetadata,
	type WebGPUPostProcessHistorySide,
	type WebGPUPostProcessMotionHistoryCopyMetadata,
} from "./renderers/webgpu/WebGPUPostProcessContracts";
export type { IWebGPUComputeFacade } from "./renderers/webgpu/ComputeFacade";
export { resolveWebGPUComputeFacade } from "./renderers/webgpu/ComputeFacade";
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
} from "./renderers/webgpu/ComputeRuntime";
export { Rasterizer } from "./renderers/software/Rasterizer";
export type {
	FrameContext,
	FramePass,
	FramePassStage,
	RendererFramePlan,
	RendererFramePlanStage,
} from "./pipeline/types";
export {
	getEnabledCustomPostProcessPassIds,
	hasEnabledCustomPostProcessPass,
	isFogPostProcessEnabled,
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
	IncrementalFrameStats,
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
export { bakeEnvironmentIBLFromEnvironmentMap } from "./pipeline/EnvironmentIBLBaker";
export type {
	EnvironmentIBLBakeAcceleration,
	EnvironmentIBLBakeOptions,
	EnvironmentIBLBakeProgress,
} from "./pipeline/EnvironmentIBLBaker";
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
	type InteractionEntityEvent,
	type InteractionPointerEventLike,
	type InteractionEvents,
	type InteractionSelectionMode,
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
export { Texture, type TextureColorSpace } from "./core/Texture";
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
