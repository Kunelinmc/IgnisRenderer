import type { Camera } from "../cameras/Camera";
import type {
	Decal,
	DecalBlendMode,
	DecalChannel,
	DecalChannelBlendModes,
} from "../decals";
import type { SceneLight } from "../lights";
import type { ParticleSystem } from "../particles";
import type {
	ParticleMeshRenderBatch,
	ParticleRenderBatch,
} from "../particles/ParticleRenderBatch";
import type { Material } from "../materials/Material";
import type { Matrix4 } from "../maths/Matrix4";
import type { Matrix3Arr, SHCoefficients } from "../maths/types";
import type { DirtyRect, IncrementalFrameContext } from "./incremental";
import type { ShadowFramePlan } from "../lights/shadows/ShadowFramePlan";
import type {
	BoundingSphere,
	IPrimitiveGeometry,
	PrimitiveDrawTopology,
} from "../core/types";
import type { Texture } from "../core/Texture";
import type { MeshInstance } from "../meshes";
import type { EnvironmentTintLinear } from "../core/Environment";
import type { PostProcessPassRegistrySnapshot } from "../postprocess/PostProcessPass";
import type { RenderTargetRegistrySnapshot } from "../rendering/CustomRenderTargets";
import type { OcclusionCandidate } from "./OcclusionCulling";
import type {
	PresentationAlphaMode,
	RenderBackendProfile,
} from "../backends/IRenderBackend";
import type {
	PrimitiveDeformationMap,
	PrimitiveDeformationMode,
} from "../simulation/animation/types";
import {
	defineTransientKey,
	type TransientStore,
} from "../foundation/TransientStore";

export const DRAW_PACKET_FLAG_TRANSPARENT = 1 << 0;
export const DRAW_PACKET_FLAG_SHADOW_CASTER = 1 << 1;
export const DRAW_PACKET_FLAG_SHADOW_TRANSMITTER = 1 << 2;
export const DRAW_PACKET_FLAG_REFLECTIVE = 1 << 3;
export const DRAW_PACKET_FLAG_SHADOW_RECEIVER = 1 << 4;

/** @internal Authoring provenance retained only for identity and diagnostics. */
export type DrawSourceRef =
	| {
		readonly kind: "mesh-instance";
		readonly instanceId: string;
	}
	| {
		readonly kind: "particle-mesh";
		readonly systemId: string;
		readonly templateIndex: number;
		readonly particleIndex: number;
	};

/** @internal Resolved geometry resource consumed by rendering backends. */
export interface DrawGeometryBinding {
	readonly resourceKey: object;
	readonly id: string;
	readonly data: IPrimitiveGeometry;
	readonly version: number;
	readonly topology: PrimitiveDrawTopology;
}

/** @internal Resolved per-instance state consumed by rendering backends. */
export interface DrawInstanceBinding {
	readonly renderLayers: number;
	readonly worldMatrix: Matrix4;
	readonly previousWorldMatrix?: Matrix4;
	readonly normalMatrix: Matrix4 | Matrix3Arr;
}

/** @internal Effective material selected during prepared-scene construction. */
export interface DrawMaterialBinding {
	readonly effective: Material;
	readonly revision: number;
	readonly pipelineKey: string;
}

export type DrawDeformationMode = PrimitiveDeformationMode;

/** @internal Current-frame animation payload routing for one submission. */
export interface DrawDeformationBinding {
	readonly mode: DrawDeformationMode;
	readonly revision: number;
	readonly jointPayloadKey: string | null;
	readonly morphPayloadKey: string | null;
}

/** @internal Camera-independent, backend-neutral draw intent. */
export interface DrawSubmission {
	readonly id: string;
	readonly source: DrawSourceRef;
	readonly geometry: DrawGeometryBinding;
	readonly instance: DrawInstanceBinding;
	readonly material: DrawMaterialBinding;
	readonly deformation: DrawDeformationBinding;
	readonly worldBounds: BoundingSphere;
	readonly passFlags: number;
}

/** @internal View-local wrapper around a camera-independent submission. */
export interface DrawPacket {
	readonly submission: DrawSubmission;
	readonly sortDepth: number;
}

export interface DecalPacket {
	readonly id: string;
	decal: Decal;
	material: Material;
	worldMatrix: Matrix4;
	inverseWorldMatrix: Matrix4;
	normalMatrix: Matrix4 | Matrix3Arr;
	worldBounds: BoundingSphere;
	receiverLayerMask: number;
	priority: number;
	opacity: number;
	edgeFade: number;
	channelBlendModes: DecalChannelBlendModes;
	sceneOrder: number;
}

export type PreparedDecalBlendMode = DecalBlendMode;
export type PreparedDecalChannel = DecalChannel;

export interface PreparedSceneSpatialIndex {
	queryOpaquePackets(rect: DirtyRect): DrawPacket[];
	queryTransparentPackets(rect: DirtyRect): DrawPacket[];
	queryOpaquePacketsInRects(rects: DirtyRect[]): DrawPacket[];
	queryTransparentPacketsInRects(rects: DirtyRect[]): DrawPacket[];
}

export const PARTICLE_TRANSIENT_BATCHES_KEY =
	defineTransientKey<ParticleRenderBatch[]>("pipeline:particle-batches");
export const PARTICLE_MESH_TRANSIENT_BATCHES_KEY =
	defineTransientKey<ParticleMeshRenderBatch[]>("pipeline:particle-mesh-batches");
export const PARTICLE_SIM_DELTA_TIME_SECONDS_KEY =
	defineTransientKey<number>("pipeline:particle-delta-time-seconds");
export const ANIMATION_SIM_DELTA_TIME_MS_KEY =
	defineTransientKey<number>("pipeline:animation-delta-time-ms");

/** Camera-independent scene work shared by every view in one renderer frame. */
export interface PreparedSceneState {
	sceneBounds: BoundingSphere;
	lights: SceneLight[];
	particleSystems: ParticleSystem[];
	hasActiveAnimations: boolean;
	environment: PreparedSceneEnvironment;
	meshInstances: MeshInstance[];
	shadowPlan: ShadowFramePlan;
	/** Camera-independent mesh submissions reused by main and offscreen views. */
	submissions: DrawSubmission[];
	shadowCasterSubmissions: DrawSubmission[];
	shadowTransmitterSubmissions: DrawSubmission[];
	/** @internal Current deformation metadata reused by secondary-camera builds. */
	deformationStates?: PrimitiveDeformationMap | null;
}

/** Camera-local visibility, ordering, and spatial data for one prepared view. */
export interface PreparedSceneView {
	camera: Camera;
	opaquePackets: DrawPacket[];
	transparentPackets: DrawPacket[];
	reflectivePackets: DrawPacket[];
	decalPackets: DecalPacket[];
	occlusion: PreparedSceneOcclusionState | null;
	spatialIndex: PreparedSceneSpatialIndex | null;
}

/** @internal Transitional combined main-view shape consumed by frame backends. */
export interface PreparedScene extends PreparedSceneState, PreparedSceneView {
	shadowCasterPackets: DrawPacket[];
	shadowTransmitterPackets: DrawPacket[];
}

export interface PreparedSceneOcclusionState {
	enabled: boolean;
	sourceFrameIndex: number;
	candidates: OcclusionCandidate[];
	culledPacketIds: string[];
	visibleCandidateCount: number;
	eligibleCandidateCount: number;
}

export interface PreparedSceneEnvironment {
	backgroundEnabled: boolean;
	lightingEnabled: boolean;
	backgroundTexture: Texture | null;
	iblTexture: Texture | null;
	backgroundStrength: number;
	diffuseStrength: number;
	specularStrength: number;
	backgroundTintLinear: EnvironmentTintLinear;
	backgroundExposure: number;
}

export interface FrameAttachments {
	pixels?: Uint8ClampedArray;
	depthBuffer?: Float32Array;
	normalBuffer?: Float32Array | null;
	motionBuffer?: Float32Array | null;
	width: number;
	height: number;
}

export interface FrameContext {
	readonly backendProfile: RenderBackendProfile;
	/** Alpha-compositing behavior of the renderer-owned presentation surface. */
	readonly presentationAlphaMode: PresentationAlphaMode;
	/**
	 * Camera that provides the active view/projection for this frame context.
	 *
	 * In renderer-driven frames this must match `scene.camera`. Secondary
	 * capture contexts must rebuild `scene` for this camera before execution.
	 */
	readonly viewCamera: Camera;
	readonly attachments: FrameAttachments;
	readonly features: ResolvedFeatureState;
	readonly postProcess: PostProcessPassRegistrySnapshot;
	readonly renderTargets: RenderTargetRegistrySnapshot;
	/** Renderer-owned target work captured before backend frame sealing. */
	readonly renderTargetJobs?: import("../rendering/CustomRenderTargets").RenderTargetJobRegistrySnapshot;
	readonly shadowPlan: ShadowFramePlan;
	readonly scene: PreparedScene;
	/** Camera-independent scene data shared with render-target views. */
	readonly sceneState: PreparedSceneState;
	/** Main camera-local view. */
	readonly view: PreparedSceneView;
	readonly shCoeffs: SHCoefficients;
	readonly shAmbientCoeffs: SHCoefficients;
	readonly worldMatrix: Matrix4;
	readonly incremental: IncrementalFrameContext;
	/** Pass-specific transient data */
	readonly transient: TransientStore;
	/**
	 * Renderer-owned pass plan for this frame. Renderer-driven frames must
	 * provide it; direct backend tests or tools may omit it, in which case
	 * backends must skip global pass-order validation.
	 */
	readonly framePlan?: RendererFramePlan;
}

export const BUILTIN_FRAME_PASS_STAGES = [
	"particle-sim",
	"shadow",
	"reflection",
	"main-opaque",
	"main-transparent",
	"particles",
	"postprocess",
] as const;

export type BuiltinFramePassStage = (typeof BUILTIN_FRAME_PASS_STAGES)[number];
export type FramePassStage = BuiltinFramePassStage | (string & {});

export interface FramePass {
	stage: FramePassStage;
	executor: "backend";
	enabled: boolean;
	dependsOn: readonly FramePassStage[];
	precompileHints?: string[];
}

export interface RendererFramePlanStage {
	readonly id: string;
	readonly kind: string;
	readonly dependsOn: readonly string[];
}

export interface RendererFramePlan {
	readonly stageOrder: readonly RendererFramePlanStage[];
	readonly backendPasses: readonly FramePass[];
}

export interface ClusteredLightingOptions {
	tileSizePx?: number;
	zSlices?: number;
	maxLights?: number;
	maxLightsPerCluster?: number;
	cullingMode?: "gather" | "scatter";
	[key: string]: unknown;
}

export interface OcclusionCullingOptions {
	minCandidateScreenAreaPx?: number;
	minOccluderScreenAreaPx?: number;
	hysteresisFrames?: number;
	maxReadbackLatencyFrames?: number;
	debug?: boolean;
	[key: string]: unknown;
}

export const DEFAULT_OCCLUSION_CULLING_OPTIONS: Required<
	Pick<
		OcclusionCullingOptions,
		| "minCandidateScreenAreaPx"
		| "minOccluderScreenAreaPx"
		| "hysteresisFrames"
		| "maxReadbackLatencyFrames"
		| "debug"
	>
> = {
	minCandidateScreenAreaPx: 64,
	minOccluderScreenAreaPx: 256,
	hysteresisFrames: 2,
	maxReadbackLatencyFrames: 3,
	debug: false,
};

export const DEFAULT_CLUSTERED_LIGHTING_OPTIONS: Required<
	Pick<
		ClusteredLightingOptions,
		| "tileSizePx"
		| "zSlices"
		| "maxLights"
		| "maxLightsPerCluster"
		| "cullingMode"
	>
> = {
	tileSizePx: 64,
	zSlices: 24,
	maxLights: 256,
	maxLightsPerCluster: 64,
	cullingMode: "gather",
};

export interface FeatureWarning {
	key: string;
	message: string;
}

export interface RendererFeatureRequest {
	enableLighting?: boolean;
	enableSH?: boolean;
	enableShadows?: boolean;
	enableReflection?: boolean;
	enableEnvironment?: boolean;
	enableOIT?: boolean;
	enableClusteredLighting?: boolean;
	clusteredLightingOptions?: ClusteredLightingOptions;
	enableOcclusionCulling?: boolean;
	occlusionCullingOptions?: OcclusionCullingOptions;
}

export type RendererFeatureFlagKey = Extract<
	keyof RendererFeatureRequest,
	`enable${string}`
>;

export type RendererFeatureFlags = {
	[K in RendererFeatureFlagKey]-?: boolean;
};

export type RendererFeatureRequestExtras = Omit<
	RendererFeatureRequest,
	RendererFeatureFlagKey
>;

export type RendererFeatureOptionKey = Extract<
	keyof RendererFeatureRequest,
	`${string}Options`
>;

export type RendererFeatureResolvedOptions = {
	[K in RendererFeatureOptionKey]-?: NonNullable<RendererFeatureRequest[K]>;
};

export type ResolvedFeatureState = RendererFeatureFlags &
	RendererFeatureRequestExtras & {
		warnings: FeatureWarning[];
	};
