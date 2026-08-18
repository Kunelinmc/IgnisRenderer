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
	IPrimitive,
	IPrimitiveGeometry,
} from "../core/types";
import type { Texture } from "../core/Texture";
import type { MeshAsset, MeshInstance } from "../meshes";
import type { EnvironmentTintLinear } from "../core/Environment";
import type { PostProcessPassRegistrySnapshot } from "../postprocess/PostProcessPass";
import type {
	CustomRenderPassRegistrySnapshot,
	RenderTargetRegistrySnapshot,
} from "../rendering/CustomRenderTargets";
import type { OcclusionCandidate } from "./OcclusionCulling";
import type { RenderBackendProfile } from "../backends/IRenderBackend";
import type { PrimitiveDeformationMap } from "../simulation/animation/types";
import {
	defineTransientKey,
	type TransientStore,
} from "../foundation/TransientStore";

export const DRAW_PACKET_FLAG_TRANSPARENT = 1 << 0;
export const DRAW_PACKET_FLAG_SHADOW_CASTER = 1 << 1;
export const DRAW_PACKET_FLAG_SHADOW_TRANSMITTER = 1 << 2;
export const DRAW_PACKET_FLAG_REFLECTIVE = 1 << 3;
export const DRAW_PACKET_FLAG_SHADOW_RECEIVER = 1 << 4;

export interface DrawPacket {
	readonly id: string;
	meshInstance: MeshInstance;
	mesh: MeshAsset;
	primitive: IPrimitive;
	material: Material;
	geometry: IPrimitiveGeometry;
	worldMatrix: Matrix4;
	previousWorldMatrix?: Matrix4;
	normalMatrix: Matrix4 | Matrix3Arr;
	worldBounds: BoundingSphere;
	deformationRevision: number;
	sortDepth: number;
	pipelineKey: string;
	passFlags: number;
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

export interface PreparedScene {
	sceneBounds: BoundingSphere;
	lights: SceneLight[];
	particleSystems: ParticleSystem[];
	hasActiveAnimations: boolean;
	camera: Camera;
	environment: PreparedSceneEnvironment;
	meshInstances: MeshInstance[];
	shadowPlan: ShadowFramePlan;
	opaquePackets: DrawPacket[];
	transparentPackets: DrawPacket[];
	shadowCasterPackets: DrawPacket[];
	shadowTransmitterPackets: DrawPacket[];
	reflectivePackets: DrawPacket[];
	decalPackets: DecalPacket[];
	occlusion: PreparedSceneOcclusionState | null;
	spatialIndex: PreparedSceneSpatialIndex | null;
	/** @internal Current deformation metadata reused by secondary-camera builds. */
	deformationStates?: PrimitiveDeformationMap | null;
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
	readonly customRenderPasses: CustomRenderPassRegistrySnapshot;
	readonly shadowPlan: ShadowFramePlan;
	readonly scene: PreparedScene;
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
