import type { Camera } from "../cameras/Camera";
import type {
	Decal,
	DecalBlendMode,
	DecalChannel,
	DecalChannelBlendModes,
} from "../decals";
import type { SceneLight, ShadowCastingLight } from "../lights";
import type { ParticleBlendMode, ParticleSystem } from "../particles";
import type { Material } from "../materials/Material";
import type { Matrix4 } from "../maths/Matrix4";
import type { IVector3, Matrix3Arr, SHCoefficients } from "../maths/types";
import type { RGBA } from "../foundation/Color";
import type { ShadowRenderSet } from "../lights/shadows/ShadowMapping";
import type { DirtyRect, IncrementalFrameContext } from "./incremental";
import type {
	BoundingSphere,
	IPrimitive,
	IPrimitiveGeometry,
} from "../core/types";
import type { Texture } from "../core/Texture";
import type { MeshAsset, MeshInstance } from "../meshes";
import type { EnvironmentTintLinear } from "../core/Environment";
import type { PostProcessPassRegistrySnapshot } from "../postprocess/PostProcessPass";
import type { OcclusionCandidate } from "./OcclusionCulling";
import type { RenderBackendProfile } from "../renderers/IRenderBackend";

export type TransientKey<TValue, TName extends string = string> = TName & {
	readonly __transientValueType?: TValue;
};

export function defineTransientKey<TValue, TName extends string = string>(
	name: TName
): TransientKey<TValue, TName> {
	return name as TransientKey<TValue, TName>;
}

export interface TransientStore extends Map<string, unknown> {
	get<TValue>(key: TransientKey<TValue>): TValue | undefined;
	get(key: string): unknown;
	set<TValue>(key: TransientKey<TValue>, value: TValue): this;
	set(key: string, value: unknown): this;
}

export function createTransientStore(
	entries?: Iterable<readonly [string, unknown]>
): TransientStore {
	return new Map<string, unknown>(entries) as TransientStore;
}

export const DRAW_PACKET_FLAG_TRANSPARENT = 1 << 0;
export const DRAW_PACKET_FLAG_SHADOW_CASTER = 1 << 1;
export const DRAW_PACKET_FLAG_SHADOW_TRANSMITTER = 1 << 2;
export const DRAW_PACKET_FLAG_REFLECTIVE = 1 << 3;

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
export const INTERACTION_TRANSIENT_STATE_KEY =
	defineTransientKey<InteractionTransientState>("pipeline:interaction-state");

export interface InteractionGizmoState {
	mode: "translate" | "rotate" | "scale";
	space: "world" | "local";
	pivot: "object-origin" | "bounds-center";
}

export interface InteractionDragRect {
	startX: number;
	startY: number;
	endX: number;
	endY: number;
	active: boolean;
}

export interface InteractionTransientState {
	selectedEntityIds: number[];
	hoveredEntityId: number | null;
	gizmo: InteractionGizmoState | null;
	dragRect: InteractionDragRect | null;
}

export interface ParticleUVRect {
	u0: number;
	v0: number;
	u1: number;
	v1: number;
}

export interface ParticleRenderItem {
	definitionIndex?: number;
	position: IVector3;
	previousPosition?: IVector3;
	size: number;
	color: RGBA;
	rotation: number;
	previousRotation?: number;
	depth: number;
	uvRect: ParticleUVRect;
}

export interface ParticleRenderBatch {
	kind?: "billboard";
	systemId: string;
	definitionIndex?: number;
	definitionId?: string;
	blendMode: ParticleBlendMode;
	texture: Texture | null;
	receiveShadows: boolean;
	castShadows: boolean;
	shadowDensity: number;
	shadowSoftness: number;
	particles: ParticleRenderItem[];
}

export interface ParticleMeshRenderItem {
	definitionIndex: number;
	position: IVector3;
	previousPosition: IVector3;
	size: number;
	color: RGBA;
	rotation: number;
	previousRotation: number;
	depth: number;
}

export interface ParticleMeshRenderBatch {
	kind: "mesh";
	systemId: string;
	definitionIndex: number;
	definitionId?: string;
	mesh: MeshAsset;
	primitive: IPrimitive;
	material: Material;
	receiveShadows: boolean;
	castShadows: boolean;
	shadowDensity: number;
	shadowSoftness: number;
	particles: ParticleMeshRenderItem[];
}

export interface PreparedScene {
	sceneBounds: BoundingSphere;
	lights: SceneLight[];
	particleSystems: ParticleSystem[];
	hasActiveAnimations: boolean;
	camera: Camera;
	environment: PreparedSceneEnvironment;
	meshInstances: MeshInstance[];
	shadowMaps: Map<ShadowCastingLight, ShadowRenderSet>;
	opaquePackets: DrawPacket[];
	transparentPackets: DrawPacket[];
	shadowCasterPackets: DrawPacket[];
	shadowTransmitterPackets: DrawPacket[];
	reflectivePackets: DrawPacket[];
	decalPackets: DecalPacket[];
	occlusion: PreparedSceneOcclusionState | null;
	spatialIndex: PreparedSceneSpatialIndex | null;
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
	readonly camera: Camera;
	readonly attachments: FrameAttachments;
	readonly features: ResolvedFeatureState;
	readonly postProcess: PostProcessPassRegistrySnapshot;
	readonly shadowMaps: Map<ShadowCastingLight, ShadowRenderSet>;
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
