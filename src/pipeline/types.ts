import type { Camera } from "../cameras/Camera";
import type { SceneLight, ShadowCastingLight } from "../lights";
import type { ParticleBlendMode, ParticleSystem } from "../particles";
import type { Material } from "../materials/Material";
import type { Matrix4 } from "../maths/Matrix4";
import type { IVector3, Matrix3Arr, SHCoefficients } from "../maths/types";
import type { RGBA } from "../foundation/Color";
import type { ShadowMap } from "../lights/ShadowMapping";
import type {
	BoundingSphere,
	IPrimitive,
	IPrimitiveGeometry,
} from "../core/types";
import type { MeshAsset, MeshInstance } from "../meshes";

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
	normalMatrix: Matrix4 | Matrix3Arr;
	worldBounds: BoundingSphere;
	sortDepth: number;
	pipelineKey: string;
	passFlags: number;
}

import type { Texture } from "../core/Texture";

export const PARTICLE_TRANSIENT_BATCHES_KEY = "pipeline:particle-batches";
export const PARTICLE_SIM_DELTA_TIME_SECONDS_KEY =
	"pipeline:particle-delta-time-seconds";
export const ANIMATION_SIM_DELTA_TIME_MS_KEY =
	"pipeline:animation-delta-time-ms";

export interface ParticleUVRect {
	u0: number;
	v0: number;
	u1: number;
	v1: number;
}

export interface ParticleRenderItem {
	position: IVector3;
	size: number;
	color: RGBA;
	rotation: number;
	depth: number;
	uvRect: ParticleUVRect;
}

export interface ParticleRenderBatch {
	systemId: string;
	blendMode: ParticleBlendMode;
	texture: Texture | null;
	receiveShadows: boolean;
	particles: ParticleRenderItem[];
}

export interface PreparedScene {
	sceneBounds: BoundingSphere;
	lights: SceneLight[];
	particleSystems: ParticleSystem[];
	hasActiveAnimations: boolean;
	camera: Camera;
	skybox?: Texture | null;
	meshInstances: MeshInstance[];
	shadowMaps: Map<ShadowCastingLight, ShadowMap>;
	opaquePackets: DrawPacket[];
	transparentPackets: DrawPacket[];
	shadowCasterPackets: DrawPacket[];
	shadowTransmitterPackets: DrawPacket[];
	reflectivePackets: DrawPacket[];
}

export interface FrameAttachments {
	pixels?: Uint8ClampedArray;
	depthBuffer?: Float32Array;
	normalBuffer?: Float32Array | null;
	width: number;
	height: number;
}

export interface FrameContext {
	readonly camera: Camera;
	readonly attachments: FrameAttachments;
	readonly features: ResolvedFeatureState;
	readonly shadowMaps: Map<ShadowCastingLight, ShadowMap>;
	readonly scene: PreparedScene;
	readonly shCoeffs: SHCoefficients;
	readonly shAmbientCoeffs: SHCoefficients;
	readonly worldMatrix: Matrix4;
	/** Pass-specific transient data */
	readonly transient: Map<string, any>;
}

export const BUILTIN_FRAME_PASS_STAGES = [
	"animation-sim",
	"particle-sim",
	"shadow",
	"reflection",
	"main-opaque",
	"main-transparent",
	"particles",
	"ssao",
	"taa",
	"ssr",
	"volumetric",
	"fxaa",
	"gamma",
] as const;

export type BuiltinFramePassStage = (typeof BUILTIN_FRAME_PASS_STAGES)[number];
export type FramePassStage = BuiltinFramePassStage | (string & {});

export interface FramePass {
	stage: FramePassStage;
	executor: "shared" | "backend";
	enabled: boolean;
	precompileHints?: string[];
}

export interface VolumetricOptions {
	samples?: number;
	downsample?: number;
	weight?: number;
	exposure?: number;
	airDensity?: number;
	anisotropy?: number;
	maxRayDistance?: number;
	scatteringAlbedo?: number;
	shadowSampleInterval?: number;
	isLinearDepth?: boolean;
	adaptiveSteps?: boolean;
	useBilateralUpscale?: boolean;
	bilateralDepthSigma?: number;
	restirCandidates?: number;
	restirTemporalWeight?: number;
	restirScaleClamp?: number;
	[key: string]: unknown;
}

export interface SSAOOptions {
	samples?: number;
	radius?: number;
	bias?: number;
	intensity?: number;
	downsample?: number;
	blurRadius?: number;
	blurSharpness?: number;
	[key: string]: unknown;
}

export interface SSROptions {
	maxSteps?: number;
	maxDistance?: number;
	thickness?: number;
	stride?: number;
	intensity?: number;
	historyWeight?: number;
	downsample?: number;
	binarySearchSteps?: number;
	edgeFade?: number;
	maxRoughness?: number;
	[key: string]: unknown;
}

export interface TAAOptions {
	jitterScale?: number;
	historyWeight?: number;
	disocclusionDepthThreshold?: number;
	motionFactor?: number;
	varianceClampGamma?: number;
	sharpen?: number;
	[key: string]: unknown;
}

export const DEFAULT_SSAO_OPTIONS: Required<
	Pick<
		SSAOOptions,
		| "samples"
		| "radius"
		| "bias"
		| "intensity"
		| "downsample"
		| "blurRadius"
		| "blurSharpness"
	>
> = {
	samples: 16,
	radius: 8,
	bias: 0.1,
	intensity: 1,
	downsample: 2,
	blurRadius: 2,
	blurSharpness: 8,
};

export const DEFAULT_TAA_OPTIONS: Required<
	Pick<
		TAAOptions,
		| "jitterScale"
		| "historyWeight"
		| "disocclusionDepthThreshold"
		| "motionFactor"
		| "varianceClampGamma"
		| "sharpen"
	>
> = {
	jitterScale: 1,
	historyWeight: 0.9,
	disocclusionDepthThreshold: 0.02,
	motionFactor: 80,
	varianceClampGamma: 1,
	sharpen: 0.1,
};

export const DEFAULT_SSR_OPTIONS: Required<
	Pick<
		SSROptions,
		| "downsample"
		| "maxSteps"
		| "binarySearchSteps"
		| "maxDistance"
		| "thickness"
		| "stride"
		| "intensity"
		| "historyWeight"
		| "edgeFade"
		| "maxRoughness"
	>
> = {
	downsample: 2,
	maxSteps: 64,
	binarySearchSteps: 6,
	maxDistance: 100,
	thickness: 0.2,
	stride: 1,
	intensity: 1,
	historyWeight: 0.85,
	edgeFade: 0.12,
	maxRoughness: 0.85,
};

export const DEFAULT_VOLUMETRIC_OPTIONS: Required<
	Pick<
		VolumetricOptions,
		| "samples"
		| "downsample"
		| "weight"
		| "exposure"
		| "airDensity"
		| "anisotropy"
		| "maxRayDistance"
		| "scatteringAlbedo"
		| "shadowSampleInterval"
		| "isLinearDepth"
		| "adaptiveSteps"
		| "useBilateralUpscale"
		| "bilateralDepthSigma"
		| "restirCandidates"
		| "restirTemporalWeight"
		| "restirScaleClamp"
	>
> = {
	samples: 32,
	downsample: 1,
	weight: 4,
	exposure: 1,
	airDensity: 1,
	anisotropy: 0.2,
	maxRayDistance: 300,
	scatteringAlbedo: 0.9,
	shadowSampleInterval: 2,
	isLinearDepth: true,
	adaptiveSteps: true,
	useBilateralUpscale: true,
	bilateralDepthSigma: 0.05,
	restirCandidates: 8,
	restirTemporalWeight: 0.8,
	restirScaleClamp: 24,
};

export interface FeatureWarning {
	key: string;
	message: string;
}

export interface RendererFeatureRequest {
	enableLighting?: boolean;
	enableGamma?: boolean;
	enableSH?: boolean;
	enableShadows?: boolean;
	enableReflection?: boolean;
	enableSkybox?: boolean;
	enableSSAO?: boolean;
	enableTAA?: boolean;
	enableSSR?: boolean;
	enableVolumetric?: boolean;
	enableFXAA?: boolean;
	ssrOptions?: SSROptions;
	ssaoOptions?: SSAOOptions;
	taaOptions?: TAAOptions;
	volumetricOptions?: VolumetricOptions;
}

export interface ResolvedFeatureState {
	enableLighting: boolean;
	enableGamma: boolean;
	enableSH: boolean;
	enableShadows: boolean;
	enableReflection: boolean;
	enableSkybox: boolean;
	enableSSAO: boolean;
	enableTAA: boolean;
	enableSSR: boolean;
	enableVolumetric: boolean;
	enableFXAA: boolean;
	warnings: FeatureWarning[];
	ssrOptions?: SSROptions;
	ssaoOptions?: SSAOOptions;
	taaOptions?: TAAOptions;
	volumetricOptions?: VolumetricOptions;
}
