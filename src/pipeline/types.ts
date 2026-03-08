import type { Camera } from "../cameras/Camera";
import type { SceneLight, ShadowCastingLight } from "../lights";
import type { ParticleBlendMode, ParticleSystem } from "../particles";
import type { Material } from "../materials/Material";
import type { Matrix4 } from "../maths/Matrix4";
import type { IVector3, Matrix3Arr, SHCoefficients } from "../maths/types";
import type { RGBA } from "../utils/Color";
import type { ShadowMap } from "../utils/ShadowMapping";
import type {
	BoundingSphere,
	IModel,
	IPrimitive,
	IPrimitiveGeometry,
} from "../core/types";

export const DRAW_PACKET_FLAG_TRANSPARENT = 1 << 0;
export const DRAW_PACKET_FLAG_SHADOW_CASTER = 1 << 1;
export const DRAW_PACKET_FLAG_SHADOW_TRANSMITTER = 1 << 2;
export const DRAW_PACKET_FLAG_REFLECTIVE = 1 << 3;

export interface DrawPacket {
	readonly id: string;
	model: IModel;
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
export const PARTICLE_SIM_DELTA_TIME_MS_KEY = "pipeline:particle-delta-time-ms";

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
	camera: Camera;
	skybox?: Texture | null;
	models: IModel[];
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

export type FramePassStage =
	| "particle-sim"
	| "shadow"
	| "reflection"
	| "main-opaque"
	| "main-transparent"
	| "particles"
	| "ssao"
	| "taa"
	| "ssr"
	| "volumetric"
	| "fxaa"
	| "gamma";

export interface FramePass {
	stage: FramePassStage;
	executor: "shared" | "backend";
	enabled: boolean;
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
