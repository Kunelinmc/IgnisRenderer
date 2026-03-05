import type { Camera } from "../../cameras/Camera";
import type { SceneLight, ShadowCastingLight } from "../../lights";
import type { Material } from "../../materials/Material";
import type { Matrix4 } from "../../maths/Matrix4";
import type { Matrix3Arr } from "../../maths/types";
import type { ShadowMap } from "../../utils/ShadowMapping";
import type {
	BoundingSphere,
	IModel,
	IPrimitive,
	IPrimitiveGeometry,
} from "../types";

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

import type { Texture } from "../Texture";

export interface PreparedScene {
	sceneBounds: BoundingSphere;
	lights: SceneLight[];
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

import type { SHCoefficients } from "../../maths/types";

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
	| "shadow"
	| "reflection"
	| "main-opaque"
	| "main-transparent"
	| "ssao"
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
	[key: string]: unknown;
}

export interface SSROptions {
	maxSteps?: number;
	maxDistance?: number;
	thickness?: number;
	stride?: number;
	intensity?: number;
	historyWeight?: number;
	[key: string]: unknown;
}

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
	enableSSR?: boolean;
	enableVolumetric?: boolean;
	enableFXAA?: boolean;
	ssrOptions?: SSROptions;
	ssaoOptions?: SSAOOptions;
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
	enableSSR: boolean;
	enableVolumetric: boolean;
	enableFXAA: boolean;
	warnings: FeatureWarning[];
	ssrOptions?: SSROptions;
	ssaoOptions?: SSAOOptions;
	volumetricOptions?: VolumetricOptions;
}
