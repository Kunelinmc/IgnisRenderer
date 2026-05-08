import type { Camera } from "../cameras/Camera";
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
import type { MeshAsset, MeshInstance } from "../meshes";
import type { EnvironmentTintLinear } from "../core/Environment";

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
	normalMatrix: Matrix4 | Matrix3Arr;
	worldBounds: BoundingSphere;
	sortDepth: number;
	pipelineKey: string;
	passFlags: number;
}

export interface PreparedSceneSpatialIndex {
	queryOpaquePackets(rect: DirtyRect): DrawPacket[];
	queryTransparentPackets(rect: DirtyRect): DrawPacket[];
	queryOpaquePacketsInRects(rects: DirtyRect[]): DrawPacket[];
	queryTransparentPacketsInRects(rects: DirtyRect[]): DrawPacket[];
}

import type { Texture } from "../core/Texture";

export const PARTICLE_TRANSIENT_BATCHES_KEY =
	defineTransientKey<ParticleRenderBatch[]>("pipeline:particle-batches");
export const PARTICLE_SIM_DELTA_TIME_SECONDS_KEY =
	defineTransientKey<number>("pipeline:particle-delta-time-seconds");
export const ANIMATION_SIM_DELTA_TIME_MS_KEY =
	defineTransientKey<number>("pipeline:animation-delta-time-ms");
export const INTERACTION_TRANSIENT_STATE_KEY =
	defineTransientKey<InteractionTransientState>("pipeline:interaction-state");

export const INTERACTION_OUTLINE_SHAPES = [
	"circle",
	"square",
	"diamond",
	"octagon",
] as const;
export type InteractionOutlineShape =
	(typeof INTERACTION_OUTLINE_SHAPES)[number];

export interface InteractionOutlineStyle {
	color: RGBA;
	thickness: number;
	opacity: number;
	xray: boolean;
	shape?: InteractionOutlineShape;
}

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
	outline: InteractionOutlineStyle;
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
	castShadows: boolean;
	shadowDensity: number;
	shadowSoftness: number;
	particles: ParticleRenderItem[];
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
	spatialIndex: PreparedSceneSpatialIndex | null;
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
	width: number;
	height: number;
}

export interface FrameContext {
	readonly camera: Camera;
	readonly attachments: FrameAttachments;
	readonly features: ResolvedFeatureState;
	readonly shadowMaps: Map<ShadowCastingLight, ShadowRenderSet>;
	readonly scene: PreparedScene;
	readonly shCoeffs: SHCoefficients;
	readonly shAmbientCoeffs: SHCoefficients;
	readonly worldMatrix: Matrix4;
	readonly incremental: IncrementalFrameContext;
	/** Pass-specific transient data */
	readonly transient: TransientStore;
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
	"ssgi",
	"taa",
	"ssr",
	"volumetric",
	"fog",
	"motion-blur",
	"dof",
	"bloom",
	"tonemap",
	"color-filter",
	"fxaa",
	"interaction-outline",
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

export const FRAME_PASS_DEPENDENCIES = new Map<
	FramePass["stage"],
	readonly FramePass["stage"][]
>([
	["shadow", ["particle-sim"]],
	["main-opaque", ["reflection", "shadow"]],
	["main-transparent", ["main-opaque"]],
	["particles", ["main-transparent"]],
	["ssao", ["particles"]],
	["ssgi", ["ssao"]],
	["taa", ["ssgi", "ssao"]],
	["ssr", ["taa"]],
	["volumetric", ["ssr"]],
	["fog", ["volumetric"]],
	["motion-blur", ["fog"]],
	["dof", ["motion-blur"]],
	["bloom", ["dof"]],
	["tonemap", ["bloom"]],
	["color-filter", ["tonemap"]],
	["fxaa", ["color-filter"]],
	["interaction-outline", ["fxaa"]],
	["gamma", ["tonemap"]],
]);

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

export interface FogOptions {
	mode?: "linear" | "exp" | "exp2";
	application?: "postprocess" | "scene";
	color?: [number, number, number];
	start?: number;
	end?: number;
	density?: number;
	strength?: number;
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

export interface SSGIOptions {
	samples?: number;
	radius?: number;
	intensity?: number;
	falloff?: number;
	depthPhi?: number;
	normalPhi?: number;
	albedoBoost?: number;
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

export interface BloomOptions {
	threshold?: number;
	softKnee?: number;
	intensity?: number;
	radius?: number;
	/** Number of downsample mip passes (1-8). Higher values produce wider bloom. */
	mipPasses?: number;
	/** Tent-filter radius used during upsample (default 1). */
	filterRadius?: number;
	[key: string]: unknown;
}

export interface MotionBlurOptions {
	shutterScale?: number;
	maxSamples?: number;
	velocityClamp?: number;
	depthReject?: number;
	centerWeight?: number;
	[key: string]: unknown;
}

export interface DOFOptions {
	focusDistance?: number;
	focusRange?: number;
	nearStrength?: number;
	farStrength?: number;
	maxBlurRadius?: number;
	depthCurve?: number;
	highlightThreshold?: number;
	highlightGain?: number;
	chromaticAberration?: number;
	[key: string]: unknown;
}

export interface ColorFilterOptions {
	brightness?: number;
	saturation?: number;
	contrast?: number;
	temperature?: number;
	tint?: number;
	[key: string]: unknown;
}

export interface ClusteredLightingOptions {
	tileSizePx?: number;
	zSlices?: number;
	maxLights?: number;
	maxLightsPerCluster?: number;
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

export const DEFAULT_SSGI_OPTIONS: Required<
	Pick<
		SSGIOptions,
		| "samples"
		| "radius"
		| "intensity"
		| "falloff"
		| "depthPhi"
		| "normalPhi"
		| "albedoBoost"
	>
> = {
	samples: 8,
	radius: 3,
	intensity: 0.35,
	falloff: 1.5,
	depthPhi: 1.25,
	normalPhi: 2,
	albedoBoost: 1,
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

export const DEFAULT_FOG_OPTIONS: Required<
	Pick<
		FogOptions,
		| "mode"
		| "application"
		| "color"
		| "start"
		| "end"
		| "density"
		| "strength"
	>
> = {
	mode: "linear",
	application: "postprocess",
	color: [0.58, 0.64, 0.72],
	start: 20,
	end: 200,
	density: 0.015,
	strength: 1,
};

export const DEFAULT_BLOOM_OPTIONS: Required<
	Pick<BloomOptions, "threshold" | "softKnee" | "intensity" | "radius" | "mipPasses" | "filterRadius">
> = {
	threshold: 1,
	softKnee: 0.5,
	intensity: 0.8,
	radius: 1,
	mipPasses: 5,
	filterRadius: 1,
};

export const DEFAULT_MOTION_BLUR_OPTIONS: Required<
	Pick<
		MotionBlurOptions,
		| "shutterScale"
		| "maxSamples"
		| "velocityClamp"
		| "depthReject"
		| "centerWeight"
	>
> = {
	shutterScale: 1,
	maxSamples: 16,
	velocityClamp: 0.06,
	depthReject: 0.025,
	centerWeight: 1,
};

export const DEFAULT_DOF_OPTIONS: Required<
	Pick<
		DOFOptions,
		| "focusDistance"
		| "focusRange"
		| "nearStrength"
		| "farStrength"
		| "maxBlurRadius"
		| "depthCurve"
		| "highlightThreshold"
		| "highlightGain"
		| "chromaticAberration"
	>
> = {
	focusDistance: 8,
	focusRange: 3,
	nearStrength: 0.85,
	farStrength: 1,
	maxBlurRadius: 12,
	depthCurve: 1.25,
	highlightThreshold: 1.2,
	highlightGain: 0.35,
	chromaticAberration: 0.2,
};

export const DEFAULT_COLOR_FILTER_OPTIONS: Required<
	Pick<
		ColorFilterOptions,
		"brightness" | "saturation" | "contrast" | "temperature" | "tint"
	>
> = {
	brightness: 0,
	saturation: 1,
	contrast: 1,
	temperature: 0,
	tint: 0,
};

export const DEFAULT_CLUSTERED_LIGHTING_OPTIONS: Required<
	Pick<
		ClusteredLightingOptions,
		"tileSizePx" | "zSlices" | "maxLights" | "maxLightsPerCluster"
	>
> = {
	tileSizePx: 64,
	zSlices: 24,
	maxLights: 256,
	maxLightsPerCluster: 64,
};

export interface FeatureWarning {
	key: string;
	message: string;
}

export interface RendererFeatureRequest {
	enableLighting?: boolean;
	enableGamma?: boolean;
	enableToneMapping?: boolean;
	enableSH?: boolean;
	enableShadows?: boolean;
	enableReflection?: boolean;
	enableEnvironment?: boolean;
	enableOIT?: boolean;
	enableSSAO?: boolean;
	enableSSGI?: boolean;
	enableTAA?: boolean;
	enableSSR?: boolean;
	enableVolumetric?: boolean;
	enableFog?: boolean;
	enableMotionBlur?: boolean;
	enableDOF?: boolean;
	enableBloom?: boolean;
	enableColorFilter?: boolean;
	enableFXAA?: boolean;
	enableClusteredLighting?: boolean;
	ssrOptions?: SSROptions;
	ssaoOptions?: SSAOOptions;
	ssgiOptions?: SSGIOptions;
	taaOptions?: TAAOptions;
	volumetricOptions?: VolumetricOptions;
	fogOptions?: FogOptions;
	bloomOptions?: BloomOptions;
	motionBlurOptions?: MotionBlurOptions;
	dofOptions?: DOFOptions;
	colorFilterOptions?: ColorFilterOptions;
	clusteredLightingOptions?: ClusteredLightingOptions;
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

/**
 * Returns whether fog should execute as a post-process pass.
 */
export function isFogPostProcessEnabled(
	features: ResolvedFeatureState
): boolean {
	return (
		features.enableFog &&
		(features.fogOptions?.application ?? "postprocess") !== "scene"
	);
}
