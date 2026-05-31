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
import type { PostProcessPassRegistrySnapshot } from "../postprocess/PostProcessPass";

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
	motionBuffer?: Float32Array | null;
	width: number;
	height: number;
}

export interface FrameContext {
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
	"animation-sim",
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
	executor: "shared" | "backend";
	enabled: boolean;
	precompileHints?: string[];
}

export interface RendererFramePlanStage {
	readonly id: string;
	readonly dependsOn: readonly string[];
}

export interface RendererFramePlan {
	readonly stageOrder: readonly RendererFramePlanStage[];
	readonly backendPasses: readonly FramePass[];
}

export const FRAME_PASS_DEPENDENCIES = new Map<
	FramePass["stage"],
	readonly FramePass["stage"][]
>([
	["shadow", ["particle-sim"]],
	["main-opaque", ["reflection", "shadow"]],
	["main-transparent", ["main-opaque"]],
	["particles", ["main-transparent"]],
]);

export interface VolumetricOptions {
	/** Ray-march step count. Higher values reduce banding at higher GPU cost. */
	samples?: number;
	/** Software volumetric grid scale divisor. Higher values improve speed. */
	downsample?: number;
	/** Overall scattering contribution added to the scene color. */
	weight?: number;
	/** Exposure multiplier applied to the accumulated light shaft result. */
	exposure?: number;
	/** Participating-media density. Higher values make fog volumes thicker. */
	airDensity?: number;
	/**
	 * Henyey-Greenstein phase anisotropy. Positive values emphasize forward
	 * scattering, negative values emphasize back scattering.
	 */
	anisotropy?: number;
	/** Maximum world-space ray distance sampled from the camera. */
	maxRayDistance?: number;
	/** Fraction of light scattered instead of absorbed, clamped to `[0, 1]`. */
	scatteringAlbedo?: number;
	/** Step interval for shadow lookups. Higher values reduce shadowing cost. */
	shadowSampleInterval?: number;
	/** Software path: whether the provided depth buffer is already linearized. */
	isLinearDepth?: boolean;
	/** Enables depth-adaptive ray marching to spend samples where detail changes. */
	adaptiveSteps?: boolean;
	/** Software path: enables bilateral upscaling for downsampled volumes. */
	useBilateralUpscale?: boolean;
	/** Depth tolerance for bilateral upscale; lower values preserve harder edges. */
	bilateralDepthSigma?: number;
	/** ReSTIR candidate count per pixel. Higher values improve light selection. */
	restirCandidates?: number;
	/** Temporal reservoir blend factor. Higher values stabilize but can ghost. */
	restirTemporalWeight?: number;
	/** Maximum ReSTIR reservoir weight scale to prevent bright outliers. */
	restirScaleClamp?: number;
	/** Allows backend-specific experimental volumetric options. */
	[key: string]: unknown;
}

export interface FogOptions {
	/** Distance falloff model used to convert depth into fog opacity. */
	mode?: "linear" | "exp" | "exp2";
	/** Executes fog in the post-process stack or during scene shading. */
	application?: "postprocess" | "scene";
	/** Linear RGB fog color mixed over the scene. */
	color?: [number, number, number];
	/** World/view depth where linear fog starts contributing. */
	start?: number;
	/** World/view depth where linear fog reaches full configured strength. */
	end?: number;
	/** Exponential fog density for `exp` and `exp2` modes. */
	density?: number;
	/** Final fog opacity multiplier. `0` disables visible fog. */
	strength?: number;
	/** Allows backend-specific experimental fog options. */
	[key: string]: unknown;
}

export interface SSAOOptions {
	/** Ambient-occlusion sample count, rounded and clamped to backend limits. */
	samples?: number;
	/** View-space sampling radius. Larger values capture wider contact shadows. */
	radius?: number;
	/** Depth bias that suppresses self-occlusion acne near flat surfaces. */
	bias?: number;
	/** Multiplier for the darkening applied by ambient occlusion. */
	intensity?: number;
	/** Internal AO buffer scale divisor. Higher values improve speed. */
	downsample?: number;
	/** Bilateral blur radius in pixels for smoothing noisy AO. */
	blurRadius?: number;
	/** Depth edge sharpness for the bilateral blur. Higher values preserve edges. */
	blurSharpness?: number;
	/** Allows backend-specific experimental SSAO options. */
	[key: string]: unknown;
}

export interface SSROptions {
	/** Maximum ray-march iterations per reflection ray. */
	maxSteps?: number;
	/** Maximum view/world-space ray distance for screen-space reflections. */
	maxDistance?: number;
	/** Depth thickness tolerance used when matching ray hits to surfaces. */
	thickness?: number;
	/** Ray step stride. Higher values improve speed but can skip thin details. */
	stride?: number;
	/** Reflection contribution multiplier mixed into the scene color. */
	intensity?: number;
	/** Temporal history blend factor. Higher values stabilize but can ghost. */
	historyWeight?: number;
	/** Internal trace buffer scale divisor. Higher values improve speed. */
	downsample?: number;
	/** Refinement iterations after a ray hit is found. */
	binarySearchSteps?: number;
	/** Screen-edge fade distance that hides reflections near missing data. */
	edgeFade?: number;
	/** Maximum material roughness that may receive SSR. */
	maxRoughness?: number;
	/** Allows backend-specific experimental SSR options. */
	[key: string]: unknown;
}

export interface SSGIOptions {
	/** Indirect-light sample count, clamped to backend limits. */
	samples?: number;
	/** Screen-space sampling radius for bounced light. */
	radius?: number;
	/** Indirect diffuse lighting multiplier. */
	intensity?: number;
	/** Distance falloff exponent for indirect samples. */
	falloff?: number;
	/** Depth sensitivity for rejecting samples across geometry breaks. */
	depthPhi?: number;
	/** Normal sensitivity for rejecting samples from unrelated surfaces. */
	normalPhi?: number;
	/** Albedo multiplier used to brighten diffuse bounce color. */
	albedoBoost?: number;
	/** Allows backend-specific experimental SSGI options. */
	[key: string]: unknown;
}

export interface TAAOptions {
	/** Sub-pixel camera jitter amplitude. `0` disables temporal jitter. */
	jitterScale?: number;
	/** Temporal color history blend factor. Higher values stabilize but can ghost. */
	historyWeight?: number;
	/** Depth delta threshold that rejects history after disocclusion. */
	disocclusionDepthThreshold?: number;
	/** Motion-vector sensitivity that lowers history weight on fast movement. */
	motionFactor?: number;
	/** Neighborhood variance clamp width. Higher values preserve detail and noise. */
	varianceClampGamma?: number;
	/** Post-TAA sharpening strength used to restore softened edges. */
	sharpen?: number;
	/** Allows backend-specific experimental TAA options. */
	[key: string]: unknown;
}

export interface BloomOptions {
	/** Luminance threshold above which pixels contribute to bloom. */
	threshold?: number;
	/** Soft threshold width that smooths the transition into bloom. */
	softKnee?: number;
	/** Final bloom contribution mixed back into the HDR scene color. */
	intensity?: number;
	/** Single-pass blur radius used by the WebGL bloom path. */
	radius?: number;
	/** Number of downsample mip passes (1-8). Higher values produce wider bloom. */
	mipPasses?: number;
	/** Tent-filter radius used during upsample (default 1). */
	filterRadius?: number;
	/** Allows backend-specific experimental bloom options. */
	[key: string]: unknown;
}

export interface MotionBlurOptions {
	/** Virtual shutter duration multiplier. Higher values lengthen blur trails. */
	shutterScale?: number;
	/** Maximum samples per pixel along the motion vector. */
	maxSamples?: number;
	/** Maximum normalized screen velocity used for blur length. */
	velocityClamp?: number;
	/** Depth-difference threshold for rejecting samples across silhouettes. */
	depthReject?: number;
	/** Weight of the current pixel in the blur accumulation. */
	centerWeight?: number;
	/** Allows backend-specific experimental motion blur options. */
	[key: string]: unknown;
}

export interface DOFOptions {
	/** Focus plane distance in the depth units consumed by the backend. */
	focusDistance?: number;
	/** Depth range around the focus plane that remains sharp. */
	focusRange?: number;
	/** Blur strength for pixels closer than the focus plane. */
	nearStrength?: number;
	/** Blur strength for pixels farther than the focus plane. */
	farStrength?: number;
	/** Maximum circle-of-confusion blur radius in pixels. */
	maxBlurRadius?: number;
	/** Curve exponent for mapping depth error to blur amount. */
	depthCurve?: number;
	/** Luminance threshold for bokeh highlight boost. */
	highlightThreshold?: number;
	/** Intensity of boosted highlights inside blurred regions. */
	highlightGain?: number;
	/** Color-channel separation amount applied to out-of-focus samples. */
	chromaticAberration?: number;
	/** Allows backend-specific experimental depth-of-field options. */
	[key: string]: unknown;
}

export interface ColorFilterOptions {
	/** Additive brightness shift in normalized color space. */
	brightness?: number;
	/** Saturation multiplier. `0` is grayscale, `1` preserves source color. */
	saturation?: number;
	/** Contrast multiplier around mid-gray. */
	contrast?: number;
	/** Warm/cool color balance shift; positive values warm the image. */
	temperature?: number;
	/** Green/magenta tint shift; positive values bias toward magenta. */
	tint?: number;
	/** Allows backend-specific experimental color-filter options. */
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
	Pick<
		BloomOptions,
		"threshold" | "softKnee" | "intensity" | "radius" | "mipPasses" | "filterRadius"
	>
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
	enableSH?: boolean;
	enableShadows?: boolean;
	enableReflection?: boolean;
	enableEnvironment?: boolean;
	enableOIT?: boolean;
	enableClusteredLighting?: boolean;
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
