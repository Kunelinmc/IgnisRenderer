import { CameraType } from "../../cameras/Camera";
import { isShadowCastingLight } from "../../lights";
import { ParticleBlendMode } from "../../particles";
import {
	AlphaMode,
	type Material,
} from "../../materials/Material";
import { ShaderMaterial } from "../../materials/ShaderMaterial";
import { clamp, sRGBToLinear } from "../../maths/Common";
import { Matrix4 } from "../../maths/Matrix4";
import type { SHCoefficients } from "../../maths/types";
import {
	resolveShadowCasterBounds,
	syncShadowMapRegistry,
	updateShadowMapMetadata,
} from "../../pipeline/ShadowMetadata";
import {
	PARTICLE_TRANSIENT_BATCHES_KEY,
	DEFAULT_BLOOM_OPTIONS,
	DEFAULT_DOF_OPTIONS,
	DEFAULT_FOG_OPTIONS,
	DEFAULT_MOTION_BLUR_OPTIONS,
	DEFAULT_SSAO_OPTIONS,
	DEFAULT_TAA_OPTIONS,
	INTERACTION_TRANSIENT_STATE_KEY,
	type DrawPacket,
	type BloomOptions,
	type DOFOptions,
	type FogOptions,
	type FrameContext,
	type FramePass,
	type InteractionTransientState,
	type MotionBlurOptions,
	type ParticleRenderBatch,
	type SSAOOptions,
	type TAAOptions,
} from "../../pipeline/types";
import { IBLBRDF } from "../../pipeline/IBLBRDF";
import {
	collectWebGLLights,
	type WebGLLightState,
	type WebGLClusteredLight,
	type WebGLShadowData,
} from "./WebGLLightCollector";
import { WebGLGeometryRegistry } from "./WebGLGeometryRegistry";
import {
	WEBGL_MAX_DIRECTIONAL_LIGHTS,
	WEBGL_MAX_SPOT_LIGHTS,
	WEBGL_SHADOW_ATLAS_COLUMNS,
	WEBGL_SHADOW_ATLAS_ROWS,
} from "./constants";
import {
	WebGLProgramLibrary,
	type WebGLSceneProgram,
	type WebGLShadowDepthProgram,
} from "./WebGLProgramLibrary";
import { WebGLTextureRegistry } from "./WebGLTextureRegistry";
import type { WebGLShaderSourceFactory } from "../../shaders/webgl/WebGLShaderSourceFactory";
import type {
	ShaderBackendCompileStage,
	ShaderRuntime,
} from "../../shaders/runtime";
import type { ShaderCompileError } from "../../shaders/runtime";
import type {
	WarmupPhaseCounters,
	WarmupPlan,
} from "../../pipeline/WarmupPlanner";
import { toShaderCompileError } from "../../pipeline/WarmupPlanner";
import {
	MAX_INTERACTION_OUTLINE_CIRCLES,
	collectProjectedOutlineCircles,
} from "../../interaction/outlineProjection";
import { getInteractionOutlineShapeCode } from "../../interaction/outlineShape";
import {
	WebGLPostProcessRuntime,
	type WebGLPostProcessPassPlugin,
} from "./WebGLPostProcessRuntime";
import { WebGLClusteredLightingRuntime } from "./WebGLClusteredLightingRuntime";
import {
	clampDownsample,
	computeHaltonJitterNDC,
	finiteOr,
	isFiniteMatrix,
	sanitizeFiniteClamped,
	toColumnMajorMat4,
	toSafeDimension,
} from "./WebGLFrameMath";
import {
	resolveMaterialUniforms,
	resolveTextureUVTransform,
} from "./WebGLMaterialUniformResolver";
import {
	bindWebGLPostSingleColorTarget,
	destroyWebGLFrameTargets,
	ensureWebGLFrameTargets,
	resolveWebGLPostProcessTargetTexture,
	type WebGLFrameTargetLifecycleHost,
} from "./WebGLFrameTargetLifecycle";
import {
	bindWebGLGlobalUniforms,
	uploadWebGLSHAmbientCoefficients,
	type WebGLGlobalUniformBinderHost,
	type WebGLSHAmbientUploadHost,
} from "./WebGLGlobalUniformBinder";
import {
	drawWebGLShadowPacket,
	renderWebGLShadows,
	renderWebGLShadowSlice,
	type WebGLShadowPassHost,
} from "./WebGLShadowPass";
import {
	bindWebGLShaderMaterialTextures,
	drawWebGLPacket,
	renderWebGLPackets,
	type WebGLScenePassHost,
} from "./WebGLScenePass";
import {
	bindWebGLParticleInstanceAttributes,
	destroyWebGLParticleResources,
	ensureWebGLParticleCapacity,
	ensureWebGLParticleResources,
	renderWebGLParticles,
	writeWebGLParticleInstances,
	type WebGLParticlePassHost,
} from "./WebGLParticlePass";

type WarnFn = (key: string, message: string) => void;
type WebGLFramePassHandler = (context: FrameContext) => void;
const TAA_HISTORY_WEIGHT_RANGE: [number, number] = [0, 0.99];
const TAA_DEPTH_THRESHOLD_RANGE: [number, number] = [1e-4, 1];
const TAA_MOTION_FACTOR_RANGE: [number, number] = [0, 512];
const TAA_VARIANCE_GAMMA_RANGE: [number, number] = [0, 8];
const TAA_SHARPEN_RANGE: [number, number] = [0, 2];
const MOTION_BLUR_SHUTTER_SCALE_RANGE: [number, number] = [0, 2];
const MOTION_BLUR_MAX_SAMPLES_RANGE: [number, number] = [4, 64];
const MOTION_BLUR_VELOCITY_CLAMP_RANGE: [number, number] = [0.005, 0.25];
const MOTION_BLUR_DEPTH_REJECT_RANGE: [number, number] = [0.0001, 0.25];
const MOTION_BLUR_CENTER_WEIGHT_RANGE: [number, number] = [0, 4];
const DOF_NEAR_FAR_STRENGTH_RANGE: [number, number] = [0, 2];
const DOF_MAX_BLUR_RADIUS_RANGE: [number, number] = [0, 32];
const DOF_DEPTH_CURVE_RANGE: [number, number] = [0.25, 4];
const DOF_HIGHLIGHT_GAIN_RANGE: [number, number] = [0, 3];
const DOF_CHROMATIC_ABERRATION_RANGE: [number, number] = [0, 2];
const WEBGL_TEXTURE_UNIT_BASE_MAP = 0;
const WEBGL_TEXTURE_UNIT_SHADOW_ATLAS = 1;
const WEBGL_TEXTURE_UNIT_ENV_SPECULAR = 2;
const WEBGL_TEXTURE_UNIT_BRDF_LUT = 3;
const WEBGL_TEXTURE_UNIT_SH_AMBIENT = 4;
const WEBGL_TEXTURE_UNIT_CLUSTER_HEADER = 5;
const WEBGL_TEXTURE_UNIT_CLUSTER_INDEX = 6;
const WEBGL_TEXTURE_UNIT_CLUSTER_LIGHT = 7;
const WEBGL_TEXTURE_UNIT_CUSTOM_START = 8;
const IDENTITY_MATRIX4_COLUMN_MAJOR = new Float32Array([
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1,
]);
const SH_COEFFICIENT_COUNT = 16;
const POST_PROCESS_STAGES: readonly FramePass["stage"][] = [
	"ssao",
	"ssgi",
	"taa",
	"ssr",
	"volumetric",
	"fog",
	"motion-blur",
	"dof",
	"bloom",
	"fxaa",
	"interaction-outline",
	"gamma",
] as const;

const PARTICLE_QUAD_VERTICES = new Float32Array([
	-0.5,
	-0.5,
	0,
	1,
	0.5,
	-0.5,
	1,
	1,
	0.5,
	0.5,
	1,
	0,
	-0.5,
	-0.5,
	0,
	1,
	0.5,
	0.5,
	1,
	0,
	-0.5,
	0.5,
	0,
	0,
]);

const PARTICLE_QUAD_STRIDE = 16;
const PARTICLE_INSTANCE_FLOATS = 13;
const PARTICLE_INSTANCE_STRIDE = PARTICLE_INSTANCE_FLOATS * 4;
const PARTICLE_INITIAL_CAPACITY = 256;
const PARTICLE_MAX_INSTANCES_PER_DRAW = 1 << 16;

export class WebGLFrameExecutor {
	private _gl: WebGL2RenderingContext;
	private _logWarning: WarnFn;
	private _programs: WebGLProgramLibrary;
	private _geometry: WebGLGeometryRegistry;
	private _textures: WebGLTextureRegistry;
	private _sceneFramebuffer: WebGLFramebuffer | null = null;
	private _sceneColorTexture: WebGLTexture | null = null;
	private _sceneMotionTexture: WebGLTexture | null = null;
	private _sceneNormalTexture: WebGLTexture | null = null;
	private _sceneDepthBuffer: WebGLRenderbuffer | null = null;
	private _shadowFramebuffer: WebGLFramebuffer | null = null;
	private _shadowAtlasTexture: WebGLTexture | null = null;
	private _shadowAtlasTileSize = 0;
	private _shadowMvpMatrix = Matrix4.identity();
	private _taaHistoryTextures: [WebGLTexture | null, WebGLTexture | null] = [null, null];
	private _taaMotionHistoryTextures: [WebGLTexture | null, WebGLTexture | null] = [null, null];
	private _taaHistoryIndex = 0;
	private _taaHistoryValid = false;
	private _taaJitter = new Float32Array(4); // currX, currY, prevX, prevY
	private _taaFrameIndex = 0;
	private _prevViewProjection: Float32Array | null = null;
	private _modelMatrixCache = new Map<string, Float32Array>();
	private _modelMatrixKeysThisFrame = new Set<string>();
	private _postFramebuffer: WebGLFramebuffer | null = null;
	private _postColorTexture: WebGLTexture | null = null;
	private _ssaoRawTexture: WebGLTexture | null = null;
	private _ssaoBlurTexture: WebGLTexture | null = null;
	private _presentSourceTexture: WebGLTexture | null = null;
	private _fullscreenVao: WebGLVertexArrayObject | null = null;
	private _particleVao: WebGLVertexArrayObject | null = null;
	private _particleQuadBuffer: WebGLBuffer | null = null;
	private _particleInstanceBuffer: WebGLBuffer | null = null;
	private _particleInstanceCapacity = 0;
	private _particleScratch = new Float32Array(0);
	private _width = 1;
	private _height = 1;
	private _targetWidth = 0;
	private _targetHeight = 0;
	private _targetSSAODownsample = DEFAULT_SSAO_OPTIONS.downsample;
	private _maxTextureSize: number;
	private _maxRenderbufferSize: number;
	private _maxTextureImageUnits: number;
	private _presentedInFrame = false;
	private _activeContext: FrameContext | null = null;
	private _lightState: WebGLLightState | null = null;
	private _clusteredLighting: WebGLClusteredLightingRuntime;
	private _postProcessRuntime: WebGLPostProcessRuntime;
	private _postProcessExecuted = false;
	private _shAmbientTexture: WebGLTexture | null = null;
	private _shAmbientTextureWidth = SH_COEFFICIENT_COUNT;
	private _shAmbientTextureHeight = 1;
	private _ssaoFrameIndex = 0;
	private _fogParams0 = new Float32Array(4);
	private _fogParams1 = new Float32Array(4);
	private _interactionOutlineCircles = new Float32Array(
		MAX_INTERACTION_OUTLINE_CIRCLES * 4
	);
	private readonly _passHandlers: Map<FramePass["stage"], WebGLFramePassHandler>;

	constructor(
		gl: WebGL2RenderingContext,
		warn: WarnFn,
		shaderRuntime?: ShaderRuntime,
		shaderCompileStage?: ShaderBackendCompileStage,
		shaderSourceFactory?: WebGLShaderSourceFactory
	) {
		this._gl = gl;
		this._logWarning = warn;
		this._programs = new WebGLProgramLibrary(
			gl,
			warn,
			shaderRuntime,
			shaderCompileStage,
			shaderSourceFactory
		);
		this._geometry = new WebGLGeometryRegistry(gl, warn);
		this._textures = new WebGLTextureRegistry(gl, warn);
		this._fullscreenVao = gl.createVertexArray();
		this._maxTextureSize = this._resolveLimit(gl.MAX_TEXTURE_SIZE, 4096);
		this._maxRenderbufferSize = this._resolveLimit(
			gl.MAX_RENDERBUFFER_SIZE,
			4096
		);
		this._maxTextureImageUnits = this._resolveLimit(
			gl.MAX_TEXTURE_IMAGE_UNITS,
			16
		);
		this._clusteredLighting = new WebGLClusteredLightingRuntime(gl);
		this._postProcessRuntime = new WebGLPostProcessRuntime(
			this._createDefaultPostProcessPasses()
		);
		this._passHandlers = this._createPassHandlers();
	}

	public beginFrame(context: FrameContext): void {
		this._activeContext = context;
		this._presentedInFrame = false;
		this._postProcessExecuted = false;
		this._modelMatrixKeysThisFrame.clear();
		this._width = toSafeDimension(context.attachments.width);
		this._height = toSafeDimension(context.attachments.height);
		const ssaoDownsample = clampDownsample(
			context.features.ssaoOptions?.downsample,
			DEFAULT_SSAO_OPTIONS.downsample
		);
		this._ensureFrameTargets(this._width, this._height, ssaoDownsample);
		this._presentSourceTexture = this._sceneColorTexture;
		this._syncShadowMetadata(context);
		this._lightState = collectWebGLLights(
			context.scene.lights,
			context.features.enableLighting,
			this._logWarning,
			context.features.enableShadows,
			context.shadowMaps,
			context.features.enableSH,
			context.scene.skybox,
			context.features.enableClusteredLighting
		);
		this._clusteredLighting.prepare(
			context,
			this._lightState,
			this._maxTextureSize
		);
		
		if (context.features.enableTAA) {
			this._taaJitter[2] = this._taaJitter[0];
			this._taaJitter[3] = this._taaJitter[1];
			const nextJitter = computeHaltonJitterNDC(this._taaFrameIndex++, this._width, this._height);
			this._taaJitter[0] = nextJitter[0];
			this._taaJitter[1] = nextJitter[1];
		} else {
			this._taaJitter.fill(0);
			this._taaFrameIndex = 0;
			this._taaHistoryValid = false;
		}
		if (context.incremental.temporalHistoryReset) {
			this._taaHistoryValid = false;
			this._prevViewProjection = null;
		}

		const gl = this._gl;
		gl.bindFramebuffer(gl.FRAMEBUFFER, this._sceneFramebuffer);
		gl.viewport(0, 0, this._width, this._height);
		gl.disable(gl.BLEND);
		gl.enable(gl.DEPTH_TEST);
		gl.depthMask(true);

		const drawBuffers =
			this._sceneNormalTexture ?
				[gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]
			:	[gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1];
		gl.drawBuffers(drawBuffers);

		const incrementalPartial = this._isIncrementalPartial(context);
		gl.clearColor(0, 0, 0, 1);
		gl.clearDepth(1);
		if (incrementalPartial) {
			const dirtyRects = this._resolveDirtyRects(context, this._width, this._height);
			if (dirtyRects.length > 0) {
				gl.enable(gl.SCISSOR_TEST);
				for (const rect of dirtyRects) {
					this._setScissorRect(rect.x, rect.y, rect.width, rect.height, this._height);
					gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
				}
				gl.disable(gl.SCISSOR_TEST);
			}
		} else {
			gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
		}

		if (!incrementalPartial && context.features.enableSkybox && context.scene.skybox) {
			this._renderSkybox(context);
		}
	}

	public executePass(pass: FramePass, context: FrameContext): void {
		const handler = this._passHandlers.get(pass.stage);
		if (!handler) {
			this._logWarning(
				`webgl-stage-unsupported-${pass.stage}`,
				`WebGL backend does not support pass "${pass.stage}" yet; skipping`
			);
			return;
		}
		handler(context);
	}

	public registerPostProcessPass(pass: WebGLPostProcessPassPlugin): void {
		this._postProcessRuntime.registerPass(pass);
	}

	public unregisterPostProcessPass(id: string): void {
		this._postProcessRuntime.unregisterPass(id);
	}

	public endFrame(): void {
		if (!this._presentedInFrame) {
			this._present(this._activeContext?.features.enableGamma !== false);
		}
		this._pruneModelMatrixCache();
		this._activeContext = null;
	}

	public warmup(
		context: FrameContext,
		plan: WarmupPlan
	): WarmupPhaseCounters {
		let total = 0;
		let compiled = 0;
		let skipped = 0;
		let failed = 0;
		const errors: ShaderCompileError[] = [];

		const compile = (label: string, action: () => void): void => {
			total++;
			try {
				action();
				compiled++;
			} catch (error) {
				failed++;
				errors.push(toShaderCompileError(error, "webgl", label));
			}
		};

		compile("WebGLSceneProgram:builtin", () => {
			this._programs.getSceneProgram();
		});
		for (const material of plan.materials) {
			if (!(material instanceof ShaderMaterial)) {
				continue;
			}
			compile(`WebGLSceneProgram:material:${material.shaderId}`, () => {
				this._programs.getSceneProgram(material);
			});
		}

		if (plan.enableSkybox) {
			compile("WebGLSkyboxProgram", () => {
				this._programs.getSkyboxProgram();
			});
		}
		if (plan.enableShadows) {
			compile("WebGLShadowDepthProgram", () => {
				this._programs.getShadowDepthProgram();
			});
		}
		if (plan.enableParticles) {
			compile("WebGLParticleProgram", () => {
				this._programs.getParticleProgram();
			});
		}

		const allowedPassIds = new Set(plan.postProcessPasses);
		const warmupHints = this._postProcessRuntime.collectWarmupHints(
			context.features,
			this._logWarning,
			allowedPassIds
		);
		for (const hint of warmupHints) {
			switch (hint) {
				case "postprocess:ssao":
					compile("WebGLSSAORawProgram", () => {
						this._programs.getSSAORawProgram();
					});
					compile("WebGLSSAOBlurProgram", () => {
						this._programs.getSSAOBlurProgram();
					});
					compile("WebGLSSAOCombineProgram", () => {
						this._programs.getSSAOCombineProgram();
					});
					break;
				case "postprocess:taa":
					compile("WebGLTAAProgram", () => {
						this._programs.getTAAProgram();
					});
					break;
				case "postprocess:fxaa":
					compile("WebGLFXAAProgram", () => {
						this._programs.getFXAAProgram();
					});
					break;
				case "postprocess:interaction-outline":
					compile("WebGLInteractionOutlineProgram", () => {
						this._programs.getInteractionOutlineProgram();
					});
					break;
				case "postprocess:bloom":
					compile("WebGLBloomProgram", () => {
						this._programs.getBloomProgram();
					});
					break;
				case "postprocess:motion-blur":
					compile("WebGLMotionBlurProgram", () => {
						this._programs.getMotionBlurProgram();
					});
					break;
				case "postprocess:dof":
					compile("WebGLDOFProgram", () => {
						this._programs.getDOFProgram();
					});
					break;
				case "postprocess:gamma":
					compile("WebGLPresentProgram", () => {
						this._programs.getPresentProgram();
					});
					break;
				case "postprocess:ssr":
					compile("WebGLSSRProgram", () => {
						this._programs.getSSRProgram();
					});
					break;
				case "postprocess:volumetric":
					compile("WebGLVolumetricProgram", () => {
						this._programs.getVolumetricProgram();
					});
					break;
				case "postprocess:fog":
					compile("WebGLFogProgram", () => {
						this._programs.getFogProgram();
					});
					break;
				default:
					skipped++;
					break;
			}
		}

		if (context.features.enableGamma && !plan.postProcessPasses.includes("gamma")) {
			compile("WebGLPresentProgram", () => {
				this._programs.getPresentProgram();
			});
		}

		return {
			phase: "webgl-programs",
			total,
			compiled,
			skipped,
			failed,
			errors,
		};
	}

	public resize(width: number, height: number): void {
		this._width = toSafeDimension(width);
		this._height = toSafeDimension(height);
		this._destroyFrameTargets();
	}

	public destroy(): void {
		this._destroyFrameTargets();
		this._destroyShadowTargets();
		this._destroyParticleResources();
		this._clusteredLighting.destroy();
		if (this._shAmbientTexture) {
			this._gl.deleteTexture(this._shAmbientTexture);
			this._shAmbientTexture = null;
		}
		this._modelMatrixCache.clear();
		this._modelMatrixKeysThisFrame.clear();
		if (this._fullscreenVao) {
			this._gl.deleteVertexArray(this._fullscreenVao);
			this._fullscreenVao = null;
		}
		this._geometry.destroy();
		this._textures.destroy();
		this._programs.destroy();
		this._activeContext = null;
	}

	private _createPassHandlers(): Map<
		FramePass["stage"],
		WebGLFramePassHandler
	> {
		const runPostProcess = (context: FrameContext) => {
			if (this._postProcessExecuted) {
				return;
			}
			this._runPostProcessGraph(context);
			this._postProcessExecuted = true;
		};
		const handlers = new Map<FramePass["stage"], WebGLFramePassHandler>([
			[
				"shadow",
				(context) => {
					this._renderShadows(context);
				},
			],
			[
				"main-opaque",
				(context) => {
					this._renderPackets(context, context.scene.opaquePackets, false);
				},
			],
			[
				"main-transparent",
				(context) => {
					this._renderPackets(context, context.scene.transparentPackets, true);
				},
			],
			[
				"particles",
				(context) => {
					this._renderParticles(context);
				},
			],
			[
				"ssao",
				(context) => {
					runPostProcess(context);
				},
			],
		]);
		for (const stage of POST_PROCESS_STAGES) {
			handlers.set(stage, runPostProcess);
		}
		return handlers;
	}

	private _runPostProcessGraph(context: FrameContext): void {
		this._postProcessRuntime.execute(context, context.features, this._logWarning);
	}

	private _createDefaultPostProcessPasses(): WebGLPostProcessPassPlugin[] {
		return [
			{
				id: "ssao",
				dependsOn: [],
				precompileHints: ["postprocess:ssao"],
				isEnabled: (features) => features.enableSSAO,
				execute: ({ frameContext }) => {
					this._applySSAO(frameContext.features.ssaoOptions, frameContext);
				},
			},
			{
				id: "ssgi",
				dependsOn: ["ssao"],
				precompileHints: ["postprocess:ssgi"],
				isEnabled: (features) => features.enableSSGI,
				execute: () => {},
			},
			{
				id: "taa",
				dependsOn: ["ssgi", "ssao"],
				precompileHints: ["postprocess:taa"],
				isEnabled: (features) => features.enableTAA,
				execute: ({ frameContext }) => {
					this._applyTAA(frameContext.features.taaOptions);
				},
			},
			{
				id: "ssr",
				dependsOn: ["taa"],
				precompileHints: ["postprocess:ssr"],
				isEnabled: (features) => features.enableSSR,
				execute: () => {},
			},
			{
				id: "volumetric",
				dependsOn: ["ssr"],
				precompileHints: ["postprocess:volumetric"],
				isEnabled: (features) => features.enableVolumetric,
				execute: () => {},
			},
			{
				id: "fog",
				dependsOn: ["volumetric"],
				precompileHints: ["postprocess:fog"],
				isEnabled: (features) =>
					features.enableFog &&
					(features.fogOptions?.application ?? "postprocess") !== "scene",
				execute: ({ frameContext }) => {
					this._applyFog(frameContext.features.fogOptions);
				},
			},
			{
				id: "motion-blur",
				dependsOn: ["fog"],
				precompileHints: ["postprocess:motion-blur"],
				isEnabled: (features) => features.enableMotionBlur,
				execute: ({ frameContext }) => {
					this._applyMotionBlur(frameContext.features.motionBlurOptions);
				},
			},
			{
				id: "dof",
				dependsOn: ["motion-blur"],
				precompileHints: ["postprocess:dof"],
				isEnabled: (features) => features.enableDOF,
				execute: ({ frameContext }) => {
					this._applyDOF(frameContext.features.dofOptions);
				},
			},
			{
				id: "bloom",
				dependsOn: ["dof"],
				precompileHints: ["postprocess:bloom"],
				isEnabled: (features) => features.enableBloom,
				execute: ({ frameContext }) => {
					this._applyBloom(frameContext.features.bloomOptions);
				},
			},
			{
				id: "fxaa",
				dependsOn: ["bloom"],
				precompileHints: ["postprocess:fxaa"],
				isEnabled: (features) => features.enableFXAA,
				execute: () => {
					this._applyFXAA();
				},
			},
			{
				id: "interaction-outline",
				dependsOn: ["fxaa"],
				precompileHints: ["postprocess:interaction-outline"],
				isEnabled: () => true,
				execute: ({ frameContext }) => {
					this._applyInteractionOutline(frameContext);
				},
			},
			{
				id: "gamma",
				dependsOn: ["interaction-outline"],
				precompileHints: ["postprocess:gamma"],
				isEnabled: (features) => features.enableGamma,
				execute: ({ frameContext }) => {
					this._present(frameContext.features.enableGamma !== false);
				},
			},
		];
	}

	private _syncShadowMetadata(context: FrameContext): void {
		const shadowLights = context.scene.lights.filter(isShadowCastingLight);
		syncShadowMapRegistry(context.shadowMaps, shadowLights);

		if (!context.features.enableShadows) {
			return;
		}

		const shadowCasterBounds = resolveShadowCasterBounds(
			context.scene.shadowCasterPackets,
			context.scene.sceneBounds,
			context.scene.camera
		);
		for (const light of shadowLights) {
			const shadowMap = context.shadowMaps.get(light);
			if (!shadowMap) continue;
			updateShadowMapMetadata(shadowMap, light, shadowCasterBounds);
		}
	}

	private _isIncrementalPartial(context: FrameContext | null | undefined): boolean {
		if (!context?.incremental) {
			return false;
		}
		return (
			context.incremental.enabled &&
			!context.incremental.forceFullFrame &&
			context.incremental.dirtyRects.length > 0
		);
	}

	private _resolveDirtyRects(
		context: FrameContext | null | undefined,
		viewportWidth: number,
		viewportHeight: number
	): Array<{ x: number; y: number; width: number; height: number }> {
		const width = Math.max(1, Math.floor(viewportWidth));
		const height = Math.max(1, Math.floor(viewportHeight));
		if (!this._isIncrementalPartial(context)) {
			return [{
				x: 0,
				y: 0,
				width,
				height,
			}];
		}
		const sourceRects = context.incremental.dirtyRects;
		const sourceWidth = Math.max(1, Math.floor(context.attachments.width));
		const sourceHeight = Math.max(1, Math.floor(context.attachments.height));
		const scaleX = width / sourceWidth;
		const scaleY = height / sourceHeight;
		const resolved: Array<{ x: number; y: number; width: number; height: number }> = [];
		for (const rect of sourceRects) {
			const minX = Math.max(0, Math.floor(rect.x * scaleX));
			const minY = Math.max(0, Math.floor(rect.y * scaleY));
			const maxX = Math.min(
				width,
				Math.ceil((rect.x + rect.width) * scaleX)
			);
			const maxY = Math.min(
				height,
				Math.ceil((rect.y + rect.height) * scaleY)
			);
			const rectWidth = maxX - minX;
			const rectHeight = maxY - minY;
			if (rectWidth <= 0 || rectHeight <= 0) {
				continue;
			}
			resolved.push({
				x: minX,
				y: minY,
				width: rectWidth,
				height: rectHeight,
			});
		}
		return resolved;
	}

	private _resolvePacketsForRect(
		context: FrameContext,
		packets: DrawPacket[],
		rect: { x: number; y: number; width: number; height: number }
	): DrawPacket[] {
		const spatialIndex = context.scene.spatialIndex;
		if (!spatialIndex) {
			return packets;
		}
		if (packets === context.scene.opaquePackets) {
			return spatialIndex.queryOpaquePackets(rect);
		}
		if (packets === context.scene.transparentPackets) {
			return spatialIndex.queryTransparentPackets(rect);
		}
		return packets;
	}

	private _setScissorRect(
		x: number,
		y: number,
		width: number,
		height: number,
		viewportHeight: number
	): void {
		const resolvedWidth = Math.max(0, Math.floor(width));
		const resolvedHeight = Math.max(0, Math.floor(height));
		if (resolvedWidth <= 0 || resolvedHeight <= 0) {
			return;
		}
		const resolvedX = Math.max(0, Math.floor(x));
		const resolvedY = Math.max(0, Math.floor(y));
		const maxY = Math.max(1, Math.floor(viewportHeight));
		const scissorY = Math.max(0, maxY - (resolvedY + resolvedHeight));
		this._gl.scissor(resolvedX, scissorY, resolvedWidth, resolvedHeight);
	}

	private _drawFullscreenTrianglesWithDirtyScissor(
		viewportWidth: number,
		viewportHeight: number,
		context: FrameContext | null | undefined
	): void {
		const gl = this._gl;
		if (!this._isIncrementalPartial(context)) {
			gl.disable(gl.SCISSOR_TEST);
			gl.drawArrays(gl.TRIANGLES, 0, 3);
			return;
		}
		const dirtyRects = this._resolveDirtyRects(
			context,
			viewportWidth,
			viewportHeight
		);
		if (dirtyRects.length === 0) {
			return;
		}
		gl.enable(gl.SCISSOR_TEST);
		for (const rect of dirtyRects) {
			this._setScissorRect(
				rect.x,
				rect.y,
				rect.width,
				rect.height,
				viewportHeight
			);
			gl.drawArrays(gl.TRIANGLES, 0, 3);
		}
		gl.disable(gl.SCISSOR_TEST);
	}

	private _renderShadows(context: FrameContext): void {
		renderWebGLShadows(this as unknown as WebGLShadowPassHost, context);
	}

	private _renderShadowSlice(
		shadowProgram: WebGLShadowDepthProgram,
		packets: DrawPacket[],
		shadow: WebGLShadowData | undefined,
		tileIndex: number
	): void {
		renderWebGLShadowSlice(
			this as unknown as WebGLShadowPassHost,
			shadowProgram,
			packets,
			shadow,
			tileIndex
		);
	}

	private _drawShadowPacket(
		shadowProgram: WebGLShadowDepthProgram,
		packet: DrawPacket,
		viewProjectionMatrix: Matrix4
	): void {
		drawWebGLShadowPacket(
			this as unknown as WebGLShadowPassHost,
			shadowProgram,
			packet,
			viewProjectionMatrix
		);
	}

	private _ensureShadowTargets(tileSize: number): void {
		if (
			this._shadowFramebuffer &&
			this._shadowAtlasTexture &&
			this._shadowAtlasTileSize === tileSize
		) {
			return;
		}

		const atlasWidth = tileSize * WEBGL_SHADOW_ATLAS_COLUMNS;
		const atlasHeight = tileSize * WEBGL_SHADOW_ATLAS_ROWS;
		if (atlasWidth > this._maxTextureSize || atlasHeight > this._maxTextureSize) {
			throw new Error(
				`WebGL shadow atlas ${atlasWidth}x${atlasHeight} exceeds MAX_TEXTURE_SIZE=${this._maxTextureSize}`
			);
		}

		this._destroyShadowTargets();
		const gl = this._gl;
		const shadowTexture = gl.createTexture();
		const shadowFramebuffer = gl.createFramebuffer();
		if (!shadowTexture || !shadowFramebuffer) {
			if (shadowTexture) gl.deleteTexture(shadowTexture);
			if (shadowFramebuffer) gl.deleteFramebuffer(shadowFramebuffer);
			throw new Error("Failed to create WebGL shadow atlas targets");
		}

		gl.bindTexture(gl.TEXTURE_2D, shadowTexture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			gl.DEPTH_COMPONENT24,
			atlasWidth,
			atlasHeight,
			0,
			gl.DEPTH_COMPONENT,
			gl.UNSIGNED_INT,
			null
		);
		gl.bindTexture(gl.TEXTURE_2D, null);

		gl.bindFramebuffer(gl.FRAMEBUFFER, shadowFramebuffer);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.DEPTH_ATTACHMENT,
			gl.TEXTURE_2D,
			shadowTexture,
			0
		);
		gl.drawBuffers([gl.NONE]);
		gl.readBuffer(gl.NONE);
		const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		if (status !== gl.FRAMEBUFFER_COMPLETE) {
			gl.deleteFramebuffer(shadowFramebuffer);
			gl.deleteTexture(shadowTexture);
			throw new Error(
				`WebGL shadow framebuffer is incomplete (status=0x${status.toString(16)})`
			);
		}

		this._shadowFramebuffer = shadowFramebuffer;
		this._shadowAtlasTexture = shadowTexture;
		this._shadowAtlasTileSize = tileSize;
	}

	private _destroyShadowTargets(): void {
		const gl = this._gl;
		if (this._shadowFramebuffer) {
			gl.deleteFramebuffer(this._shadowFramebuffer);
			this._shadowFramebuffer = null;
		}
		if (this._shadowAtlasTexture) {
			gl.deleteTexture(this._shadowAtlasTexture);
			this._shadowAtlasTexture = null;
		}
		this._shadowAtlasTileSize = 0;
	}

	private _renderPackets(
		context: FrameContext,
		packets: DrawPacket[],
		transparent: boolean
	): void {
		renderWebGLPackets(
			this as unknown as WebGLScenePassHost,
			context,
			packets,
			transparent
		);
	}

	private _drawPacket(
		sceneProgram: WebGLSceneProgram,
		packet: DrawPacket,
		transparentPass: boolean,
		context: FrameContext
	): void {
		drawWebGLPacket(
			this as unknown as WebGLScenePassHost,
			sceneProgram,
			packet,
			transparentPass,
			context
		);
	}

	private _bindShaderMaterialTextures(
		sceneProgram: WebGLSceneProgram,
		material: Material
	): void {
		bindWebGLShaderMaterialTextures(
			this as unknown as WebGLScenePassHost,
			sceneProgram,
			material
		);
	}

	private _renderParticles(context: FrameContext): void {
		renderWebGLParticles(this as unknown as WebGLParticlePassHost, context);
	}

	private _writeParticleInstances(batch: ParticleRenderBatch): number {
		return writeWebGLParticleInstances(
			this as unknown as WebGLParticlePassHost,
			batch
		);
	}

	private _ensureParticleResources(): void {
		ensureWebGLParticleResources(this as unknown as WebGLParticlePassHost);
	}

	private _ensureParticleCapacity(requiredInstances: number): void {
		ensureWebGLParticleCapacity(
			this as unknown as WebGLParticlePassHost,
			requiredInstances
		);
	}

	private _bindParticleInstanceAttributes(): void {
		bindWebGLParticleInstanceAttributes(
			this as unknown as WebGLParticlePassHost
		);
	}

	private _destroyParticleResources(): void {
		destroyWebGLParticleResources(this as unknown as WebGLParticlePassHost);
	}

	private _bindGlobalUniforms(
		sceneProgram: WebGLSceneProgram,
		context: FrameContext
	): void {
		bindWebGLGlobalUniforms(
			this as unknown as WebGLGlobalUniformBinderHost,
			sceneProgram,
			context
		);
	}

	private _uploadSHAmbientCoefficients(
		coeffs: SHCoefficients | null | undefined
	): boolean {
		return uploadWebGLSHAmbientCoefficients(
			this as unknown as WebGLSHAmbientUploadHost,
			coeffs
		);
	}

	private _renderSkybox(context: FrameContext): void {
		const skyboxTexture = context.scene.skybox;
		if (!skyboxTexture || !this._fullscreenVao) return;

		const gl = this._gl;
		const skyboxProgram = this._programs.getSkyboxProgram();
		const resolved = this._textures.getSkyboxTexture(skyboxTexture);
		const view = context.camera.viewMatrix.elements;
		const isOrthographic = context.camera.type === CameraType.Orthographic;
		const tanHalfFov =
			isOrthographic ? 0 : Math.tan((context.camera.fov * Math.PI) / 360);
		const aspect = context.camera.aspectRatio || this._width / this._height;

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._sceneFramebuffer);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
		gl.useProgram(skyboxProgram.program);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.BLEND);
		gl.disable(gl.DEPTH_TEST);
		gl.depthMask(false);

		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, resolved.texture);
		if (skyboxProgram.uniforms.skyboxMap) {
			gl.uniform1i(skyboxProgram.uniforms.skyboxMap, 0);
		}
		if (skyboxProgram.uniforms.skyboxBasisRight) {
			gl.uniform4f(
				skyboxProgram.uniforms.skyboxBasisRight,
				view[0][0],
				view[0][1],
				view[0][2],
				tanHalfFov
			);
		}
		if (skyboxProgram.uniforms.skyboxBasisUp) {
			gl.uniform4f(
				skyboxProgram.uniforms.skyboxBasisUp,
				view[1][0],
				view[1][1],
				view[1][2],
				aspect
			);
		}
		if (skyboxProgram.uniforms.skyboxBasisBackward) {
			gl.uniform3f(
				skyboxProgram.uniforms.skyboxBasisBackward,
				view[2][0],
				view[2][1],
				view[2][2]
			);
		}
		if (skyboxProgram.uniforms.skyboxIsOrthographic) {
			gl.uniform1f(
				skyboxProgram.uniforms.skyboxIsOrthographic,
				isOrthographic ? 1 : 0
			);
		}
		if (skyboxProgram.uniforms.skyboxMapIsLinear) {
			gl.uniform1i(
				skyboxProgram.uniforms.skyboxMapIsLinear,
				resolved.isLinear ? 1 : 0
			);
		}
		gl.drawArrays(gl.TRIANGLES, 0, 3);

		gl.depthMask(true);
		gl.enable(gl.DEPTH_TEST);
		gl.bindVertexArray(null);
	}

	private _applySSAO(options: SSAOOptions | undefined, context: FrameContext): void {
		if (
			!this._postFramebuffer ||
			!this._sceneColorTexture ||
			!this._postColorTexture ||
			!this._sceneMotionTexture ||
			!this._sceneNormalTexture ||
			!this._ssaoRawTexture ||
			!this._ssaoBlurTexture ||
			!this._fullscreenVao
		) {
			return;
		}

		const sourceTexture = this._presentSourceTexture ?? this._sceneColorTexture;
		if (!sourceTexture) {
			return;
		}
		const targetTexture = this._resolvePostProcessTargetTexture(sourceTexture);
		if (!targetTexture) {
			return;
		}

		const gl = this._gl;
		const rawProgram = this._programs.getSSAORawProgram();
		const blurProgram = this._programs.getSSAOBlurProgram();
		const combineProgram = this._programs.getSSAOCombineProgram();
		const radius = Math.max(1, finiteOr(options?.radius, DEFAULT_SSAO_OPTIONS.radius));
		const bias = Math.max(1e-4, finiteOr(options?.bias, DEFAULT_SSAO_OPTIONS.bias));
		const intensity = Math.max(
			0,
			finiteOr(options?.intensity, DEFAULT_SSAO_OPTIONS.intensity)
		);
		const blurRadius = clamp(
			finiteOr(options?.blurRadius, DEFAULT_SSAO_OPTIONS.blurRadius),
			1,
			4
		);
		const blurSharpness = Math.max(
			1e-3,
			finiteOr(options?.blurSharpness, DEFAULT_SSAO_OPTIONS.blurSharpness)
		);
		const samples = clamp(
			Math.round(finiteOr(options?.samples, DEFAULT_SSAO_OPTIONS.samples)),
			4,
			48
		);
		const isOrthographic = context.camera.type === CameraType.Orthographic;
		const tanHalfFov = isOrthographic ? 0 : Math.tan((context.camera.fov * Math.PI) / 360);
		const aspect =
			context.camera.aspectRatio || this._width / Math.max(this._height, 1);
		const fullInvW = 1 / Math.max(this._width, 1);
		const fullInvH = 1 / Math.max(this._height, 1);
		const aoWidth = Math.max(
			1,
			Math.floor(this._width / Math.max(this._targetSSAODownsample, 1))
		);
		const aoHeight = Math.max(
			1,
			Math.floor(this._height / Math.max(this._targetSSAODownsample, 1))
		);
		const aoInvW = 1 / aoWidth;
		const aoInvH = 1 / aoHeight;
		this._ssaoFrameIndex = (this._ssaoFrameIndex + 1) % 1024;
		const frameJitter = this._ssaoFrameIndex / 1024;
		const view = context.camera.viewMatrix.elements;
		const cameraPosition = context.camera.getWorldPosition();

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._postFramebuffer);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);

		this._bindPostSingleColorTarget(this._ssaoRawTexture);
		gl.viewport(0, 0, aoWidth, aoHeight);
		gl.useProgram(rawProgram.program);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, this._sceneNormalTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, this._sceneMotionTexture);
		if (rawProgram.uniforms.normalMap) gl.uniform1i(rawProgram.uniforms.normalMap, 0);
		if (rawProgram.uniforms.depthMap) gl.uniform1i(rawProgram.uniforms.depthMap, 1);
		if (rawProgram.uniforms.invSize)
			gl.uniform4f(rawProgram.uniforms.invSize, fullInvW, fullInvH, aoInvW, aoInvH);
		if (rawProgram.uniforms.gtao)
			gl.uniform4f(rawProgram.uniforms.gtao, radius, bias, intensity, samples);
		if (rawProgram.uniforms.blurProj)
			gl.uniform4f(
				rawProgram.uniforms.blurProj,
				blurRadius,
				blurSharpness,
				tanHalfFov,
				aspect
			);
		if (rawProgram.uniforms.pass)
			gl.uniform4f(
				rawProgram.uniforms.pass,
				1,
				0,
				isOrthographic ? 1 : 0,
				frameJitter
			);
		if (rawProgram.uniforms.cameraPosition)
			gl.uniform3f(
				rawProgram.uniforms.cameraPosition,
				cameraPosition.x,
				cameraPosition.y,
				cameraPosition.z
			);
		if (rawProgram.uniforms.basisRight)
			gl.uniform3f(
				rawProgram.uniforms.basisRight,
				view[0][0],
				view[0][1],
				view[0][2]
			);
		if (rawProgram.uniforms.basisUp)
			gl.uniform3f(rawProgram.uniforms.basisUp, view[1][0], view[1][1], view[1][2]);
		if (rawProgram.uniforms.basisBackward)
			gl.uniform3f(
				rawProgram.uniforms.basisBackward,
				view[2][0],
				view[2][1],
				view[2][2]
			);
		this._drawFullscreenTrianglesWithDirtyScissor(
			aoWidth,
			aoHeight,
			context
		);

		this._bindPostSingleColorTarget(this._ssaoBlurTexture);
		gl.viewport(0, 0, aoWidth, aoHeight);
		gl.useProgram(blurProgram.program);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, this._ssaoRawTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, this._sceneMotionTexture);
		if (blurProgram.uniforms.sourceMap) gl.uniform1i(blurProgram.uniforms.sourceMap, 0);
		if (blurProgram.uniforms.depthMap) gl.uniform1i(blurProgram.uniforms.depthMap, 1);
		if (blurProgram.uniforms.invSize)
			gl.uniform4f(blurProgram.uniforms.invSize, fullInvW, fullInvH, aoInvW, aoInvH);
		if (blurProgram.uniforms.blurProj)
			gl.uniform4f(
				blurProgram.uniforms.blurProj,
				blurRadius,
				blurSharpness,
				tanHalfFov,
				aspect
			);
		if (blurProgram.uniforms.pass)
			gl.uniform4f(
				blurProgram.uniforms.pass,
				1,
				0,
				isOrthographic ? 1 : 0,
				frameJitter
			);
		this._drawFullscreenTrianglesWithDirtyScissor(
			aoWidth,
			aoHeight,
			context
		);

		this._bindPostSingleColorTarget(this._ssaoRawTexture);
		gl.viewport(0, 0, aoWidth, aoHeight);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, this._ssaoBlurTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, this._sceneMotionTexture);
		if (blurProgram.uniforms.pass)
			gl.uniform4f(
				blurProgram.uniforms.pass,
				0,
				1,
				isOrthographic ? 1 : 0,
				frameJitter
			);
		this._drawFullscreenTrianglesWithDirtyScissor(
			this._width,
			this._height,
			this._activeContext
		);

		this._bindPostSingleColorTarget(targetTexture);
		gl.viewport(0, 0, this._width, this._height);
		gl.useProgram(combineProgram.program);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, this._ssaoRawTexture);
		if (combineProgram.uniforms.sceneColor)
			gl.uniform1i(combineProgram.uniforms.sceneColor, 0);
		if (combineProgram.uniforms.aoMap) gl.uniform1i(combineProgram.uniforms.aoMap, 1);
		if (combineProgram.uniforms.invSize)
			gl.uniform4f(
				combineProgram.uniforms.invSize,
				fullInvW,
				fullInvH,
				aoInvW,
				aoInvH
			);
		this._drawFullscreenTrianglesWithDirtyScissor(
			this._width,
			this._height,
			this._activeContext
		);
		gl.bindVertexArray(null);

		this._presentSourceTexture = targetTexture;
	}

	private _applyFog(options?: FogOptions): void {
		if (
			!this._sceneMotionTexture ||
			!this._postFramebuffer ||
			!this._sceneColorTexture ||
			!this._postColorTexture ||
			!this._fullscreenVao
		) {
			return;
		}
		const sourceTexture = this._presentSourceTexture ?? this._sceneColorTexture;
		if (!sourceTexture) {
			return;
		}
		const targetTexture = this._resolvePostProcessTargetTexture(sourceTexture);
		if (!targetTexture) {
			return;
		}

		this._updateFogParams(options, true);

		const gl = this._gl;
		const fogProgram = this._programs.getFogProgram();
		gl.bindFramebuffer(gl.FRAMEBUFFER, this._postFramebuffer);
		this._bindPostSingleColorTarget(targetTexture);
		gl.viewport(0, 0, this._width, this._height);
		gl.useProgram(fogProgram.program);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, this._sceneMotionTexture);

		const uniforms = fogProgram.uniforms;
		if (uniforms.sceneColor) {
			gl.uniform1i(uniforms.sceneColor, 0);
		}
		if (uniforms.motionDepthMap) {
			gl.uniform1i(uniforms.motionDepthMap, 1);
		}
		if (uniforms.fogParams0) {
			gl.uniform4fv(uniforms.fogParams0, this._fogParams0);
		}
		if (uniforms.fogParams1) {
			gl.uniform4fv(uniforms.fogParams1, this._fogParams1);
		}
		this._drawFullscreenTrianglesWithDirtyScissor(
			this._width,
			this._height,
			this._activeContext
		);
		gl.bindVertexArray(null);
		this._presentSourceTexture = targetTexture;
	}

	private _applyMotionBlur(options?: MotionBlurOptions): void {
		if (
			!this._sceneMotionTexture ||
			!this._postFramebuffer ||
			!this._sceneColorTexture ||
			!this._postColorTexture ||
			!this._fullscreenVao
		) {
			return;
		}
		const sourceTexture = this._presentSourceTexture ?? this._sceneColorTexture;
		if (!sourceTexture) {
			return;
		}
		const targetTexture = this._resolvePostProcessTargetTexture(sourceTexture);
		if (!targetTexture) {
			return;
		}

		const shutterScale = sanitizeFiniteClamped(
			options?.shutterScale,
			DEFAULT_MOTION_BLUR_OPTIONS.shutterScale,
			MOTION_BLUR_SHUTTER_SCALE_RANGE[0],
			MOTION_BLUR_SHUTTER_SCALE_RANGE[1]
		);
		const maxSamples = clamp(
			Math.round(
				finiteOr(options?.maxSamples, DEFAULT_MOTION_BLUR_OPTIONS.maxSamples)
			),
			MOTION_BLUR_MAX_SAMPLES_RANGE[0],
			MOTION_BLUR_MAX_SAMPLES_RANGE[1]
		);
		const velocityClamp = sanitizeFiniteClamped(
			options?.velocityClamp,
			DEFAULT_MOTION_BLUR_OPTIONS.velocityClamp,
			MOTION_BLUR_VELOCITY_CLAMP_RANGE[0],
			MOTION_BLUR_VELOCITY_CLAMP_RANGE[1]
		);
		const depthReject = sanitizeFiniteClamped(
			options?.depthReject,
			DEFAULT_MOTION_BLUR_OPTIONS.depthReject,
			MOTION_BLUR_DEPTH_REJECT_RANGE[0],
			MOTION_BLUR_DEPTH_REJECT_RANGE[1]
		);
		const centerWeight = sanitizeFiniteClamped(
			options?.centerWeight,
			DEFAULT_MOTION_BLUR_OPTIONS.centerWeight,
			MOTION_BLUR_CENTER_WEIGHT_RANGE[0],
			MOTION_BLUR_CENTER_WEIGHT_RANGE[1]
		);

		const gl = this._gl;
		const motionBlurProgram = this._programs.getMotionBlurProgram();
		gl.bindFramebuffer(gl.FRAMEBUFFER, this._postFramebuffer);
		this._bindPostSingleColorTarget(targetTexture);
		gl.viewport(0, 0, this._width, this._height);
		gl.useProgram(motionBlurProgram.program);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, this._sceneMotionTexture);

		const uniforms = motionBlurProgram.uniforms;
		if (uniforms.sourceMap) gl.uniform1i(uniforms.sourceMap, 0);
		if (uniforms.motionDepthMap) gl.uniform1i(uniforms.motionDepthMap, 1);
		if (uniforms.texelSize) {
			gl.uniform2f(
				uniforms.texelSize,
				1 / Math.max(1, this._width),
				1 / Math.max(1, this._height)
			);
		}
		if (uniforms.motionParams) {
			gl.uniform4f(
				uniforms.motionParams,
				shutterScale,
				maxSamples,
				velocityClamp,
				depthReject
			);
		}
		if (uniforms.centerWeight) {
			gl.uniform1f(uniforms.centerWeight, centerWeight);
		}
		this._drawFullscreenTrianglesWithDirtyScissor(
			this._width,
			this._height,
			this._activeContext
		);
		gl.bindVertexArray(null);

		this._presentSourceTexture = targetTexture;
	}

	private _applyDOF(options?: DOFOptions): void {
		if (
			!this._sceneMotionTexture ||
			!this._postFramebuffer ||
			!this._sceneColorTexture ||
			!this._postColorTexture ||
			!this._fullscreenVao
		) {
			return;
		}
		const sourceTexture = this._presentSourceTexture ?? this._sceneColorTexture;
		if (!sourceTexture) {
			return;
		}
		const targetTexture = this._resolvePostProcessTargetTexture(sourceTexture);
		if (!targetTexture) {
			return;
		}

		const focusDistance = Math.max(
			0.01,
			finiteOr(options?.focusDistance, DEFAULT_DOF_OPTIONS.focusDistance)
		);
		const focusRange = Math.max(
			0.001,
			finiteOr(options?.focusRange, DEFAULT_DOF_OPTIONS.focusRange)
		);
		const nearStrength = sanitizeFiniteClamped(
			options?.nearStrength,
			DEFAULT_DOF_OPTIONS.nearStrength,
			DOF_NEAR_FAR_STRENGTH_RANGE[0],
			DOF_NEAR_FAR_STRENGTH_RANGE[1]
		);
		const farStrength = sanitizeFiniteClamped(
			options?.farStrength,
			DEFAULT_DOF_OPTIONS.farStrength,
			DOF_NEAR_FAR_STRENGTH_RANGE[0],
			DOF_NEAR_FAR_STRENGTH_RANGE[1]
		);
		const maxBlurRadius = sanitizeFiniteClamped(
			options?.maxBlurRadius,
			DEFAULT_DOF_OPTIONS.maxBlurRadius,
			DOF_MAX_BLUR_RADIUS_RANGE[0],
			DOF_MAX_BLUR_RADIUS_RANGE[1]
		);
		const depthCurve = sanitizeFiniteClamped(
			options?.depthCurve,
			DEFAULT_DOF_OPTIONS.depthCurve,
			DOF_DEPTH_CURVE_RANGE[0],
			DOF_DEPTH_CURVE_RANGE[1]
		);
		const highlightThreshold = Math.max(
			0,
			finiteOr(options?.highlightThreshold, DEFAULT_DOF_OPTIONS.highlightThreshold)
		);
		const highlightGain = sanitizeFiniteClamped(
			options?.highlightGain,
			DEFAULT_DOF_OPTIONS.highlightGain,
			DOF_HIGHLIGHT_GAIN_RANGE[0],
			DOF_HIGHLIGHT_GAIN_RANGE[1]
		);
		const chromaticAberration = sanitizeFiniteClamped(
			options?.chromaticAberration,
			DEFAULT_DOF_OPTIONS.chromaticAberration,
			DOF_CHROMATIC_ABERRATION_RANGE[0],
			DOF_CHROMATIC_ABERRATION_RANGE[1]
		);

		const gl = this._gl;
		const dofProgram = this._programs.getDOFProgram();
		gl.bindFramebuffer(gl.FRAMEBUFFER, this._postFramebuffer);
		this._bindPostSingleColorTarget(targetTexture);
		gl.viewport(0, 0, this._width, this._height);
		gl.useProgram(dofProgram.program);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, this._sceneMotionTexture);

		const uniforms = dofProgram.uniforms;
		if (uniforms.sourceMap) gl.uniform1i(uniforms.sourceMap, 0);
		if (uniforms.motionDepthMap) gl.uniform1i(uniforms.motionDepthMap, 1);
		if (uniforms.texelSize) {
			gl.uniform2f(
				uniforms.texelSize,
				1 / Math.max(1, this._width),
				1 / Math.max(1, this._height)
			);
		}
		if (uniforms.focusParams) {
			gl.uniform4f(
				uniforms.focusParams,
				focusDistance,
				focusRange,
				nearStrength,
				farStrength
			);
		}
		if (uniforms.dofParams) {
			gl.uniform4f(
				uniforms.dofParams,
				maxBlurRadius,
				depthCurve,
				highlightThreshold,
				highlightGain
			);
		}
		if (uniforms.chromaticAberration) {
			gl.uniform1f(uniforms.chromaticAberration, chromaticAberration);
		}
		this._drawFullscreenTrianglesWithDirtyScissor(
			this._width,
			this._height,
			this._activeContext
		);
		gl.bindVertexArray(null);

		this._presentSourceTexture = targetTexture;
	}

	private _applyBloom(options?: BloomOptions): void {
		if (
			!this._postFramebuffer ||
			!this._sceneColorTexture ||
			!this._postColorTexture
		) {
			return;
		}
		if (!this._fullscreenVao) {
			return;
		}
		const sourceTexture = this._presentSourceTexture ?? this._sceneColorTexture;
		if (!sourceTexture) {
			return;
		}
		const targetTexture = this._resolvePostProcessTargetTexture(sourceTexture);
		if (!targetTexture) {
			return;
		}

		const gl = this._gl;
		const bloomProgram = this._programs.getBloomProgram();
		const threshold = Math.max(
			0,
			finiteOr(options?.threshold, DEFAULT_BLOOM_OPTIONS.threshold)
		);
		const softKnee = Math.max(
			1e-4,
			finiteOr(options?.softKnee, DEFAULT_BLOOM_OPTIONS.softKnee)
		);
		const intensity = Math.max(
			0,
			finiteOr(options?.intensity, DEFAULT_BLOOM_OPTIONS.intensity)
		);
		const radius = clamp(
			finiteOr(options?.radius, DEFAULT_BLOOM_OPTIONS.radius),
			0.5,
			4
		);

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._postFramebuffer);
		this._bindPostSingleColorTarget(targetTexture);
		gl.viewport(0, 0, this._width, this._height);
		gl.useProgram(bloomProgram.program);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		if (bloomProgram.uniforms.sourceMap) {
			gl.uniform1i(bloomProgram.uniforms.sourceMap, 0);
		}
		if (bloomProgram.uniforms.texelSize) {
			gl.uniform2f(
				bloomProgram.uniforms.texelSize,
				1 / Math.max(1, this._width),
				1 / Math.max(1, this._height)
			);
		}
		if (bloomProgram.uniforms.bloomParams) {
			gl.uniform4f(
				bloomProgram.uniforms.bloomParams,
				threshold,
				softKnee,
				intensity,
				radius
			);
		}
		this._drawFullscreenTrianglesWithDirtyScissor(
			this._width,
			this._height,
			this._activeContext
		);
		gl.bindVertexArray(null);

		this._presentSourceTexture = targetTexture;
	}

	private _applyFXAA(): void {
		if (
			!this._postFramebuffer ||
			!this._sceneColorTexture ||
			!this._postColorTexture
		) {
			return;
		}
		if (!this._fullscreenVao) {
			return;
		}
		const sourceTexture = this._presentSourceTexture ?? this._sceneColorTexture;
		if (!sourceTexture) {
			return;
		}
		const targetTexture = this._resolvePostProcessTargetTexture(sourceTexture);
		if (!targetTexture) {
			return;
		}

		const gl = this._gl;
		const fxaaProgram = this._programs.getFXAAProgram();

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._postFramebuffer);
		this._bindPostSingleColorTarget(targetTexture);
		gl.viewport(0, 0, this._width, this._height);
		gl.useProgram(fxaaProgram.program);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		if (fxaaProgram.uniforms.sourceMap) {
			gl.uniform1i(fxaaProgram.uniforms.sourceMap, 0);
		}
		if (fxaaProgram.uniforms.texelSize) {
			gl.uniform2f(
				fxaaProgram.uniforms.texelSize,
				1 / Math.max(1, this._width),
				1 / Math.max(1, this._height)
			);
		}
		this._drawFullscreenTrianglesWithDirtyScissor(
			this._width,
			this._height,
			this._activeContext
		);
		gl.bindVertexArray(null);

		this._presentSourceTexture = targetTexture;
	}

	private _updateFogParams(options: FogOptions | undefined, enabled: boolean): void {
		const source = options ?? DEFAULT_FOG_OPTIONS;
		const color = source.color ?? DEFAULT_FOG_OPTIONS.color;
		const start = Math.max(
			0,
			finiteOr(source.start, DEFAULT_FOG_OPTIONS.start)
		);
		const end = Math.max(
			start + 1e-4,
			finiteOr(source.end, DEFAULT_FOG_OPTIONS.end)
		);
		const density = Math.max(
			0,
			finiteOr(source.density, DEFAULT_FOG_OPTIONS.density)
		);
		const strength = enabled ?
			Math.max(0, finiteOr(source.strength, DEFAULT_FOG_OPTIONS.strength))
		:	0;

		this._fogParams0[0] = this._resolveFogMode(source.mode);
		this._fogParams0[1] = start;
		this._fogParams0[2] = end;
		this._fogParams0[3] = density;

		this._fogParams1[0] = clamp(
			finiteOr(color[0], DEFAULT_FOG_OPTIONS.color[0]),
			0,
			1
		);
		this._fogParams1[1] = clamp(
			finiteOr(color[1], DEFAULT_FOG_OPTIONS.color[1]),
			0,
			1
		);
		this._fogParams1[2] = clamp(
			finiteOr(color[2], DEFAULT_FOG_OPTIONS.color[2]),
			0,
			1
		);
		this._fogParams1[3] = strength;
	}

	private _resolveFogMode(mode: FogOptions["mode"] | undefined): number {
		switch (mode) {
			case "exp":
				return 1;
			case "exp2":
				return 2;
			default:
				return 0;
		}
	}

	private _applyInteractionOutline(context: FrameContext): void {
		if (
			!this._postFramebuffer ||
			!this._sceneColorTexture ||
			!this._postColorTexture
		) {
			return;
		}
		if (!this._fullscreenVao) {
			return;
		}
		const state = context.transient.get(
			INTERACTION_TRANSIENT_STATE_KEY
		) as InteractionTransientState | null | undefined;
		const selectedEntityIds = state?.selectedEntityIds ?? [];
		if (selectedEntityIds.length === 0) {
			return;
		}

		const circles = collectProjectedOutlineCircles(
			context,
			selectedEntityIds,
			MAX_INTERACTION_OUTLINE_CIRCLES
		);
		if (circles.length === 0) {
			return;
		}

		const sourceTexture = this._presentSourceTexture ?? this._sceneColorTexture;
		if (!sourceTexture) {
			return;
		}
		const targetTexture = this._resolvePostProcessTargetTexture(sourceTexture);
		if (!targetTexture) {
			return;
		}

		const gl = this._gl;
		const program = this._programs.getInteractionOutlineProgram();
		const circleData = this._interactionOutlineCircles;
		let writeOffset = 0;
		for (const circle of circles) {
			circleData[writeOffset] = circle.centerX;
			circleData[writeOffset + 1] = circle.centerY;
			circleData[writeOffset + 2] = circle.radius;
			circleData[writeOffset + 3] = 0;
			writeOffset += 4;
		}

		const outlineColor = state?.outline?.color ?? { r: 255, g: 196, b: 64, a: 1 };
		const colorScale =
			Math.max(outlineColor.r, outlineColor.g, outlineColor.b) > 1 ? 255 : 1;
		const linearR = sRGBToLinear(
			clamp(outlineColor.r / Math.max(1, colorScale), 0, 1)
		);
		const linearG = sRGBToLinear(
			clamp(outlineColor.g / Math.max(1, colorScale), 0, 1)
		);
		const linearB = sRGBToLinear(
			clamp(outlineColor.b / Math.max(1, colorScale), 0, 1)
		);
		const alpha = clamp(
			finiteOr(state?.outline?.opacity, 0.9) *
				finiteOr(outlineColor.a, 1),
			0,
			1
		);
		const thickness = Math.max(1, finiteOr(state?.outline?.thickness, 2));
		const shapeCode = getInteractionOutlineShapeCode(state?.outline?.shape);

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._postFramebuffer);
		this._bindPostSingleColorTarget(targetTexture);
		gl.viewport(0, 0, this._width, this._height);
		gl.useProgram(program.program);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);

		if (program.uniforms.sourceMap) {
			gl.uniform1i(program.uniforms.sourceMap, 0);
		}
		if (program.uniforms.outlineColor) {
			gl.uniform4f(program.uniforms.outlineColor, linearR, linearG, linearB, 1);
		}
		if (program.uniforms.outlineParams) {
			gl.uniform3f(program.uniforms.outlineParams, alpha, thickness, shapeCode);
		}
		if (program.uniforms.viewportSize) {
			gl.uniform2f(
				program.uniforms.viewportSize,
				Math.max(1, this._width),
				Math.max(1, this._height)
			);
		}
		if (program.uniforms.circleCount) {
			gl.uniform1i(program.uniforms.circleCount, circles.length);
		}
		if (program.uniforms.circles) {
			gl.uniform4fv(
				program.uniforms.circles,
				circleData.subarray(0, circles.length * 4)
			);
		}

		this._drawFullscreenTrianglesWithDirtyScissor(
			this._width,
			this._height,
			this._activeContext
		);
		gl.bindVertexArray(null);
		this._presentSourceTexture = targetTexture;
	}

	private _applyTAA(options?: TAAOptions): void {
		if (
			!this._sceneMotionTexture ||
			!this._postFramebuffer ||
			!this._taaHistoryTextures[0] ||
			!this._fullscreenVao
		) {
			return;
		}

		const gl = this._gl;
		const taaProgram = this._programs.getTAAProgram();
		const historyIndex = this._taaHistoryIndex;
		const currentHistory = this._taaHistoryTextures[historyIndex];
		const nextHistory = this._taaHistoryTextures[1 - historyIndex];
		const currentMotionHistory = this._taaMotionHistoryTextures[historyIndex];
		const nextMotionHistory = this._taaMotionHistoryTextures[1 - historyIndex];
		const sourceTexture = this._presentSourceTexture ?? this._sceneColorTexture;
		if (!sourceTexture) {
			return;
		}

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._postFramebuffer);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT0,
			gl.TEXTURE_2D,
			nextHistory!,
			0
		);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT1,
			gl.TEXTURE_2D,
			nextMotionHistory!,
			0
		);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);

		gl.viewport(0, 0, this._width, this._height);
		gl.useProgram(taaProgram.program);
		gl.bindVertexArray(this._fullscreenVao);

		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);

		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, currentHistory!);
		gl.activeTexture(gl.TEXTURE2);
		gl.bindTexture(gl.TEXTURE_2D, this._sceneMotionTexture);
		gl.activeTexture(gl.TEXTURE3);
		gl.bindTexture(gl.TEXTURE_2D, currentMotionHistory!);

		const uniforms = taaProgram.uniforms;
		if (uniforms.sceneColor) gl.uniform1i(uniforms.sceneColor, 0);
		if (uniforms.historyMap) gl.uniform1i(uniforms.historyMap, 1);
		if (uniforms.motionMap) gl.uniform1i(uniforms.motionMap, 2);
		if (uniforms.motionHistory) gl.uniform1i(uniforms.motionHistory, 3);
		if (uniforms.texelSize) {
			gl.uniform2f(
				uniforms.texelSize,
				1 / Math.max(1, this._width),
				1 / Math.max(1, this._height)
			);
		}

		const weight = sanitizeFiniteClamped(
			options?.historyWeight,
			DEFAULT_TAA_OPTIONS.historyWeight,
			TAA_HISTORY_WEIGHT_RANGE[0],
			TAA_HISTORY_WEIGHT_RANGE[1]
		);
		const depthThreshold = sanitizeFiniteClamped(
			options?.disocclusionDepthThreshold,
			DEFAULT_TAA_OPTIONS.disocclusionDepthThreshold,
			TAA_DEPTH_THRESHOLD_RANGE[0],
			TAA_DEPTH_THRESHOLD_RANGE[1]
		);
		const motionFactor = sanitizeFiniteClamped(
			options?.motionFactor,
			DEFAULT_TAA_OPTIONS.motionFactor,
			TAA_MOTION_FACTOR_RANGE[0],
			TAA_MOTION_FACTOR_RANGE[1]
		);
		const varianceClampGamma = sanitizeFiniteClamped(
			options?.varianceClampGamma,
			DEFAULT_TAA_OPTIONS.varianceClampGamma,
			TAA_VARIANCE_GAMMA_RANGE[0],
			TAA_VARIANCE_GAMMA_RANGE[1]
		);
		const sharpen = sanitizeFiniteClamped(
			options?.sharpen,
			DEFAULT_TAA_OPTIONS.sharpen,
			TAA_SHARPEN_RANGE[0],
			TAA_SHARPEN_RANGE[1]
		);

		if (uniforms.historyWeight) gl.uniform1f(uniforms.historyWeight, weight);
		if (uniforms.depthThreshold)
			gl.uniform1f(uniforms.depthThreshold, depthThreshold);
		if (uniforms.motionFactor) gl.uniform1f(uniforms.motionFactor, motionFactor);
		if (uniforms.varianceClampGamma)
			gl.uniform1f(uniforms.varianceClampGamma, varianceClampGamma);
		if (uniforms.sharpen) gl.uniform1f(uniforms.sharpen, sharpen);
		if (uniforms.historyValid)
			gl.uniform1f(uniforms.historyValid, this._taaHistoryValid ? 1.0 : 0.0);

		gl.drawArrays(gl.TRIANGLES, 0, 3);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT1,
			gl.TEXTURE_2D,
			null,
			0
		);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
		gl.bindVertexArray(null);

		this._taaHistoryIndex = 1 - historyIndex;
		this._taaHistoryValid = true;
		this._presentSourceTexture = nextHistory;
	}

	private _present(
		applyGamma: boolean,
		context: FrameContext | null = this._activeContext
	): void {
		const sourceTexture = this._presentSourceTexture ?? this._sceneColorTexture;
		if (!sourceTexture || !this._fullscreenVao) return;
		const gl = this._gl;
		const presentProgram = this._programs.getPresentProgram();
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		gl.viewport(0, 0, this._width, this._height);
		gl.useProgram(presentProgram.program);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		if (presentProgram.uniforms.sourceMap) {
			gl.uniform1i(presentProgram.uniforms.sourceMap, 0);
		}
		if (presentProgram.uniforms.applyGamma) {
			gl.uniform1i(presentProgram.uniforms.applyGamma, applyGamma ? 1 : 0);
		}
		this._drawFullscreenTrianglesWithDirtyScissor(
			this._width,
			this._height,
			context
		);
		gl.bindVertexArray(null);
		this._presentedInFrame = true;
	}

	private _resolvePostProcessTargetTexture(
		sourceTexture: WebGLTexture
	): WebGLTexture | null {
		return resolveWebGLPostProcessTargetTexture(
			this as unknown as WebGLFrameTargetLifecycleHost,
			sourceTexture
		);
	}

	private _bindPostSingleColorTarget(texture: WebGLTexture): void {
		bindWebGLPostSingleColorTarget(
			this as unknown as WebGLFrameTargetLifecycleHost,
			texture
		);
	}

	private _ensureFrameTargets(
		width: number,
		height: number,
		ssaoDownsample: number
	): void {
		ensureWebGLFrameTargets(
			this as unknown as WebGLFrameTargetLifecycleHost,
			width,
			height,
			ssaoDownsample
		);
	}

	private _destroyFrameTargets(): void {
		destroyWebGLFrameTargets(
			this as unknown as WebGLFrameTargetLifecycleHost
		);
	}

	private _pruneModelMatrixCache(): void {
		if (this._modelMatrixCache.size <= this._modelMatrixKeysThisFrame.size) {
			return;
		}
		for (const cacheKey of this._modelMatrixCache.keys()) {
			if (!this._modelMatrixKeysThisFrame.has(cacheKey)) {
				this._modelMatrixCache.delete(cacheKey);
			}
		}
	}

	private _resolveLimit(parameter: number, fallback: number): number {
		try {
			const value = this._gl.getParameter(parameter);
			if (typeof value === "number" && Number.isFinite(value) && value > 0) {
				return Math.floor(value);
			}
		} catch {}
		return fallback;
	}

	private _setCullMode(material: Material): void {
		const gl = this._gl;
		if (material.doubleSided || material.cullMode === "none") {
			gl.disable(gl.CULL_FACE);
			return;
		}
		gl.enable(gl.CULL_FACE);
		gl.frontFace(gl.CCW);
		if (material.cullMode === "front") {
			gl.cullFace(gl.FRONT);
		} else {
			gl.cullFace(gl.BACK);
		}
	}
}
