import { CameraType } from "../../cameras/Camera";
import { isShadowCastingLight } from "../../lights";
import { ParticleBlendMode } from "../../particles";
import {
	AlphaMode,
	type Material,
} from "../../materials/Material";
import { materialUsesTransmission } from "../../materials/transparency";
import {
	ShaderMaterial,
	type ShaderTargetMode,
} from "../../materials/ShaderMaterial";
import { Matrix4 } from "../../maths/Matrix4";
import type { SHCoefficients } from "../../maths/types";
import {
	resolveShadowCasterBounds,
	syncShadowMapRegistry,
	updateShadowMapMetadata,
} from "../../pipeline/ShadowMetadata";
import {
	mergeParticleShadowBounds,
	createParticleShadowVolumeGrid,
	hasParticleShadowCastingBatches,
	injectParticleBatchIntoShadowVolume,
	resolveParticleShadowCasterBounds,
} from "../../pipeline/ParticleShadowVolume";
import { selectCSMDirectionalLights } from "../../pipeline/ShadowStrategyRegistry";
import {
	PARTICLE_TRANSIENT_BATCHES_KEY,
	type DrawPacket,
	type FrameContext,
	type FramePass,
	type ParticleRenderBatch,
} from "../../pipeline/types";
import type {
	LogicalGBufferBridge,
	PostProcessPass,
	PostProcessPassExecutionContextRequest,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "../../postprocess";
import {
	DEFAULT_SSAO_OPTIONS,
	resolveSSAODownsample,
	type SSAOOptions,
	type WebGLSSAOContext,
} from "../../postprocess/passes/ScreenSpaceAmbientOcclusionPass";
import type { WebGLBloomContext } from "../../postprocess/passes/BloomPass";
import {
	DEFAULT_FOG_OPTIONS,
	resolveFogUniformParams,
	type FogOptions,
	type WebGLFogContext,
} from "../../postprocess/passes/FogPass";
import {
	DEFAULT_TAA_OPTIONS,
	type TAAOptions,
} from "../../postprocess/passes/TemporalAntiAliasingPass";
import { IBLBRDF } from "../../pipeline/IBLBRDF";
import {
	collectWebGLLights,
	type WebGLLightState,
	type WebGLClusteredLight,
	type WebGLShadowData,
} from "./WebGLLightCollector";
import { WebGLGeometryRegistry } from "./WebGLGeometryRegistry";
import { POST_PROCESS_STAGES } from "../constants";
import {
	IDENTITY_MATRIX4_COLUMN_MAJOR,
	SH_COEFFICIENT_COUNT,
	WEBGL_MAX_DIRECTIONAL_LIGHTS,
	WEBGL_MAX_SPOT_LIGHTS,
	WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS,
	WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH,
	WEBGL_PARTICLE_SHADOW_VOLUME_GRID_HEIGHT,
	WEBGL_PARTICLE_SHADOW_VOLUME_GRID_WIDTH,
	WEBGL_PARTICLE_SHADOW_VOLUME_MAX_SLICES,
	WEBGL_REFLECTION_PROBE_CAMERA_WORLD_POSITION_SCRATCH,
	WEBGL_SHADOW_CAPABILITIES,
	WEBGL_SHADOW_ATLAS_COLUMNS,
	WEBGL_SHADOW_ATLAS_ROWS,
	WEBGL_TEXTURE_UNIT_PARTICLE_SHADOW_VOLUME,
} from "./constants";
import type { ShadowMap } from "../../lights/shadows/ShadowMapping";
import {
	WebGLProgramLibrary,
	type WebGLProgramWarmupHandle,
	type WebGLSceneProgram,
	type WebGLShadowDepthProgram,
	type WebGLShadowTransmittanceProgram,
} from "./WebGLProgramLibrary";
import { WebGLTextureRegistry } from "./WebGLTextureRegistry";
import type {
	ShaderBackendCompileStage,
	ShaderRuntime,
} from "../../shaders/runtime";
import type { ShaderCompileError } from "../../shaders/runtime";
import type {
	WarmupPhaseCounters,
	WarmupPlan,
} from "../../pipeline/WarmupPlanner";
import {
	WARMUP_POST_PROCESS_DESCRIPTORS_TRANSIENT_KEY,
	toShaderCompileError,
} from "../../pipeline/WarmupPlanner";
import { WebGLClusteredLightingRuntime } from "./WebGLClusteredLightingRuntime";
import {
	isFiniteMatrix,
	toColumnMajorMat4,
	toSafeDimension,
} from "./WebGLFrameMath";
import type { WebGLFXAAContext } from "../../postprocess/passes/FastApproximateAntiAliasingPass";
import type {
	WebGLColorFilterContext,
	WebGLDepthOfFieldContext,
	WebGLGammaContext,
	WebGLInteractionOutlineContext,
	WebGLMotionBlurContext,
	WebGLScreenPostProcessContext,
	WebGLToneMappingContext,
} from "../../postprocess/passes/BuiltinFallbackPasses";
import {
	resolveMaterialUniforms,
	resolveTextureUVTransform,
} from "./WebGLMaterialUniformResolver";
import { Logger } from "../../foundation/Logger";
import {
	bindWebGLOITSingleColorTarget,
	bindWebGLPostSingleColorTarget,
	destroyWebGLFrameTargets,
	ensureWebGLFrameTargets,
	resolveWebGLPostProcessTargetTexture,
	type WebGLFrameTargetFormat,
	type WebGLFrameTargetLifecycleHost,
} from "./WebGLFrameTargetLifecycle";
import {
	bindWebGLGlobalUniforms,
	uploadWebGLLocalLightProbeCoefficients,
	uploadWebGLSHAmbientCoefficients,
	type WebGLGlobalUniformBinderHost,
	type WebGLLocalLightProbeUploadHost,
	type WebGLSHAmbientUploadHost,
} from "./WebGLGlobalUniformBinder";
import {
	drawWebGLShadowPacket,
	drawWebGLShadowTransmittancePacket,
	renderWebGLShadows,
	renderWebGLShadowSlice,
	type WebGLShadowPassHost,
} from "./WebGLShadowPass";
import {
	bindWebGLShaderMaterialUniforms,
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
import { TemporalJitterState } from "../temporal/TemporalJitterState";
import type { WebGLTAAContext } from "../../postprocess/passes/TemporalAntiAliasingPass";

type WebGLFramePassHandler = (context: FrameContext) => void;

export interface WebGLFrameExecutorOptions {
	validatePrograms?: boolean;
}

const WEBGL_POSTPROCESS_WARMUP_HINTS_BY_PASS: Readonly<
	Record<string, readonly string[]>
> = {
	ssao: [],
	ssgi: [],
	taa: ["postprocess:taa"],
	ssr: [],
	volumetric: [],
	fog: [],
	"motion-blur": [],
	dof: [],
	bloom: [],
	tonemap: [],
	"color-filter": [],
	fxaa: [],
	"interaction-outline": [],
	gamma: [],
};

export class WebGLFrameExecutor {
	private _gl: WebGL2RenderingContext;
	private _programs: WebGLProgramLibrary;
	private _geometry: WebGLGeometryRegistry;
	private _textures: WebGLTextureRegistry;
	private _sceneFramebuffer: WebGLFramebuffer | null = null;
	private _sceneColorTexture: WebGLTexture | null = null;
	private _sceneColorFormat: WebGLFrameTargetFormat = "rgba8unorm";
	private _sceneMotionTexture: WebGLTexture | null = null;
	private _sceneMotionFormat: WebGLFrameTargetFormat = "rgba8unorm";
	private _sceneNormalTexture: WebGLTexture | null = null;
	private _sceneNormalFormat: WebGLFrameTargetFormat = "rgba8unorm";
	private _sceneDepthBuffer: WebGLRenderbuffer | null = null;
	private _oitFramebuffer: WebGLFramebuffer | null = null;
	private _oitAccumTexture: WebGLTexture | null = null;
	private _oitRevealTexture: WebGLTexture | null = null;
	private _shadowFramebuffer: WebGLFramebuffer | null = null;
	private _shadowAtlasTexture: WebGLTexture | null = null;
	private _shadowTransmittanceTexture: WebGLTexture | null = null;
	private _shadowAtlasTileSize = 0;
	private _shadowMvpMatrix = Matrix4.identity();
	private _particleShadowVolumeTexture: WebGLTexture | null = null;
	private _particleShadowVolumeAtlasWidth = 0;
	private _particleShadowVolumeAtlasHeight = 0;
	private _particleShadowVolumeAtlasSize = new Float32Array(2);
	private _particleShadowVolumeGridSize = new Float32Array(4);
	private _particleShadowVolumeSliceParams = new Float32Array(
		WEBGL_PARTICLE_SHADOW_VOLUME_MAX_SLICES * 4
	);
	private _particleShadowVolumePixels = new Float32Array(0);
	private _taaHistoryTextures: [WebGLTexture | null, WebGLTexture | null] = [null, null];
	private _taaMotionHistoryTextures: [WebGLTexture | null, WebGLTexture | null] = [null, null];
	private _taaHistoryIndex = 0;
	private _taaHistoryValid = false;
	private _taaJitter = new Float32Array(4); // currX, currY, prevX, prevY
	private _temporalJitterState = new TemporalJitterState();
	private _prevViewProjection: Float32Array | null = null;
	private _modelMatrixCache = new Map<string, Float32Array>();
	private _modelMatrixKeysThisFrame = new Set<string>();
	private _postFramebuffer: WebGLFramebuffer | null = null;
	private _postColorTexture: WebGLTexture | null = null;
	private _postColorFormat: WebGLFrameTargetFormat = "rgba8unorm";
	private _ssaoRawTexture: WebGLTexture | null = null;
	private _ssaoBlurTexture: WebGLTexture | null = null;
	private _ssaoColorFormat: WebGLFrameTargetFormat = "rgba8unorm";
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
	private _shAmbientTexture: WebGLTexture | null = null;
	private _shAmbientTextureWidth = SH_COEFFICIENT_COUNT;
	private _shAmbientTextureHeight = 1;
	private _localLightProbeSHTexture: WebGLTexture | null = null;
	private _localLightProbeSHTextureWidth = SH_COEFFICIENT_COUNT;
	private _localLightProbeSHTextureHeight = 1;
	private _ssaoFrameIndex = 0;
	private _fogParams0 = new Float32Array(4);
	private _fogParams1 = new Float32Array(4);
	private _oitPassMode: 0 | 1 | 2 = 0;
	private _oitActive = false;
	private _oitHasContributors = false;
	private _oitLegacyTransparentPackets: DrawPacket[] = [];
	private _oitNeedsLegacyAfterParticles = false;
	private _supportsFloatColorBuffer: boolean | null = null;
	private readonly _passHandlers: Map<FramePass["stage"], WebGLFramePassHandler>;

	constructor(
		gl: WebGL2RenderingContext,
		shaderRuntime?: ShaderRuntime,
		shaderCompileStage?: ShaderBackendCompileStage,
		options: WebGLFrameExecutorOptions = {},
	) {
		this._gl = gl;
		this._programs = new WebGLProgramLibrary(
			gl,
			shaderRuntime,
			shaderCompileStage,
			{
				validatePrograms: options.validatePrograms === true,
			},
		);
		this._geometry = new WebGLGeometryRegistry(gl);
		this._textures = new WebGLTextureRegistry(gl);
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
		this._passHandlers = this._createPassHandlers();
	}

	public beginFrame(context: FrameContext): void {
		this._activeContext = context;
		this._presentedInFrame = false;
		this._modelMatrixKeysThisFrame.clear();
		this._width = toSafeDimension(context.attachments.width);
		this._height = toSafeDimension(context.attachments.height);
		const ssaoOptions =
			context.postProcess.getOptions<SSAOOptions>("ssao") ??
			DEFAULT_SSAO_OPTIONS;
		const ssaoDownsample = resolveSSAODownsample(
			ssaoOptions.downsample
		);
		this._ensureFrameTargets(this._width, this._height, ssaoDownsample);
		this._configureOIT(context);
		this._presentSourceTexture = this._sceneColorTexture;
		this._oitPassMode = 0;
		this._oitHasContributors = false;
		this._oitLegacyTransparentPackets = [];
		this._oitNeedsLegacyAfterParticles = false;
		this._particleShadowVolumeAtlasSize.fill(0);
		this._particleShadowVolumeGridSize.fill(0);
		this._particleShadowVolumeSliceParams.fill(0);
		this._syncShadowMetadata(context);
			this._lightState = collectWebGLLights(
				context.scene.lights,
				context.features.enableLighting,
				context.features.enableShadows,
				context.shadowMaps,
				context.features.enableSH,
				context.scene.environment.lightingEnabled ?
					context.scene.environment.iblTexture
				:	null,
				context.features.enableClusteredLighting,
				context.camera.getWorldPosition(
					WEBGL_REFLECTION_PROBE_CAMERA_WORLD_POSITION_SCRATCH
				)
			);
		this._clusteredLighting.prepare(
			context,
			this._lightState,
			this._maxTextureSize
		);
		
		const taaEnabled = context.postProcess.isEnabled("taa");
		const taaOptions =
			context.postProcess.getOptions<TAAOptions>("taa") ?? DEFAULT_TAA_OPTIONS;
		const temporalJitter = this._temporalJitterState.next({
			enabled: taaEnabled,
			isOrthographic: context.camera.type === CameraType.Orthographic,
			width: this._width,
			height: this._height,
			jitterScale: taaOptions.jitterScale ?? DEFAULT_TAA_OPTIONS.jitterScale,
			reset: context.incremental.temporalHistoryReset,
		});
		this._taaJitter[0] = temporalJitter[0];
		this._taaJitter[1] = temporalJitter[1];
		this._taaJitter[2] = temporalJitter[2];
		this._taaJitter[3] = temporalJitter[3];
		if (!taaEnabled) {
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

		if (
			!incrementalPartial &&
			context.features.enableEnvironment &&
			context.scene.environment.backgroundEnabled &&
			context.scene.environment.backgroundTexture
		) {
			this._renderEnvironment(context);
		}
	}

	public executePass(pass: FramePass, context: FrameContext): void {
		const handler = this._passHandlers.get(pass.stage);
		if (!handler) {
			const key = `webgl-stage-unsupported-${pass.stage}`;
			Logger.warn(
				`[${key}] WebGL backend does not support pass "${pass.stage}" yet; skipping`,
				{ scope: "WebGLFrameExecutor", onceKey: key }
			);
			return;
		}
		handler(context);
	}

	public createPostProcessResource(
		desc: PostProcessResourceDescriptor
	): PostProcessResourceHandle {
		const gl = this._gl;
		const texture = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		const requestedFloat = desc.format !== "rgba8unorm";
		const actualFormat: WebGLFrameTargetFormat =
			requestedFloat && this._supportsWebGLFloatColorBuffer() ?
				"rgba16float"
			:	"rgba8unorm";
		if (requestedFloat && actualFormat === "rgba8unorm") {
			this._warnFloatColorFallback();
		}
		const internalFormat =
			actualFormat === "rgba8unorm" ? gl.RGBA8 : gl.RGBA16F;
		const type =
			actualFormat === "rgba8unorm" ? gl.UNSIGNED_BYTE : gl.HALF_FLOAT;
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			internalFormat,
			desc.width,
			desc.height,
			0,
			gl.RGBA,
			type,
			null
		);
		gl.bindTexture(gl.TEXTURE_2D, null);
		return {
			id: desc.id,
			backend: "webgl",
			width: desc.width,
			height: desc.height,
			format: actualFormat,
			resource: texture,
		};
	}

	public destroyPostProcessResource(handle: PostProcessResourceHandle): void {
		this._gl.deleteTexture(handle.resource as WebGLTexture | null);
	}

	public createGBufferBridge(context: FrameContext): LogicalGBufferBridge {
		const width = Math.max(1, context.attachments.width);
		const height = Math.max(1, context.attachments.height);
		return {
			width,
			height,
			normalSpace: "world",
			depthEncoding: "hardware",
			motionEncoding: "ndc-delta",
			channels: {
				color: this._sceneColorTexture ?
					{
						semantic: "color",
						handle: { backend: "webgl", texture: this._sceneColorTexture },
						width,
						height,
						format: this._sceneColorFormat,
					}
				:	undefined,
				depth: this._sceneMotionTexture ?
					{
						semantic: "depth",
						handle: { backend: "webgl", texture: this._sceneMotionTexture },
						width,
						height,
						format: this._sceneMotionFormat,
						encoding: "motion-depth.z",
					}
				:	undefined,
				motion: this._sceneMotionTexture ?
					{
						semantic: "motion",
						handle: { backend: "webgl", texture: this._sceneMotionTexture },
						width,
						height,
						format: this._sceneMotionFormat,
						encoding: "motion-depth.xy",
					}
				:	undefined,
				normal: this._sceneNormalTexture ?
					{
						semantic: "normal",
						handle: { backend: "webgl", texture: this._sceneNormalTexture },
						width,
						height,
						format: this._sceneNormalFormat,
						encoding: "encoded-world-normal",
					}
				:	undefined,
			},
			worldPosition: {
				source: "derived",
				available: !!this._sceneMotionTexture,
			},
		};
	}

	public executePostProcessPass(
		passId: string,
		request: PostProcessPassRequest
	): PostProcessPassResult {
		const context = request.frameContext;
		switch (passId) {
			default:
				void context;
				return { ran: false };
		}
	}

	public getPassExecutionContext(
		request: PostProcessPassExecutionContextRequest
	): unknown {
		if (!request.pass.builtIn) {
			return undefined;
		}
		switch (request.passId) {
			case "motion-blur": {
				const context: WebGLMotionBlurContext =
					this._createWebGLScreenPostProcessContext();
				return context;
			}
			case "dof": {
				const context: WebGLDepthOfFieldContext =
					this._createWebGLScreenPostProcessContext();
				return context;
			}
			case "tonemap": {
				const context: WebGLToneMappingContext =
					this._createWebGLScreenPostProcessContext();
				return context;
			}
			case "color-filter": {
				const context: WebGLColorFilterContext =
					this._createWebGLScreenPostProcessContext();
				return context;
			}
			case "interaction-outline": {
				const context: WebGLInteractionOutlineContext =
					this._createWebGLScreenPostProcessContext();
				return context;
			}
			case "gamma": {
				const context: WebGLGammaContext =
					this._createWebGLGammaPostProcessContext();
				return context;
			}
			case "ssao": {
				const context: WebGLSSAOContext = {
					gl: this._gl,
					programs: this._programs,
					fullscreenVao: this._fullscreenVao,
					postFramebuffer: this._postFramebuffer,
					sceneColorTexture: this._sceneColorTexture,
					sceneMotionTexture: this._sceneMotionTexture,
					sceneNormalTexture: this._sceneNormalTexture,
					ssaoRawTexture: this._ssaoRawTexture,
					ssaoBlurTexture: this._ssaoBlurTexture,
					width: this._width,
					height: this._height,
					ssaoDownsample: this._targetSSAODownsample,
					getSourceTexture: () =>
						this._presentSourceTexture ?? this._sceneColorTexture,
					resolveTargetTexture: (sourceTexture) =>
						resolveWebGLPostProcessTargetTexture(
							this as unknown as WebGLFrameTargetLifecycleHost,
							sourceTexture
						),
					bindColorTarget: (texture) => this._bindPostSingleColorTarget(texture),
					nextFrameJitter: () => this._nextSSAOFrameJitter(),
					drawFullscreen: (width, height, frameContext) =>
						this._drawFullscreenTrianglesWithDirtyScissor(
							width,
							height,
							frameContext
						),
					publishColorTexture: (texture) => {
						this._presentSourceTexture = texture;
					},
				};
				return context;
			}
			case "taa": {
				this._applyPipelineHistories(request);
				const context: WebGLTAAContext = {
					gl: this._gl,
					programs: this._programs,
					fullscreenVao: this._fullscreenVao,
					postFramebuffer: this._postFramebuffer,
					sceneColorTexture: this._sceneColorTexture,
					sceneMotionTexture: this._sceneMotionTexture,
					width: this._width,
					height: this._height,
					historyRead: request.histories.taa?.read.resource as WebGLTexture | null,
					historyWrite: request.histories.taa?.write.resource as WebGLTexture | null,
					motionHistoryRead: request.histories.motion?.read
						.resource as WebGLTexture | null,
					motionHistoryWrite: request.histories.motion?.write
						.resource as WebGLTexture | null,
					getSourceTexture: () =>
						this._presentSourceTexture ?? this._sceneColorTexture,
					resolveTargetTexture: (sourceTexture) =>
						resolveWebGLPostProcessTargetTexture(
							this as unknown as WebGLFrameTargetLifecycleHost,
							sourceTexture
						),
					publishColorTexture: (texture) => {
						this._presentSourceTexture = texture;
						this._taaHistoryValid = true;
					},
					warn: (key, message) =>
						Logger.warn(`[${key}] ${message}`, {
							scope: "WebGLFrameExecutor",
							onceKey: key,
						}),
				};
				return context;
			}
			case "fxaa": {
				const context: WebGLFXAAContext = {
					gl: this._gl,
					programs: this._programs,
					fullscreenVao: this._fullscreenVao,
					postFramebuffer: this._postFramebuffer,
					sceneColorTexture: this._sceneColorTexture,
					width: this._width,
					height: this._height,
					getSourceTexture: () =>
						this._presentSourceTexture ?? this._sceneColorTexture,
					resolveTargetTexture: (sourceTexture) =>
						resolveWebGLPostProcessTargetTexture(
							this as unknown as WebGLFrameTargetLifecycleHost,
							sourceTexture
						),
					bindColorTarget: (texture) => this._bindPostSingleColorTarget(texture),
					drawFullscreen: () =>
						this._drawFullscreenTrianglesWithDirtyScissor(
							this._width,
							this._height,
							this._activeContext
						),
					publishColorTexture: (texture) => {
						this._presentSourceTexture = texture;
					},
				};
				return context;
			}
			case "fog": {
				const context: WebGLFogContext = {
					gl: this._gl,
					programs: this._programs,
					fullscreenVao: this._fullscreenVao,
					postFramebuffer: this._postFramebuffer,
					sceneColorTexture: this._sceneColorTexture,
					sceneMotionTexture: this._sceneMotionTexture,
					width: this._width,
					height: this._height,
					getSourceTexture: () =>
						this._presentSourceTexture ?? this._sceneColorTexture,
					resolveTargetTexture: (sourceTexture) =>
						resolveWebGLPostProcessTargetTexture(
							this as unknown as WebGLFrameTargetLifecycleHost,
							sourceTexture
						),
					bindColorTarget: (texture) => this._bindPostSingleColorTarget(texture),
					drawFullscreen: () =>
						this._drawFullscreenTrianglesWithDirtyScissor(
							this._width,
							this._height,
							this._activeContext
						),
					publishColorTexture: (texture) => {
						this._presentSourceTexture = texture;
					},
				};
				return context;
			}
			case "bloom": {
				const context: WebGLBloomContext = {
					gl: this._gl,
					programs: this._programs,
					fullscreenVao: this._fullscreenVao,
					postFramebuffer: this._postFramebuffer,
					sceneColorTexture: this._sceneColorTexture,
					width: this._width,
					height: this._height,
					getSourceTexture: () =>
						this._presentSourceTexture ?? this._sceneColorTexture,
					resolveTargetTexture: (sourceTexture) =>
						resolveWebGLPostProcessTargetTexture(
							this as unknown as WebGLFrameTargetLifecycleHost,
							sourceTexture
						),
					bindColorTarget: (texture) => this._bindPostSingleColorTarget(texture),
					drawFullscreen: () =>
						this._drawFullscreenTrianglesWithDirtyScissor(
							this._width,
							this._height,
							this._activeContext
						),
					publishColorTexture: (texture) => {
						this._presentSourceTexture = texture;
					},
				};
				return context;
			}
			default:
				return undefined;
		}
	}

	private _createWebGLScreenPostProcessContext(): WebGLScreenPostProcessContext {
		return {
			gl: this._gl,
			programs: this._programs,
			fullscreenVao: this._fullscreenVao,
			postFramebuffer: this._postFramebuffer,
			sceneColorTexture: this._sceneColorTexture,
			sceneMotionTexture: this._sceneMotionTexture,
			width: this._width,
			height: this._height,
			getSourceTexture: () => this._presentSourceTexture ?? this._sceneColorTexture,
			resolveTargetTexture: (sourceTexture) =>
				resolveWebGLPostProcessTargetTexture(
					this as unknown as WebGLFrameTargetLifecycleHost,
					sourceTexture
				),
			bindColorTarget: (texture) => this._bindPostSingleColorTarget(texture),
			drawFullscreen: () =>
				this._drawFullscreenTrianglesWithDirtyScissor(
					this._width,
					this._height,
					this._activeContext
				),
			publishColorTexture: (texture) => {
				this._presentSourceTexture = texture;
			},
		};
	}

	private _createWebGLGammaPostProcessContext(): WebGLGammaContext {
		return {
			gl: this._gl,
			programs: this._programs,
			fullscreenVao: this._fullscreenVao,
			width: this._width,
			height: this._height,
			getSourceTexture: () => this._presentSourceTexture ?? this._sceneColorTexture,
			drawFullscreen: () =>
				this._drawFullscreenTrianglesWithDirtyScissor(
					this._width,
					this._height,
					this._activeContext
				),
			markPresented: () => {
				this._presentedInFrame = true;
			},
		};
	}

	public endFrame(): void {
		if (!this._presentedInFrame) {
			this._present(this._activeContext?.postProcess.isEnabled("gamma") !== false);
		}
		this._pruneModelMatrixCache();
		this._activeContext = null;
	}

	public abortFrame(): void {
		this._activeContext = null;
		this._presentedInFrame = false;
		this._presentSourceTexture = this._sceneColorTexture;
		this._lightState = null;
		this._oitPassMode = 0;
		this._oitActive = false;
		this._oitHasContributors = false;
		this._oitLegacyTransparentPackets = [];
		this._oitNeedsLegacyAfterParticles = false;
		this._modelMatrixKeysThisFrame.clear();
	}

	public async warmup(
		context: FrameContext,
		plan: WarmupPlan
	): Promise<WarmupPhaseCounters> {
		let skipped = 0;
		const errors: ShaderCompileError[] = [];
		const handles: WebGLProgramWarmupHandle[] = [];
		let enqueueFailures = 0;

		const enqueue = async (
			label: string,
			action: () => unknown | Promise<unknown>
		): Promise<void> => {
			try {
				handles.push(...(await this._collectWarmupHandles(action)));
			} catch (error) {
				enqueueFailures++;
				errors.push(toShaderCompileError(error, "webgl", label));
			}
		};

		await enqueue("WebGLSceneProgram:builtin", () => {
			return this._warmupProgramHandle(
				"warmupSceneProgram",
				"getSceneProgram",
			);
		});
		const materialWarmupModes: ShaderTargetMode[] =
			plan.sceneTargetMode === "mrt" ? ["mrt", "single"] : ["single"];
		for (const material of plan.materials) {
			if (!(material instanceof ShaderMaterial)) {
				continue;
			}
			for (const mode of materialWarmupModes) {
				await enqueue(`WebGLSceneProgram:material:${material.shaderId}:${mode}`, () => {
					return this._warmupProgramHandle(
						"warmupSceneProgram",
						"getSceneProgram",
						material,
						mode,
					);
				});
			}
		}

		if (plan.enableEnvironment) {
			await enqueue("WebGLEnvironmentProgram", () => {
				return this._warmupProgramHandle(
					"warmupEnvironmentProgram",
					"getEnvironmentProgram",
				);
			});
		}
		if (plan.enableShadows) {
			await enqueue("WebGLShadowDepthProgram", () => {
				return this._warmupProgramHandle(
					"warmupShadowDepthProgram",
					"getShadowDepthProgram",
				);
			});
		}
		if (plan.enableParticles) {
			await enqueue("WebGLParticleProgram", () => {
				return this._warmupProgramHandle(
					"warmupParticleProgram",
					"getParticleProgram",
				);
			});
		}
		if (context.features.enableOIT) {
			await enqueue("WebGLOITResolveProgram", () => {
				return this._warmupProgramHandle(
					"warmupOITResolveProgram",
					"getOITResolveProgram",
				);
			});
		}

		const warmupHints = new Set<string>();
		for (const passId of plan.postProcessPasses) {
			const hints = WEBGL_POSTPROCESS_WARMUP_HINTS_BY_PASS[passId];
			if (!hints) {
				continue;
			}
			for (const hint of hints) {
				warmupHints.add(hint);
			}
		}
		for (const hint of warmupHints) {
			switch (hint) {
				case "postprocess:taa":
					await enqueue("WebGLTAAProgram", () => {
						return this._warmupProgramHandle(
							"warmupTAAProgram",
							"getTAAProgram",
						);
					});
					break;
				case "postprocess:fxaa":
					await enqueue("WebGLFXAAProgram", () => {
						return this._warmupProgramHandle(
							"warmupFXAAProgram",
							"getFXAAProgram",
						);
					});
					break;
				case "postprocess:tonemap":
					await enqueue("WebGLToneMappingProgram", () => {
						return this._warmupProgramHandle(
							"warmupToneMappingProgram",
							"getToneMappingProgram",
						);
					});
					break;
				case "postprocess:color-filter":
					await enqueue("WebGLColorFilterProgram", () => {
						return this._warmupProgramHandle(
							"warmupColorFilterProgram",
							"getColorFilterProgram",
						);
					});
					break;
				case "postprocess:interaction-outline":
					await enqueue("WebGLInteractionOutlineProgram", () => {
						return this._warmupProgramHandle(
							"warmupInteractionOutlineProgram",
							"getInteractionOutlineProgram",
						);
					});
					break;
				case "postprocess:motion-blur":
					await enqueue("WebGLMotionBlurProgram", () => {
						return this._warmupProgramHandle(
							"warmupMotionBlurProgram",
							"getMotionBlurProgram",
						);
					});
					break;
				case "postprocess:dof":
					await enqueue("WebGLDOFProgram", () => {
						return this._warmupProgramHandle(
							"warmupDOFProgram",
							"getDOFProgram",
						);
					});
					break;
				case "postprocess:gamma":
					await enqueue("WebGLPresentProgram", () => {
						return this._warmupProgramHandle(
							"warmupPresentProgram",
							"getPresentProgram",
						);
					});
					break;
				default:
					skipped++;
					break;
			}
		}

		const descriptorById = this._getWarmupPostProcessDescriptorMap(context);
		const warmedPassImplementations = new Set<string>();
		for (const passId of plan.postProcessPasses) {
			if (warmedPassImplementations.has(passId)) {
				continue;
			}
			const implementation = descriptorById.get(passId)?.getImplementation("webgl");
			if (typeof implementation?.warmup !== "function") {
				continue;
			}
			warmedPassImplementations.add(passId);
			await enqueue(`WebGLPostWarmup:${passId}`, async () => {
				const warmupContext = this._getPassWarmupExecutionContext(passId);
				await implementation.warmup?.(
					warmupContext,
					{
						frameContext: context,
						postProcess: context.postProcess,
						backend: "webgl",
						context: warmupContext,
						options:
							context.postProcess.getOptions(passId) ??
							descriptorById.get(passId)?.normalizeOptions({
								frameContext: context,
								postProcess: context.postProcess,
								backend: "webgl",
							}),
					}
				);
			});
		}

		if (
			context.postProcess.isEnabled("gamma") &&
			!plan.postProcessPasses.includes("gamma")
		) {
			await enqueue("WebGLPresentProgram", () => {
				return this._warmupProgramHandle(
					"warmupPresentProgram",
					"getPresentProgram",
				);
			});
		}

		const finalized = await this._finalizeWarmupHandles(handles);
		errors.push(...finalized.errors);
		const failed = enqueueFailures + finalized.failed;
		const compiled = finalized.compiled;
		const total = handles.length + enqueueFailures + skipped;

		return {
			phase: "webgl-programs",
			total,
			compiled,
			skipped,
			failed,
			errors,
		};
	}

	private async _collectWarmupHandles(
		action: () => unknown | Promise<unknown>
	): Promise<WebGLProgramWarmupHandle[]> {
		const programs = this._programs as unknown as {
			markWarmupHandles?: () => number;
			collectWarmupHandlesSince?: (
				mark: number
			) => WebGLProgramWarmupHandle[];
		};
		const mark =
			typeof programs.markWarmupHandles === "function" ?
				programs.markWarmupHandles()
			:	null;
		const result = await action();
		if (
			mark !== null &&
			typeof programs.collectWarmupHandlesSince === "function"
		) {
			const logged = programs.collectWarmupHandlesSince(mark);
			if (logged.length > 0) {
				return logged;
			}
		}
		if (isWebGLProgramWarmupHandle(result)) {
			return [result];
		}
		if (
			Array.isArray(result) &&
			result.every((entry) => isWebGLProgramWarmupHandle(entry))
		) {
			return result;
		}
		return [];
	}

	private _warmupProgramHandle(
		warmupMethod: string,
		getMethod: string,
		...args: unknown[]
	): WebGLProgramWarmupHandle {
		const programs = this._programs as unknown as Record<string, unknown>;
		const warmup = programs[warmupMethod];
		if (typeof warmup === "function") {
			return warmup.apply(this._programs, args) as WebGLProgramWarmupHandle;
		}
		const get = programs[getMethod];
		if (typeof get === "function") {
			get.apply(this._programs, args);
			return createCompletedWebGLWarmupHandle(getMethod);
		}
		throw new Error(`WebGL program library does not expose ${warmupMethod}.`);
	}

	private async _finalizeWarmupHandles(
		handles: WebGLProgramWarmupHandle[]
	): Promise<{
		compiled: number;
		failed: number;
		errors: ShaderCompileError[];
	}> {
		const pending = handles.slice();
		const errors: ShaderCompileError[] = [];
		let compiled = 0;
		let failed = 0;

		if (pending.length > 0) {
			await yieldWebGLWarmupFrame();
		}
		while (pending.length > 0) {
			let progressed = false;
			for (let i = 0; i < pending.length; ) {
				const handle = pending[i];
				let complete = false;
				try {
					complete = handle.isComplete();
				} catch (error) {
					failed++;
					errors.push(toShaderCompileError(error, "webgl", handle.label));
					pending.splice(i, 1);
					progressed = true;
					continue;
				}
				if (!complete) {
					i++;
					continue;
				}
				try {
					handle.finalize();
					compiled++;
				} catch (error) {
					failed++;
					errors.push(toShaderCompileError(error, "webgl", handle.label));
				}
				pending.splice(i, 1);
				progressed = true;
				await yieldWebGLWarmupFrame();
			}
			if (pending.length > 0 && !progressed) {
				await yieldWebGLWarmupFrame();
			}
		}

		return {
			compiled,
			failed,
			errors,
		};
	}

	private _getWarmupPostProcessDescriptorMap(
		context: FrameContext
	): Map<string, PostProcessPass> {
		const descriptors =
			context.transient?.get(WARMUP_POST_PROCESS_DESCRIPTORS_TRANSIENT_KEY) ??
			context.postProcess.getEnabledPasses().map((pass) => pass.pass);
		return new Map(descriptors.map((pass) => [pass.id, pass]));
	}

	private _getPassWarmupExecutionContext(passId: string): unknown {
		switch (passId) {
			case "motion-blur":
			case "dof":
			case "tonemap":
			case "color-filter":
			case "interaction-outline":
				return this._createWebGLScreenPostProcessContext();
			case "gamma":
				return this._createWebGLGammaPostProcessContext();
			case "ssao": {
				const context: WebGLSSAOContext = {
					gl: this._gl,
					programs: this._programs,
					fullscreenVao: this._fullscreenVao,
					postFramebuffer: this._postFramebuffer,
					sceneColorTexture: this._sceneColorTexture,
					sceneMotionTexture: this._sceneMotionTexture,
					sceneNormalTexture: this._sceneNormalTexture,
					ssaoRawTexture: this._ssaoRawTexture,
					ssaoBlurTexture: this._ssaoBlurTexture,
					width: this._width,
					height: this._height,
					ssaoDownsample: this._targetSSAODownsample,
					getSourceTexture: () =>
						this._presentSourceTexture ?? this._sceneColorTexture,
					resolveTargetTexture: (sourceTexture) =>
						resolveWebGLPostProcessTargetTexture(
							this as unknown as WebGLFrameTargetLifecycleHost,
							sourceTexture
						),
					bindColorTarget: (texture) => this._bindPostSingleColorTarget(texture),
					nextFrameJitter: () => this._nextSSAOFrameJitter(),
					drawFullscreen: (width, height, frameContext) =>
						this._drawFullscreenTrianglesWithDirtyScissor(
							width,
							height,
							frameContext
						),
					publishColorTexture: (texture) => {
						this._presentSourceTexture = texture;
					},
				};
				return context;
			}
			case "fxaa": {
				const context: WebGLFXAAContext = {
					gl: this._gl,
					programs: this._programs,
					fullscreenVao: this._fullscreenVao,
					postFramebuffer: this._postFramebuffer,
					sceneColorTexture: this._sceneColorTexture,
					width: this._width,
					height: this._height,
					getSourceTexture: () =>
						this._presentSourceTexture ?? this._sceneColorTexture,
					resolveTargetTexture: (sourceTexture) =>
						resolveWebGLPostProcessTargetTexture(
							this as unknown as WebGLFrameTargetLifecycleHost,
							sourceTexture
						),
					bindColorTarget: (texture) => this._bindPostSingleColorTarget(texture),
					drawFullscreen: () =>
						this._drawFullscreenTrianglesWithDirtyScissor(
							this._width,
							this._height,
							this._activeContext
						),
					publishColorTexture: (texture) => {
						this._presentSourceTexture = texture;
					},
				};
				return context;
			}
			case "fog": {
				const context: WebGLFogContext = {
					gl: this._gl,
					programs: this._programs,
					fullscreenVao: this._fullscreenVao,
					postFramebuffer: this._postFramebuffer,
					sceneColorTexture: this._sceneColorTexture,
					sceneMotionTexture: this._sceneMotionTexture,
					width: this._width,
					height: this._height,
					getSourceTexture: () =>
						this._presentSourceTexture ?? this._sceneColorTexture,
					resolveTargetTexture: (sourceTexture) =>
						resolveWebGLPostProcessTargetTexture(
							this as unknown as WebGLFrameTargetLifecycleHost,
							sourceTexture
						),
					bindColorTarget: (texture) => this._bindPostSingleColorTarget(texture),
					drawFullscreen: () =>
						this._drawFullscreenTrianglesWithDirtyScissor(
							this._width,
							this._height,
							this._activeContext
						),
					publishColorTexture: (texture) => {
						this._presentSourceTexture = texture;
					},
				};
				return context;
			}
			case "bloom": {
				const context: WebGLBloomContext = {
					gl: this._gl,
					programs: this._programs,
					fullscreenVao: this._fullscreenVao,
					postFramebuffer: this._postFramebuffer,
					sceneColorTexture: this._sceneColorTexture,
					width: this._width,
					height: this._height,
					getSourceTexture: () =>
						this._presentSourceTexture ?? this._sceneColorTexture,
					resolveTargetTexture: (sourceTexture) =>
						resolveWebGLPostProcessTargetTexture(
							this as unknown as WebGLFrameTargetLifecycleHost,
							sourceTexture
						),
					bindColorTarget: (texture) => this._bindPostSingleColorTarget(texture),
					drawFullscreen: () =>
						this._drawFullscreenTrianglesWithDirtyScissor(
							this._width,
							this._height,
							this._activeContext
						),
					publishColorTexture: (texture) => {
						this._presentSourceTexture = texture;
					},
				};
				return context;
			}
			default:
				return undefined;
		}
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
		if (this._localLightProbeSHTexture) {
			this._gl.deleteTexture(this._localLightProbeSHTexture);
			this._localLightProbeSHTexture = null;
		}
		if (this._particleShadowVolumeTexture) {
			this._gl.deleteTexture(this._particleShadowVolumeTexture);
			this._particleShadowVolumeTexture = null;
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
		const runPostProcess = (_context: FrameContext) => {};
		const handlers = new Map<FramePass["stage"], WebGLFramePassHandler>([
			[
				"shadow",
				(context) => {
					this._updateParticleShadowVolumes(context);
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
					if (this._oitActive) {
						this._renderOITTransparentPass(context);
						return;
					}
					this._renderPackets(context, context.scene.transparentPackets, true);
				},
			],
			[
				"particles",
				(context) => {
					if (this._oitActive) {
						this._renderOITParticlePass(context);
						return;
					}
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

	private _updateParticleShadowVolumes(context: FrameContext): void {
		this._particleShadowVolumeAtlasSize.fill(0);
		this._particleShadowVolumeGridSize.fill(0);
		this._particleShadowVolumeSliceParams.fill(0);

		const batches = context.transient.get(PARTICLE_TRANSIENT_BATCHES_KEY) as
			| readonly ParticleRenderBatch[]
			| undefined;
		const directionalShadow = this._lightState?.directionalShadows[0];
		if (
			!context.features.enableShadows ||
			!directionalShadow?.enabled ||
			!hasParticleShadowCastingBatches(batches)
		) {
			return;
		}
		if (this._maxTextureImageUnits <= WEBGL_TEXTURE_UNIT_PARTICLE_SHADOW_VOLUME) {
			Logger.warn(
				"[webgl-particle-shadow-volume-texture-units] WebGL fragment texture unit budget is too small for particle shadow volumes; disabling particle volume shadows for this frame.",
				{
					scope: "WebGLFrameExecutor",
					onceKey: "webgl-particle-shadow-volume-texture-units",
				}
			);
			return;
		}

		const atlasWidth =
			WEBGL_PARTICLE_SHADOW_VOLUME_GRID_WIDTH *
			WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS;
		const atlasRows = Math.ceil(
			(
				WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH *
				WEBGL_PARTICLE_SHADOW_VOLUME_MAX_SLICES
			) / WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS
		);
		const atlasHeight =
			WEBGL_PARTICLE_SHADOW_VOLUME_GRID_HEIGHT * atlasRows;
		if (atlasWidth > this._maxTextureSize || atlasHeight > this._maxTextureSize) {
			Logger.warn(
				`[webgl-particle-shadow-volume-atlas-limit] WebGL particle shadow volume atlas ${atlasWidth}x${atlasHeight} exceeds MAX_TEXTURE_SIZE=${this._maxTextureSize}; disabling particle volume shadows for this frame.`,
				{
					scope: "WebGLFrameExecutor",
					onceKey: "webgl-particle-shadow-volume-atlas-limit",
				}
			);
			return;
		}

		const texture = this._ensureParticleShadowVolumeTexture(
			atlasWidth,
			atlasHeight
		);
		if (!texture) {
			return;
		}

		const requiredPixels = atlasWidth * atlasHeight;
		if (this._particleShadowVolumePixels.length !== requiredPixels) {
			this._particleShadowVolumePixels = new Float32Array(requiredPixels);
		}
		this._particleShadowVolumePixels.fill(0);

		const matrices =
			directionalShadow.strategyType === "csm" ?
				directionalShadow.cascadeViewProjectionMatrices
			:	[directionalShadow.viewProjectionMatrix];
		const cascadeCount =
			directionalShadow.strategyType === "csm" ?
				Math.max(1, Math.min(4, directionalShadow.cascadeCount | 0))
			:	1;
		let activeSliceCount = 0;
		for (
			let sliceIndex = 0;
			sliceIndex < Math.min(WEBGL_PARTICLE_SHADOW_VOLUME_MAX_SLICES, cascadeCount);
			sliceIndex++
		) {
			const matrix = matrices[sliceIndex];
			if (!matrix) {
				continue;
			}
			const grid = createParticleShadowVolumeGrid({
				width: WEBGL_PARTICLE_SHADOW_VOLUME_GRID_WIDTH,
				height: WEBGL_PARTICLE_SHADOW_VOLUME_GRID_HEIGHT,
				depth: WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH,
			});
			const shadowMap = {
				viewProjectionMatrix: matrix,
			} as ShadowMap;
			for (const batch of batches ?? []) {
				injectParticleBatchIntoShadowVolume(grid, shadowMap, batch);
			}
			if (!grid.active) {
				continue;
			}
			this._packParticleShadowVolumeSlice(
				grid.density,
				sliceIndex,
				atlasWidth
			);
			const sliceOffset = sliceIndex * 4;
			this._particleShadowVolumeSliceParams[sliceOffset] = 1;
			this._particleShadowVolumeSliceParams[sliceOffset + 1] =
				sliceIndex * WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH;
			activeSliceCount++;
		}

		if (activeSliceCount <= 0) {
			this._particleShadowVolumeSliceParams.fill(0);
			return;
		}

		this._particleShadowVolumeAtlasSize[0] = atlasWidth;
		this._particleShadowVolumeAtlasSize[1] = atlasHeight;
		this._particleShadowVolumeGridSize[0] =
			WEBGL_PARTICLE_SHADOW_VOLUME_GRID_WIDTH;
		this._particleShadowVolumeGridSize[1] =
			WEBGL_PARTICLE_SHADOW_VOLUME_GRID_HEIGHT;
		this._particleShadowVolumeGridSize[2] =
			WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH;
		this._particleShadowVolumeGridSize[3] =
			WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS;

		const gl = this._gl;
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			gl.R32F,
			atlasWidth,
			atlasHeight,
			0,
			gl.RED,
			gl.FLOAT,
			this._particleShadowVolumePixels
		);
	}

	private _ensureParticleShadowVolumeTexture(
		width: number,
		height: number
	): WebGLTexture | null {
		const gl = this._gl;
		if (
			this._particleShadowVolumeTexture &&
			this._particleShadowVolumeAtlasWidth === width &&
			this._particleShadowVolumeAtlasHeight === height
		) {
			return this._particleShadowVolumeTexture;
		}
		if (this._particleShadowVolumeTexture) {
			gl.deleteTexture(this._particleShadowVolumeTexture);
		}
		const texture = gl.createTexture();
		if (!texture) {
			Logger.warn(
				"[webgl-particle-shadow-volume-create-failed] Failed to create WebGL particle shadow volume atlas; disabling particle volume shadows for this frame.",
				{
					scope: "WebGLFrameExecutor",
					onceKey: "webgl-particle-shadow-volume-create-failed",
				}
			);
			this._particleShadowVolumeTexture = null;
			this._particleShadowVolumeAtlasWidth = 0;
			this._particleShadowVolumeAtlasHeight = 0;
			return null;
		}
		this._particleShadowVolumeTexture = texture;
		this._particleShadowVolumeAtlasWidth = width;
		this._particleShadowVolumeAtlasHeight = height;

		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		return texture;
	}

	private _packParticleShadowVolumeSlice(
		density: Float32Array,
		sliceIndex: number,
		atlasWidth: number
	): void {
		const width = WEBGL_PARTICLE_SHADOW_VOLUME_GRID_WIDTH;
		const height = WEBGL_PARTICLE_SHADOW_VOLUME_GRID_HEIGHT;
		const depth = WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH;
		for (let z = 0; z < depth; z++) {
			const tileIndex = sliceIndex * depth + z;
			const tileX = tileIndex % WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS;
			const tileY = Math.floor(
				tileIndex / WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS
			);
			for (let y = 0; y < height; y++) {
				const sourceOffset = z * width * height + y * width;
				const targetOffset =
					(tileY * height + y) * atlasWidth + tileX * width;
				this._particleShadowVolumePixels.set(
					density.subarray(sourceOffset, sourceOffset + width),
					targetOffset
				);
			}
		}
	}

	private _applyPipelineHistories(request: PostProcessPassRequest): void {
		const taa = request.histories.taa;
		if (taa) {
			this._taaHistoryTextures = [
				taa.read.resource as WebGLTexture | null,
				taa.write.resource as WebGLTexture | null,
			];
			this._taaHistoryIndex = 0;
			this._taaHistoryValid = taa.valid;
		}
		const motion = request.histories.motion;
		if (motion) {
			this._taaMotionHistoryTextures = [
				motion.read.resource as WebGLTexture | null,
				motion.write.resource as WebGLTexture | null,
			];
		}
	}

	private _configureOIT(context: FrameContext): void {
		if (context.features.enableOIT !== true) {
			this._oitActive = false;
			return;
		}
		if (
			!this._oitFramebuffer ||
			!this._oitAccumTexture ||
			!this._oitRevealTexture ||
			!this._postFramebuffer ||
			!this._postColorTexture
		) {
			const key = "webgl-oit-disabled-runtime";
			Logger.warn(
				`[${key}] WebGL OIT requires float color-buffer render targets; falling back to legacy transparent rendering.`,
				{
					scope: "WebGLFrameExecutor",
					onceKey: key,
				}
			);
			this._oitActive = false;
			return;
		}
		this._oitActive = true;
	}

	private _partitionTransparentPackets(packets: DrawPacket[]): {
		oitPackets: DrawPacket[];
		legacyPackets: DrawPacket[];
	} {
		const oitPackets: DrawPacket[] = [];
		const legacyPackets: DrawPacket[] = [];
		for (const packet of packets) {
			if (
				materialUsesTransmission(packet.material) ||
				packet.material instanceof ShaderMaterial
			) {
				legacyPackets.push(packet);
				continue;
			}
			oitPackets.push(packet);
		}
		return {
			oitPackets,
			legacyPackets,
		};
	}

	private _syncShadowMetadata(context: FrameContext): void {
		const shadowLights = context.scene.lights.filter(isShadowCastingLight);
		syncShadowMapRegistry(context.shadowMaps, shadowLights);

		if (!context.features.enableShadows) {
			return;
		}

		const shadowCasterBounds = resolveShadowCasterBounds(
			context.scene.shadowCasterPackets,
			context.scene.sceneBounds
		);
		const combinedShadowCasterBounds = mergeParticleShadowBounds(
			shadowCasterBounds,
			resolveParticleShadowCasterBounds(context.scene.particleSystems)
		);
		const selectedCSMLights = selectCSMDirectionalLights(
			shadowLights,
			WEBGL_SHADOW_CAPABILITIES.maxCsmDirectionalLights
		);
		for (const light of shadowLights) {
			const shadowRenderSet = context.shadowMaps.get(light);
			if (!shadowRenderSet) continue;
			updateShadowMapMetadata(
				shadowRenderSet,
				light,
				combinedShadowCasterBounds,
				{
					camera: context.scene.camera,
					backendCapabilities: WEBGL_SHADOW_CAPABILITIES,
					allowCSMDirectionalLights: selectedCSMLights,
					onWarning: (key, message) =>
						Logger.warn(`[${key}] ${message}`, {
							scope: "WebGLFrameExecutor",
							onceKey: key,
						}),
				}
			);
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
		tileIndex: number,
		cascadeIndex: number = 0
	): void {
		renderWebGLShadowSlice(
			this as unknown as WebGLShadowPassHost,
			shadowProgram,
			packets,
			shadow,
			tileIndex,
			cascadeIndex
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

	private _drawShadowTransmittancePacket(
		shadowProgram: WebGLShadowTransmittanceProgram,
		packet: DrawPacket,
		viewProjectionMatrix: Matrix4
	): void {
		drawWebGLShadowTransmittancePacket(
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
		const transmittanceTexture = gl.createTexture();
		const shadowFramebuffer = gl.createFramebuffer();
		if (!shadowTexture || !transmittanceTexture || !shadowFramebuffer) {
			if (shadowTexture) gl.deleteTexture(shadowTexture);
			if (transmittanceTexture) gl.deleteTexture(transmittanceTexture);
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

		gl.bindTexture(gl.TEXTURE_2D, transmittanceTexture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			gl.RGBA8,
			atlasWidth,
			atlasHeight,
			0,
			gl.RGBA,
			gl.UNSIGNED_BYTE,
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
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT0,
			gl.TEXTURE_2D,
			transmittanceTexture,
			0
		);
		gl.drawBuffers([gl.NONE]);
		gl.readBuffer(gl.NONE);
		const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		if (status !== gl.FRAMEBUFFER_COMPLETE) {
			gl.deleteFramebuffer(shadowFramebuffer);
			gl.deleteTexture(shadowTexture);
			gl.deleteTexture(transmittanceTexture);
			throw new Error(
				`WebGL shadow framebuffer is incomplete (status=0x${status.toString(16)})`
			);
		}

		this._shadowFramebuffer = shadowFramebuffer;
		this._shadowAtlasTexture = shadowTexture;
		this._shadowTransmittanceTexture = transmittanceTexture;
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
		if (this._shadowTransmittanceTexture) {
			gl.deleteTexture(this._shadowTransmittanceTexture);
			this._shadowTransmittanceTexture = null;
		}
		this._shadowAtlasTileSize = 0;
	}

	private _renderPackets(
		context: FrameContext,
		packets: DrawPacket[],
		transparent: boolean,
		options: {
			framebuffer?: WebGLFramebuffer | null;
			drawBuffers?: number[];
			blendMode?: "legacy" | "oit-accum" | "oit-reveal";
			oitPassMode?: 0 | 1 | 2;
		} = {}
	): void {
		renderWebGLPackets(
			this as unknown as WebGLScenePassHost,
			context,
			packets,
			transparent,
			options
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

	private _bindShaderMaterialUniforms(
		sceneProgram: WebGLSceneProgram,
		material: Material
	): void {
		bindWebGLShaderMaterialUniforms(
			this as unknown as WebGLScenePassHost,
			sceneProgram,
			material
		);
	}

	private _renderParticles(
		context: FrameContext,
		options: {
			framebuffer?: WebGLFramebuffer | null;
			drawBuffers?: number[];
			includeBlendModes?: ParticleBlendMode[];
			oitPassMode?: 0 | 1 | 2;
		} = {}
	): void {
		renderWebGLParticles(
			this as unknown as WebGLParticlePassHost,
			context,
			options
		);
	}

	private _clearOITTargets(): void {
		if (!this._oitFramebuffer || !this._oitAccumTexture || !this._oitRevealTexture) {
			return;
		}
		const gl = this._gl;
		gl.bindFramebuffer(gl.FRAMEBUFFER, this._oitFramebuffer);
		this._bindOITSingleColorTarget(this._oitAccumTexture);
		gl.viewport(0, 0, this._width, this._height);
		gl.disable(gl.BLEND);
		gl.clearColor(0, 0, 0, 0);
		gl.clear(gl.COLOR_BUFFER_BIT);
		this._bindOITSingleColorTarget(this._oitRevealTexture);
		gl.clearColor(1, 1, 1, 1);
		gl.clear(gl.COLOR_BUFFER_BIT);
	}

	private _copySceneColorForOITResolve(context: FrameContext): boolean {
		if (
			!this._postFramebuffer ||
			!this._postColorTexture ||
			!this._sceneColorTexture ||
			!this._fullscreenVao
		) {
			return false;
		}
		const gl = this._gl;
		const copyProgram = this._programs.getCopyProgram();
		gl.bindFramebuffer(gl.FRAMEBUFFER, this._postFramebuffer);
		this._bindPostSingleColorTarget(this._postColorTexture);
		gl.viewport(0, 0, this._width, this._height);
		gl.useProgram(copyProgram.program);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, this._sceneColorTexture);
		if (copyProgram.uniforms.sourceMap) {
			gl.uniform1i(copyProgram.uniforms.sourceMap, 0);
		}
		this._drawFullscreenTrianglesWithDirtyScissor(
			this._width,
			this._height,
			context
		);
		gl.bindVertexArray(null);
		return true;
	}

	private _resolveOITComposition(context: FrameContext): void {
		if (
			!this._oitHasContributors ||
			!this._sceneFramebuffer ||
			!this._sceneColorTexture ||
			!this._postColorTexture ||
			!this._oitAccumTexture ||
			!this._oitRevealTexture ||
			!this._fullscreenVao
		) {
			return;
		}
		if (!this._copySceneColorForOITResolve(context)) {
			return;
		}
		const gl = this._gl;
		const resolveProgram = this._programs.getOITResolveProgram();
		gl.bindFramebuffer(gl.FRAMEBUFFER, this._sceneFramebuffer);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
		gl.viewport(0, 0, this._width, this._height);
		gl.useProgram(resolveProgram.program);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, this._postColorTexture);
		if (resolveProgram.uniforms.sceneColor) {
			gl.uniform1i(resolveProgram.uniforms.sceneColor, 0);
		}
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, this._oitAccumTexture);
		if (resolveProgram.uniforms.oitAccumMap) {
			gl.uniform1i(resolveProgram.uniforms.oitAccumMap, 1);
		}
		gl.activeTexture(gl.TEXTURE2);
		gl.bindTexture(gl.TEXTURE_2D, this._oitRevealTexture);
		if (resolveProgram.uniforms.oitRevealMap) {
			gl.uniform1i(resolveProgram.uniforms.oitRevealMap, 2);
		}
		this._drawFullscreenTrianglesWithDirtyScissor(
			this._width,
			this._height,
			context
		);
		gl.bindVertexArray(null);
		gl.activeTexture(gl.TEXTURE0);
		this._presentSourceTexture = this._sceneColorTexture;
	}

	private _renderOITTransparentPass(context: FrameContext): void {
		if (!this._oitActive || !this._oitFramebuffer) {
			this._renderPackets(context, context.scene.transparentPackets, true);
			return;
		}
		const { oitPackets, legacyPackets } = this._partitionTransparentPackets(
			context.scene.transparentPackets
		);
		this._oitLegacyTransparentPackets = legacyPackets;
		this._oitNeedsLegacyAfterParticles =
			(context.scene.particleSystems?.length ?? 0) > 0;
		this._oitHasContributors = false;
		if (oitPackets.length > 0) {
			this._clearOITTargets();
			// WebGL cannot assign different blend states per attachment, so OIT
			// accum and reveal are emitted as separate draws.
			this._renderPackets(context, oitPackets, true, {
				framebuffer: this._oitFramebuffer,
				drawBuffers: [this._gl.COLOR_ATTACHMENT0],
				blendMode: "oit-accum",
				oitPassMode: 1,
			});
			this._renderPackets(context, oitPackets, true, {
				framebuffer: this._oitFramebuffer,
				drawBuffers: [this._gl.COLOR_ATTACHMENT0],
				blendMode: "oit-reveal",
				oitPassMode: 2,
			});
			this._oitHasContributors = true;
		}
		if (!this._oitNeedsLegacyAfterParticles) {
			if (this._oitHasContributors) {
				this._resolveOITComposition(context);
			}
			if (this._oitLegacyTransparentPackets.length > 0) {
				this._renderPackets(
					context,
					this._oitLegacyTransparentPackets,
					true
				);
			}
			this._oitLegacyTransparentPackets = [];
			this._oitHasContributors = false;
		}
	}

	private _renderOITParticlePass(context: FrameContext): void {
		if (!this._oitActive || !this._oitFramebuffer) {
			this._renderParticles(context);
			return;
		}
		if (!this._oitHasContributors) {
			this._clearOITTargets();
		}
		// Match mesh OIT routing: alpha particles render accum/reveal
		// sequentially, additive particles remain on the legacy path.
		this._renderParticles(context, {
			framebuffer: this._oitFramebuffer,
			drawBuffers: [this._gl.COLOR_ATTACHMENT0],
			includeBlendModes: [ParticleBlendMode.Alpha],
			oitPassMode: 1,
		});
		this._renderParticles(context, {
			framebuffer: this._oitFramebuffer,
			drawBuffers: [this._gl.COLOR_ATTACHMENT0],
			includeBlendModes: [ParticleBlendMode.Alpha],
			oitPassMode: 2,
		});
		this._oitHasContributors = true;
		if (this._oitHasContributors) {
			this._resolveOITComposition(context);
		}
		if (this._oitLegacyTransparentPackets.length > 0) {
			this._renderPackets(context, this._oitLegacyTransparentPackets, true);
		}
		this._renderParticles(context, {
			includeBlendModes: [ParticleBlendMode.Additive],
		});
		this._oitLegacyTransparentPackets = [];
		this._oitHasContributors = false;
		this._oitNeedsLegacyAfterParticles = false;
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

	private _uploadLocalLightProbeCoefficients(
		probes: NonNullable<WebGLLightState["localLightProbes"]>
	): boolean {
		return uploadWebGLLocalLightProbeCoefficients(
			this as unknown as WebGLLocalLightProbeUploadHost,
			probes
		);
	}

	private _renderEnvironment(context: FrameContext): void {
		const environmentBackgroundTexture =
			context.scene.environment.backgroundTexture;
		if (!environmentBackgroundTexture || !this._fullscreenVao) return;
		const environment = context.scene.environment;

		const gl = this._gl;
		const environmentProgram = this._programs.getEnvironmentProgram();
		const resolved = this._textures.getEnvironmentTexture(environmentBackgroundTexture);
		const view = context.camera.viewMatrix.elements;
		const isOrthographic = context.camera.type === CameraType.Orthographic;
		const tanHalfFov =
			isOrthographic ? 0 : Math.tan((context.camera.fov * Math.PI) / 360);
		const aspect = context.camera.aspectRatio || this._width / this._height;

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._sceneFramebuffer);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
		gl.useProgram(environmentProgram.program);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.BLEND);
		gl.disable(gl.DEPTH_TEST);
		gl.depthMask(false);

		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, resolved.texture);
		if (environmentProgram.uniforms.environmentMap) {
			gl.uniform1i(environmentProgram.uniforms.environmentMap, 0);
		}
		if (environmentProgram.uniforms.environmentBasisRight) {
			gl.uniform4f(
				environmentProgram.uniforms.environmentBasisRight,
				view[0][0],
				view[0][1],
				view[0][2],
				tanHalfFov
			);
		}
		if (environmentProgram.uniforms.environmentBasisUp) {
			gl.uniform4f(
				environmentProgram.uniforms.environmentBasisUp,
				view[1][0],
				view[1][1],
				view[1][2],
				aspect
			);
		}
		if (environmentProgram.uniforms.environmentBasisBackward) {
			gl.uniform3f(
				environmentProgram.uniforms.environmentBasisBackward,
				view[2][0],
				view[2][1],
				view[2][2]
			);
		}
		if (environmentProgram.uniforms.environmentIsOrthographic) {
			gl.uniform1f(
				environmentProgram.uniforms.environmentIsOrthographic,
				isOrthographic ? 1 : 0
			);
		}
		if (environmentProgram.uniforms.environmentMapIsLinear) {
			gl.uniform1i(
				environmentProgram.uniforms.environmentMapIsLinear,
				resolved.isLinear ? 1 : 0
			);
		}
		if (environmentProgram.uniforms.environmentBackgroundTint) {
			gl.uniform3f(
				environmentProgram.uniforms.environmentBackgroundTint,
				environment.backgroundTintLinear.r,
				environment.backgroundTintLinear.g,
				environment.backgroundTintLinear.b
			);
		}
		if (environmentProgram.uniforms.environmentBackgroundExposure) {
			gl.uniform1f(
				environmentProgram.uniforms.environmentBackgroundExposure,
				Math.max(1e-6, environment.backgroundExposure)
			);
		}
		if (environmentProgram.uniforms.environmentBackgroundStrength) {
			gl.uniform1f(
				environmentProgram.uniforms.environmentBackgroundStrength,
				Math.max(0, environment.backgroundStrength)
			);
		}
		gl.drawArrays(gl.TRIANGLES, 0, 3);

		gl.depthMask(true);
		gl.enable(gl.DEPTH_TEST);
		gl.bindVertexArray(null);
	}

	private _updateFogParams(options: FogOptions | undefined, enabled: boolean): void {
		resolveFogUniformParams(options, enabled, this._fogParams0, this._fogParams1);
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

	private _nextSSAOFrameJitter(): number {
		this._ssaoFrameIndex = (this._ssaoFrameIndex + 1) % 1024;
		return this._ssaoFrameIndex / 1024;
	}

	private _bindOITSingleColorTarget(texture: WebGLTexture): void {
		bindWebGLOITSingleColorTarget(
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

	private _supportsWebGLFloatColorBuffer(): boolean {
		if (this._supportsFloatColorBuffer === null) {
			this._supportsFloatColorBuffer = !!this._gl.getExtension(
				"EXT_color_buffer_float"
			);
		}
		return this._supportsFloatColorBuffer;
	}

	private _warnFloatColorFallback(): void {
		const key = "webgl-hdr-float-unsupported";
		Logger.warn(
			`[${key}] EXT_color_buffer_float is unavailable; falling back to RGBA8 color, motion, and post-process attachments.`,
			{
				scope: "WebGLFrameExecutor",
				onceKey: key,
			}
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

function isWebGLProgramWarmupHandle(
	value: unknown
): value is WebGLProgramWarmupHandle {
	return (
		typeof value === "object" &&
		value !== null &&
		"label" in value &&
		typeof (value as { isComplete?: unknown }).isComplete === "function" &&
		typeof (value as { finalize?: unknown }).finalize === "function"
	);
}

function createCompletedWebGLWarmupHandle(
	label: string
): WebGLProgramWarmupHandle {
	return {
		label,
		isComplete: () => true,
		finalize: () => {},
	};
}

function yieldWebGLWarmupFrame(): Promise<void> {
	return new Promise((resolve) => {
		const requestFrame = (globalThis as {
			requestAnimationFrame?: (callback: () => void) => unknown;
		}).requestAnimationFrame;
		if (typeof requestFrame === "function") {
			requestFrame(() => resolve());
			return;
		}
		setTimeout(resolve, 0);
	});
}
