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
import { clamp, sRGBToLinear } from "../../maths/Common";
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
	DEFAULT_BLOOM_OPTIONS,
	DEFAULT_COLOR_FILTER_OPTIONS,
	DEFAULT_DOF_OPTIONS,
	DEFAULT_FOG_OPTIONS,
	DEFAULT_MOTION_BLUR_OPTIONS,
	DEFAULT_SSAO_OPTIONS,
	INTERACTION_TRANSIENT_STATE_KEY,
	type DrawPacket,
	type BloomOptions,
	type ColorFilterOptions,
	type DOFOptions,
	type FogOptions,
	type FrameContext,
	type FramePass,
	type MotionBlurOptions,
	type ParticleRenderBatch,
	type SSAOOptions,
} from "../../pipeline/types";
import type {
	LogicalGBufferBridge,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessPassDescriptor,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "../../postprocess";
import { getBuiltinPostProcessPasses } from "../../postprocess/PostProcessPipeline";
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
	DOF_CHROMATIC_ABERRATION_RANGE,
	DOF_DEPTH_CURVE_RANGE,
	DOF_HIGHLIGHT_GAIN_RANGE,
	DOF_MAX_BLUR_RADIUS_RANGE,
	DOF_NEAR_FAR_STRENGTH_RANGE,
	IDENTITY_MATRIX4_COLUMN_MAJOR,
	MOTION_BLUR_CENTER_WEIGHT_RANGE,
	MOTION_BLUR_DEPTH_REJECT_RANGE,
	MOTION_BLUR_MAX_SAMPLES_RANGE,
	MOTION_BLUR_SHUTTER_SCALE_RANGE,
	MOTION_BLUR_VELOCITY_CLAMP_RANGE,
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
	type WebGLSceneProgram,
	type WebGLShadowDepthProgram,
	type WebGLShadowTransmittanceProgram,
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
import {
	WARMUP_POST_PROCESS_DESCRIPTORS_TRANSIENT_KEY,
	toShaderCompileError,
} from "../../pipeline/WarmupPlanner";
import {
	MAX_INTERACTION_OUTLINE_CIRCLES,
	collectProjectedOutlineCircles,
} from "../../interaction/outlineProjection";
import { getInteractionOutlineShapeCode } from "../../interaction/outlineShape";
import { WebGLClusteredLightingRuntime } from "./WebGLClusteredLightingRuntime";
import {
	clampDownsample,
	finiteOr,
	isFiniteMatrix,
	sanitizeFiniteClamped,
	toColumnMajorMat4,
	toSafeDimension,
} from "./WebGLFrameMath";
import type { WebGLFXAAContext } from "../../postprocess/passes/FastApproximateAntiAliasingPass";
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

const WEBGL_POSTPROCESS_WARMUP_HINTS_BY_PASS: Readonly<
	Record<string, readonly string[]>
> = {
	ssao: ["postprocess:ssao"],
	ssgi: ["postprocess:ssgi"],
	taa: ["postprocess:taa"],
	ssr: [],
	volumetric: ["postprocess:volumetric"],
	fog: ["postprocess:fog"],
	"motion-blur": ["postprocess:motion-blur"],
	dof: ["postprocess:dof"],
	bloom: ["postprocess:bloom"],
	tonemap: ["postprocess:tonemap"],
	"color-filter": ["postprocess:color-filter"],
	fxaa: [],
	"interaction-outline": ["postprocess:interaction-outline"],
	gamma: ["postprocess:gamma"],
};

export class WebGLFrameExecutor {
	private _gl: WebGL2RenderingContext;
	private _programs: WebGLProgramLibrary;
	private _geometry: WebGLGeometryRegistry;
	private _textures: WebGLTextureRegistry;
	private _sceneFramebuffer: WebGLFramebuffer | null = null;
	private _sceneColorTexture: WebGLTexture | null = null;
	private _sceneMotionTexture: WebGLTexture | null = null;
	private _sceneNormalTexture: WebGLTexture | null = null;
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
	private _shAmbientTexture: WebGLTexture | null = null;
	private _shAmbientTextureWidth = SH_COEFFICIENT_COUNT;
	private _shAmbientTextureHeight = 1;
	private _localLightProbeSHTexture: WebGLTexture | null = null;
	private _localLightProbeSHTextureWidth = SH_COEFFICIENT_COUNT;
	private _localLightProbeSHTextureHeight = 1;
	private _ssaoFrameIndex = 0;
	private _fogParams0 = new Float32Array(4);
	private _fogParams1 = new Float32Array(4);
	private _interactionOutlineCircles = new Float32Array(
		MAX_INTERACTION_OUTLINE_CIRCLES * 4
	);
	private _oitPassMode: 0 | 1 | 2 = 0;
	private _oitActive = false;
	private _oitHasContributors = false;
	private _oitLegacyTransparentPackets: DrawPacket[] = [];
	private _oitNeedsLegacyAfterParticles = false;
	private readonly _passHandlers: Map<FramePass["stage"], WebGLFramePassHandler>;

	constructor(
		gl: WebGL2RenderingContext,
		shaderRuntime?: ShaderRuntime,
		shaderCompileStage?: ShaderBackendCompileStage,
		shaderSourceFactory?: WebGLShaderSourceFactory
	) {
		this._gl = gl;
		this._programs = new WebGLProgramLibrary(
			gl,
			shaderRuntime,
			shaderCompileStage,
			shaderSourceFactory,
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
		const ssaoDownsample = clampDownsample(
			context.postProcess.options.ssao.downsample,
			DEFAULT_SSAO_OPTIONS.downsample
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
		
		const temporalJitter = this._temporalJitterState.next({
			enabled: context.postProcess.enabled.taa,
			isOrthographic: context.camera.type === CameraType.Orthographic,
			width: this._width,
			height: this._height,
			jitterScale: context.postProcess.options.taa.jitterScale,
			reset: context.incremental.temporalHistoryReset,
		});
		this._taaJitter[0] = temporalJitter[0];
		this._taaJitter[1] = temporalJitter[1];
		this._taaJitter[2] = temporalJitter[2];
		this._taaJitter[3] = temporalJitter[3];
		if (!context.postProcess.enabled.taa) {
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
		const internalFormat =
			desc.format === "rgba8unorm" ? gl.RGBA8 : gl.RGBA16F;
		const type = desc.format === "rgba8unorm" ? gl.UNSIGNED_BYTE : gl.HALF_FLOAT;
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
			format: desc.format,
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
						format: "rgba16float",
					}
				:	undefined,
				depth: this._sceneMotionTexture ?
					{
						semantic: "depth",
						handle: { backend: "webgl", texture: this._sceneMotionTexture },
						width,
						height,
						format: "rgba16float",
						encoding: "motion-depth.z",
					}
				:	undefined,
				motion: this._sceneMotionTexture ?
					{
						semantic: "motion",
						handle: { backend: "webgl", texture: this._sceneMotionTexture },
						width,
						height,
						format: "rgba16float",
						encoding: "motion-depth.xy",
					}
				:	undefined,
				normal: this._sceneNormalTexture ?
					{
						semantic: "normal",
						handle: { backend: "webgl", texture: this._sceneNormalTexture },
						width,
						height,
						format: "rgba16float",
						encoding: "world-normal",
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
			case "ssao":
				this._applySSAO(context.postProcess.options.ssao, context);
				return { ran: true };
			case "fog":
				this._applyFog(context.postProcess.options.fog);
				return { ran: true };
			case "motion-blur":
				this._applyMotionBlur(context.postProcess.options["motion-blur"]);
				return { ran: true };
			case "dof":
				this._applyDOF(context.postProcess.options.dof);
				return { ran: true };
			case "bloom":
				this._applyBloom(context.postProcess.options.bloom);
				return { ran: true };
			case "tonemap":
				this._applyToneMapping();
				return { ran: true };
			case "color-filter":
				this._applyColorFilter(context.postProcess.options["color-filter"]);
				return { ran: true };
			case "interaction-outline":
				this._applyInteractionOutline(context);
				return { ran: true };
			case "gamma":
				this._present(context.postProcess.enabled.gamma);
				return { ran: true };
			default:
				return { ran: false };
		}
	}

	public getPassExecutionContext(
		passId: string,
		request: PostProcessPassRequest
	): unknown {
		switch (passId) {
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
			default:
				return undefined;
		}
	}

	public endFrame(): void {
		if (!this._presentedInFrame) {
			this._present(this._activeContext?.postProcess.enabled.gamma !== false);
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
		const materialWarmupModes: ShaderTargetMode[] =
			plan.sceneTargetMode === "mrt" ? ["mrt", "single"] : ["single"];
		for (const material of plan.materials) {
			if (!(material instanceof ShaderMaterial)) {
				continue;
			}
			for (const mode of materialWarmupModes) {
				compile(`WebGLSceneProgram:material:${material.shaderId}:${mode}`, () => {
					this._programs.getSceneProgram(material, mode);
				});
			}
		}

		if (plan.enableEnvironment) {
			compile("WebGLEnvironmentProgram", () => {
				this._programs.getEnvironmentProgram();
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
		if (context.features.enableOIT) {
			compile("WebGLOITResolveProgram", () => {
				this._programs.getOITResolveProgram();
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
				case "postprocess:tonemap":
					compile("WebGLToneMappingProgram", () => {
						this._programs.getToneMappingProgram();
					});
					break;
				case "postprocess:color-filter":
					compile("WebGLColorFilterProgram", () => {
						this._programs.getColorFilterProgram();
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

		const descriptorById = this._getWarmupPostProcessDescriptorMap(context);
		const warmedPassImplementations = new Set<string>();
		for (const passId of plan.postProcessPasses) {
			if (warmedPassImplementations.has(passId)) {
				continue;
			}
			const implementation = descriptorById.get(passId)?.implementations.webgl;
			if (typeof implementation?.warmup !== "function") {
				continue;
			}
			warmedPassImplementations.add(passId);
			compile(`WebGLPostWarmup:${passId}`, () => {
				implementation.warmup?.(this._getPassWarmupExecutionContext(passId));
			});
		}

		if (
			context.postProcess.enabled.gamma &&
			!plan.postProcessPasses.includes("gamma")
		) {
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

	private _getWarmupPostProcessDescriptorMap(
		context: FrameContext
	): Map<string, PostProcessPassDescriptor> {
		const descriptors =
			context.transient?.get(WARMUP_POST_PROCESS_DESCRIPTORS_TRANSIENT_KEY) ??
			getBuiltinPostProcessPasses();
		return new Map(descriptors.map((pass) => [pass.id, pass]));
	}

	private _getPassWarmupExecutionContext(passId: string): unknown {
		switch (passId) {
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

	private _applyToneMapping(): void {
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
		const toneMappingProgram = this._programs.getToneMappingProgram();

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._postFramebuffer);
		this._bindPostSingleColorTarget(targetTexture);
		gl.viewport(0, 0, this._width, this._height);
		gl.useProgram(toneMappingProgram.program);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		if (toneMappingProgram.uniforms.sourceMap) {
			gl.uniform1i(toneMappingProgram.uniforms.sourceMap, 0);
		}
		this._drawFullscreenTrianglesWithDirtyScissor(
			this._width,
			this._height,
			this._activeContext
		);
		gl.bindVertexArray(null);

		this._presentSourceTexture = targetTexture;
	}

	private _applyColorFilter(options?: ColorFilterOptions): void {
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

		const brightness = sanitizeFiniteClamped(
			options?.brightness,
			DEFAULT_COLOR_FILTER_OPTIONS.brightness,
			-1,
			1
		);
		const saturation = sanitizeFiniteClamped(
			options?.saturation,
			DEFAULT_COLOR_FILTER_OPTIONS.saturation,
			0,
			2
		);
		const contrast = sanitizeFiniteClamped(
			options?.contrast,
			DEFAULT_COLOR_FILTER_OPTIONS.contrast,
			0,
			2
		);
		const temperature = sanitizeFiniteClamped(
			options?.temperature,
			DEFAULT_COLOR_FILTER_OPTIONS.temperature,
			-1,
			1
		);
		const tint = sanitizeFiniteClamped(
			options?.tint,
			DEFAULT_COLOR_FILTER_OPTIONS.tint,
			-1,
			1
		);

		const gl = this._gl;
		const program = this._programs.getColorFilterProgram();

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
		if (program.uniforms.filterParams0) {
			gl.uniform4f(
				program.uniforms.filterParams0,
				brightness,
				saturation,
				contrast,
				temperature
			);
		}
		if (program.uniforms.filterParams1) {
			gl.uniform4f(program.uniforms.filterParams1, tint, 0, 0, 0);
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
		);
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
