import { CameraType } from "../../cameras/Camera";
import { isShadowCastingLight } from "../../lights";
import { ParticleBlendMode } from "../../particles";
import {
	AlphaMode,
	type Material,
} from "../../materials/Material";
import {
	isMaterialTransparentPass,
	materialUsesTransmission,
} from "../../materials/transparency";
import {
	ShaderMaterial,
	type ShaderTargetMode,
} from "../../materials/ShaderMaterial";
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
import { selectCSMDirectionalLights } from "../../pipeline/ShadowMetadata";
import {
	PARTICLE_TRANSIENT_BATCHES_KEY,
	type DrawPacket,
	type FrameContext,
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
} from "../../postprocess/passes/ScreenSpaceAmbientOcclusionPass";
import {
	DEFAULT_FOG_OPTIONS,
	resolveFogUniformParams,
	type FogOptions,
} from "../../postprocess/passes/FogPass";
import {
	DEFAULT_TAA_OPTIONS,
	type TAAOptions,
} from "../../postprocess/passes/TemporalAntiAliasingPass";
import { IBLBRDF } from "../../lights/ibl/IBLBRDF";
import {
	collectWebGLLights,
	type WebGLLightState,
	type WebGLClusteredLight,
} from "./WebGLLightCollector";
import { WebGLGeometryRegistry } from "./WebGLGeometryRegistry";
import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_POINT_LIGHTS,
	MAX_SPOT_LIGHTS,
} from "../constants";
import {
	IDENTITY_MATRIX4_COLUMN_MAJOR,
	SH_COEFFICIENT_COUNT,
	WEBGL_PARTICLE_SHADOW_VOLUME_ATLAS_COLUMNS,
	WEBGL_PARTICLE_SHADOW_VOLUME_GRID_DEPTH,
	WEBGL_PARTICLE_SHADOW_VOLUME_GRID_HEIGHT,
	WEBGL_PARTICLE_SHADOW_VOLUME_GRID_WIDTH,
	WEBGL_PARTICLE_SHADOW_VOLUME_MAX_SLICES,
	WEBGL_REFLECTION_PROBE_CAMERA_WORLD_POSITION_SCRATCH,
	WEBGL_SHADOW_CAPABILITIES,
	WEBGL_TEXTURE_UNIT_PARTICLE_SHADOW_VOLUME,
} from "./constants";
import type { ShadowMap } from "../../lights/shadows/ShadowMapping";
import {
	WebGLProgramLibrary,
	type WebGLSceneProgram,
} from "./WebGLProgramLibrary";
import {
	WebGLProgramCompiler,
	type WebGLProgramWarmupHandle,
} from "./WebGLProgramCompiler";
import {
	WebGLProgramWarmupQueue,
	type WebGLProgramWarmupPriority,
} from "./WebGLProgramWarmupQueue";
import {
	DEFAULT_DEFERRED_UPLOAD_BYTES_PER_FRAME,
	DEFAULT_DEFERRED_UPLOADS_PER_FRAME,
	WebGLTextureRegistry,
} from "./WebGLTextureRegistry";
import type {
	ShaderBackendCompileStage,
	ShaderRuntime,
} from "../../shaders/runtime";
import { ShaderSource, type WebGLSceneLightLimits } from "../../shaders/ShaderSource";
import type {
	WarmupPhaseCounters,
	WarmupPlan,
} from "../../pipeline/WarmupPlanner";
import { toShaderCompileError } from "../../pipeline/WarmupPlanner";
import { createWarmupYieldController } from "../../pipeline/WarmupScheduler";
import type { WarmupOptions } from "../IRenderBackend";
import { WebGLClusteredLightingRuntime } from "./WebGLClusteredLightingRuntime";
import {
	isFiniteMatrix,
	toColumnMajorMat4,
	toSafeDimension,
} from "./WebGLFrameMath";
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
	uploadWebGLIrradianceProbeGridCoefficients,
	uploadWebGLLocalLightProbeCoefficients,
	uploadWebGLSHAmbientCoefficients,
	type WebGLGlobalUniformBinderHost,
	type WebGLIrradianceProbeGridUploadHost,
	type WebGLLocalLightProbeUploadHost,
	type WebGLSHAmbientUploadHost,
} from "./WebGLGlobalUniformBinder";
import { WebGLShadowPass } from "./WebGLShadowPass";
import {
	bindWebGLShaderMaterialUniforms,
	bindWebGLShaderMaterialTextures,
	drawWebGLPacket,
	renderWebGLEarlyZPrepass,
	renderWebGLPackets,
	type WebGLPacketDrawOptions,
	type WebGLScenePassHost,
} from "./WebGLScenePass";
import {
	WebGLParticlePass,
	type WebGLParticleRenderOptions,
} from "./WebGLParticlePass";
import { BackendPostProcessRuntime } from "../../postprocess/BackendPostProcessRuntime";
import { WebGLPostProcessExecutor } from "./WebGLPostProcessExecutor";
import { TemporalJitterState } from "../cross/TemporalJitterState";
import { WebGLPostProcessBridge } from "./WebGLPostProcessBridge";
import {
	getWebGLSceneDepthVariantKey,
	getWebGLSceneVariantKey,
	resolveWebGLBuiltinDepthVariant,
	resolveWebGLBuiltinSceneVariant,
	type WebGLSceneDepthVariantDescriptor,
	type WebGLSceneVariantDescriptor,
} from "./WebGLSceneProgramVariants";
import { WebGLCustomRenderTargetRuntime } from "./WebGLCustomRenderTargetRuntime";
import type { FramePass } from "../../pipeline/types";
import type { RenderTargetReadbackOptions } from "../CustomRenderTargets";
import type { TextureReadbackResult } from "../IComputeRuntime";

export interface WebGLFrameExecutorOptions {
	validatePrograms?: boolean;
	enableEarlyZPrepass?: boolean;
	onProgramCompilePending?: () => void;
	/**
	 * Called when deferred WebGL texture uploads remain queued after the current
	 * upload budget. The callback should mark the renderer dirty so another frame
	 * can continue processing the queue.
	 */
	onTextureUploadPending?: () => void;
	postProcessRuntime?: BackendPostProcessRuntime;
}

export class WebGLFrameExecutor {
	private _gl: WebGL2RenderingContext;
	private _programCompiler: WebGLProgramCompiler;
	private _programs: WebGLProgramLibrary;
	private _geometry: WebGLGeometryRegistry;
	private _textures: WebGLTextureRegistry;
	private _shadowPass: WebGLShadowPass;
	private _particlePass: WebGLParticlePass;
	private _sceneFramebuffer: WebGLFramebuffer | null = null;
	private _sceneColorTexture: WebGLTexture | null = null;
	private _sceneColorFormat: WebGLFrameTargetFormat = "rgba8unorm";
	private _sceneMotionTexture: WebGLTexture | null = null;
	private _sceneMotionFormat: WebGLFrameTargetFormat = "rgba8unorm";
	private _sceneNormalTexture: WebGLTexture | null = null;
	private _sceneNormalFormat: WebGLFrameTargetFormat = "rgba8unorm";
	private _sceneAlbedoTexture: WebGLTexture | null = null;
	private _sceneAlbedoFormat: WebGLFrameTargetFormat = "rgba8unorm";
	private _sceneSpecularTexture: WebGLTexture | null = null;
	private _sceneSpecularFormat: WebGLFrameTargetFormat = "rgba8unorm";
	private _materialGBufferEnabled = false;
	private _sceneDepthBuffer: WebGLRenderbuffer | null = null;
	private _oitFramebuffer: WebGLFramebuffer | null = null;
	private _oitAccumTexture: WebGLTexture | null = null;
	private _oitRevealTexture: WebGLTexture | null = null;
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
	private _width = 1;
	private _height = 1;
	private _targetWidth = 0;
	private _targetHeight = 0;
	private _targetSSAODownsample = DEFAULT_SSAO_OPTIONS.downsample;
	private _targetMaterialGBufferEnabled = false;
	private _maxTextureSize: number;
	private _maxRenderbufferSize: number;
	private _maxTextureImageUnits: number;
	private _irradianceProbeGridSamplingSupported = false;
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
	private _irradianceProbeGridSHTexture: WebGLTexture | null = null;
	private _irradianceProbeGridSHTextureWidth = SH_COEFFICIENT_COUNT;
	private _irradianceProbeGridSHTextureHeight = 1;
	private _ssaoFrameIndex = 0;
	private _fogParams0 = new Float32Array(4);
	private _fogParams1 = new Float32Array(4);
	private _oitPassMode: 0 | 1 | 2 = 0;
	private _oitActive = false;
	private _oitHasContributors = false;
	private _oitTransparentPackets: DrawPacket[] = [];
	private _oitLegacyTransparentPackets: DrawPacket[] = [];
	private _oitNeedsLegacyAfterParticles = false;
	private _supportsFloatColorBuffer: boolean | null = null;
	private _enableEarlyZPrepass = true;
	private _postProcessRuntime: BackendPostProcessRuntime;
	private _postProcessBridge: WebGLPostProcessBridge;
	private _customRenderTargets: WebGLCustomRenderTargetRuntime;

	constructor(
		gl: WebGL2RenderingContext,
		shaderRuntime?: ShaderRuntime,
		shaderCompileStage?: ShaderBackendCompileStage,
		options: WebGLFrameExecutorOptions = {},
	) {
		this._gl = gl;
		this._postProcessRuntime = options.postProcessRuntime ?? new BackendPostProcessRuntime({
			executor: new WebGLPostProcessExecutor({
				getFrameExecutor: () => this,
			}),
			backend: {
				type: "webgl",
				profile: {
					id: "webgl",
					capabilities: {},
					frameScheduling: "on-demand",
					shadow: {},
					lighting: {},
				},
				extensions: {
					getBackendExtension: () => null,
					requireBackendExtension: () => { throw new Error("Mock"); },
				},
			} as any,
			warn: () => {},
		});
		this._enableEarlyZPrepass = options.enableEarlyZPrepass !== false;
		this._programCompiler = new WebGLProgramCompiler(
			gl,
			shaderRuntime,
			shaderCompileStage,
			{
				validatePrograms: options.validatePrograms === true,
				onProgramCompilePending: options.onProgramCompilePending,
			}
		);
		this._programs = new WebGLProgramLibrary(
			gl,
			shaderRuntime,
			shaderCompileStage,
			{
				validatePrograms: options.validatePrograms === true,
				onProgramCompilePending: options.onProgramCompilePending,
				compiler: this._programCompiler,
			},
		);
		this._irradianceProbeGridSamplingSupported =
			this._programs.supportsIrradianceProbeGridSampling();
		this._geometry = new WebGLGeometryRegistry(gl);
		this._textures = new WebGLTextureRegistry(gl, undefined, {
			uploadScheduling: "deferred",
			maxUploadsPerFrame: DEFAULT_DEFERRED_UPLOADS_PER_FRAME,
			maxUploadBytesPerFrame: DEFAULT_DEFERRED_UPLOAD_BYTES_PER_FRAME,
			onUploadPending: options.onTextureUploadPending,
		});
		this._shadowPass = new WebGLShadowPass({
			gl,
			programs: this._programs,
			geometry: this._geometry,
			getLightState: () => this._lightState,
			getSceneFramebuffer: () => this._sceneFramebuffer,
			getViewportSize: () => ({ width: this._width, height: this._height }),
			getMaxTextureSize: () => this._maxTextureSize,
		});
		this._particlePass = new WebGLParticlePass({
			gl,
			programs: this._programs,
			textures: this._textures,
			getSceneFramebuffer: () => this._sceneFramebuffer,
			getViewportSize: () => ({ width: this._width, height: this._height }),
			getFogParams: () => ({
				params0: this._fogParams0,
				params1: this._fogParams1,
			}),
			isIncrementalPartial: (context) => this._isIncrementalPartial(context),
			resolveDirtyRects: (context, width, height) =>
				this._resolveDirtyRects(context, width, height),
			setScissorRect: (x, y, width, height, viewportHeight) =>
				this._setScissorRect(x, y, width, height, viewportHeight),
			updateFogParams: (fogOptions, enabled) =>
				this._updateFogParams(fogOptions, enabled),
		});
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
		this._customRenderTargets = new WebGLCustomRenderTargetRuntime(gl);
		this._postProcessBridge = new WebGLPostProcessBridge({
			getGL: () => this._gl,
			getProgramCompiler: () => this._programCompiler,
			getFullscreenVao: () => this._fullscreenVao,
			getPostFramebuffer: () => this._postFramebuffer,
			getSceneColorTexture: () => this._sceneColorTexture,
			getSceneMotionTexture: () => this._sceneMotionTexture,
			getSceneNormalTexture: () => this._sceneNormalTexture,
			getSSAORawTexture: () => this._ssaoRawTexture,
			getSSAOBlurTexture: () => this._ssaoBlurTexture,
			getWidth: () => this._width,
			getHeight: () => this._height,
			getSSAODownsample: () => this._targetSSAODownsample,
			getActiveContext: () => this._activeContext,
			getSourceTexture: () => this._presentSourceTexture ?? this._sceneColorTexture,
			resolveTargetTexture: (sourceTexture) =>
				this._resolvePostProcessTargetTexture(sourceTexture),
			bindColorTarget: (texture) => this._bindPostSingleColorTarget(texture),
			drawFullscreen: (width, height, frameContext) =>
				this._drawFullscreenTrianglesWithDirtyScissor(
					width,
					height,
					frameContext
				),
			publishColorTexture: (texture) => {
				this._presentSourceTexture = texture;
			},
			markTAAHistoryValid: () => {
				this._taaHistoryValid = true;
			},
			nextFrameJitter: () => this._nextSSAOFrameJitter(),
			applyPipelineHistories: (request) => this._applyPipelineHistories(request),
			warn: (key, message) =>
				Logger.warn(`[${key}] ${message}`, {
					scope: "WebGLFrameExecutor",
					onceKey: key,
				}),
		});
	}

	private get _shadowAtlasTexture(): WebGLTexture | null {
		return this._shadowPass.getTargets().atlasTexture;
	}

	private get _shadowTransmittanceTexture(): WebGLTexture | null {
		return this._shadowPass.getTargets().transmittanceTexture;
	}

	private get _shadowAtlasTileSize(): number {
		return this._shadowPass.getTargets().atlasTileSize;
	}

	public beginFrame(context: FrameContext): void {
		this._programCompiler.beginFrame();
		this._textures.beginFrame();
		this._customRenderTargets.sync(context);
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
		this._ensureFrameTargets(
			this._width,
			this._height,
			ssaoDownsample,
			this._requiresMaterialGBuffer(context)
		);
		this._configureOIT(context);
		this._presentSourceTexture = this._sceneColorTexture;
		this._oitPassMode = 0;
		this._oitHasContributors = false;
		this._oitTransparentPackets = [];
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
				context.viewCamera.getWorldPosition(
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
			isOrthographic: context.viewCamera.type === CameraType.Orthographic,
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
	}

	public clearFrameTargets(context: FrameContext): void {
		const gl = this._gl;
		gl.bindFramebuffer(gl.FRAMEBUFFER, this._sceneFramebuffer);
		gl.viewport(0, 0, this._width, this._height);
		gl.disable(gl.BLEND);
		gl.enable(gl.DEPTH_TEST);
		gl.depthMask(true);

		const drawBuffers = this._materialGBufferEnabled ?
			[
				gl.COLOR_ATTACHMENT0,
				gl.COLOR_ATTACHMENT1,
				gl.COLOR_ATTACHMENT2,
				gl.COLOR_ATTACHMENT3,
				gl.COLOR_ATTACHMENT4,
			]
		:	this._sceneNormalTexture ?
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
	}

	public renderEnvironmentNode(context: FrameContext): void {
		const incrementalPartial = this._isIncrementalPartial(context);
		if (
			!incrementalPartial &&
			context.features.enableEnvironment &&
			context.scene.environment.backgroundEnabled &&
			context.scene.environment.backgroundTexture
		) {
			this._renderEnvironment(context);
		}
	}

	public isOITActive(): boolean {
		return this._oitActive && !!this._oitFramebuffer;
	}

	public hasPresentedInFrame(): boolean {
		return this._presentedInFrame;
	}

	public collectFrameGraphResources(): readonly string[] {
		const resources = new Set<string>();
		if (this._sceneColorTexture) {
			resources.add("frame:scene-color");
			resources.add("frame:present-source");
		}
		if (this._sceneMotionTexture) resources.add("frame:motion-depth");
		if (this._sceneNormalTexture) resources.add("frame:normal");
		if (this._sceneDepthBuffer) resources.add("frame:depth");
		if (this._postColorTexture) resources.add("post:color");
		if (this._ssaoRawTexture) resources.add("post:ssao-raw");
		if (this._ssaoBlurTexture) resources.add("post:ssao-blur");
		if (this._oitAccumTexture) resources.add("oit:accum");
		if (this._oitRevealTexture) resources.add("oit:reveal");
		if (this._shadowAtlasTexture) resources.add("shadow:atlas");
		if (this._shadowTransmittanceTexture) {
			resources.add("shadow:transmittance");
		}
		return Array.from(resources);
	}

	public renderShadowNode(context: FrameContext): void {
		this._updateParticleShadowVolumes(context);
		this._renderShadows(context);
	}

	public renderOpaqueDepthPrepass(context: FrameContext): Set<string> {
		return this._renderEarlyZPrepass(context, context.scene.opaquePackets);
	}

	public renderOpaqueScene(
		context: FrameContext,
		earlyZPacketIds: ReadonlySet<string>
	): void {
		this._renderPackets(context, context.scene.opaquePackets, false, {
			earlyZPacketIds,
		});
	}

	public renderTransparentLegacy(context: FrameContext): void {
		this._renderPackets(context, context.scene.transparentPackets, true);
	}

	public prepareOITTransparent(context: FrameContext): void {
		if (!this.isOITActive()) {
			this.renderTransparentLegacy(context);
			return;
		}
		const { oitPackets, legacyPackets } = this._partitionTransparentPackets(
			context.scene.transparentPackets
		);
		this._oitTransparentPackets = oitPackets;
		this._oitLegacyTransparentPackets = legacyPackets;
		this._oitNeedsLegacyAfterParticles =
			(context.scene.particleSystems?.length ?? 0) > 0;
		this._oitHasContributors = false;
		if (oitPackets.length > 0) {
			this._clearOITTargets();
		}
	}

	public renderOITTransparentAccum(context: FrameContext): void {
		if (!this.isOITActive() || this._oitTransparentPackets.length <= 0) {
			return;
		}
		this._renderPackets(context, this._oitTransparentPackets, true, {
			framebuffer: this._oitFramebuffer,
			drawBuffers: [this._gl.COLOR_ATTACHMENT0],
			blendMode: "oit-accum",
			oitPassMode: 1,
		});
	}

	public renderOITTransparentReveal(context: FrameContext): void {
		if (!this.isOITActive() || this._oitTransparentPackets.length <= 0) {
			return;
		}
		this._renderPackets(context, this._oitTransparentPackets, true, {
			framebuffer: this._oitFramebuffer,
			drawBuffers: [this._gl.COLOR_ATTACHMENT0],
			blendMode: "oit-reveal",
			oitPassMode: 2,
		});
		this._oitHasContributors = true;
	}

	public resolveOIT(context: FrameContext): void {
		if (this._oitHasContributors) {
			this._resolveOITComposition(context);
		}
	}

	public renderOITLegacyTransparent(context: FrameContext): void {
		if (this._oitLegacyTransparentPackets.length > 0) {
			this._renderPackets(context, this._oitLegacyTransparentPackets, true);
		}
		this._oitTransparentPackets = [];
		this._oitLegacyTransparentPackets = [];
		this._oitHasContributors = false;
		this._oitNeedsLegacyAfterParticles = false;
	}

	public prepareOITParticles(): void {
		if (!this.isOITActive()) {
			return;
		}
		if (!this._oitHasContributors) {
			this._clearOITTargets();
		}
	}

	public renderOITParticleAccum(context: FrameContext): void {
		if (!this.isOITActive()) {
			this._renderParticles(context);
			return;
		}
		this._renderParticles(context, {
			framebuffer: this._oitFramebuffer,
			drawBuffers: [this._gl.COLOR_ATTACHMENT0],
			includeBlendModes: [ParticleBlendMode.Alpha],
			oitPassMode: 1,
		});
	}

	public renderOITParticleReveal(context: FrameContext): void {
		if (!this.isOITActive()) {
			return;
		}
		this._renderParticles(context, {
			framebuffer: this._oitFramebuffer,
			drawBuffers: [this._gl.COLOR_ATTACHMENT0],
			includeBlendModes: [ParticleBlendMode.Alpha],
			oitPassMode: 2,
		});
		this._oitHasContributors = true;
	}

	public renderParticlesLegacy(context: FrameContext): void {
		this._renderParticles(context);
	}

	public renderOITAdditiveParticles(context: FrameContext): void {
		this._renderParticles(context, {
			includeBlendModes: [ParticleBlendMode.Additive],
		});
	}

	public presentFrame(): void {
		if (!this._presentedInFrame) {
			this._present(this._activeContext, true);
		}
	}

	public finishFrame(): void {
		this._customRenderTargets.markFrameCommitted();
		this._pruneModelMatrixCache();
		this._activeContext = null;
	}

	public hasCustomRenderPass(pass: FramePass, context: FrameContext): boolean {
		return this._customRenderTargets.hasPass(pass, context);
	}

	public executeCustomRenderPass(
		pass: FramePass,
		context: FrameContext
	): Promise<void> {
		return this._customRenderTargets.executePass(pass, context);
	}

	public readCustomRenderTargetColor(
		id: string,
		attachmentIndex?: number,
		options?: RenderTargetReadbackOptions
	): Promise<TextureReadbackResult> {
		return this._customRenderTargets.readColor(id, attachmentIndex, options);
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
				albedo: this._sceneAlbedoTexture ?
					{
						semantic: "albedo",
						handle: { backend: "webgl", texture: this._sceneAlbedoTexture },
						width,
						height,
						format: this._sceneAlbedoFormat,
						encoding: "linear-rgb-alpha",
					}
				:	undefined,
				roughness: this._sceneNormalTexture && this._materialGBufferEnabled ?
					{
						semantic: "roughness",
						handle: { backend: "webgl", texture: this._sceneNormalTexture },
						width,
						height,
						format: this._sceneNormalFormat,
						encoding: "normal-roughness-metallic.z",
					}
				:	undefined,
				metallic: this._sceneNormalTexture && this._materialGBufferEnabled ?
					{
						semantic: "metallic",
						handle: { backend: "webgl", texture: this._sceneNormalTexture },
						width,
						height,
						format: this._sceneNormalFormat,
						encoding: "normal-roughness-metallic.w",
					}
				:	undefined,
				specular: this._sceneSpecularTexture ?
					{
						semantic: "specular",
						handle: { backend: "webgl", texture: this._sceneSpecularTexture },
						width,
						height,
						format: this._sceneSpecularFormat,
						encoding: "specular-color-factor.rgba",
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
		return this._postProcessBridge.getPassExecutionContext(request);
	}

	/** @internal Opens controlled output publication for a runtime post-process frame. */
	public beginPostProcessFrame(): void {
		this._postProcessBridge.beginFrameTransaction();
	}

	/** @internal Closes controlled output publication for a runtime post-process frame. */
	public endPostProcessFrame(): void {
		this._postProcessBridge.endFrameTransaction();
	}

	/** @internal Commits validated WebGL post-process output for one pass. */
	public completePostProcessPass(
		request: PostProcessPassRequest,
		result: PostProcessPassResult
	): void {
		this._postProcessBridge.completePass(request, result);
	}

	public endFrame(): void {
		this.presentFrame();
		this.finishFrame();
	}

	public abortFrame(): void {
		this._postProcessBridge.clearPendingFrameState();
		this._customRenderTargets.markFrameAborted();
		this._activeContext = null;
		this._presentedInFrame = false;
		this._presentSourceTexture = this._sceneColorTexture;
		this._lightState = null;
		this._oitPassMode = 0;
		this._oitActive = false;
		this._oitHasContributors = false;
		this._oitTransparentPackets = [];
		this._oitLegacyTransparentPackets = [];
		this._oitNeedsLegacyAfterParticles = false;
		this._modelMatrixKeysThisFrame.clear();
	}

	public async warmup(
		context: FrameContext,
		plan: WarmupPlan,
		options: WarmupOptions = {}
	): Promise<WarmupPhaseCounters> {
		let skipped = 0;
		const yieldController = createWarmupYieldController(options);
		const queue = new WebGLProgramWarmupQueue();

		const enqueue = (
			label: string,
			priority: WebGLProgramWarmupPriority,
			action: () => unknown | Promise<unknown>
		): void => {
			queue.enqueue({
				label,
				priority,
				action: () => this._collectWarmupHandles(action),
			});
		};

		const materialWarmupModes: ShaderTargetMode[] =
			plan.sceneTargetMode === "mrt" ? ["mrt", "single"] : ["single"];
		const warmupLightState = this._collectWarmupLightState(context);
		const builtinSceneVariants = this._collectWarmupSceneVariants(
			context,
			plan.materials,
			materialWarmupModes,
			warmupLightState
		);
		const builtinDepthVariants = this._collectWarmupDepthVariants(
			plan.materials
		);
		enqueue("WebGLSceneSource:builtin", "core", () =>
			ShaderSource.prepareMany(
				[...builtinSceneVariants.values()].flatMap((variant) => [
					{
						key: "webgl.scene.raw" as const,
						params: {
							limits: this._getWarmupSceneLightLimits(),
							variant,
						},
					},
					{
						key: "webgl.scene.composite" as const,
						params: {
							limits: this._getWarmupSceneLightLimits(),
							variant,
						},
					},
				])
			)
		);
		for (const [variantKey, variant] of builtinSceneVariants) {
			enqueue(`WebGLSceneProgram:builtin:${variantKey}`, "core", () => {
				return this._warmupProgramHandle(
					"warmupSceneProgram",
					"getSceneProgram",
					undefined,
					variant.output,
					variant
				);
			});
		}
		if (this._enableEarlyZPrepass) {
			for (const [variantKey, variant] of builtinDepthVariants) {
				enqueue(
					`WebGLSceneDepthPrepassProgram:builtin:${variantKey}`,
					"core",
					() => {
						return this._warmupProgramHandle(
							"warmupSceneDepthPrepassProgram",
							"getSceneDepthPrepassProgram",
							undefined,
							"single",
							variant
						);
					}
				);
			}
		}
		for (const material of plan.materials) {
			if (!(material instanceof ShaderMaterial)) {
				continue;
			}
			for (const mode of materialWarmupModes) {
				enqueue(
					`WebGLSceneProgram:material:${material.shaderId}:${mode}`,
					"core",
					() => {
						return this._warmupProgramHandle(
							"warmupSceneProgram",
							"getSceneProgram",
							material,
							mode,
						);
					}
				);
				if (this._enableEarlyZPrepass) {
					enqueue(
						`WebGLSceneDepthPrepassProgram:material:${material.shaderId}:${mode}`,
						"core",
						() => {
							return this._warmupProgramHandle(
								"warmupSceneDepthPrepassProgram",
								"getSceneDepthPrepassProgram",
								material,
								mode,
							);
						}
					);
				}
			}
		}

		if (plan.enableEnvironment) {
			enqueue("WebGLEnvironmentProgram", "optional", () => {
				return this._warmupProgramHandle(
					"warmupEnvironmentProgram",
					"getEnvironmentProgram",
				);
			});
		}
		if (plan.enableShadows) {
			enqueue("WebGLShadowDepthProgram", "optional", () => {
				return this._warmupProgramHandle(
					"warmupShadowDepthProgram",
					"getShadowDepthProgram",
				);
			});
		}
		if (plan.enableParticles) {
			enqueue("WebGLParticleProgram", "optional", () => {
				return this._warmupProgramHandle(
					"warmupParticleProgram",
					"getParticleProgram",
				);
			});
		}
		if (context.features?.enableOIT) {
			enqueue("WebGLOITResolveProgram", "optional", () => {
				return this._warmupProgramHandle(
					"warmupOITResolveProgram",
					"getOITResolveProgram",
				);
			});
		}

		const warmupGraph = this._postProcessRuntime!.compileWarmupGraph(context);
		const warmedPassImplementations = new Set<string>();
		for (const passId of plan.postProcessPasses) {
			if (warmedPassImplementations.has(passId)) {
				continue;
			}
			const compiled = warmupGraph.passes.find((p) => p.id === passId);
			const implementation = compiled?.implementation;
			if (typeof implementation?.warmup !== "function") {
				continue;
			}
			warmedPassImplementations.add(passId);
			enqueue(`WebGLPostWarmup:${passId}`, "postprocess", async () => {
				const warmupContext =
					this._postProcessBridge.getPassWarmupExecutionContext(
						implementation
					);
				await implementation.warmup?.(
					warmupContext,
					{
						frameContext: context,
						postProcess: context.postProcess,
						backend: "webgl",
						context: warmupContext,
						options: compiled.options,
					}
				);
			});
		}

		enqueue("WebGLPresentProgram", "core", () => {
			return this._warmupProgramHandle(
				"warmupPresentProgram",
				"getPresentProgram",
			);
		});

		const result = await queue.run(yieldController, options);
		const errors = result.errors.map((entry) =>
			toShaderCompileError(entry.error, "webgl", entry.label)
		);
		const failed = result.enqueueFailures + result.failed;
		const compiled = result.compiled;
		const total = result.handles + result.enqueueFailures + skipped;

		return {
			phase: "webgl-programs",
			total,
			compiled,
			skipped,
			failed,
			errors,
		};
	}

	private _collectWarmupLightState(context: FrameContext): WebGLLightState {
		const scene = context.scene;
		const environment = scene?.environment;
		return collectWebGLLights(
			scene?.lights ?? [],
			context.features?.enableLighting ?? false,
			context.features?.enableShadows ?? false,
			context.shadowMaps ?? new Map(),
			context.features?.enableSH ?? false,
			environment?.lightingEnabled ?
				environment.iblTexture
			:	null,
			context.features?.enableClusteredLighting ?? false,
			context.viewCamera?.getWorldPosition ?
				context.viewCamera.getWorldPosition(
					WEBGL_REFLECTION_PROBE_CAMERA_WORLD_POSITION_SCRATCH
				)
			:	null
		);
	}

	private _collectWarmupSceneVariants(
		context: FrameContext,
		materials: readonly Material[],
		modes: readonly ShaderTargetMode[],
		lightState: WebGLLightState
	): Map<string, WebGLSceneVariantDescriptor> {
		const variants = new Map<string, WebGLSceneVariantDescriptor>();
		for (const material of materials) {
			if (material instanceof ShaderMaterial) {
				continue;
			}
			for (const mode of modes) {
				this._addWarmupSceneVariant(
					variants,
					context,
					material,
					mode,
					0,
					lightState
				);
			}
			if (context.features?.enableOIT && isMaterialTransparentPass(material)) {
				this._addWarmupSceneVariant(
					variants,
					context,
					material,
					"single",
					1,
					lightState
				);
				this._addWarmupSceneVariant(
					variants,
					context,
					material,
					"single",
					2,
					lightState
				);
			}
		}
		return variants;
	}

	private _addWarmupSceneVariant(
		variants: Map<string, WebGLSceneVariantDescriptor>,
		context: FrameContext,
		material: Material,
		mode: ShaderTargetMode,
		oitPassMode: 0 | 1 | 2,
		lightState: WebGLLightState
	): void {
		const variant = resolveWebGLBuiltinSceneVariant(
			context,
			material,
			mode,
			oitPassMode,
			{
				lightState,
				enableShadowTransmittance: this._maxTextureImageUnits >= 17,
				enableIrradianceProbeGrid: this._irradianceProbeGridSamplingSupported,
			},
			mode === "mrt"
		);
		if (!variant) {
			return;
		}
		variants.set(getWebGLSceneVariantKey(variant), variant);
	}

	private _collectWarmupDepthVariants(
		materials: readonly Material[]
	): Map<string, WebGLSceneDepthVariantDescriptor> {
		const variants = new Map<string, WebGLSceneDepthVariantDescriptor>();
		for (const material of materials) {
			if (
				material instanceof ShaderMaterial ||
				isMaterialTransparentPass(material) ||
				material.depthWrite === false
			) {
				continue;
			}
			const variant = resolveWebGLBuiltinDepthVariant(material);
			if (!variant) {
				continue;
			}
			variants.set(getWebGLSceneDepthVariantKey(variant), variant);
		}
		return variants;
	}

	private _getWarmupSceneLightLimits(): WebGLSceneLightLimits {
		return {
			maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
			maxPointLights: MAX_POINT_LIGHTS,
			maxSpotLights: MAX_SPOT_LIGHTS,
			enableShadowTransmittance: this._maxTextureImageUnits >= 17,
			enableIrradianceProbeGrid: this._irradianceProbeGridSamplingSupported,
		};
	}

	private async _collectWarmupHandles(
		action: () => unknown | Promise<unknown>
	): Promise<WebGLProgramWarmupHandle[]> {
		const mark = this._programCompiler.markWarmupHandles();
		const result = await action();
		const logged = this._programCompiler.collectWarmupHandlesSince(mark);
		if (logged.length > 0) {
			return logged;
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
	): WebGLProgramWarmupHandle | null {
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

	private _getWarmupPostProcessDescriptorMap(
		context: FrameContext,
		plan: WarmupPlan
	): Map<string, PostProcessPass> {
		const descriptors =
			plan.postProcessDescriptors ??
			context.postProcess.getEnabledPasses().map((pass) => pass.pass);
		return new Map(descriptors.map((pass) => [pass.id, pass]));
	}

	public resize(width: number, height: number): void {
		this._width = toSafeDimension(width);
		this._height = toSafeDimension(height);
		this._destroyFrameTargets();
	}

	public destroy(): void {
		this._customRenderTargets.destroy();
		this._destroyFrameTargets();
		this._shadowPass.destroy();
		this._particlePass.destroy();
		this._clusteredLighting.destroy();
		if (this._shAmbientTexture) {
			this._gl.deleteTexture(this._shAmbientTexture);
			this._shAmbientTexture = null;
		}
		if (this._localLightProbeSHTexture) {
			this._gl.deleteTexture(this._localLightProbeSHTexture);
			this._localLightProbeSHTexture = null;
		}
		if (this._irradianceProbeGridSHTexture) {
			this._gl.deleteTexture(this._irradianceProbeGridSHTexture);
			this._irradianceProbeGridSHTexture = null;
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
		this._programCompiler.destroy();
		this._activeContext = null;
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
		this._shadowPass.render(context);
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
			earlyZPacketIds?: ReadonlySet<string>;
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

	private _renderEarlyZPrepass(
		context: FrameContext,
		packets: DrawPacket[]
	): Set<string> {
		if (!this._enableEarlyZPrepass || packets.length === 0) {
			return new Set<string>();
		}
		return renderWebGLEarlyZPrepass(
			this as unknown as WebGLScenePassHost,
			context,
			packets
		);
	}

	private _resolveSceneProgramVariant(
		context: FrameContext,
		packet: DrawPacket,
		mode: ShaderTargetMode
	): WebGLSceneVariantDescriptor | null {
		return resolveWebGLBuiltinSceneVariant(
			context,
			packet.material,
			mode,
			this._oitPassMode,
			{
				lightState: this._lightState,
				enableShadowTransmittance: !!this._shadowTransmittanceTexture,
				enableIrradianceProbeGrid: this._irradianceProbeGridSamplingSupported,
			},
			this._materialGBufferEnabled
		);
	}

	private _resolveSceneDepthPrepassVariant(
		packet: DrawPacket
	): WebGLSceneDepthVariantDescriptor | null {
		return resolveWebGLBuiltinDepthVariant(packet.material);
	}

	private _drawPacket(
		sceneProgram: WebGLSceneProgram,
		packet: DrawPacket,
		transparentPass: boolean,
		context: FrameContext,
		options?: WebGLPacketDrawOptions
	): void {
		drawWebGLPacket(
			this as unknown as WebGLScenePassHost,
			sceneProgram,
			packet,
			transparentPass,
			context,
			options
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
		options: WebGLParticleRenderOptions = {}
	): void {
		this._particlePass.render(context, options);
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
		const copyProgram = this._programs.tryGetCopyProgram();
		if (!copyProgram) {
			return false;
		}
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
		const resolveProgram = this._programs.tryGetOITResolveProgram();
		if (!resolveProgram) {
			return;
		}
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

	private _uploadIrradianceProbeGridCoefficients(
		grid: WebGLLightState["irradianceProbeGrid"]
	): boolean {
		return uploadWebGLIrradianceProbeGridCoefficients(
			this as unknown as WebGLIrradianceProbeGridUploadHost,
			grid
		);
	}

	private _renderEnvironment(context: FrameContext): void {
		const environmentBackgroundTexture =
			context.scene.environment.backgroundTexture;
		if (!environmentBackgroundTexture || !this._fullscreenVao) return;
		const environment = context.scene.environment;

		const gl = this._gl;
		const environmentProgram = this._programs.tryGetEnvironmentProgram();
		if (!environmentProgram) {
			return;
		}
		const resolved = this._textures.getEnvironmentTexture(environmentBackgroundTexture);
		const view = context.viewCamera.viewMatrix.elements;
		const isOrthographic = context.viewCamera.type === CameraType.Orthographic;
		const tanHalfFov =
			isOrthographic ? 0 : Math.tan((context.viewCamera.fov * Math.PI) / 360);
		const aspect = context.viewCamera.aspectRatio || this._width / this._height;

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
		context: FrameContext | null = this._activeContext,
		nonBlocking = false
	): boolean {
		const sourceTexture = this._presentSourceTexture ?? this._sceneColorTexture;
		if (!sourceTexture || !this._fullscreenVao) return false;
		const gl = this._gl;
		const presentProgram =
			nonBlocking ?
				this._programs.tryGetPresentProgram()
			:	this._programs.getPresentProgram();
		if (!presentProgram) {
			return false;
		}
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
		this._drawFullscreenTrianglesWithDirtyScissor(
			this._width,
			this._height,
			context
		);
		gl.bindVertexArray(null);
		this._presentedInFrame = true;
		return true;
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
		ssaoDownsample: number,
		materialGBufferRequested: boolean
	): void {
		ensureWebGLFrameTargets(
			this as unknown as WebGLFrameTargetLifecycleHost,
			width,
			height,
			ssaoDownsample,
			materialGBufferRequested
		);
	}

	private _requiresMaterialGBuffer(context: FrameContext): boolean {
		for (const resolved of context.postProcess.getEnabledPasses()) {
			const requirements = resolved.pass.getRequirements({
				frameContext: context,
				postProcess: context.postProcess,
				backend: "webgl",
				options: resolved.options,
			});
			if (requirements.gBuffer?.some((semantic) =>
				semantic === "albedo" ||
				semantic === "roughness" ||
				semantic === "metallic" ||
				semantic === "specular"
			)) {
				return true;
			}
		}
		return false;
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
