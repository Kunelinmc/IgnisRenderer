import { CameraType } from "../../cameras/Camera";
import {
	AlphaMode,
	type Material,
} from "../../materials/Material";
import type { ShaderTargetMode } from "../../materials/ShaderMaterial";
import type { SHCoefficients } from "../../maths/types";
import {
	type DrawPacket,
	type FrameContext,
} from "../../pipeline/types";
import type {
	LogicalGBufferBridge,
	PostProcessPassExecutionContextRequest,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessPassCompletion,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "../../postprocess";
import type { FramePreparationRequirements } from "../../pipeline/FrameRequirements";
import {
	DEFAULT_FOG_OPTIONS,
	resolveFogUniformParams,
	type FogOptions,
} from "../../postprocess/passes/FogPass";
import { IBLBRDF } from "../../lights/ibl/IBLBRDF";
import {
	collectWebGLLights,
	type WebGLLightState,
	type WebGLClusteredLight,
} from "./WebGLLightCollector";
import { WebGLGeometryRegistry } from "./WebGLGeometryRegistry";
import {
	IDENTITY_MATRIX4_COLUMN_MAJOR,
	SH_COEFFICIENT_COUNT,
	WEBGL_REFLECTION_PROBE_CAMERA_WORLD_POSITION_SCRATCH,
} from "./constants";
import {
	WebGLProgramLibrary,
	type WebGLSceneProgram,
} from "./WebGLProgramLibrary";
import { WebGLProgramCompiler } from "./WebGLProgramCompiler";
import {
	DEFAULT_DEFERRED_UPLOAD_BYTES_PER_FRAME,
	DEFAULT_DEFERRED_UPLOADS_PER_FRAME,
	WebGLTextureRegistry,
} from "./WebGLTextureRegistry";
import type {
	ShaderBackendCompileStage,
	ShaderRuntime,
} from "../../shaders/runtime";
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
import { WebGLFrameTargetManager } from "./WebGLFrameTargetManager";
import { WebGLFrameSession } from "./WebGLFrameSession";
import { WebGLFullscreenRenderer } from "./WebGLFullscreenRenderer";
import {
	bindWebGLGlobalUniforms,
	uploadWebGLIrradianceProbeGridCoefficients,
	uploadWebGLLocalLightProbeCoefficients,
	uploadWebGLSHAmbientCoefficients,
} from "./WebGLGlobalUniformBinder";
import {
	WebGLShadowRuntime,
	type WebGLShadowSamplingState,
} from "./WebGLShadowRuntime";
import {
	bindWebGLShaderMaterialUniforms,
	bindWebGLShaderMaterialTextures,
	drawWebGLPacket,
	type WebGLPacketDrawOptions,
} from "./WebGLScenePass";
import {
	WebGLParticlePass,
	type WebGLParticleRenderOptions,
} from "./WebGLParticlePass";
import { BackendPostProcessRuntime } from "../../postprocess/BackendPostProcessRuntime";
import { WebGLPostProcessServices } from "./WebGLPostProcessServices";
import { WebGLTemporalFrameState } from "./WebGLTemporalFrameState";
import { WebGLWarmupCoordinator } from "./WebGLWarmupCoordinator";
import { WebGLTransparencyRuntime } from "./WebGLTransparencyRuntime";
import {
	resolveWebGLBuiltinDepthVariant,
	resolveWebGLBuiltinSceneVariant,
	type WebGLSceneDepthVariantDescriptor,
	type WebGLSceneVariantDescriptor,
} from "./WebGLSceneProgramVariants";
import { WebGLCustomRenderTargetRuntime } from "./WebGLCustomRenderTargetRuntime";
import type { FramePass } from "../../pipeline/types";
import type {
	RenderTargetReadbackOptions,
	RenderTargetReadbackResult,
} from "../../rendering/CustomRenderTargets";
import { WebGLSceneRuntime } from "./WebGLSceneRuntime";

export interface WebGLFrameServiceOwnerOptions {
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

export class WebGLFrameServiceOwner {
	public _gl: WebGL2RenderingContext;
	private _programCompiler: WebGLProgramCompiler;
	public _programs: WebGLProgramLibrary;
	public _geometry: WebGLGeometryRegistry;
	public _textures: WebGLTextureRegistry;
	private _shadow: WebGLShadowRuntime;
	private _particlePass: WebGLParticlePass;
	private _targets: WebGLFrameTargetManager;
	private readonly _temporalFrameState = new WebGLTemporalFrameState();
	private _scene: WebGLSceneRuntime;
	private _fullscreen: WebGLFullscreenRenderer;
	private readonly _session = new WebGLFrameSession();
	private _maxTextureSize: number;
	private _maxRenderbufferSize: number;
	public _maxTextureImageUnits: number;
	public _irradianceProbeGridSamplingSupported = false;
	public _clusteredLighting: WebGLClusteredLightingRuntime;
	public _shAmbientTexture: WebGLTexture | null = null;
	public _shAmbientTextureWidth = SH_COEFFICIENT_COUNT;
	public _shAmbientTextureHeight = 1;
	public _localLightProbeSHTexture: WebGLTexture | null = null;
	public _localLightProbeSHTextureWidth = SH_COEFFICIENT_COUNT;
	public _localLightProbeSHTextureHeight = 1;
	public _irradianceProbeGridSHTexture: WebGLTexture | null = null;
	public _irradianceProbeGridSHTextureWidth = SH_COEFFICIENT_COUNT;
	public _irradianceProbeGridSHTextureHeight = 1;
	public _fogParams0 = new Float32Array(4);
	public _fogParams1 = new Float32Array(4);
	public _oitPassMode: 0 | 1 | 2 = 0;
	private _transparency: WebGLTransparencyRuntime;
	private _enableEarlyZPrepass = true;
	private _postProcessRuntime: BackendPostProcessRuntime;
	private _postProcess: WebGLPostProcessServices;
	private _warmup: WebGLWarmupCoordinator;
	private _customRenderTargets: WebGLCustomRenderTargetRuntime;

	public get enableEarlyZPrepass(): boolean {
		return this._enableEarlyZPrepass;
	}

	public get warmupCoordinator(): WebGLWarmupCoordinator {
		return this._warmup;
	}

	public get transparency(): WebGLTransparencyRuntime {
		return this._transparency;
	}

	public get scene(): WebGLSceneRuntime {
		return this._scene;
	}

	public get _modelMatrixCache(): Map<string, Float32Array> {
		return this._scene.modelMatrixCache;
	}

	public get _modelMatrixKeysThisFrame(): Set<string> {
		return this._scene.modelMatrixKeysThisFrame;
	}

	/** @internal Transitional access for typed WebGL draw-helper contracts. */
	public get _width(): number {
		return this._session.width;
	}
	public set _width(value: number) {
		this._session.width = value;
	}
	public get _height(): number {
		return this._session.height;
	}
	public set _height(value: number) {
		this._session.height = value;
	}
	public get _activeContext(): FrameContext | null {
		return this._session.context;
	}
	public set _activeContext(value: FrameContext | null) {
		this._session.context = value;
	}
	public get _presentedInFrame(): boolean {
		return this._session.presented;
	}
	public set _presentedInFrame(value: boolean) {
		this._session.presented = value;
	}
	public get _lightState(): WebGLLightState | null {
		return this._session.lightState;
	}
	public set _lightState(value: WebGLLightState | null) {
		this._session.lightState = value;
	}
	public get _temporalJitterCurrentPrev(): Float32Array {
		return this._temporalFrameState.jitterCurrentPrev;
	}
	public get _previousViewProjection(): Float32Array | null {
		return this._temporalFrameState.previousViewProjection;
	}
	public get _fullscreenVao(): WebGLVertexArrayObject | null {
		return this._fullscreen._vao;
	}
	public set _fullscreenVao(value: WebGLVertexArrayObject | null) {
		this._fullscreen._vao = value;
	}

	/** @internal Transitional frame-target access for WebGL draw helpers. */
	public get _sceneFramebuffer(): WebGLFramebuffer | null {
		return this._targets._sceneFramebuffer;
	}
	public set _sceneFramebuffer(value: WebGLFramebuffer | null) {
		this._targets._sceneFramebuffer = value;
	}
	public get _sceneColorTexture(): WebGLTexture | null {
		return this._targets._sceneColorTexture;
	}
	public set _sceneColorTexture(value: WebGLTexture | null) {
		this._targets._sceneColorTexture = value;
	}
	public get _sceneMotionTexture(): WebGLTexture | null {
		return this._targets._sceneMotionTexture;
	}
	public set _sceneMotionTexture(value: WebGLTexture | null) {
		this._targets._sceneMotionTexture = value;
	}
	public get _sceneNormalTexture(): WebGLTexture | null {
		return this._targets._sceneNormalTexture;
	}
	public set _sceneNormalTexture(value: WebGLTexture | null) {
		this._targets._sceneNormalTexture = value;
	}
	public get _materialGBufferEnabled(): boolean {
		return this._targets._materialGBufferEnabled;
	}
	public set _materialGBufferEnabled(value: boolean) {
		this._targets._materialGBufferEnabled = value;
	}
	public get _oitFramebuffer(): WebGLFramebuffer | null {
		return this._targets._oitFramebuffer;
	}
	public set _oitFramebuffer(value: WebGLFramebuffer | null) {
		this._targets._oitFramebuffer = value;
	}
	public get _oitAccumTexture(): WebGLTexture | null {
		return this._targets._oitAccumTexture;
	}
	public set _oitAccumTexture(value: WebGLTexture | null) {
		this._targets._oitAccumTexture = value;
	}
	public get _oitRevealTexture(): WebGLTexture | null {
		return this._targets._oitRevealTexture;
	}
	public set _oitRevealTexture(value: WebGLTexture | null) {
		this._targets._oitRevealTexture = value;
	}
	public get _postFramebuffer(): WebGLFramebuffer | null {
		return this._targets._postFramebuffer;
	}
	public set _postFramebuffer(value: WebGLFramebuffer | null) {
		this._targets._postFramebuffer = value;
	}
	public get _postColorTexture(): WebGLTexture | null {
		return this._targets._postColorTexture;
	}
	public set _postColorTexture(value: WebGLTexture | null) {
		this._targets._postColorTexture = value;
	}
	public get _presentSourceTexture(): WebGLTexture | null {
		return this._targets._presentSourceTexture;
	}
	public set _presentSourceTexture(value: WebGLTexture | null) {
		this._targets._presentSourceTexture = value;
	}
	constructor(
		gl: WebGL2RenderingContext,
		shaderRuntime?: ShaderRuntime,
		shaderCompileStage?: ShaderBackendCompileStage,
		options: WebGLFrameServiceOwnerOptions = {},
	) {
		this._gl = gl;
		if (!options.postProcessRuntime) {
			throw new Error(
				"WebGL frame services require an explicitly owned post-process runtime.",
			);
		}
		this._postProcessRuntime = options.postProcessRuntime;
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
		this._particlePass = new WebGLParticlePass({
			gl,
			programs: this._programs,
			textures: this._textures,
			getSceneFramebuffer: () => this._targets._sceneFramebuffer,
			getViewportSize: () => ({ width: this._session.width, height: this._session.height }),
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
		this._maxTextureSize = this._resolveLimit(gl.MAX_TEXTURE_SIZE, 4096);
		this._maxRenderbufferSize = this._resolveLimit(
			gl.MAX_RENDERBUFFER_SIZE,
			4096
		);
		this._maxTextureImageUnits = this._resolveLimit(
			gl.MAX_TEXTURE_IMAGE_UNITS,
			16
		);
		this._targets = new WebGLFrameTargetManager(
			gl,
			this._maxTextureSize,
			this._maxRenderbufferSize,
		);
		this._fullscreen = new WebGLFullscreenRenderer({
			gl: this._gl,
			targets: this._targets,
			getPrograms: () => this._programs,
			getWidth: () => this._session.width,
			getHeight: () => this._session.height,
			isIncrementalPartial: (context) => this._isIncrementalPartial(context),
			resolveDirtyRects: (context, width, height) =>
				this._resolveDirtyRects(context, width, height),
			setScissorRect: (x, y, width, height, viewportHeight) =>
				this._setScissorRect(x, y, width, height, viewportHeight),
			markPresented: () => {
				this._session.presented = true;
			},
		});
		this._scene = new WebGLSceneRuntime({
			gl: this._gl,
			host: this,
			targets: this._targets,
			enableEarlyZPrepass: this._enableEarlyZPrepass,
			getWidth: () => this._session.width,
			getHeight: () => this._session.height,
			isIncrementalPartial: (context) => this._isIncrementalPartial(context),
			resolveDirtyRects: (context, width, height) =>
				this._resolveDirtyRects(context, width, height),
			setScissorRect: (x, y, width, height, viewportHeight) =>
				this._setScissorRect(x, y, width, height, viewportHeight),
			renderEnvironment: (context) => this._renderEnvironment(context),
			renderLegacyTransparent: (context) =>
				this._transparency.renderLegacyTransparent(context),
		});
		this._shadow = new WebGLShadowRuntime({
			gl: this._gl,
			programs: this._programs,
			geometry: this._geometry,
			maxTextureSize: this._maxTextureSize,
			maxTextureImageUnits: this._maxTextureImageUnits,
			getSceneFramebuffer: () => this._targets._sceneFramebuffer,
			getWidth: () => this._session.width,
			getHeight: () => this._session.height,
		});
		this._clusteredLighting = new WebGLClusteredLightingRuntime(gl);
		this._customRenderTargets = new WebGLCustomRenderTargetRuntime(gl, {
			restoreFrameState: (context) => this._restoreCustomRenderPassState(context),
		});
		this._postProcess = new WebGLPostProcessServices({
			gl: this._gl,
			targets: this._targets,
			getProgramCompiler: () => this._programCompiler,
			getFullscreenVao: () => this._fullscreen._vao,
			getWidth: () => this._session.width,
			getHeight: () => this._session.height,
			getActiveContext: () => this._session.context,
			drawFullscreen: (width, height, context) =>
				this._fullscreen.draw(width, height, context),
		});
		this._transparency = new WebGLTransparencyRuntime({
			gl: this._gl,
			targets: this._targets,
			getPrograms: () => this._programs,
			getFullscreenVao: () => this._fullscreen._vao,
			getWidth: () => this._session.width,
			getHeight: () => this._session.height,
			renderPackets: (context, packets, transparent, options) =>
				this._scene.renderPackets(context, packets, transparent, options),
			renderParticles: (context, options) => this._renderParticles(context, options),
			drawFullscreen: (width, height, context) =>
				this._fullscreen.draw(width, height, context),
		});
		this._warmup = new WebGLWarmupCoordinator({
			getPrograms: () => this._programs,
			getCompiler: () => this._programCompiler,
			postProcessRuntime: this._postProcessRuntime,
			postProcess: this._postProcess,
			enableEarlyZPrepass: this._enableEarlyZPrepass,
			maxTextureImageUnits: this._maxTextureImageUnits,
			irradianceProbeGridSamplingSupported:
				this._irradianceProbeGridSamplingSupported,
		});
	}

	/** @internal Returns the single readonly shadow-consumer dependency. */
	public getShadowSamplingState(): WebGLShadowSamplingState {
		return this._shadow.getSamplingState();
	}

	public beginFrame(
		context: FrameContext,
		materialGBufferRequested = false,
	): void {
		this._programCompiler.beginFrame();
		this._textures.beginFrame();
		this._customRenderTargets.sync(context);
		this._session.begin(context);
		this._scene.beginFrame();
		this._ensureFrameTargets(
			this._session.width,
			this._session.height,
			materialGBufferRequested,
		);
		this._transparency.beginFrame(context);
		this._targets._presentSourceTexture = this._targets._sceneColorTexture;
		this._oitPassMode = 0;
		this._shadow.beginFrame(context);
		this._session.lightState = collectWebGLLights(
			context.scene.lights,
			{
				enableLighting: context.features.enableLighting,
				enableShadows: context.features.enableShadows,
				shadowPlan: context.shadowPlan,
				enableSH: context.features.enableSH,
				environmentTexture: context.scene.environment.lightingEnabled ?
					context.scene.environment.iblTexture
				:	null,
				enableClusteredLighting:
					context.features.enableClusteredLighting,
				cameraWorldPosition: context.viewCamera.getWorldPosition(
					WEBGL_REFLECTION_PROBE_CAMERA_WORLD_POSITION_SCRATCH,
				),
			}
		);
		this._shadow.prepareFrame(context, this._session.lightState);
		this._clusteredLighting.prepare(
			context,
			this._session.lightState,
			this._maxTextureSize
		);

	}

	public beginTemporalFrame(
		context: FrameContext,
		frameRequirements: FramePreparationRequirements,
	): void {
		this._temporalFrameState.beginFrame(context, frameRequirements);
	}

	public commitTemporalFrame(): void {
		this._temporalFrameState.commitFrame();
	}

	public clearFrameTargets(context: FrameContext): void {
		this._scene.clearFrameTargets(context);
	}

	public renderEnvironmentNode(context: FrameContext): void {
		this._scene.renderEnvironment(context);
	}

	public isOITActive(): boolean {
		return this._transparency.isActive();
	}

	public hasPresentedInFrame(): boolean {
		return this._session.presented;
	}

	public collectFrameGraphResources(): readonly string[] {
		const resources = new Set(this._targets.collectGraphResources());
		for (const descriptor of this._shadow.describeGraphResources().resources) {
			resources.add(descriptor.id);
		}
		return Array.from(resources);
	}

	public collectFrameGraphResourceCatalog(includeShadowResources = true) {
		const frameCatalog = this._targets.collectGraphResourceCatalog();
		if (!includeShadowResources) return frameCatalog;
		const shadowCatalog = this._shadow.describeGraphResources();
		return Object.freeze({
			resources: Object.freeze([
				...frameCatalog.resources,
				...shadowCatalog.resources,
			]),
			bindings: Object.freeze([
				...frameCatalog.bindings,
				...shadowCatalog.bindings,
			]),
		});
	}

	public renderShadowNode(context: FrameContext): void {
		this._shadow.renderPreparedFrame(context);
	}

	public renderOpaqueDepthPrepass(context: FrameContext): Set<string> {
		return this._scene.renderOpaqueDepthPrepass(context);
	}

	public renderOpaqueScene(
		context: FrameContext,
		earlyZPacketIds: ReadonlySet<string>
	): void {
		this._scene.renderOpaque(context, earlyZPacketIds);
	}

	public renderTransparentLegacy(context: FrameContext): void {
		this._scene.renderLegacyTransparent(context);
	}

	public prepareOITTransparent(context: FrameContext): void {
		this._transparency.prepareTransparent(context);
	}

	public renderOITTransparentAccum(context: FrameContext): void {
		this._transparency.renderTransparentAccum(context);
	}

	public renderOITTransparentReveal(context: FrameContext): void {
		this._transparency.renderTransparentReveal(context);
	}

	public copySceneColorForOIT(context: FrameContext): void {
		this._transparency.copySceneColor(context);
	}

	public resolveOIT(context: FrameContext): void {
		this._transparency.resolve(context);
	}

	public renderOITLegacyTransparent(context: FrameContext): void {
		this._transparency.renderLegacy(context);
	}

	public prepareOITParticles(): void {
		this._transparency.prepareParticles();
	}

	public renderOITParticleAccum(context: FrameContext): void {
		this._transparency.renderParticleAccum(context);
	}

	public renderOITParticleReveal(context: FrameContext): void {
		this._transparency.renderParticleReveal(context);
	}

	public renderParticlesLegacy(context: FrameContext): void {
		this._transparency.renderParticlesLegacy(context);
	}

	public renderOITAdditiveParticles(context: FrameContext): void {
		this._transparency.renderAdditiveParticles(context);
	}

	public presentFrame(): void {
		if (!this._session.presented) {
			this._fullscreen.present(this._session.context, true);
		}
	}

	public finishFrame(): void {
		this._customRenderTargets.markFrameCommitted();
		this._scene.finishFrame();
		this._shadow.abortFrame();
		this._session.finish();
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

	private _restoreCustomRenderPassState(context: FrameContext): void {
		const gl = this._gl;
		const sceneFramebuffer = this._targets._sceneFramebuffer;
		gl.bindFramebuffer(gl.FRAMEBUFFER, sceneFramebuffer);
		gl.viewport(0, 0, context.attachments.width, context.attachments.height);
		if (sceneFramebuffer) {
			const drawBuffers =
				this._targets._materialGBufferEnabled ?
					[
						gl.COLOR_ATTACHMENT0,
						gl.COLOR_ATTACHMENT1,
						gl.COLOR_ATTACHMENT2,
						gl.COLOR_ATTACHMENT3,
						gl.COLOR_ATTACHMENT4,
					]
				:	this._targets._sceneNormalTexture ?
					[
						gl.COLOR_ATTACHMENT0,
						gl.COLOR_ATTACHMENT1,
						gl.COLOR_ATTACHMENT2,
					]
				:	[gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1];
			gl.drawBuffers(drawBuffers);
			gl.readBuffer(gl.COLOR_ATTACHMENT0);
		} else {
			gl.drawBuffers([gl.BACK]);
			gl.readBuffer(gl.BACK);
		}
		gl.disable(gl.SCISSOR_TEST);
		gl.disable(gl.BLEND);
		gl.disable(gl.CULL_FACE);
		gl.enable(gl.DEPTH_TEST);
		gl.depthMask(true);
		gl.depthFunc(gl.LESS);
		gl.colorMask(true, true, true, true);
		gl.useProgram(null);
		gl.bindVertexArray(null);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, null);
	}

	/** @internal Restores the queue-owned baseline after auxiliary context work. */
	public restoreContextWorkBaseline(): void {
		if (this._session.context) {
			this._restoreCustomRenderPassState(this._session.context);
			return;
		}
		const gl = this._gl;
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		gl.viewport(0, 0, this._session.width, this._session.height);
		gl.drawBuffers([gl.BACK]);
		gl.readBuffer(gl.BACK);
		gl.disable(gl.SCISSOR_TEST);
		gl.disable(gl.BLEND);
		gl.disable(gl.CULL_FACE);
		gl.enable(gl.DEPTH_TEST);
		gl.depthMask(true);
		gl.depthFunc(gl.LESS);
		gl.colorMask(true, true, true, true);
		gl.useProgram(null);
		gl.bindVertexArray(null);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, null);
	}

	public readCustomRenderTargetColor(
		id: string,
		attachmentIndex?: number,
		options?: RenderTargetReadbackOptions
	): Promise<RenderTargetReadbackResult> {
		return this._customRenderTargets.readColor(id, attachmentIndex, options);
	}

	public createPostProcessResource(
		desc: PostProcessResourceDescriptor
	): PostProcessResourceHandle {
		return this._postProcess.createResource(desc);
	}

	public destroyPostProcessResource(handle: PostProcessResourceHandle): void {
		this._postProcess.destroyResource(handle);
	}

	public createGBufferBridge(context: FrameContext): LogicalGBufferBridge {
		return this._targets.createGBufferBridge(context);
	}

	public createPassExecutionContext(
		request: PostProcessPassExecutionContextRequest
	): unknown {
		return this._postProcess.createPassExecutionContext(request);
	}

	/** @internal Opens controlled output publication for a runtime post-process frame. */
	public beginPostProcessFrame(): void {
		this._postProcess.beginFrame();
	}

	/** @internal Closes controlled output publication for a runtime post-process frame. */
	public endPostProcessFrame(): void {
		this._postProcess.endFrame();
	}

	/** @internal Aborts controlled output publication without aborting the renderer frame. */
	public abortPostProcessFrame(): void {
		this._postProcess.abortFrame();
	}

	/** @internal Commits validated WebGL post-process output for one pass. */
	public completePostProcessPass(
		request: PostProcessPassRequest,
		result: PostProcessPassResult
	): PostProcessPassCompletion {
		return this._postProcess.completePass(request, result);
	}

	public abortFrame(): void {
		this._shadow.abortFrame();
		this._temporalFrameState.abortFrame();
		this._postProcess.abortFrame();
		this._customRenderTargets.markFrameAborted();
		this._session.abort();
		this._targets._presentSourceTexture = this._targets._sceneColorTexture;
		this._oitPassMode = 0;
		this._transparency.abortFrame();
		this._scene.abortFrame();
	}

	public resize(width: number, height: number): void {
		this._session.width = toSafeDimension(width);
		this._session.height = toSafeDimension(height);
		this._temporalFrameState.reset();
		this._destroyFrameTargets();
	}

	public destroy(): void {
		this._customRenderTargets.destroy();
		this._destroyFrameTargets();
		this._shadow.destroy();
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
		this._scene.destroy();
		this._fullscreen.destroy();
		this._geometry.destroy();
		this._textures.destroy();
		this._programs.destroy();
		this._programCompiler.destroy();
		this._temporalFrameState.reset();
		this._session.finish();
	}

	public _isIncrementalPartial(context: FrameContext | null | undefined): boolean {
		if (!context?.incremental) {
			return false;
		}
		return (
			context.incremental.enabled &&
			!context.incremental.forceFullFrame &&
			context.incremental.dirtyRects.length > 0
		);
	}

	public _resolveDirtyRects(
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

	public _resolvePacketsForRect(
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

	public _setScissorRect(
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

	public _resolveSceneProgramVariant(
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
				lightState: this._session.lightState,
				enableShadowTransmittance:
					this.getShadowSamplingState().transmittanceAvailable,
				enableIrradianceProbeGrid: this._irradianceProbeGridSamplingSupported,
			},
			this._targets._materialGBufferEnabled
		);
	}

	public _resolveSceneDepthPrepassVariant(
		packet: DrawPacket
	): WebGLSceneDepthVariantDescriptor | null {
		return resolveWebGLBuiltinDepthVariant(packet.material);
	}

	public _drawPacket(
		sceneProgram: WebGLSceneProgram,
		packet: DrawPacket,
		transparentPass: boolean,
		context: FrameContext,
		options?: WebGLPacketDrawOptions
	): void {
		drawWebGLPacket(
			this,
			sceneProgram,
			packet,
			transparentPass,
			context,
			options
		);
	}

	public _bindShaderMaterialTextures(
		sceneProgram: WebGLSceneProgram,
		material: Material
	): void {
		bindWebGLShaderMaterialTextures(
			this,
			sceneProgram,
			material
		);
	}

	public _bindShaderMaterialUniforms(
		sceneProgram: WebGLSceneProgram,
		material: Material
	): void {
		bindWebGLShaderMaterialUniforms(
			this,
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

	public _bindGlobalUniforms(
		sceneProgram: WebGLSceneProgram,
		context: FrameContext
	): void {
		bindWebGLGlobalUniforms(
			this,
			sceneProgram,
			context
		);
	}

	public _uploadSHAmbientCoefficients(
		coeffs: SHCoefficients | null | undefined
	): boolean {
		return uploadWebGLSHAmbientCoefficients(
			this,
			coeffs
		);
	}

	public _uploadLocalLightProbeCoefficients(
		probes: NonNullable<WebGLLightState["localLightProbes"]>
	): boolean {
		return uploadWebGLLocalLightProbeCoefficients(
			this,
			probes
		);
	}

	public _uploadIrradianceProbeGridCoefficients(
		grid: WebGLLightState["irradianceProbeGrid"]
	): boolean {
		return uploadWebGLIrradianceProbeGridCoefficients(
			this,
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
		const aspect = context.viewCamera.aspectRatio || this._session.width / this._session.height;

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._targets._sceneFramebuffer);
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

	public _updateFogParams(options: FogOptions | undefined, enabled: boolean): void {
		resolveFogUniformParams(options, enabled, this._fogParams0, this._fogParams1);
	}

	private _resolvePostProcessTargetTexture(
		sourceTexture: WebGLTexture
	): WebGLTexture | null {
		return this._targets.resolvePostProcessTargetTexture(sourceTexture);
	}

	private _bindPostSingleColorTarget(texture: WebGLTexture): void {
		this._targets.bindPostSingleColorTarget(texture);
	}

	private _bindOITSingleColorTarget(texture: WebGLTexture): void {
		this._targets.bindOITSingleColorTarget(texture);
	}

	private _ensureFrameTargets(
		width: number,
		height: number,
		materialGBufferRequested: boolean
	): void {
		this._targets.ensure(
			width,
			height,
			materialGBufferRequested,
		);
	}

	private _destroyFrameTargets(): void {
		this._targets.destroy();
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

	public _setCullMode(material: Material): void {
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
