import {
	AlphaMode,
	type Material,
} from "../../materials/Material";
import type { ShaderTargetMode } from "../../materials/ShaderMaterial";
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
import { IBLBRDF } from "../../lights/ibl/IBLBRDF";
import {
	collectWebGLLights,
	type WebGLLightState,
	type WebGLClusteredLight,
} from "./WebGLLightCollector";
import { WebGLGeometryRegistry } from "./WebGLGeometryRegistry";
import {
	DEFAULT_DEFERRED_GEOMETRY_UPLOAD_BYTES_PER_FRAME,
	DEFAULT_DEFERRED_GEOMETRY_UPLOADS_PER_FRAME,
} from "./WebGLGeometryRegistry";
import {
	IDENTITY_MATRIX4_COLUMN_MAJOR,
	WEBGL_REFLECTION_PROBE_CAMERA_WORLD_POSITION_SCRATCH,
} from "./constants";
import { WebGLSceneProgramRepository } from "./WebGLSceneProgramRepository";
import type { WebGLSceneProgram } from "./WebGLSceneProgram";
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
	resolveMaterialUniforms,
	resolveTextureUVTransform,
} from "./WebGLMaterialUniformResolver";
import { Logger } from "../../foundation/Logger";
import { WebGLFrameTargetManager } from "./WebGLFrameTargetManager";
import { WebGLFrameSession } from "./WebGLFrameSession";
import { WebGLFullscreenRenderer } from "./WebGLFullscreenRenderer";
import { WebGLEnvironmentRenderer } from "./WebGLEnvironmentRenderer";
import { bindWebGLGlobalUniforms } from "./WebGLGlobalUniformBinder";
import {
	drawWebGLPacket,
	createWebGLSceneDrawState,
	type WebGLScenePassDeps,
} from "./WebGLScenePass";
import { WebGLProbeSHTextures } from "./WebGLProbeSHTextures";
import { WebGLFogState } from "./WebGLFogState";
import {
	WebGLShadowRuntime,
	type WebGLShadowSamplingState,
} from "./WebGLShadowRuntime";
import {
	WebGLParticlePass,
	type WebGLParticleRenderOptions,
} from "./WebGLParticlePass";
import { BackendPostProcessRuntime } from "../../postprocess/BackendPostProcessRuntime";
import {
	WebGLPostProcessServices,
	WebGLPostProcessWarmupContributor,
} from "./WebGLPostProcessServices";
import { WebGLTemporalFrameState } from "./WebGLTemporalFrameState";
import { WebGLWarmupCoordinator } from "./WebGLWarmupCoordinator";
import {
	WebGLTransparencyRuntime,
	WebGLTransparencyWarmupContributor,
} from "./WebGLTransparencyRuntime";
import {
	planWebGLScenePrograms,
	WebGLSceneProgramWarmupContributor,
} from "./WebGLSceneProgramPlanner";
import { WebGLCustomRenderTargetRuntime } from "./WebGLCustomRenderTargetRuntime";
import type { FramePass } from "../../pipeline/types";
import type {
	RenderTargetReadbackOptions,
	RenderTargetReadbackResult,
} from "../../rendering/CustomRenderTargets";
import type { DisplayOutputState } from "../../rendering/DisplayOutput";
import type { PostProcessColorDomain } from "../../postprocess/PostProcessPass";
import { WebGLSceneRuntime } from "./WebGLSceneRuntime";
import { WebGLAnimationPayloadPool } from "./WebGLAnimationPayloadPool";
import {
	createWebGLVertexTextureUnitLayout,
	type WebGLVertexTextureUnitLayout,
} from "./WebGLVertexTextureUnits";

export interface WebGLFrameServicesOptions {
	validatePrograms?: boolean;
	enableEarlyZPrepass?: boolean;
	onProgramCompilePending?: () => void;
	/**
	 * Called when deferred WebGL texture uploads remain queued after the current
	 * upload budget. The callback should mark the renderer dirty so another frame
	 * can continue processing the queue.
	 */
	onTextureUploadPending?: () => void;
	/**
	 * Called when deferred WebGL geometry uploads remain queued after the
	 * current upload budget. The callback should mark the renderer dirty so
	 * another frame can continue processing the queue.
	 */
	onGeometryUploadPending?: (packets: readonly DrawPacket[]) => void;
	postProcessRuntime?: BackendPostProcessRuntime;
	getDisplayOutputState?: () => DisplayOutputState;
}

export class WebGLFrameServices {
	public _gl: WebGL2RenderingContext;
	private _programCompiler: WebGLProgramCompiler;
	public _scenePrograms: WebGLSceneProgramRepository;
	public _geometry: WebGLGeometryRegistry;
	public _animationPayloads: WebGLAnimationPayloadPool | null;
	public _textures: WebGLTextureRegistry;
	private _shadow: WebGLShadowRuntime;
	private _particlePass: WebGLParticlePass;
	private _targets: WebGLFrameTargetManager;
	private readonly _temporalFrameState = new WebGLTemporalFrameState();
	private _scene: WebGLSceneRuntime;
	private _fullscreen: WebGLFullscreenRenderer;
	private _environment: WebGLEnvironmentRenderer;
	private readonly _session = new WebGLFrameSession();
	private _maxTextureSize: number;
	private readonly _vertexTextureUnits: WebGLVertexTextureUnitLayout;
	private _maxRenderbufferSize: number;
	public _clusteredLighting: WebGLClusteredLightingRuntime;
	public _probeSHTextures: WebGLProbeSHTextures;
	public _fog = new WebGLFogState();
	private readonly _sceneDrawState = createWebGLSceneDrawState();
	/** @internal Scene-pass dependency bundle; exposed for test stubbing. */
	public readonly _sceneDeps: WebGLScenePassDeps;
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

	public async prepareSceneProgramSources(context: FrameContext): Promise<void> {
		const packets = [
			...(context.scene?.opaquePackets ?? []),
			...(context.scene?.transparentPackets ?? []),
		];
		const plan = planWebGLScenePrograms(
			context,
			packets,
			["mrt", "single"],
		);
		await this._scenePrograms.prepareBuiltinSceneVariants(
			plan.sceneVariants.values(),
		);
		// Issue compiles for planned variants ahead of the draw loop so
		// first-use resolution rarely blocks on link finalization.
		this._scenePrograms.issuePlannedSceneProgramCompiles(plan);
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

	public get _lightState(): WebGLLightState | null {
		return this._session.lightState;
	}
	public get _temporalJitterCurrentPrev(): Float32Array {
		return this._temporalFrameState.jitterCurrentPrev;
	}
	public get _previousViewProjection(): Float32Array | null {
		return this._temporalFrameState.previousViewProjection;
	}

	constructor(
		gl: WebGL2RenderingContext,
		shaderRuntime?: ShaderRuntime,
		shaderCompileStage?: ShaderBackendCompileStage,
		options: WebGLFrameServicesOptions = {},
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
		this._scenePrograms = new WebGLSceneProgramRepository({
			compiler: this._programCompiler,
			shaderRuntime,
			shaderCompileStage,
		});
		this._geometry = new WebGLGeometryRegistry(gl, undefined, {
			uploadScheduling: "deferred",
			maxUploadsPerFrame: DEFAULT_DEFERRED_GEOMETRY_UPLOADS_PER_FRAME,
			maxUploadBytesPerFrame: DEFAULT_DEFERRED_GEOMETRY_UPLOAD_BYTES_PER_FRAME,
			onUploadPending: options.onGeometryUploadPending,
		});
		this._textures = new WebGLTextureRegistry(gl, undefined, {
			uploadScheduling: "deferred",
			maxUploadsPerFrame: DEFAULT_DEFERRED_UPLOADS_PER_FRAME,
			maxUploadBytesPerFrame: DEFAULT_DEFERRED_UPLOAD_BYTES_PER_FRAME,
			onUploadPending: options.onTextureUploadPending,
		});
		this._probeSHTextures = new WebGLProbeSHTextures(gl);
		const fogUniformView = {
			params0: this._fog.params0,
			params1: this._fog.params1,
		};
		this._particlePass = new WebGLParticlePass({
			gl,
			programCompiler: this._programCompiler,
			textures: this._textures,
			getSceneFramebuffer: () => this._targets._sceneFramebuffer,
			getViewportSize: () => ({ width: this._session.width, height: this._session.height }),
			getFogParams: () => fogUniformView,
			isIncrementalPartial: (context) => this._isIncrementalPartial(context),
			resolveDirtyRects: (context, width, height) =>
				this._resolveDirtyRects(context, width, height),
			setScissorRect: (x, y, width, height, viewportHeight) =>
				this._setScissorRect(x, y, width, height, viewportHeight),
			updateFogParams: (fogOptions, enabled) =>
				this._fog.update(fogOptions, enabled),
		});
		this._maxTextureSize = this._resolveLimit(gl.MAX_TEXTURE_SIZE, 4096);
		this._vertexTextureUnits = createWebGLVertexTextureUnitLayout(gl);
		this._animationPayloads = typeof gl.createTexture === "function" ?
			new WebGLAnimationPayloadPool(
				gl,
				this._vertexTextureUnits,
				this._maxTextureSize,
			) : null;
		this._maxRenderbufferSize = this._resolveLimit(
			gl.MAX_RENDERBUFFER_SIZE,
			4096
		);
		this._targets = new WebGLFrameTargetManager(
			gl,
			this._maxTextureSize,
			this._maxRenderbufferSize,
		);
		this._fullscreen = new WebGLFullscreenRenderer({
			gl: this._gl,
			targets: this._targets,
			programCompiler: this._programCompiler,
			getWidth: () => this._session.width,
			getHeight: () => this._session.height,
			isIncrementalPartial: (context) => this._isIncrementalPartial(context),
			resolveDirtyRects: (context, width, height) =>
				this._resolveDirtyRects(context, width, height),
			setScissorRect: (x, y, width, height, viewportHeight) =>
				this._setScissorRect(x, y, width, height, viewportHeight),
			getDisplayOutputState: () => options.getDisplayOutputState?.(),
			getColorDomain: () => this._postProcess.outputColorDomain,
			markPresented: () => {
				this._session.presented = true;
			},
		});
		this._environment = new WebGLEnvironmentRenderer({
			gl: this._gl,
			programCompiler: this._programCompiler,
			targets: this._targets,
			textures: this._textures,
			getFullscreenVao: () => this._fullscreen._vao,
			getWidth: () => this._session.width,
			getHeight: () => this._session.height,
		});
		const modelMatrixCache = new Map<string, Float32Array>();
		const modelMatrixKeysThisFrame = new Set<string>();
		const sceneDeps: WebGLScenePassDeps = {
			gl: this._gl,
			targets: this._targets,
			drawState: this._sceneDrawState,
			scenePrograms: this._scenePrograms,
			geometry: this._geometry,
			textures: this._textures,
			animationPayloads: this._animationPayloads,
			modelMatrixCache,
			modelMatrixKeysThisFrame,
			getWidth: () => this._session.width,
			getHeight: () => this._session.height,
			isIncrementalPartial: (context) => this._isIncrementalPartial(context),
			resolveDirtyRects: (context, width, height) =>
				this._resolveDirtyRects(context, width, height),
			resolvePacketsForRect: (context, packets, rect) =>
				this._resolvePacketsForRect(context, packets, rect),
			setScissorRect: (x, y, width, height, viewportHeight) =>
				this._setScissorRect(x, y, width, height, viewportHeight),
			bindGlobalUniforms: (sceneProgram, context) =>
				this._bindGlobalUniforms(sceneProgram, context),
			bindAnimationPayload: (sceneProgram, packet) =>
				this._bindAnimationPayload(sceneProgram, packet),
			drawPacket: (sceneProgram, packet, transparentPass, context, options) =>
				drawWebGLPacket(
					this._sceneDeps,
					sceneProgram,
					packet,
					transparentPass,
					context,
					options
				),
			getLightState: () => this._session.lightState,
			getShadowSamplingState: () => this.getShadowSamplingState(),
		};
		this._sceneDeps = sceneDeps;
		this._scene = new WebGLSceneRuntime({
			gl: this._gl,
			deps: sceneDeps,
			modelMatrixCache,
			modelMatrixKeysThisFrame,
			targets: this._targets,
			enableEarlyZPrepass: this._enableEarlyZPrepass,
			getWidth: () => this._session.width,
			getHeight: () => this._session.height,
			isIncrementalPartial: (context) => this._isIncrementalPartial(context),
			resolveDirtyRects: (context, width, height) =>
				this._resolveDirtyRects(context, width, height),
			setScissorRect: (x, y, width, height, viewportHeight) =>
				this._setScissorRect(x, y, width, height, viewportHeight),
			renderEnvironment: (context) => {
				this._environment.render(context);
			},
			renderLegacyTransparent: (context) =>
				this._transparency.renderLegacyTransparent(context),
		});
		this._shadow = new WebGLShadowRuntime({
			gl: this._gl,
			programCompiler: this._programCompiler,
			geometry: this._geometry,
			animationPayloads: this._animationPayloads ?? undefined,
			maxTextureSize: this._maxTextureSize,
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
			getDisplayOutputState: () => options.getDisplayOutputState?.(),
			drawFullscreen: (width, height, context) =>
				this._fullscreen.draw(width, height, context),
		});
		this._transparency = new WebGLTransparencyRuntime({
			gl: this._gl,
			targets: this._targets,
			programCompiler: this._programCompiler,
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
			compiler: this._programCompiler,
			contributors: [
				new WebGLSceneProgramWarmupContributor(
					this._scenePrograms,
					this._enableEarlyZPrepass,
				),
				this._environment,
				this._shadow,
				this._particlePass,
				new WebGLTransparencyWarmupContributor(this._transparency),
				new WebGLPostProcessWarmupContributor(
					this._postProcessRuntime,
					this._postProcess,
				),
				this._fullscreen,
			],
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
		this._geometry.beginFrame();
		this._textures.beginFrame();
		this._customRenderTargets.sync(context);
		this._session.begin(context);
		this._animationPayloads?.beginFrame(context);
		this._scene.beginFrame();
		this._ensureFrameTargets(
			this._session.width,
			this._session.height,
			materialGBufferRequested,
		);
		this._transparency.beginFrame(context);
		this._targets._presentSourceTexture = this._targets._sceneColorTexture;
		this._postProcess.setInitialColorDomain("scene-linear-hdr");
		this._sceneDrawState.oitPassMode = 0;
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
		this._transparency.renderLegacyTransparent(context);
	}

	public prepareTransmissionDepth(context: FrameContext): void {
		this._transparency.prepareTransmissionDepth(context);
	}

	public renderLegacyTransparentSegment(
		context: FrameContext,
		start: number,
		end: number,
	): void {
		this._transparency.renderLegacyTransparentSegment(context, start, end);
	}

	public copyTransmissionBackground(context: FrameContext): void {
		this._transparency.copyTransmissionBackground(context);
	}

	public renderTransmissionPacket(context: FrameContext, index: number): void {
		this._transparency.renderTransmissionPacket(context, index);
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

	public setPostProcessInitialColorDomain(domain: PostProcessColorDomain): void {
		this._postProcess.setInitialColorDomain(domain);
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
		this._sceneDrawState.oitPassMode = 0;
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
		this._transparency.destroy();
		this._customRenderTargets.destroy();
		this._destroyFrameTargets();
		this._shadow.destroy();
		this._particlePass.destroy();
		this._clusteredLighting.destroy();
		this._probeSHTextures.destroy();
		this._scene.destroy();
		this._environment.destroy();
		this._fullscreen.destroy();
		this._geometry.destroy();
		this._animationPayloads?.destroy();
		this._textures.destroy();
		this._scenePrograms.destroy();
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

	public _bindAnimationPayload(
		sceneProgram: WebGLSceneProgram,
		packet: DrawPacket,
	): boolean {
		if (!sceneProgram.uniforms.animationCounts) return true;
		if (!this._animationPayloads) {
			return !packet.meshInstance.skeleton &&
				(packet.primitive.geometry.morphTargets?.length ?? 0) === 0;
		}
		const geometry = this._geometry.getGeometry(packet);
		if (!geometry) return false;
		return this._animationPayloads.bind(sceneProgram.uniforms, packet, geometry);
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
}

function toSafeDimension(value: unknown): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return 1;
	}
	return Math.max(1, Math.floor(value));
}
