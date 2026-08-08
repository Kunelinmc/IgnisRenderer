import type { Texture } from "../../core/Texture";
import type {
	DrawPacket,
	FrameContext,
} from "../../pipeline/types";
import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../../core/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import {
	TextureFormat,
	type IBindingGroup,
	type IComputePipeline,
	type IRenderPipeline,
	type IRenderTexture,
	type ISampler,
} from "../types";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import {
	createBaselineFramePacketSet,
	type PreparedFramePacketSet,
} from "../../pipeline/FramePacketContributorRegistry";
import type { WebGPUResourceManager } from "./WebGPUResourceManager";
import type { IWebGPUComputeFacade } from "./ComputeFacade";
import {
	ANIMATION_WEBGPU_JOINT_MATRICES_KEY,
	ANIMATION_WEBGPU_MORPH_WEIGHTS_KEY,
} from "../../simulation/animation/types";
import {
	collectWebGPUEnvironment,
	collectWebGPULightingCatalog,
	createWebGPULightingState,
	createWebGPUMaterialUniformData,
	type WebGPUEnvironmentState,
	type WebGPUFeatureState,
	type WebGPULightingState,
} from "./";
import type { WebGPUFrameFeatureDataStore } from "./FrameFeatures";
import {
	WEBGPU_CLUSTERED_LIGHTING_DATA,
	canPrepareClusteredLighting,
	createWebGPUFrameFeatureRegistry,
} from "./WebGPUFrameFeatureModules";
import { createWebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";
import { WebGPUFrameBindingCache } from "./WebGPUFrameBindingCache";
import { WebGPUClusteredLightingRuntime } from "./WebGPUClusteredLightingRuntime";
import { WebGPUGeometryRegistry, type WebGPUGeometryHandle } from "./WebGPUGeometryRegistry";
import {
	WebGPUMaterialBindingCache,
	type WebGPUModelAnimationBindingState,
} from "./WebGPUMaterialBindingCache";
import { WebGPUPipelineLibrary } from "./WebGPUPipelineLibrary";
import { WebGPUDeferredResources } from "./WebGPUDeferredResources";
import type {
	WebGPUScenePipelineDrawMode,
	WebGPUSceneTargetMode,
	WebGPUTransparentPipelineMode,
} from "./WebGPUPipelineLibrary";
import {
	type WebGPUPagedShadowFrameRequest,
} from "./WebGPUPagedShadowTechnique";
import { WebGPUShadowRuntime } from "./WebGPUShadowRuntime";
import type { ShadowCastingLight } from "../../lights";
import { WebGPUTextureRegistry } from "./WebGPUTextureRegistry";
import type { WarmupPhaseCounters, WarmupPlan } from "../../pipeline/WarmupPlanner";
import { toShaderCompileError } from "../../pipeline/WarmupPlanner";
import { createWarmupYieldController } from "../../pipeline/WarmupScheduler";
import type { ShaderCompileError } from "../../shaders/runtime";
import { Logger } from "../../foundation/Logger";
import type { WarmupOptions } from "../IRenderBackend";
import type {
	WebGPUDrawResourceOptions,
	WebGPUDrawResources,
	WebGPUEnvironmentDrawResources,
	WebGPUEnvironmentResourceOptions,
	WebGPUFrameResourceScope as WebGPUFrameResourceScopeContract,
	WebGPUFrameScopePrepareOptions,
	WebGPUParticleBillboardRenderer,
	WebGPUPreparedFrameResources as WebGPUPreparedFrameResourcesContract,
} from "./WebGPUResourceContracts";
import { WebGPUParticleRenderResources } from "./WebGPUParticleRenderResources";

interface WebGPUFrameServicePrepareOptions extends WebGPUFrameScopePrepareOptions {
	readonly scopeKey: string;
}

type WebGPUFrameServicePreparedResources = WebGPUPreparedFrameResourcesContract & {
	readonly scopeKey: string;
};

interface WebGPUFrameResourceScope {
	frameBindings: WebGPUFrameBindingCache;
	clusteredLighting: WebGPUClusteredLightingRuntime;
	prepared: WebGPUFrameServicePreparedResources | null;
}

class WebGPUFrameResourceScopeHandle implements WebGPUFrameResourceScopeContract {
	private _prepared: WebGPUFrameServicePreparedResources | null = null;
	private _destroyed = false;

	constructor(
		private readonly _owner: WebGPUFrameServiceOwner,
		private readonly _scopeKey: string,
	) {}

	public prepare(
		context: FrameContext,
		options: WebGPUFrameScopePrepareOptions,
	): WebGPUPreparedFrameResourcesContract {
		if (this._destroyed) {
			throw new Error("WebGPU frame resource scope has been destroyed.");
		}
		this._prepared = this._owner.prepareFrame(context, {
			...options,
			scopeKey: this._scopeKey,
		});
		return this._prepared;
	}

	public updateParticleShadowVolumes(context: FrameContext): void {
		if (!this._prepared) {
			throw new Error("WebGPU frame resource scope has not been prepared.");
		}
		this._owner.updateParticleShadowVolumes(this._prepared, context);
	}

	public destroy(): void {
		if (this._destroyed) {
			return;
		}
		this._destroyed = true;
		this._prepared = null;
		this._owner.releaseScope(this._scopeKey);
	}
}

/** @internal WebGPU backend-private composition and shared-resource owner. */
export class WebGPUFrameServiceOwner {
	private _backend: WebGPUDeviceResourceHost;
	private _resourceManager: WebGPUResourceManager;
	private _computeFacade: IWebGPUComputeFacade;
	private _layouts: ReturnType<typeof createWebGPUPipelineLayouts>;
	private _geometryRegistry: WebGPUGeometryRegistry;
	private _textureRegistry: WebGPUTextureRegistry;
	private _pipelineLibrary: WebGPUPipelineLibrary;
	private _materialBindings: WebGPUMaterialBindingCache;
	private _shadowRuntime: WebGPUShadowRuntime;
	private _shadowRuntimeDestroyed = false;
	private _frameFeatureRegistry = createWebGPUFrameFeatureRegistry();
	private _frameScopes = new Map<string, WebGPUFrameResourceScope>();
	private _nextFrameScopeId = 0;
	private _deferredResources: WebGPUDeferredResources;
	private _particleRenderResources: WebGPUParticleRenderResources;
	private _destroyed = false;

	constructor(
		backend: WebGPUDeviceResourceHost,
		resourceManager: WebGPUResourceManager,
		computeFacade: IWebGPUComputeFacade,
	) {
		this._backend = backend;
		this._resourceManager = resourceManager;
		this._computeFacade = computeFacade;
		const device = backend.device;
		if (!device) {
			throw new Error("WebGPU backend is not initialized; cannot create render resources.");
		}
		this._layouts = createWebGPUPipelineLayouts(device);
		this._geometryRegistry = new WebGPUGeometryRegistry(backend);
		this._textureRegistry = new WebGPUTextureRegistry(backend, resourceManager);
		this._pipelineLibrary = new WebGPUPipelineLibrary(backend, this._layouts, {
			listenToShaderRuntime: false,
		});
		this._particleRenderResources = new WebGPUParticleRenderResources(
			backend,
			this._layouts,
			this._textureRegistry,
		);
		this._deferredResources = new WebGPUDeferredResources(backend, this._layouts, () =>
			this._pipelineLibrary.getDeferredLightingPipeline(),
		);
		this._materialBindings = new WebGPUMaterialBindingCache(backend, this._layouts);
		this._shadowRuntime = new WebGPUShadowRuntime(
			backend,
			resourceManager,
			this._geometryRegistry
		);
	}

	public async init(): Promise<void> {
		await this._pipelineLibrary.init();
	}

	/** @internal Creates an isolated frame-binding ownership scope. */
	public createFrameScope(): WebGPUFrameResourceScopeContract {
		if (this._destroyed) {
			throw new Error("WebGPU frame service owner has been destroyed.");
		}
		this._nextFrameScopeId++;
		return new WebGPUFrameResourceScopeHandle(this, `frame-scope-${this._nextFrameScopeId}`);
	}

	public async warmup(
		context: FrameContext,
		plan: WarmupPlan,
		options: WarmupOptions = {},
		framePackets: PreparedFramePacketSet = createBaselineFramePacketSet(context),
	): Promise<WarmupPhaseCounters> {
		let total = 0;
		let compiled = 0;
		let skipped = 0;
		let failed = 0;
		const errors: ShaderCompileError[] = [];
		const yieldController = createWarmupYieldController(options);
		const warmupScopeKey = "warmup-main";
		let warmupResources: WebGPUFrameServicePreparedResources | null = null;

		try {
			warmupResources = this.prepareFrame(context, {
				scopeKey: warmupScopeKey,
				sceneTargetMode: plan.sceneTargetMode,
				framePackets,
				temporalStateMode: "disabled",
			});
		} catch (error) {
			failed++;
			errors.push(toShaderCompileError(error, "webgpu", "WebGPUPrepareFrame"));
		}

		if (plan.enableEnvironment && warmupResources) {
			total++;
			try {
				await this.getEnvironmentResources(warmupResources, plan.sceneTargetMode, {
					sampleCount: 1,
				});
				compiled++;
			} catch (error) {
				failed++;
				errors.push(toShaderCompileError(error, "webgpu", "WebGPUEnvironmentWarmup"));
			}
			await yieldController.yieldIfNeeded();
		}

		const drawPackets = [...context.scene.opaquePackets, ...context.scene.transparentPackets];
		for (const packet of drawPackets) {
			total++;
			try {
				const resources = warmupResources
					? await this.getDrawResources(packet, warmupResources, { sampleCount: 1 })
					: null;
				if (resources && resources.length > 0) {
					compiled++;
				} else {
					skipped++;
				}
			} catch (error) {
				failed++;
				errors.push(toShaderCompileError(error, "webgpu", `WebGPUDrawWarmup:${packet.id}`));
			}
			await yieldController.yieldIfNeeded();
		}

		if (context.features.enableReflection && context.scene.reflectivePackets.length > 0) {
			let reflectionResources: WebGPUFrameServicePreparedResources | null = null;
			try {
				reflectionResources = this.prepareFrame(context, {
					scopeKey: "warmup-planar-reflection",
					sceneTargetMode: "color",
					framePackets,
					temporalStateMode: "disabled",
				});

				for (const packet of drawPackets) {
					total++;
					try {
						const resources = reflectionResources
							? await this.getDrawResources(packet, reflectionResources, {
									sceneTargetMode: "color",
									drawMode: "reflection-capture",
									sampleCount: 1,
								})
							: null;
						if (resources && resources.length > 0) {
							compiled++;
						} else {
							skipped++;
						}
					} catch (error) {
						failed++;
						errors.push(
							toShaderCompileError(
								error,
								"webgpu",
								`WebGPUPlanarReflectionCaptureWarmup:${packet.id}`,
							),
						);
					}
					await yieldController.yieldIfNeeded();
				}

				for (const packet of context.scene.reflectivePackets) {
					total++;
					try {
						const resources = reflectionResources
							? await this.getDrawResources(packet, reflectionResources, {
									sceneTargetMode: "mrt",
									drawMode: "planar-reflection-composite",
									sampleCount: 1,
								})
							: null;
						if (resources && resources.length > 0) {
							compiled++;
						} else {
							skipped++;
						}
					} catch (error) {
						failed++;
						errors.push(
							toShaderCompileError(
								error,
								"webgpu",
								`WebGPUPlanarReflectionCompositeWarmup:${packet.id}`,
							),
						);
					}
					await yieldController.yieldIfNeeded();
				}
			} finally {
				this.releaseScope("warmup-planar-reflection");
			}
		}

		if (plan.enableShadows) {
			total++;
			try {
				await this._shadowRuntime.warmup();
				compiled++;
			} catch (error) {
				failed++;
				errors.push(toShaderCompileError(error, "webgpu", "WebGPUShadowWarmup"));
			}
			await yieldController.yieldIfNeeded();
		}

		if (plan.enableParticles) {
			total++;
			try {
				await this._particleRenderResources.warmup(plan.sceneTargetMode);
				compiled++;
			} catch (error) {
				failed++;
				errors.push(toShaderCompileError(error, "webgpu", "WebGPUParticleWarmup"));
			}
			await yieldController.yieldIfNeeded();
		}

		this.releaseScope(warmupScopeKey);
		return {
			phase: "webgpu-resources",
			total,
			compiled,
			skipped,
			failed,
			errors,
		};
	}

	public async renderShadows(
		context: FrameContext,
		framePackets: WebGPUFrameServicePrepareOptions["framePackets"],
		encoder?: ICommandEncoder | null,
	): Promise<void> {
		await this._shadowRuntime.renderAtlas(
			{
				...context,
				scene: {
					...context.scene,
					shadowCasterPackets: framePackets.shadowCasters.slice(),
					shadowTransmitterPackets: framePackets.shadowTransmitters.slice(),
				},
			},
			encoder,
		);
	}

	/**
	 * Advances caches that are shared across prepared resource scopes.
	 *
	 * @sideEffects Increments material/particle cache frame counters and evicts
	 * stale per-frame cache entries.
	 */
	public beginFrameResourceLifecycle(): void {
		this._materialBindings.beginFrame();
		this._particleRenderResources.beginFrame();
	}

	public prepareFrame(
		context: FrameContext,
		options: WebGPUFrameServicePrepareOptions,
	): WebGPUFrameServicePreparedResources {
		const resolvedOptions = this._resolvePrepareFrameOptions(context, options);
		const jointMatrixMap = context.transient.get(ANIMATION_WEBGPU_JOINT_MATRICES_KEY) ?? null;
		const morphWeightMap = context.transient.get(ANIMATION_WEBGPU_MORPH_WEIGHTS_KEY) ?? null;
		const scene = context.scene;
		const features = context.features;
		const postProcess = context.postProcess;
		const shAmbientCoeffs = context.shAmbientCoeffs;
		const renderWidth = Math.max(1, context.attachments.width || 1);
		const renderHeight = Math.max(1, context.attachments.height || 1);
		const temporalHistoryReset = context.incremental?.temporalHistoryReset === true;
		const shadowCasterPackets = resolvedOptions.framePackets.shadowCasters.slice();
		const shadowTransmitterPackets = resolvedOptions.framePackets.shadowTransmitters.slice();
		const temporalStateMode = resolvedOptions.temporalStateMode ?? "advance";
		const featureState: WebGPUFeatureState = {
			enableLighting: features.enableLighting,
			enableSH: features.enableSH,
			enableShadows: features.enableShadows,
			enableReflection: features.enableReflection,
			enableEnvironment: features.enableEnvironment,
			enableOIT: features.enableOIT,
			enableClusteredLighting: features.enableClusteredLighting,
			clusteredLightingOptions: features.clusteredLightingOptions,
			postProcess,
			warnings: [],
		};

		this._shadowRuntime.preparePaged({
			context,
			encoder: null,
			shadowPlan: context.shadowPlan,
			shadowCasterPackets,
			shadowTransmitterPackets,
		});

		const lightingCatalog = collectWebGPULightingCatalog(
			scene.lights,
			features.enableLighting,
			features.enableSH,
			features.enableShadows,
			context.shadowPlan,
		);
		const enableClusteredSurfaceLighting = canPrepareClusteredLighting({
			scene,
			featureState,
		});
		const lightingState = createWebGPULightingState(
			lightingCatalog,
			enableClusteredSurfaceLighting,
		);
		for (const warning of lightingState.warnings) {
			Logger.warn(`[${warning.key}] ${warning.message}`, {
				scope: "WebGPUFrameServiceOwner",
				onceKey: warning.key,
			});
		}
		const featureData = this._frameFeatureRegistry.prepareFrame({
			frameContext: context,
			scene,
			featureState,
			lightingCatalog,
			lightingState,
			renderWidth,
			renderHeight,
		});
		const clusteredLightingData = featureData.get(WEBGPU_CLUSTERED_LIGHTING_DATA) ?? null;
		for (const warning of clusteredLightingData?.warnings ?? []) {
			Logger.warn(`[${warning.key}] ${warning.message}`, {
				scope: "WebGPUFrameServiceOwner",
				onceKey: warning.key,
			});
		}

		const environmentState = collectWebGPUEnvironment(
			scene,
			featureState.enableSH,
			shAmbientCoeffs,
		);
		for (const warning of environmentState.warnings) {
			Logger.warn(`[${warning.key}] ${warning.message}`, {
				scope: "WebGPUFrameServiceOwner",
				onceKey: warning.key,
			});
		}

		this._shadowRuntime.prepareAtlas(
			lightingState,
			this._resolveShadowAtlasTileSize(
				context.shadowPlan,
				features.enableShadows,
				context.shadowPlan
			),
		);
		const scope = this._getOrCreateFrameScope(resolvedOptions.scopeKey);
		scope.frameBindings.prepare(
			scene,
			lightingState,
			environmentState,
			featureState,
			renderWidth,
			renderHeight,
			resolvedOptions.sceneTargetMode,
			{
				temporalStateMode,
				temporalHistoryReset,
				frameRequirements: resolvedOptions.frameRequirements,
			},
		);
		scope.clusteredLighting.prepareFrame(
			scene,
			featureState,
			clusteredLightingData,
			renderWidth,
			renderHeight,
		);

		const frameResources: WebGPUFrameServicePreparedResources = {
			scopeKey: resolvedOptions.scopeKey,
			sceneTargetMode: resolvedOptions.sceneTargetMode,
			frameBinding: scope.frameBindings.getSceneBinding(),
			decalFrameBinding: scope.frameBindings.getDecalFrameBinding(),
			environmentBinding: scope.frameBindings.getEnvironmentBinding(),
			clusteredSceneBinding: scope.clusteredLighting.getSceneBinding(),
			lightingState,
			featureData,
			featureState,
			environmentState,
			jointMatrixMap,
			morphWeightMap,
		};
		scope.prepared = frameResources;
		return frameResources;
	}

	/**
	 * Releases all GPU resources owned by a prepared frame scope.
	 *
	 * @param scopeKey Scope identifier previously passed to `prepareFrame()`.
	 * @sideEffects Destroys bind groups and buffers for the scope.
	 */
	public releaseScope(scopeKey: string): void {
		const scope = this._frameScopes.get(scopeKey);
		if (!scope) {
			return;
		}
		scope.frameBindings.destroy();
		scope.clusteredLighting.destroy();
		this._frameScopes.delete(scopeKey);
	}

	/**
	 * Returns the bind group layout used by G-buffer geometry shaders to write
	 * deferred storage payload textures.
	 *
	 * @returns The WebGPU bind group layout for `gMaterialExt0/3` writes.
	 * @sideEffects None.
	 */
	public getGBufferWriteLayout(): GPUBindGroupLayout {
		return this._deferredResources.getGBufferWriteLayout();
	}

	/**
	 * Returns the bind group layout used by the deferred lighting shader to read
	 * color and storage G-buffer payload textures.
	 *
	 * @returns The WebGPU bind group layout for deferred G-buffer reads.
	 * @sideEffects None.
	 */
	public getGBufferReadLayout(): GPUBindGroupLayout {
		return this._deferredResources.getGBufferReadLayout();
	}

	/**
	 * Returns the bind group layout used by the WebGPU deferred decal pass.
	 *
	 * @returns The WebGPU bind group layout for decal material resources.
	 * @sideEffects None.
	 */
	public getDecalBindGroupLayout(): GPUBindGroupLayout {
		return this._deferredResources.getDecalBindGroupLayout();
	}

	/**
	 * Returns the bind group layout used by fallback deferred decal draws to
	 * write extended G-buffer storage payload textures.
	 *
	 * @returns The WebGPU bind group layout for decal storage outputs.
	 * @sideEffects None.
	 */
	public getDecalOutputBindGroupLayout(): GPUBindGroupLayout {
		return this._deferredResources.getDecalOutputBindGroupLayout();
	}

	/**
	 * Returns the bind group layout used by WebGPU deferred decal batch data.
	 *
	 * @returns The WebGPU bind group layout for tile-binned decal resources.
	 * @sideEffects None.
	 */
	public getDecalBatchBindGroupLayout(): GPUBindGroupLayout {
		return this._deferredResources.getDecalBatchBindGroupLayout();
	}

	/**
	 * Returns the bind group layout used by planar reflection composite draws.
	 *
	 * @returns The WebGPU bind group layout for reflection texture sampling.
	 * @sideEffects None.
	 */
	public getPlanarReflectionLayout(): GPUBindGroupLayout {
		return this._layouts.planarReflectionBindGroupLayout;
	}

	/**
	 * Returns the empty placeholder bind group used to preserve deferred shader
	 * group indices without binding per-model material resources.
	 *
	 * Constraints: The returned bind group is only valid for the deferred
	 * lighting layout's empty group index 1.
	 *
	 * @returns A bind group compatible with deferred layout group index 1.
	 * @sideEffects May create and cache the placeholder bind group.
	 */
	public getDeferredUnusedBinding(): IBindingGroup {
		return this._deferredResources.getDeferredUnusedBinding();
	}

	/** @internal Returns device-lifetime deferred placeholder textures. */
	public getDeferredPlaceholderTextures(): {
		readonly rgba16Float: IRenderTexture;
		readonly rgba8Unorm: IRenderTexture;
		readonly rgba16Uint: IRenderTexture;
	} {
		return this._deferredResources.getDeferredPlaceholderTextures();
	}

	/**
	 * Resolves the fullscreen deferred lighting render pipeline.
	 *
	 * @returns A pipeline that reads the G-buffer and writes `sceneColorMain`.
	 * @sideEffects May compile and cache the deferred lighting pipeline.
	 */
	public async getDeferredLightingPipeline(): Promise<IRenderPipeline> {
		return this._deferredResources.getDeferredLightingPipeline();
	}

	/**
	 * Resolves the fullscreen deferred decal render pipeline.
	 *
	 * @returns A pipeline that reads G-buffer snapshots and writes updated
	 * G-buffer channels.
	 * @sideEffects May compile and cache the decal shader module and pipeline.
	 */
	public async getDecalPipeline(): Promise<IRenderPipeline> {
		return this._deferredResources.getDecalPipeline();
	}

	/**
	 * Resolves the tile-binned deferred decal compute pipeline.
	 *
	 * @returns A compute pipeline that applies compatible decal segments.
	 * @sideEffects May compile and cache the decal shader module and pipeline.
	 */
	public async getDecalBatchPipeline(): Promise<IComputePipeline> {
		return this._deferredResources.getDecalBatchPipeline();
	}

	public updateParticleShadowVolumes(
		frameResources: WebGPUFrameServicePreparedResources,
		context: FrameContext,
	): void {
		const prepared = this._requirePreparedFrameResources(
			frameResources,
			"updateParticleShadowVolumes",
		);
		if (!context) {
			return;
		}
		const scope = this._requireFrameScope(prepared.scopeKey);
		scope.frameBindings.updateParticleShadowVolumes(context, prepared.lightingState);
		prepared.frameBinding = scope.frameBindings.getSceneBinding();
	}

	/**
	 * @internal WebGPU paged shadow frame graph hook.
	 */
	public preparePagedShadowFrame(request: WebGPUPagedShadowFrameRequest): void {
		this._shadowRuntime.preparePaged(request);
	}

	/**
	 * @internal WebGPU paged shadow frame graph hook.
	 */
	public recordPagedShadowPageMarkPass(
		request: WebGPUPagedShadowFrameRequest,
	): void | Promise<void> {
		return this._shadowRuntime.recordPageMark(request);
	}

	/**
	 * @internal WebGPU paged shadow frame graph hook.
	 */
	public recordPagedShadowPageAllocationPass(
		request: WebGPUPagedShadowFrameRequest,
	): void | Promise<void> {
		return this._shadowRuntime.recordPageAllocation(request);
	}

	/**
	 * @internal WebGPU paged shadow frame graph hook.
	 */
	public recordPagedShadowPageTableCopyPass(
		request: WebGPUPagedShadowFrameRequest,
	): void | Promise<void> {
		return this._shadowRuntime.recordPageTableCopy(request);
	}

	/**
	 * @internal WebGPU paged shadow frame graph hook.
	 */
	public recordPagedShadowDepthPass(request: WebGPUPagedShadowFrameRequest): Promise<void> {
		return this._shadowRuntime.recordPagedDepth(request);
	}

	/**
	 * @internal WebGPU paged shadow delayed feedback hook.
	 */
	public recordPagedShadowFeedbackPass(
		request: WebGPUPagedShadowFrameRequest,
	): void | Promise<void> {
		return this._shadowRuntime.recordFeedback(request);
	}

	public async buildClusteredLighting(
		encoder: ICommandEncoder,
		frameResources: WebGPUFrameServicePreparedResources,
	): Promise<void> {
		const scope = this._requireFrameScope(frameResources.scopeKey);
		await scope.clusteredLighting.build(encoder, frameResources.frameBinding);
		frameResources.clusteredSceneBinding = scope.clusteredLighting.getSceneBinding();
	}

	public onShaderRuntimeChanged(): void {
		if (this._destroyed) {
			return;
		}
		this._pipelineLibrary.invalidateShaderRuntimeCaches();
		this._particleRenderResources.onShaderRuntimeChanged();
		this._deferredResources.onShaderRuntimeChanged();
		this.onShadowRuntimeShaderChanged();
		for (const scope of this._frameScopes.values()) {
			scope.clusteredLighting.onShaderRuntimeChanged();
		}
	}

	/** @internal Owned by the WebGPU deferred frame runtime. */
	public invalidateDeferredRuntimeResources(): void {
		this._deferredResources.onShaderRuntimeChanged();
	}

	/** @internal Owned by the WebGPU shadow frame runtime. */
	public onShadowRuntimeShaderChanged(): void {
		if (this._shadowRuntimeDestroyed) {
			return;
		}
		this._shadowRuntime.onShaderRuntimeChanged();
	}

	/** @internal Owned by the WebGPU shadow frame runtime. */
	public destroyShadowRuntimeResources(): void {
		if (this._shadowRuntimeDestroyed) {
			return;
		}
		this._shadowRuntimeDestroyed = true;
		this._shadowRuntime.destroy();
	}

	public destroy(): void {
		if (this._destroyed) {
			return;
		}
		this._destroyed = true;
		this.destroyShadowRuntimeResources();
		this._particleRenderResources.destroy();
		this.invalidateDeferredRuntimeResources();
		this._frameFeatureRegistry.destroy();
		for (const scope of this._frameScopes.values()) {
			scope.frameBindings.destroy();
			scope.clusteredLighting.destroy();
		}
		this._frameScopes.clear();
		this._materialBindings.destroy();
		this._pipelineLibrary.destroy();
		this._textureRegistry.destroy();
		this._geometryRegistry.destroy();
	}

	public getTextureForSlot(texture: Texture | null, slotIndex: number): IRenderTexture {
		return this._textureRegistry.getTextureForSlot(texture, slotIndex);
	}

	public getTextureForSlotAsync(
		texture: Texture | null,
		slotIndex: number,
	): Promise<IRenderTexture> {
		return this._textureRegistry.getTextureForSlotAsync(texture, slotIndex);
	}

	public getSamplerForTexture(texture: Texture | null): ISampler {
		return this._textureRegistry.getSamplerForTexture(texture);
	}

	public registerExternalTexture(
		texture: Texture,
		resource: IRenderTexture,
		uploadedVersion: number = texture.version,
		mipLevelCount: number = 1,
	): void {
		this._textureRegistry.registerExternalTexture(
			texture,
			resource,
			uploadedVersion,
			mipLevelCount,
		);
	}

	public unregisterExternalTexture(texture: Texture): void {
		this._textureRegistry.unregisterExternalTexture(texture);
	}

	/** @internal Returns the owner-managed particle rendering capability. */
	public getParticleBillboardRenderer(): WebGPUParticleBillboardRenderer {
		return this._particleRenderResources;
	}

	public get sceneFrameLayout(): GPUBindGroupLayout {
		return this._layouts.sceneFrameBindGroupLayout;
	}

	private _resolvePrepareFrameOptions(
		context: FrameContext,
		options: WebGPUFrameServicePrepareOptions | undefined,
	): WebGPUFrameServicePrepareOptions {
		if (!options) {
			throw new Error("WebGPU frame preparation requires explicit frame options.");
		}
		if (typeof options.scopeKey !== "string" || options.scopeKey.trim() === "") {
			throw new Error(
				"WebGPU frame preparation requires a non-empty internal scope identifier.",
			);
		}
		if (!options.sceneTargetMode) {
			throw new Error("WebGPU frame preparation requires sceneTargetMode.");
		}
		return {
			scopeKey: options.scopeKey,
			sceneTargetMode: options.sceneTargetMode,
			framePackets: options.framePackets ?? createBaselineFramePacketSet(context),
			temporalStateMode: options?.temporalStateMode,
			frameRequirements: options.frameRequirements,
		};
	}

	public commitTemporalFrame(): void {
		for (const scope of this._frameScopes.values()) {
			scope.frameBindings.commitTemporalFrame();
		}
	}

	public abortTemporalFrame(): void {
		for (const scope of this._frameScopes.values()) {
			scope.frameBindings.abortTemporalFrame();
		}
	}

	/** @internal Owned by the WebGPU device and resize lifecycle. */
	public resetTemporalState(): void {
		for (const scope of this._frameScopes.values()) {
			scope.frameBindings.resetTemporalState();
		}
	}

	private _getOrCreateFrameScope(scopeKey: string): WebGPUFrameResourceScope {
		let scope = this._frameScopes.get(scopeKey);
		if (scope) {
			return scope;
		}
		const clusteredLighting = new WebGPUClusteredLightingRuntime(
			this._computeFacade,
			this._layouts.clusteredSceneBindGroupLayout,
			this._layouts.sceneFrameBindGroupLayout,
		);
		clusteredLighting.onWarn((key, message) =>
			Logger.warn(`[${key}] ${message}`, {
				scope: "WebGPUClusteredLightingRuntime",
			})
		);
		scope = {
			frameBindings: new WebGPUFrameBindingCache(
				this._backend,
				this._resourceManager,
				this._layouts,
				this._textureRegistry,
				this._shadowRuntime,
			),
			clusteredLighting,
			prepared: null,
		};
		this._frameScopes.set(scopeKey, scope);
		return scope;
	}

	private _requireFrameScope(scopeKey: string): WebGPUFrameResourceScope {
		const scope = this._frameScopes.get(scopeKey);
		if (!scope) {
			throw new Error(`WebGPU frame scope "${scopeKey}" is not prepared.`);
		}
		return scope;
	}

	private _requirePreparedFrameResources(
		frameResources: WebGPUFrameServicePreparedResources | undefined,
		methodName: string,
	): WebGPUFrameServicePreparedResources {
		if (!this._isPreparedFrameResources(frameResources)) {
			throw new Error(
				`WebGPUFrameServiceOwner.${methodName}() requires explicit prepared frame resources.`,
			);
		}
		return frameResources;
	}

	private _isPreparedFrameResources(
		value: unknown,
	): value is WebGPUFrameServicePreparedResources {
		return (
			!!value &&
			typeof value === "object" &&
			typeof (value as { scopeKey?: unknown }).scopeKey === "string" &&
			"frameBinding" in value &&
			"decalFrameBinding" in value &&
			"clusteredSceneBinding" in value
		);
	}

	private _resolveShadowAtlasTileSize(
		shadowPlan: FrameContext["shadowPlan"],
		enableShadows: boolean,
		plan: FrameContext["shadowPlan"],
	): number {
		if (!enableShadows) {
			return 1;
		}

		const atlasLightIds = new Set(
			(plan?.jobs ?? [])
				.filter((job) =>
					job.technique === "atlas" || job.technique === "atlas-fallback"
				)
				.map((job) => plan.lights[job.lightIndex]?.lightId)
				.filter((lightId): lightId is string => typeof lightId === "string")
		);
		let tileSize = 0;
		for (const prepared of shadowPlan?.lights ?? []) {
			if (!atlasLightIds.has(prepared.lightId)) continue;
			const hasValidSlice = prepared.slices.length > 0;
			if (!hasValidSlice) {
				continue;
			}
			tileSize = Math.max(tileSize, prepared.effectiveResolution | 0);
		}

		return Math.max(1, tileSize);
	}

	public async getDrawResources(
		packet: DrawPacket,
		frameResources: WebGPUFrameServicePreparedResources,
		options: WebGPUDrawResourceOptions,
	): Promise<WebGPUDrawResources[] | null> {
		const prepared = this._requirePreparedFrameResources(frameResources, "getDrawResources");
		const transparentPipelineMode = options.transparentPipelineMode ?? "default";
		const sceneTargetMode = options.sceneTargetMode ?? prepared.sceneTargetMode;
		const drawMode = options.drawMode ?? "default";
		const sampleCount = options.sampleCount;
		const results: WebGPUDrawResources[] = [];
		const geometry = this._geometryRegistry.getGeometry(packet.primitive);
		const topology = geometry.topology;
		const frameBinding = prepared.frameBinding;
		const clusteredBinding = prepared.clusteredSceneBinding;
		const animationState = this._resolveAnimationState(packet, geometry, prepared);

		// ----- SOLID OBJECT -----
		const solidMaterialData = createWebGPUMaterialUniformData(packet.material, false);
		for (const warning of solidMaterialData.warnings) {
			Logger.warn(`[${warning.key}] ${warning.message}`, {
				scope: "WebGPUFrameServiceOwner",
				onceKey: warning.key,
			});
		}

		const solidPipeline =
			drawMode === "early-z-prepass"
				? await this._pipelineLibrary.getEarlyZPrepassPipeline(
						packet.material,
						sceneTargetMode,
						false,
						topology,
						sampleCount,
					)
				: await this._pipelineLibrary.getPipeline(
						packet.material,
						sceneTargetMode,
						false,
						topology,
						transparentPipelineMode,
						drawMode,
						sampleCount,
						options.deferredGBufferLayout,
					);
		if (!solidPipeline) {
			return null;
		}
		const solidTextures = await Promise.all(
			solidMaterialData.textureSlots.map((slot, index) =>
				this._textureRegistry.getTextureForSlotAsync(slot.map, index),
			),
		);
		const solidSamplers = solidMaterialData.textureSlots.map((slot) =>
			this._textureRegistry.getSamplerForTexture(slot.map),
		);
		const solidAnisotropyTexture = await this._textureRegistry.getTextureForSlotAsync(
			solidMaterialData.anisotropyTexture.map,
			-1,
		);
		const solidModelBinding = this._materialBindings.getBinding(
			packet,
			solidPipeline,
			solidMaterialData,
			solidTextures,
			solidSamplers,
			solidAnisotropyTexture,
			animationState,
		);

		results.push({
			pipeline: solidPipeline,
			frameBinding,
			modelBinding: solidModelBinding,
			clusteredBinding,
			vertexBuffer: geometry.vertexBuffer,
			indexBuffer: geometry.indexBuffer,
			indexCount: geometry.indexCount,
		});

		// ----- WIREFRAME OVERLAY -----
		if (
			drawMode !== "early-z-prepass" &&
			packet.material.wireframe &&
			topology === DEFAULT_PRIMITIVE_DRAW_TOPOLOGY
		) {
			const wireMaterialData = createWebGPUMaterialUniformData(packet.material, true);
			const wirePipeline = await this._pipelineLibrary.getPipeline(
				packet.material,
				sceneTargetMode,
				true,
				topology,
				transparentPipelineMode,
				drawMode,
				sampleCount,
			);
			const wireTextures = await Promise.all(
				wireMaterialData.textureSlots.map((slot, index) =>
					this._textureRegistry.getTextureForSlotAsync(slot.map, index),
				),
			);
			const wireSamplers = wireMaterialData.textureSlots.map((slot) =>
				this._textureRegistry.getSamplerForTexture(slot.map),
			);
			const wireAnisotropyTexture = await this._textureRegistry.getTextureForSlotAsync(
				wireMaterialData.anisotropyTexture.map,
				-1,
			);
			const wireModelBinding = this._materialBindings.getBinding(
				packet,
				wirePipeline,
				wireMaterialData,
				wireTextures,
				wireSamplers,
				wireAnisotropyTexture,
				animationState,
			);

			results.push({
				pipeline: wirePipeline,
				frameBinding,
				modelBinding: wireModelBinding,
				clusteredBinding,
				vertexBuffer: geometry.vertexBuffer,
				indexBuffer: geometry.wireframeIndexBuffer,
				indexCount: geometry.wireframeIndexCount,
			});
		}

		return results;
	}

	public async getEnvironmentResources(
		frameResources: WebGPUFrameServicePreparedResources,
		sceneTargetMode: WebGPUSceneTargetMode,
		options: WebGPUEnvironmentResourceOptions,
	): Promise<WebGPUEnvironmentDrawResources | null> {
		const prepared = this._requirePreparedFrameResources(
			frameResources,
			"getEnvironmentResources",
		);
		const resolvedSceneTargetMode = sceneTargetMode;
		if (
			!prepared.featureState.enableEnvironment ||
			!prepared.environmentState.environmentTexture
		) {
			return null;
		}

		const pipeline = await this._pipelineLibrary.getEnvironmentPipeline(
			resolvedSceneTargetMode,
			options.sampleCount,
		);
		const frameBinding = prepared.environmentBinding;

		return {
			pipeline,
			frameBinding,
		};
	}

	private _resolveAnimationState(
		packet: DrawPacket,
		geometry: WebGPUGeometryHandle,
		frameResources: WebGPUFrameServicePreparedResources,
	): WebGPUModelAnimationBindingState {
		const runtimeJoint = frameResources.jointMatrixMap?.get(packet.meshInstance.id) ?? null;
		let jointMatrices: Float32Array | null = null;
		if (runtimeJoint?.skeleton) {
			runtimeJoint.skeleton.updateJointMatrices(packet.meshInstance.worldMatrix);
			jointMatrices = runtimeJoint.skeleton.toFloat32Array(runtimeJoint.matrices);
		} else if (packet.meshInstance.skeleton) {
			packet.meshInstance.skeleton.updateJointMatrices(packet.meshInstance.worldMatrix);
			jointMatrices = packet.meshInstance.skeleton.toFloat32Array();
		}

		const runtimeMorph = frameResources.morphWeightMap?.get(packet.primitive.id) ?? null;
		let morphTargetCount = Math.max(0, runtimeMorph?.targetCount ?? 0);
		let sourceMorphWeights: Float32Array | null = runtimeMorph?.weights ?? null;
		if (!sourceMorphWeights || morphTargetCount <= 0) {
			const primitiveIndex = packet.mesh.primitives.indexOf(packet.primitive);
			const instanceWeights =
				primitiveIndex >= 0 ? packet.meshInstance.morphWeights[primitiveIndex] : null;
			sourceMorphWeights = instanceWeights ?? null;
			morphTargetCount = sourceMorphWeights?.length ?? 0;
		}

		morphTargetCount = Math.min(Math.max(0, morphTargetCount), geometry.morphTargetCount);
		let morphWeights: Float32Array | null = null;
		if (sourceMorphWeights && morphTargetCount > 0) {
			morphWeights = new Float32Array(morphTargetCount);
			morphWeights.set(sourceMorphWeights.subarray(0, morphTargetCount));
		}

		return {
			jointMatrices,
			morphWeights,
			morphTargetCount,
			morphPositionBuffer: geometry.morphPositionBuffer,
			morphNormalBuffer: geometry.morphNormalBuffer,
		};
	}
}
