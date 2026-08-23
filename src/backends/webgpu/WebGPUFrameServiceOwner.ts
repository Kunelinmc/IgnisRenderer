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
} from "../../pipeline/FramePackets";
import type { WebGPUResourceManager } from "./WebGPUResourceManager";
import type { IWebGPUComputeFacade } from "./ComputeFacade";
import {
	ANIMATION_JOINT_MATRICES_KEY,
	ANIMATION_MORPH_WEIGHTS_KEY,
} from "../../simulation/animation/types";
import {
	collectWebGPUEnvironment,
	collectWebGPULightingCatalog,
	createWebGPULightingState,
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
import { WebGPUGeometryRegistry } from "./WebGPUGeometryRegistry";
import { WebGPUAnimationPayloadPool } from "./WebGPUAnimationPayloadPool";
import { WebGPUMaterialBindingCache } from "./WebGPUMaterialBindingCache";
import { WebGPUMaterialSnapshotCache } from "./WebGPUMaterialSnapshotCache";
import { WebGPUStaticMeshBatcher } from "./WebGPUStaticMeshBatcher";
import { WebGPUScenePipelineResources } from "./WebGPUScenePipelineResources";
import { WebGPUDeferredResources } from "./WebGPUDeferredResources";
import { WebGPUEnvironmentResources } from "./WebGPUEnvironmentResources";
import { WebGPUDrawResourceAssembler } from "./WebGPUDrawResourceAssembler";
import { WebGPUMaterialPipelineResolver } from "./WebGPUMaterialPipelineResolver";
import { WebGPUPlanarReflectionDrawResources } from "./WebGPUPlanarReflectionDrawResources";
import type {
	WebGPUSceneTargetMode,
} from "./WebGPUScenePassDescriptors";
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
	private _scenePipelines: WebGPUScenePipelineResources;
	private _animationPayloads: WebGPUAnimationPayloadPool;
	private _materialBindings: WebGPUMaterialBindingCache;
	private _materialSnapshots: WebGPUMaterialSnapshotCache;
	private _staticBatcher: WebGPUStaticMeshBatcher;
	private _shadowRuntime: WebGPUShadowRuntime;
	private _shadowRuntimeDestroyed = false;
	private _frameFeatureRegistry = createWebGPUFrameFeatureRegistry();
	private _frameScopes = new Map<string, WebGPUFrameResourceScope>();
	private _nextFrameScopeId = 0;
	private _deferredResources: WebGPUDeferredResources;
	private _environmentResources: WebGPUEnvironmentResources;
	private _materialPipelineResolver: WebGPUMaterialPipelineResolver;
	private _drawResourceAssembler: WebGPUDrawResourceAssembler;
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
		this._materialSnapshots = new WebGPUMaterialSnapshotCache(this._textureRegistry);
		this._scenePipelines = new WebGPUScenePipelineResources(
			backend,
			this._layouts,
		);
		this._particleRenderResources = new WebGPUParticleRenderResources(
			backend,
			this._layouts,
			this._textureRegistry,
		);
		this._deferredResources = new WebGPUDeferredResources(backend, this._layouts);
		this._environmentResources = new WebGPUEnvironmentResources(
			backend,
			this._layouts,
		);
		this._animationPayloads = new WebGPUAnimationPayloadPool(backend);
		this._staticBatcher = new WebGPUStaticMeshBatcher(
			backend,
			this._layouts,
			this._animationPayloads,
		);
		this._materialBindings = new WebGPUMaterialBindingCache(
			backend,
			this._layouts,
			this._animationPayloads
		);
		this._materialPipelineResolver = new WebGPUMaterialPipelineResolver();
		this._drawResourceAssembler = new WebGPUDrawResourceAssembler(
			backend,
			this._geometryRegistry,
			this._animationPayloads,
			this._materialSnapshots,
			this._materialBindings,
			this._staticBatcher,
			this._materialPipelineResolver,
		);
		this._shadowRuntime = new WebGPUShadowRuntime(
			backend,
			resourceManager,
			this._geometryRegistry,
			this._animationPayloads
		);
	}

	/** @internal Creates reflection-owned composite draw resources. */
	public createPlanarReflectionDrawResources(): WebGPUPlanarReflectionDrawResources {
		return new WebGPUPlanarReflectionDrawResources(
			this._backend,
			this._layouts,
			this._drawResourceAssembler,
		);
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
			const reflectionDraws = this.createPlanarReflectionDrawResources();
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
							? await reflectionDraws.getDrawResources(packet, reflectionResources, {
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
				reflectionDraws.destroy();
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
		this._animationPayloads.beginFrame();
		this._materialBindings.beginFrame();
		this._materialSnapshots.beginFrame();
		this._particleRenderResources.beginFrame();
		this._staticBatcher.beginFrame();
	}

	public prepareFrame(
		context: FrameContext,
		options: WebGPUFrameServicePrepareOptions,
	): WebGPUFrameServicePreparedResources {
		const resolvedOptions = this._resolvePrepareFrameOptions(context, options);
		this._staticBatcher.preparePackets(resolvedOptions.framePackets.opaque);
		const jointMatrixMap = context.transient.get(ANIMATION_JOINT_MATRICES_KEY) ?? null;
		const morphWeightMap = context.transient.get(ANIMATION_MORPH_WEIGHTS_KEY) ?? null;
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
		this._scenePipelines.invalidateShaderRuntimeCaches();
		this._materialPipelineResolver.clear();
		this._environmentResources.onShaderRuntimeChanged();
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
		this._environmentResources.destroy();
		this._frameFeatureRegistry.destroy();
		for (const scope of this._frameScopes.values()) {
			scope.frameBindings.destroy();
			scope.clusteredLighting.destroy();
		}
		this._frameScopes.clear();
		this._materialBindings.destroy();
		this._materialSnapshots.clear();
		this._staticBatcher.destroy();
		this._animationPayloads.destroy();
		this._scenePipelines.destroy();
		this._textureRegistry.destroy();
		this._geometryRegistry.destroy();
	}

	public getTextureForSlot(texture: Texture | null, slotIndex: number): IRenderTexture {
		return this._textureRegistry.getTextureForSlot(texture, slotIndex);
	}

	/** @internal Test and benchmark diagnostics for CPU-side draw preparation. */
	public getDebugStats() {
		return {
			materialSnapshots: this._materialSnapshots.getDebugStats(),
			staticBatching: this._staticBatcher.getDebugStats(),
			animationPayloads: this._animationPayloads.getDebugStats(),
		};
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
		this._staticBatcher.commitFrame();
	}

	public abortTemporalFrame(): void {
		for (const scope of this._frameScopes.values()) {
			scope.frameBindings.abortTemporalFrame();
		}
		this._staticBatcher.abortFrame();
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
		return this._drawResourceAssembler.getDrawResources(
			packet,
			prepared,
			options,
			this._scenePipelines,
		);
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

		const pipeline = await this._environmentResources.getPipeline(
			resolvedSceneTargetMode,
			options.sampleCount,
		);
		const frameBinding = prepared.environmentBinding;

		return {
			pipeline,
			frameBinding,
		};
	}

}
