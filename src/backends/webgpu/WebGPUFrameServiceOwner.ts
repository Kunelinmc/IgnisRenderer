import type { Texture } from "../../core/Texture";
import type {
	DrawPacket,
	FrameContext,
} from "../../pipeline/types";
import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../../core/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import { TextureFormat } from "../../core/TextureFormat";
import {
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
import { WebGPUMaterialBufferCache } from "./WebGPUMaterialBufferCache";
import { WebGPUMaterialSnapshotCache } from "./WebGPUMaterialSnapshotCache";
import { WebGPUStaticMeshBatcher } from "./WebGPUStaticMeshBatcher";
import { WebGPUScenePipelineResources } from "./WebGPUScenePipelineResources";
import { WebGPUDeferredResources } from "./WebGPUDeferredResources";
import { WebGPUEnvironmentResources } from "./WebGPUEnvironmentResources";
import { WebGPUDrawResourceAssembler } from "./WebGPUDrawResourceAssembler";
import { WebGPUMaterialPipelineResolver } from "./WebGPUMaterialPipelineResolver";
import { WebGPUPlanarReflectionDrawResources } from "./WebGPUPlanarReflectionDrawResources";
import { WebGPUSceneDrawResources } from "./WebGPUSceneDrawResources";
import type {
	WebGPUSceneTargetMode,
} from "./WebGPUScenePassDescriptors";
import { WebGPUShadowRuntime } from "./WebGPUShadowRuntime";
import type { ShadowCastingLight } from "../../lights";
import { WebGPUTextureRegistry } from "./WebGPUTextureRegistry";
import type { WarmupPhaseCounters, WarmupPlan } from "../../pipeline/WarmupPlanner";
import { toShaderCompileError } from "../../pipeline/WarmupPlanner";
import { createWarmupYieldController } from "../../pipeline/WarmupScheduler";
import { Logger } from "../../foundation/Logger";
import type { WarmupOptions } from "../IRenderBackend";
import type {
	WebGPUDrawResourceOptions,
	WebGPUDrawResources,
	WebGPUEnvironmentDrawResources,
	WebGPUEnvironmentResourceOptions,
	WebGPUFrameResourceScope as WebGPUFrameResourceScopeContract,
	WebGPUFrameScopeRole,
	WebGPUFrameScopePrepareOptions,
	WebGPUParticleBillboardRenderer,
	WebGPUPreparedFrameResources as WebGPUPreparedFrameResourcesContract,
	WebGPUShadowRenderProvider,
} from "./WebGPUResourceContracts";
import { WebGPUParticleRenderResources } from "./WebGPUParticleRenderResources";
import { materialSupportsWebGPUDeferredLighting } from "./material";
import {
	analyzeWebGPUDeferredFeatures,
	resolveWebGPUDeferredConfiguration,
} from "./rendergraph/WebGPUDeferredFrameModule";

interface WebGPUFrameServicePrepareOptions extends WebGPUFrameScopePrepareOptions {
	readonly scopeKey: string;
	readonly scopeRole: WebGPUFrameScopeRole;
}

export interface WebGPUFrameServiceWarmupRuntimeOptions {
	readonly enableEarlyZPrepass: boolean;
	readonly enableDeferredLighting: boolean;
	readonly sampleCount: number;
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
		private readonly _scopeRole: WebGPUFrameScopeRole,
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
			scopeRole: this._scopeRole,
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
	private _materialBuffers: WebGPUMaterialBufferCache;
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
	private _sceneDraws: WebGPUSceneDrawResources;
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
		this._materialBuffers = new WebGPUMaterialBufferCache(backend);
		this._staticBatcher = new WebGPUStaticMeshBatcher(
			backend,
			this._layouts,
			this._animationPayloads,
			this._materialBuffers,
		);
		this._materialBindings = new WebGPUMaterialBindingCache(
			backend,
			this._layouts,
			this._animationPayloads,
			this._materialBuffers,
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
		this._sceneDraws = new WebGPUSceneDrawResources(
			this._drawResourceAssembler,
			this._scenePipelines,
		);
		this._shadowRuntime = new WebGPUShadowRuntime(
			backend,
			this._geometryRegistry,
			this._animationPayloads,
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
	public createFrameScope(
		role: WebGPUFrameScopeRole = "auxiliary",
	): WebGPUFrameResourceScopeContract {
		if (this._destroyed) {
			throw new Error("WebGPU frame service owner has been destroyed.");
		}
		this._nextFrameScopeId++;
		return new WebGPUFrameResourceScopeHandle(
			this,
			`frame-scope-${this._nextFrameScopeId}`,
			role,
		);
	}

	/** @internal Provides the frame module with the owning shadow runtime. */
	public getShadowRenderProvider(): WebGPUShadowRenderProvider {
		return this._shadowRuntime;
	}

	public async warmup(
		context: FrameContext,
		plan: WarmupPlan,
		options: WarmupOptions = {},
		framePackets: PreparedFramePacketSet = createBaselineFramePacketSet(context),
		runtimeOptions: WebGPUFrameServiceWarmupRuntimeOptions = {
			enableEarlyZPrepass: true,
			enableDeferredLighting: true,
			sampleCount: 1,
		},
	): Promise<WarmupPhaseCounters[]> {
		const phases: WarmupPhaseCounters[] = [];
		const yieldController = createWarmupYieldController(options);
		const deferredAnalysis = analyzeWebGPUDeferredFeatures(context, framePackets);
		const deviceLimits = this._backend.device?.limits;
		const deferredConfiguration = resolveWebGPUDeferredConfiguration(
			deferredAnalysis,
			{
				capabilities: {
					maxColorAttachments: deviceLimits?.maxColorAttachments ?? 8,
					maxColorAttachmentBytesPerSample:
						deviceLimits?.maxColorAttachmentBytesPerSample ?? 32,
					maxStorageTexturesPerShaderStage:
						deviceLimits?.maxStorageTexturesPerShaderStage ?? 4,
				},
				options: {
					sampleCount: runtimeOptions.sampleCount,
					enableDeferredLighting: runtimeOptions.enableDeferredLighting,
					forceDeferredFallback: false,
				},
			},
		);
		const mainSceneTargetMode = deferredConfiguration.active
			? "mrt"
			: plan.sceneTargetMode;
		const mainScope = this.createFrameScope();
		let warmupResources: WebGPUPreparedFrameResourcesContract | null = null;
		try {
			warmupResources = mainScope.prepare(context, {
				sceneTargetMode: mainSceneTargetMode,
				framePackets,
				temporalStateMode: "disabled",
			});
		} catch (error) {
			phases.push({
				phase: "webgpu-prepare",
				total: 1,
				compiled: 0,
				skipped: 0,
				failed: 1,
				errors: [toShaderCompileError(
					error,
					"webgpu",
					"WebGPUPrepareFrame",
				)],
			});
		}
		try {
			const drawPackets = [
				...context.scene.opaquePackets,
				...context.scene.transparentPackets,
			];
			const forwardPackets = deferredConfiguration.active
				? [
					...context.scene.opaquePackets.filter((packet) =>
						!materialSupportsWebGPUDeferredLighting(packet.submission.material.effective)),
					...context.scene.transparentPackets,
				]
				: drawPackets;
			if (warmupResources) {
				phases.push(await this._sceneDraws.warmup({
					packets: forwardPackets,
					frameResources: warmupResources,
					sceneTargetMode: mainSceneTargetMode,
					sampleCount: runtimeOptions.sampleCount,
					enableEarlyZPrepass: runtimeOptions.enableEarlyZPrepass,
					yieldIfNeeded: () => yieldController.yieldIfNeeded(),
				}));
			}
			if (plan.enableEnvironment) {
				phases.push(await this._environmentResources.warmup({
					modes: [
						deferredConfiguration.active ? "gbuffer" : mainSceneTargetMode,
					],
					sampleCount: runtimeOptions.sampleCount,
					yieldIfNeeded: () => yieldController.yieldIfNeeded(),
				}));
			}

			if (deferredConfiguration.active) {
				const deferredScope = this.createFrameScope();
				try {
					const deferredResources = deferredScope.prepare(context, {
						sceneTargetMode: "gbuffer",
						framePackets,
						temporalStateMode: "disabled",
					});
					phases.push(await this._sceneDraws.warmup({
						phase: "webgpu-deferred-scene",
						packets: context.scene.opaquePackets.filter((packet) =>
							materialSupportsWebGPUDeferredLighting(packet.submission.material.effective)),
						frameResources: deferredResources,
						sceneTargetMode: "gbuffer",
						sampleCount: 1,
						enableEarlyZPrepass: runtimeOptions.enableEarlyZPrepass,
						deferredGBufferLayout:
							deferredConfiguration.deferredGBufferLayout,
						yieldIfNeeded: () => yieldController.yieldIfNeeded(),
					}));
				} finally {
					deferredScope.destroy();
				}
			}
			phases.push(await this._deferredResources.warmup({
				active: deferredConfiguration.active,
				hasDecals: (context.scene.decalPackets?.length ?? 0) > 0,
				yieldIfNeeded: () => yieldController.yieldIfNeeded(),
			}));
		} finally {
			mainScope.destroy();
		}

		if (plan.enableShadows) {
			const phase: WarmupPhaseCounters = {
				phase: "webgpu-shadows",
				total: 1,
				compiled: 0,
				skipped: 0,
				failed: 0,
				errors: [],
			};
			try {
				await this._shadowRuntime.warmup();
				phase.compiled++;
			} catch (error) {
				phase.failed++;
				phase.errors.push(toShaderCompileError(
					error,
					"webgpu",
					"WebGPUShadowWarmup",
				));
			}
			await yieldController.yieldIfNeeded();
			phases.push(phase);
		}

		if (plan.enableParticles) {
			const phase: WarmupPhaseCounters = {
				phase: "webgpu-particles",
				total: 1,
				compiled: 0,
				skipped: 0,
				failed: 0,
				errors: [],
			};
			try {
				await this._particleRenderResources.warmup(plan.sceneTargetMode);
				phase.compiled++;
			} catch (error) {
				phase.failed++;
				phase.errors.push(toShaderCompileError(
					error,
					"webgpu",
					"WebGPUParticleWarmup",
				));
			}
			await yieldController.yieldIfNeeded();
			phases.push(phase);
		}

		return phases;
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
			enableShadows: context.shadowPlan?.hasRasterWork === true,
			enableReflection: features.enableReflection,
			enableEnvironment: features.enableEnvironment,
			enableOIT: features.enableOIT,
			enableClusteredLighting: features.enableClusteredLighting,
			clusteredLightingOptions: features.clusteredLightingOptions,
			postProcess,
			warnings: [],
		};

		const lightingCatalog = collectWebGPULightingCatalog(
			scene.lights,
			features.enableLighting,
			features.enableSH,
			context.shadowPlan?.hasRasterWork === true,
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
			this._resolveShadowAtlasTileSize(context.shadowPlan),
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
		this._sceneDraws.onShaderRuntimeChanged();
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
		this._materialBuffers.destroy();
		this._animationPayloads.destroy();
		this._sceneDraws.destroy();
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
			materialBuffers: this._materialBuffers.getDebugStats(),
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
			scopeRole: options.scopeRole,
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
	): number {
		let tileSize = 0;
		for (const prepared of shadowPlan?.lights ?? []) {
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
		return this._sceneDraws.getDrawResources(
			packet,
			prepared,
			options,
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
