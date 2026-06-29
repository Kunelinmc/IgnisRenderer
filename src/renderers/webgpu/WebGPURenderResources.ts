import type { Texture } from "../../core/Texture";
import {
	DRAW_PACKET_FLAG_REFLECTIVE,
	DRAW_PACKET_FLAG_SHADOW_CASTER,
	DRAW_PACKET_FLAG_SHADOW_TRANSMITTER,
	DRAW_PACKET_FLAG_TRANSPARENT,
	PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
	PARTICLE_TRANSIENT_BATCHES_KEY,
} from "../../pipeline/types";
import type {
	DrawPacket,
	FrameContext,
	ParticleMeshRenderBatch,
	ParticleMeshRenderItem,
	ParticleRenderBatch,
} from "../../pipeline/types";
import { ParticleBlendMode } from "../../particles";
import { AlphaMode } from "../../materials/Material";
import { isMaterialTransparentPass } from "../../materials/transparency";
import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../../core/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import {
	BufferUsage,
	TextureFormat,
	type IBindingGroup,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderPipeline,
	type IRenderTexture,
	type ISampler,
	type IShaderModule,
} from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
import { resolveWebGPUComputeFacade } from "./ComputeFacade";
import type { IWebGPUComputeFacade } from "./ComputeFacade";
import {
	ANIMATION_WEBGPU_JOINT_MATRICES_KEY,
	ANIMATION_WEBGPU_MORPH_WEIGHTS_KEY,
	type JointMatrixMap,
	type MorphWeightMap,
} from "../../simulation/animation/types";
import {
	collectWebGPUEnvironment,
	collectWebGPULighting,
	createWebGPUMaterialUniformData,
	type WebGPUEnvironmentState,
	type WebGPUFeatureState,
	type WebGPULightingState,
} from "./";
import { createWebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";
import {
	WebGPUFrameBindingCache,
	type WebGPUTemporalStateMode,
} from "./WebGPUFrameBindingCache";
import { WebGPUClusteredLightingRuntime } from "./WebGPUClusteredLightingRuntime";
import {
	WebGPUGeometryRegistry,
	type WebGPUGeometryHandle,
} from "./WebGPUGeometryRegistry";
import {
	WebGPUMaterialBindingCache,
	type WebGPUModelAnimationBindingState,
} from "./WebGPUMaterialBindingCache";
import { WebGPUPipelineLibrary } from "./WebGPUPipelineLibrary";
import type {
	WebGPUScenePipelineDrawMode,
	WebGPUSceneTargetMode,
	WebGPUTransparentPipelineMode,
} from "./WebGPUPipelineLibrary";
import { WebGPUShadowAtlasAllocator } from "./WebGPUShadowAtlasAllocator";
import { WebGPUShadowPass } from "./WebGPUShadowPass";
import {
	WebGPUPagedShadowRuntime,
	type WebGPUPagedShadowFrameRequest,
	type WebGPUPagedShadowResources,
} from "./WebGPUPagedShadowRuntime";
import {
	resolveShadowCasterBounds,
	syncShadowMapRegistry,
	updateShadowMapMetadata,
} from "../../pipeline/ShadowMetadata";
import {
	mergeParticleShadowBounds,
	resolveParticleShadowCasterBounds,
} from "../../pipeline/ParticleShadowVolume";
import {
	selectCSMDirectionalLights,
	type ShadowBackendCapabilities,
} from "../../pipeline/ShadowStrategyRegistry";
import { isShadowCastingLight, type ShadowCastingLight } from "../../lights";
import type { ShadowRenderSet } from "../../lights/shadows/ShadowMapping";
import { WebGPUTextureRegistry } from "./WebGPUTextureRegistry";
import { ShaderSource } from "../../shaders/ShaderSource";
import { clamp } from "../../maths/Common";
import { Matrix4 } from "../../maths/Matrix4";
import { Quaternion } from "../../maths/Quaternion";
import type { Matrix3Arr } from "../../maths/types";
import type { MeshInstance } from "../../meshes";
import {
	WEBGPU_PARTICLE_BINDING_SAMPLER,
	WEBGPU_PARTICLE_BINDING_TEXTURE,
	WEBGPU_PARTICLE_BINDING_UV_TRANSFORM,
	WEBGPU_PARTICLE_INSTANCE_FLOATS,
	WEBGPU_PARTICLE_INSTANCE_STRIDE,
	WEBGPU_PARTICLE_QUAD_VERTICES,
	WEBGPU_PARTICLE_UV_UNIFORM_SIZE,
} from "./constants";
import {
	WEBGPU_PARTICLE_VERTEX_LAYOUTS,
} from "./bufferLayouts";
import {
	WEBGPU_PARTICLE_DRAW_BATCHES_KEY,
	type WebGPUParticleDrawBatch,
} from "./particleTransient";
import type {
	WarmupPhaseCounters,
	WarmupPlan,
} from "../../pipeline/WarmupPlanner";
import { toShaderCompileError } from "../../pipeline/WarmupPlanner";
import { createWarmupYieldController } from "../../pipeline/WarmupScheduler";
import type { ShaderCompileError } from "../../shaders/runtime";
import { Logger } from "../../foundation/Logger";
import type { WarmupOptions } from "../IRenderBackend";

const WEBGPU_SHADOW_CAPABILITIES: ShadowBackendCapabilities = {
	backendKey: "webgpu",
	supportsSingleMap: true,
	supportsDirectionalCSM: true,
	supportsSpotCSM: false,
	supportsPointCSM: false,
	maxCsmDirectionalLights: 1,
	maxDynamicShadowCost: 48,
	supportsPagedShadows: true,
	supportsPagedShadowRendering: true,
	maxPagedShadowPages: 2048,
	pagedShadowPageSizeRange: [64, 256],
};

export interface WebGPUDrawResources {
	pipeline: any;
	frameBinding: any;
	modelBinding: any;
	clusteredBinding: any;
	vertexBuffer: any;
	indexBuffer: any;
	indexCount: number;
}

export interface WebGPUEnvironmentDrawResources {
	pipeline: any;
	frameBinding: any;
}

export interface WebGPUParticlePassTargets {
	colorAttachments: Array<{
		view: any;
		resolveTarget?: any;
		clearValue?: { r: number; g: number; b: number; a: number };
		loadOp: "clear" | "load";
		storeOp: "store" | "discard";
	}>;
	depth: any;
	label: string;
}

const WEBGPU_PARTICLE_BINDING_CACHE_MAX_AGE_FRAMES = 120;
const WEBGPU_PARTICLE_BINDING_CACHE_MAX_ENTRIES = 128;

interface WebGPUParticleBindingCacheEntry {
	group: IBindingGroup;
	texture: any;
	sampler: any;
	uvTransformBuffer: IRenderBuffer;
	lastUsedFrame: number;
}

export interface WebGPUDrawResourceOptions {
	transparentPipelineMode?: WebGPUTransparentPipelineMode;
	sceneTargetMode?: WebGPUSceneTargetMode;
	drawMode?: WebGPUScenePipelineDrawMode;
	sampleCountOverride?: number;
}

export interface WebGPUEnvironmentResourceOptions {
	sampleCountOverride?: number;
}

interface WebGPUParticleRenderOptions {
	includeBlendModes?: readonly ParticleBlendMode[];
	pipelineMode?: "legacy" | "oit";
	sampleCountOverride?: number;
}

export interface WebGPUParticleMeshPacketOptions {
	includeOpaque?: boolean;
	includeTransparent?: boolean;
	includeShadowCasters?: boolean;
	includeShadowTransmitters?: boolean;
	includeReflective?: boolean;
}

export interface WebGPUPrepareFrameOptions {
	readonly scopeKey: string;
	readonly sceneTargetMode: WebGPUSceneTargetMode;
	readonly temporalStateMode?: WebGPUTemporalStateMode;
}

export interface WebGPUPreparedFrameResources {
	readonly scopeKey: string;
	readonly sceneTargetMode: WebGPUSceneTargetMode;
	frameBinding: IBindingGroup;
	decalFrameBinding: IBindingGroup;
	environmentBinding: IBindingGroup;
	clusteredSceneBinding: IBindingGroup;
	readonly lightingState: WebGPULightingState;
	readonly featureState: WebGPUFeatureState;
	readonly environmentState: WebGPUEnvironmentState;
	readonly jointMatrixMap: JointMatrixMap | null;
	readonly morphWeightMap: MorphWeightMap | null;
}

interface WebGPUFrameResourceScope {
	frameBindings: WebGPUFrameBindingCache;
	clusteredLighting: WebGPUClusteredLightingRuntime;
	prepared: WebGPUPreparedFrameResources | null;
}

export class WebGPURenderResources {
	private _backend: WebGPUBackend;
	private _computeFacade: IWebGPUComputeFacade;
	private _layouts: ReturnType<typeof createWebGPUPipelineLayouts>;
	private _geometryRegistry: WebGPUGeometryRegistry;
	private _textureRegistry: WebGPUTextureRegistry;
	private _shadowAtlases: WebGPUShadowAtlasAllocator;
	private _pipelineLibrary: WebGPUPipelineLibrary;
	private _materialBindings: WebGPUMaterialBindingCache;
	private _shadowPass: WebGPUShadowPass;
	private _pagedShadowRuntime: WebGPUPagedShadowRuntime;
	private _frameScopes = new Map<string, WebGPUFrameResourceScope>();
	private _particleShaderModule: IShaderModule | null = null;
	private _decalShaderModule: IShaderModule | null = null;
	private _decalPipeline: IRenderPipeline | null = null;
	private _decalBatchPipeline: IComputePipeline | null = null;
	private _particleQuadBuffer: IRenderBuffer | null = null;
	private _particleInstanceBuffer: IRenderBuffer | null = null;
	private _particleInstanceCapacity = 0;
	private _particlePipelineAlpha = new Map<string, IRenderPipeline>();
	private _particlePipelineOITAlpha = new Map<string, IRenderPipeline>();
	private _particlePipelineAdditive = new Map<string, IRenderPipeline>();
	private _particleBindingCache = new Map<
		string,
		WebGPUParticleBindingCacheEntry
	>();
	private _deferredUnusedBinding: IBindingGroup | null = null;
	private _frameId = 0;
	private _destroyed = false;

	constructor(backend: WebGPUBackend) {
		this._backend = backend;
		this._computeFacade = resolveWebGPUComputeFacade(backend);
		const device = backend.device;
		if (!device) {
			throw new Error(
				"WebGPU backend is not initialized; cannot create render resources."
			);
		}
		this._layouts = createWebGPUPipelineLayouts(device);
		this._geometryRegistry = new WebGPUGeometryRegistry(backend);
		this._textureRegistry = new WebGPUTextureRegistry(backend);
		this._shadowAtlases = new WebGPUShadowAtlasAllocator(backend);
		this._pipelineLibrary = new WebGPUPipelineLibrary(
			backend,
			this._layouts,
			{ listenToShaderRuntime: false }
		);
		this._materialBindings = new WebGPUMaterialBindingCache(
			backend,
			this._layouts
		);
		this._shadowPass = new WebGPUShadowPass(
			backend,
			this._geometryRegistry,
			this._shadowAtlases
		);
		this._pagedShadowRuntime = new WebGPUPagedShadowRuntime(
			backend,
			this._shadowPass
		);
	}

	public async init(): Promise<void> {
		await this._pipelineLibrary.init();
	}

	public async warmup(
		context: FrameContext,
		plan: WarmupPlan,
		options: WarmupOptions = {}
	): Promise<WarmupPhaseCounters> {
		let total = 0;
		let compiled = 0;
		let skipped = 0;
		let failed = 0;
		const errors: ShaderCompileError[] = [];
		const yieldController = createWarmupYieldController(options);
		const warmupScopeKey = "warmup-main";
		let warmupResources: WebGPUPreparedFrameResources | null = null;

		try {
			warmupResources = this.prepareFrame(context, {
				scopeKey: warmupScopeKey,
				sceneTargetMode: plan.sceneTargetMode,
				temporalStateMode: "disabled",
			});
		} catch (error) {
			failed++;
			errors.push(toShaderCompileError(error, "webgpu", "WebGPUPrepareFrame"));
		}

		if (plan.enableEnvironment && warmupResources) {
			total++;
			try {
				await this.getEnvironmentResources(warmupResources);
				compiled++;
			} catch (error) {
				failed++;
				errors.push(
					toShaderCompileError(error, "webgpu", "WebGPUEnvironmentWarmup")
				);
			}
			await yieldController.yieldIfNeeded();
		}

		const drawPackets = [
			...context.scene.opaquePackets,
			...context.scene.transparentPackets,
		];
		for (const packet of drawPackets) {
			total++;
			try {
				const resources = warmupResources ?
					await this.getDrawResources(packet, warmupResources)
				:	null;
				if (resources && resources.length > 0) {
					compiled++;
				} else {
					skipped++;
				}
			} catch (error) {
				failed++;
				errors.push(
					toShaderCompileError(error, "webgpu", `WebGPUDrawWarmup:${packet.id}`)
				);
			}
			await yieldController.yieldIfNeeded();
		}

		if (
			context.features.enableReflection &&
			context.scene.reflectivePackets.length > 0
		) {
			let reflectionResources: WebGPUPreparedFrameResources | null = null;
			try {
				reflectionResources = this.prepareFrame(context, {
					scopeKey: "warmup-planar-reflection",
					sceneTargetMode: "color",
					temporalStateMode: "disabled",
				});

				for (const packet of drawPackets) {
					total++;
					try {
						const resources = reflectionResources ?
							await this.getDrawResources(packet, reflectionResources, {
								sceneTargetMode: "color",
								drawMode: "reflection-capture",
							})
						:	null;
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
								`WebGPUPlanarReflectionCaptureWarmup:${packet.id}`
							)
						);
					}
					await yieldController.yieldIfNeeded();
				}

				for (const packet of context.scene.reflectivePackets) {
					total++;
					try {
						const resources = reflectionResources ?
							await this.getDrawResources(packet, reflectionResources, {
								sceneTargetMode: "mrt",
								drawMode: "planar-reflection-composite",
							})
						:	null;
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
								`WebGPUPlanarReflectionCompositeWarmup:${packet.id}`
							)
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
				await this._shadowPass.warmup();
				compiled++;
			} catch (error) {
				failed++;
				errors.push(
					toShaderCompileError(error, "webgpu", "WebGPUShadowWarmup")
				);
			}
			await yieldController.yieldIfNeeded();
		}

		if (plan.enableParticles) {
			total++;
			try {
				await this._ensureParticleResources(
					plan.sceneTargetMode,
					1,
					"legacy"
				);
				compiled++;
			} catch (error) {
				failed++;
				errors.push(
					toShaderCompileError(error, "webgpu", "WebGPUParticleWarmup")
				);
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
		encoder?: ICommandEncoder | null
	): Promise<void> {
		const shadowPackets = this.buildParticleMeshDrawPackets(context, {
			includeOpaque: false,
			includeTransparent: false,
			includeShadowCasters: true,
			includeShadowTransmitters: true,
		});
		if (shadowPackets.length <= 0) {
			await this._shadowPass.render(context, encoder);
			return;
		}
		const particleShadowCasters = shadowPackets.filter(
			(packet) => (packet.passFlags & DRAW_PACKET_FLAG_SHADOW_CASTER) !== 0
		);
		const particleShadowTransmitters = shadowPackets.filter(
			(packet) =>
				(packet.passFlags & DRAW_PACKET_FLAG_SHADOW_TRANSMITTER) !== 0
		);
		await this._shadowPass.render(
			{
				...context,
				scene: {
					...context.scene,
					shadowCasterPackets: [
						...context.scene.shadowCasterPackets,
						...particleShadowCasters,
					],
					shadowTransmitterPackets: [
						...context.scene.shadowTransmitterPackets,
						...particleShadowTransmitters,
					],
				},
			},
			encoder
		);
	}

	/**
	 * Advances caches that are shared across prepared resource scopes.
	 *
	 * @sideEffects Increments material/particle cache frame counters and evicts
	 * stale per-frame cache entries.
	 */
	public beginFrameResourceLifecycle(): void {
		this._frameId++;
		this._materialBindings.beginFrame();
		this._evictParticleBindings();
	}

	public prepareFrame(
		context: FrameContext,
		options: WebGPUPrepareFrameOptions
	): WebGPUPreparedFrameResources {
		const resolvedOptions = this._resolvePrepareFrameOptions(options);
		const jointMatrixMap =
			context.transient.get(ANIMATION_WEBGPU_JOINT_MATRICES_KEY) ?? null;
		const morphWeightMap =
			context.transient.get(ANIMATION_WEBGPU_MORPH_WEIGHTS_KEY) ?? null;
		const scene = context.scene;
		const features = context.features;
		const postProcess = context.postProcess;
		const shAmbientCoeffs = context.shAmbientCoeffs;
		const renderWidth = Math.max(1, context.attachments.width || 1);
		const renderHeight = Math.max(1, context.attachments.height || 1);
		const temporalHistoryReset =
			context.incremental?.temporalHistoryReset === true;
		const shadowMaps = context.shadowMaps;
		const particleMeshShadowPackets =
			this.buildParticleMeshDrawPackets(context, {
				includeOpaque: false,
				includeTransparent: false,
				includeShadowCasters: true,
				includeShadowTransmitters: true,
			});
		const shadowCasterPackets = [
			...scene.shadowCasterPackets,
			...particleMeshShadowPackets.filter(
				(packet) =>
					(packet.passFlags & DRAW_PACKET_FLAG_SHADOW_CASTER) !== 0
			),
		];
		const shadowTransmitterPackets = [
			...scene.shadowTransmitterPackets,
			...particleMeshShadowPackets.filter(
				(packet) =>
					(packet.passFlags & DRAW_PACKET_FLAG_SHADOW_TRANSMITTER) !== 0
			),
		];
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

		const shadowLights = scene.lights.filter(isShadowCastingLight);
		syncShadowMapRegistry(shadowMaps, shadowLights);
		const shadowCasterBounds = resolveShadowCasterBounds(
			shadowCasterPackets,
			scene.sceneBounds
		);
		const combinedShadowCasterBounds = mergeParticleShadowBounds(
			shadowCasterBounds,
			resolveParticleShadowCasterBounds(scene.particleSystems)
		);
		const selectedCSMLights = selectCSMDirectionalLights(
			shadowLights,
			WEBGPU_SHADOW_CAPABILITIES.maxCsmDirectionalLights
		);
		if (features.enableShadows) {
			for (const light of shadowLights) {
				const shadowRenderSet = shadowMaps.get(light);
				if (shadowRenderSet) {
					updateShadowMapMetadata(
						shadowRenderSet,
						light,
						combinedShadowCasterBounds,
						{
							camera: scene.camera,
							backendCapabilities: WEBGPU_SHADOW_CAPABILITIES,
							allowCSMDirectionalLights: selectedCSMLights,
							onWarning: (key, message) =>
								Logger.warn(`[${key}] ${message}`, {
									scope: "WebGPURenderResources",
									onceKey: key,
								}),
						}
					);
				}
			}
		}
		this._pagedShadowRuntime.prepareFrame({
			context,
			encoder: null,
			renderSets: shadowMaps,
			shadowCasterPackets,
			shadowTransmitterPackets,
		});

		const lightingState = collectWebGPULighting(
			scene.lights,
			features.enableLighting,
			features.enableSH,
			features.enableShadows,
			shadowMaps,
			featureState.enableClusteredLighting
		);
		for (const warning of lightingState.warnings) {
			Logger.warn(`[${warning.key}] ${warning.message}`, {
				scope: "WebGPURenderResources",
			});
		}

		const environmentState = collectWebGPUEnvironment(
			scene,
			featureState.enableSH,
			shAmbientCoeffs
		);
		for (const warning of environmentState.warnings) {
			Logger.warn(`[${warning.key}] ${warning.message}`, {
				scope: "WebGPURenderResources",
			});
		}

		this._shadowAtlases.prepare(
			lightingState,
			this._resolveShadowAtlasTileSize(scene.shadowMaps, features.enableShadows)
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
			}
		);
		scope.clusteredLighting.prepareFrame(
			scene,
			featureState,
			lightingState,
			renderWidth,
			renderHeight
		);

		const frameResources: WebGPUPreparedFrameResources = {
			scopeKey: resolvedOptions.scopeKey,
			sceneTargetMode: resolvedOptions.sceneTargetMode,
			frameBinding: scope.frameBindings.getSceneBinding(),
			decalFrameBinding: scope.frameBindings.getDecalFrameBinding(),
			environmentBinding: scope.frameBindings.getEnvironmentBinding(),
			clusteredSceneBinding: scope.clusteredLighting.getSceneBinding(),
			lightingState,
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

	public getFrameBinding(
		frameResources: WebGPUPreparedFrameResources
	): IBindingGroup {
		return this._requirePreparedFrameResources(
			frameResources,
			"getFrameBinding"
		).frameBinding;
	}

	public getClusteredSceneBinding(
		frameResources: WebGPUPreparedFrameResources
	): IBindingGroup {
		return this._requirePreparedFrameResources(
			frameResources,
			"getClusteredSceneBinding"
		).clusteredSceneBinding;
	}

	/**
	 * Returns the bind group layout used by G-buffer geometry shaders to write
	 * deferred storage payload textures.
	 *
	 * @returns The WebGPU bind group layout for `gMaterialExt0/1/2` writes.
	 * @sideEffects None.
	 */
	public getGBufferWriteLayout(): GPUBindGroupLayout {
		return this._layouts.gbufferWriteBindGroupLayout;
	}

	/**
	 * Returns the bind group layout used by the deferred lighting shader to read
	 * color and storage G-buffer payload textures.
	 *
	 * @returns The WebGPU bind group layout for deferred G-buffer reads.
	 * @sideEffects None.
	 */
	public getGBufferReadLayout(): GPUBindGroupLayout {
		return this._layouts.gbufferReadBindGroupLayout;
	}

	/**
	 * Returns the bind group layout used by the WebGPU deferred decal pass.
	 *
	 * @returns The WebGPU bind group layout for decal material resources.
	 * @sideEffects None.
	 */
	public getDecalBindGroupLayout(): GPUBindGroupLayout {
		return this._layouts.decalBindGroupLayout;
	}

	/**
	 * Returns the bind group layout used by fallback deferred decal draws to
	 * write extended G-buffer storage payload textures.
	 *
	 * @returns The WebGPU bind group layout for decal storage outputs.
	 * @sideEffects None.
	 */
	public getDecalOutputBindGroupLayout(): GPUBindGroupLayout {
		return this._layouts.decalOutputBindGroupLayout;
	}

	/**
	 * Returns the bind group layout used by WebGPU deferred decal batch data.
	 *
	 * @returns The WebGPU bind group layout for tile-binned decal resources.
	 * @sideEffects None.
	 */
	public getDecalBatchBindGroupLayout(): GPUBindGroupLayout {
		return this._layouts.decalBatchBindGroupLayout;
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
		if (!this._deferredUnusedBinding) {
			this._deferredUnusedBinding = this._backend.createBindingGroup({
				layout: this._layouts.deferredUnusedBindGroupLayout,
				entries: [],
				label: "WebGPUDeferredUnusedBinding",
			});
		}
		return this._deferredUnusedBinding;
	}

	/**
	 * Resolves the fullscreen deferred lighting render pipeline.
	 *
	 * @returns A pipeline that reads the G-buffer and writes `sceneColorMain`.
	 * @sideEffects May compile and cache the deferred lighting pipeline.
	 */
	public async getDeferredLightingPipeline(): Promise<IRenderPipeline> {
		return this._pipelineLibrary.getDeferredLightingPipeline();
	}

	/**
	 * Resolves the fullscreen deferred decal render pipeline.
	 *
	 * @returns A pipeline that reads G-buffer snapshots and writes updated
	 * G-buffer channels.
	 * @sideEffects May compile and cache the decal shader module and pipeline.
	 */
	public async getDecalPipeline(): Promise<IRenderPipeline> {
		if (this._decalPipeline) {
			return this._decalPipeline;
		}
		if (!this._decalShaderModule) {
			const shader = await ShaderSource.load("webgpu.utility.decal.composite");
			this._decalShaderModule = await this._backend.createShaderModule({
				code: shader.code,
				sourceMap: shader.sourceMap,
				label: "WebGPUDecalShader",
				language: "wgsl",
				stage: "unknown",
				sourceKind: "decal",
			});
		}
		this._decalPipeline = await this._backend.createPipeline({
			layout: this._layouts.decalPipelineLayout,
			label: "WebGPUDeferredDecalPipeline",
			vertex: {
				module: this._decalShaderModule,
				entryPoint: "vsMain",
			},
			fragment: {
				module: this._decalShaderModule,
				entryPoint: "fsMain",
				targets: [
					{ format: TextureFormat.RGBA8Unorm },
					{ format: TextureFormat.RGBA16Float },
					{ format: TextureFormat.RGBA16Float },
					{ format: TextureFormat.RGBA16Float },
					{ format: TextureFormat.RGBA16Float },
					{ format: TextureFormat.RGBA16Float },
					{ format: TextureFormat.RGBA16Float },
				],
			},
			primitive: {
				topology: "triangle-list" as any,
				cullMode: "none",
				frontFace: "ccw",
			},
			sampleCount: 1,
		} as any);
		return this._decalPipeline;
	}

	/**
	 * Resolves the tile-binned deferred decal compute pipeline.
	 *
	 * @returns A compute pipeline that applies compatible decal segments.
	 * @sideEffects May compile and cache the decal shader module and pipeline.
	 */
	public async getDecalBatchPipeline(): Promise<IComputePipeline> {
		if (this._decalBatchPipeline) {
			return this._decalBatchPipeline;
		}
		if (!this._decalShaderModule) {
			const shader = await ShaderSource.load("webgpu.utility.decal.composite");
			this._decalShaderModule = await this._backend.createShaderModule({
				code: shader.code,
				sourceMap: shader.sourceMap,
				label: "WebGPUDecalShader",
				language: "wgsl",
				stage: "unknown",
				sourceKind: "decal",
			});
		}
		this._decalBatchPipeline = await this._backend.createComputePipeline({
			layout: this._layouts.decalBatchPipelineLayout,
			label: "WebGPUDeferredDecalBatchPipeline",
			compute: {
				module: this._decalShaderModule,
				entryPoint: "csMainBatch",
			},
		} as any);
		return this._decalBatchPipeline;
	}

	public updateParticleShadowVolumes(
		frameResources: WebGPUPreparedFrameResources,
		context: FrameContext
	): void {
		const prepared = this._requirePreparedFrameResources(
			frameResources,
			"updateParticleShadowVolumes"
		);
		if (!context) {
			return;
		}
		const scope = this._requireFrameScope(prepared.scopeKey);
		scope.frameBindings.updateParticleShadowVolumes(
			context,
			prepared.lightingState
		);
		prepared.frameBinding = scope.frameBindings.getSceneBinding();
	}

	/**
	 * @internal WebGPU paged shadow frame graph hook.
	 */
	public preparePagedShadowFrame(
		request: WebGPUPagedShadowFrameRequest
	): void {
		this._pagedShadowRuntime.prepareFrame(request);
	}

	/**
	 * @internal WebGPU paged shadow frame graph hook.
	 */
	public recordPagedShadowPageMarkPass(
		request: WebGPUPagedShadowFrameRequest
	): void | Promise<void> {
		return this._pagedShadowRuntime.recordPageMarkPass(request);
	}

	/**
	 * @internal WebGPU paged shadow frame graph hook.
	 */
	public recordPagedShadowPageAllocationPass(
		request: WebGPUPagedShadowFrameRequest
	): void | Promise<void> {
		return this._pagedShadowRuntime.recordPageAllocationPass(request);
	}

	/**
	 * @internal WebGPU paged shadow frame graph hook.
	 */
	public recordPagedShadowPageTableCopyPass(
		request: WebGPUPagedShadowFrameRequest
	): void | Promise<void> {
		return this._pagedShadowRuntime.recordPageTableCopyPass(request);
	}

	/**
	 * @internal WebGPU paged shadow frame graph hook.
	 */
	public recordPagedShadowDepthPass(
		request: WebGPUPagedShadowFrameRequest
	): Promise<void> {
		return this._pagedShadowRuntime.recordDepthPass(request);
	}

	/**
	 * @internal WebGPU paged shadow delayed feedback hook.
	 */
	public recordPagedShadowFeedbackPass(
		request: WebGPUPagedShadowFrameRequest
	): void | Promise<void> {
		return this._pagedShadowRuntime.recordFeedbackPass(request);
	}

	/**
	 * @internal WebGPU frame binding hook.
	 */
	public getPagedShadowResources(): WebGPUPagedShadowResources {
		return this._pagedShadowRuntime.getResources();
	}

	public async buildClusteredLighting(
		encoder: ICommandEncoder,
		frameResources: WebGPUPreparedFrameResources
	): Promise<void> {
		const scope = this._requireFrameScope(frameResources.scopeKey);
		await scope.clusteredLighting.build(
			encoder,
			frameResources.frameBinding
		);
		frameResources.clusteredSceneBinding =
			scope.clusteredLighting.getSceneBinding();
	}

	public onShaderRuntimeChanged(): void {
		if (this._destroyed) {
			return;
		}
		this._pipelineLibrary.invalidateShaderRuntimeCaches();
		this._particleShaderModule = null;
		this._particlePipelineAlpha.clear();
		this._particlePipelineOITAlpha.clear();
		this._particlePipelineAdditive.clear();
		this._clearParticleBindingCache();
		for (const scope of this._frameScopes.values()) {
			scope.clusteredLighting.onShaderRuntimeChanged();
		}
		this._decalShaderModule = null;
		this._decalPipeline = null;
		this._decalBatchPipeline = null;
		this._shadowPass.onShaderRuntimeChanged();
	}

	public destroy(): void {
		if (this._destroyed) {
			return;
		}
		this._destroyed = true;
		this._clearParticleBindingCache();
		this._particleShaderModule = null;
		this._decalShaderModule = null;
		this._decalPipeline = null;
		this._decalBatchPipeline = null;
		this._particlePipelineAlpha.clear();
		this._particlePipelineOITAlpha.clear();
		this._particlePipelineAdditive.clear();
		this._particleQuadBuffer?.destroy();
		this._particleQuadBuffer = null;
		this._particleInstanceBuffer?.destroy();
		this._particleInstanceBuffer = null;
		this._particleInstanceCapacity = 0;
		this._destroyBindingGroup(this._deferredUnusedBinding);
		this._deferredUnusedBinding = null;
		for (const scope of this._frameScopes.values()) {
			scope.frameBindings.destroy();
			scope.clusteredLighting.destroy();
		}
		this._frameScopes.clear();
		this._materialBindings.destroy();
		this._pagedShadowRuntime.destroy();
		this._shadowPass.destroy();
		this._pipelineLibrary.destroy();
		this._shadowAtlases.destroy();
		this._textureRegistry.destroy();
		this._geometryRegistry.destroy();
	}

	public getLightingState(
		frameResources: WebGPUPreparedFrameResources
	): WebGPULightingState {
		return this._requirePreparedFrameResources(
			frameResources,
			"getLightingState"
		).lightingState;
	}

	public getTextureForSlot(
		texture: Texture | null,
		slotIndex: number
	): IRenderTexture {
		return this._textureRegistry.getTextureForSlot(texture, slotIndex);
	}

	public getTextureForSlotAsync(
		texture: Texture | null,
		slotIndex: number
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
		mipLevelCount: number = 1
	): void {
		this._textureRegistry.registerExternalTexture(
			texture,
			resource,
			uploadedVersion,
			mipLevelCount
		);
	}

	public unregisterExternalTexture(texture: Texture): void {
		this._textureRegistry.unregisterExternalTexture(texture);
	}

	public get sceneFrameLayout(): GPUBindGroupLayout {
		return this._layouts.sceneFrameBindGroupLayout;
	}

	private _resolvePrepareFrameOptions(
		options: WebGPUPrepareFrameOptions | undefined
	): WebGPUPrepareFrameOptions {
		if (!options) {
			throw new Error(
				"WebGPURenderResources.prepareFrame() requires explicit frame options."
			);
		}
		if (typeof options.scopeKey !== "string" || options.scopeKey.trim() === "") {
			throw new Error(
				"WebGPURenderResources.prepareFrame() requires a non-empty scopeKey."
			);
		}
		if (!options.sceneTargetMode) {
			throw new Error(
				"WebGPURenderResources.prepareFrame() requires sceneTargetMode."
			);
		}
		return {
			scopeKey: options.scopeKey,
			sceneTargetMode: options.sceneTargetMode,
			temporalStateMode: options?.temporalStateMode,
		};
	}

	private _getOrCreateFrameScope(scopeKey: string): WebGPUFrameResourceScope {
		let scope = this._frameScopes.get(scopeKey);
		if (scope) {
			return scope;
		}
		scope = {
			frameBindings: new WebGPUFrameBindingCache(
				this._backend,
				this._layouts,
				this._textureRegistry,
				this._shadowAtlases,
				this._pagedShadowRuntime
			),
			clusteredLighting: new WebGPUClusteredLightingRuntime(
				this._computeFacade,
				this._layouts.clusteredSceneBindGroupLayout,
				this._layouts.sceneFrameBindGroupLayout,
				(key, message) =>
					Logger.warn(`[${key}] ${message}`, {
						scope: "WebGPUClusteredLightingRuntime",
					})
			),
			prepared: null,
		};
		this._frameScopes.set(scopeKey, scope);
		return scope;
	}

	private _requireFrameScope(scopeKey: string): WebGPUFrameResourceScope {
		const scope = this._frameScopes.get(scopeKey);
		if (!scope) {
			throw new Error(
				`WebGPURenderResources frame scope "${scopeKey}" is not prepared.`
			);
		}
		return scope;
	}

	private _requirePreparedFrameResources(
		frameResources: WebGPUPreparedFrameResources | undefined,
		methodName: string
	): WebGPUPreparedFrameResources {
		if (!this._isPreparedFrameResources(frameResources)) {
			throw new Error(
				`WebGPURenderResources.${methodName}() requires explicit prepared frame resources.`
			);
		}
		return frameResources;
	}

	private _isPreparedFrameResources(
		value: unknown
	): value is WebGPUPreparedFrameResources {
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
		shadowMaps: ReadonlyMap<ShadowCastingLight, ShadowRenderSet>,
		enableShadows: boolean
	): number {
		if (!enableShadows) {
			return 1;
		}

		let tileSize = 0;
		for (const renderSet of shadowMaps.values()) {
			const hasValidSlice = renderSet.slices.some(
				(slice) => !!slice.shadowMap.viewProjectionMatrix
			);
			if (!hasValidSlice) {
				continue;
			}
			tileSize = Math.max(tileSize, renderSet.size | 0);
		}

		return Math.max(1, tileSize);
	}

	public async getDrawResources(
		packet: DrawPacket,
		frameResources: WebGPUPreparedFrameResources,
		options: WebGPUDrawResourceOptions = {}
	): Promise<WebGPUDrawResources[] | null> {
		const prepared = this._requirePreparedFrameResources(
			frameResources,
			"getDrawResources"
		);
		const transparentPipelineMode =
			options.transparentPipelineMode ?? "default";
		const sceneTargetMode =
			options.sceneTargetMode ?? prepared.sceneTargetMode;
		const drawMode = options.drawMode ?? "default";
		const sampleCountOverride = options.sampleCountOverride;
		const results: WebGPUDrawResources[] = [];
		const geometry = this._geometryRegistry.getGeometry(packet.primitive);
		const topology = geometry.topology;
		const frameBinding = prepared.frameBinding;
		const clusteredBinding = prepared.clusteredSceneBinding;
		const animationState = this._resolveAnimationState(
			packet,
			geometry,
			prepared
		);

		// ----- SOLID OBJECT -----
		const solidMaterialData = createWebGPUMaterialUniformData(
			packet.material,
			false
		);
		for (const warning of solidMaterialData.warnings) {
			Logger.warn(`[${warning.key}] ${warning.message}`, {
				scope: "WebGPURenderResources",
			});
		}

		const solidPipeline =
			drawMode === "early-z-prepass" ?
				await this._pipelineLibrary.getEarlyZPrepassPipeline(
					packet.material,
					sceneTargetMode,
					false,
					topology,
					sampleCountOverride
				)
			:	await this._pipelineLibrary.getPipeline(
					packet.material,
					sceneTargetMode,
					false,
					topology,
					transparentPipelineMode,
					drawMode,
					sampleCountOverride
				);
		if (!solidPipeline) {
			return null;
		}
		const solidTextures = await Promise.all(
			solidMaterialData.textureSlots.map((slot, index) =>
				this._textureRegistry.getTextureForSlotAsync(slot.map, index)
			)
		);
		const solidSamplers = solidMaterialData.textureSlots.map((slot) =>
			this._textureRegistry.getSamplerForTexture(slot.map)
		);
		const solidAnisotropyTexture = await this._textureRegistry.getTextureForSlotAsync(
			solidMaterialData.anisotropyTexture.map,
			-1
		);
		const solidModelBinding = this._materialBindings.getBinding(
			packet,
			solidPipeline,
			solidMaterialData,
			solidTextures,
			solidSamplers,
			solidAnisotropyTexture,
			animationState
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
			const wireMaterialData = createWebGPUMaterialUniformData(
				packet.material,
				true
			);
			const wirePipeline = await this._pipelineLibrary.getPipeline(
				packet.material,
				sceneTargetMode,
				true,
				topology,
				transparentPipelineMode,
				drawMode,
				sampleCountOverride
			);
			const wireTextures = await Promise.all(
				wireMaterialData.textureSlots.map((slot, index) =>
					this._textureRegistry.getTextureForSlotAsync(slot.map, index)
				)
			);
			const wireSamplers = wireMaterialData.textureSlots.map((slot) =>
				this._textureRegistry.getSamplerForTexture(slot.map)
			);
			const wireAnisotropyTexture = await this._textureRegistry.getTextureForSlotAsync(
				wireMaterialData.anisotropyTexture.map,
				-1
			);
			const wireModelBinding = this._materialBindings.getBinding(
				packet,
				wirePipeline,
				wireMaterialData,
				wireTextures,
				wireSamplers,
				wireAnisotropyTexture,
				animationState
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
		frameResources: WebGPUPreparedFrameResources,
		sceneTargetMode?: WebGPUSceneTargetMode,
		options: WebGPUEnvironmentResourceOptions = {}
	): Promise<WebGPUEnvironmentDrawResources | null> {
		const prepared = this._requirePreparedFrameResources(
			frameResources,
			"getEnvironmentResources"
		);
		const resolvedSceneTargetMode = sceneTargetMode ?? prepared.sceneTargetMode;
		if (
			!prepared.featureState.enableEnvironment ||
			!prepared.environmentState.environmentTexture
		) {
			return null;
		}

		const pipeline = await this._pipelineLibrary.getEnvironmentPipeline(
			resolvedSceneTargetMode,
			options.sampleCountOverride
		);
		const frameBinding = prepared.environmentBinding;

		return {
			pipeline,
			frameBinding,
		};
	}

	/**
	 * Converts simulated mesh-particle batches into transient draw packets that
	 * reuse the regular WebGPU mesh material pipeline.
	 *
	 * @param context Current frame context containing particle mesh transients.
	 * @param options Pass filters controlling which packet classes are returned.
	 * @returns New draw packets for the requested particle mesh passes.
	 * @constraints This does not mutate the scene graph or ECS; packets are valid
	 * only for the current frame.
	 * @sideEffects None.
	 */
	public buildParticleMeshDrawPackets(
		context: FrameContext,
		options: WebGPUParticleMeshPacketOptions = {}
	): DrawPacket[] {
		const batches = context.transient.get(PARTICLE_MESH_TRANSIENT_BATCHES_KEY);
		if (!batches || batches.length === 0) {
			return [];
		}
		const includeOpaque = options.includeOpaque ?? true;
		const includeTransparent = options.includeTransparent ?? true;
		const includeShadowCasters = options.includeShadowCasters ?? false;
		const includeShadowTransmitters =
			options.includeShadowTransmitters ?? false;
		const includeReflective = options.includeReflective ?? false;
		const packets: DrawPacket[] = [];

		for (const batch of batches) {
			for (let particleIndex = 0; particleIndex < batch.particles.length; particleIndex++) {
				const packet = this._createParticleMeshPacket(
					batch,
					batch.particles[particleIndex],
					particleIndex
				);
				const flags = packet.passFlags;
				const isTransparent =
					(flags & DRAW_PACKET_FLAG_TRANSPARENT) !== 0;
				const matchesMain =
					(isTransparent && includeTransparent) ||
					(!isTransparent && includeOpaque);
				const matchesShadow =
					(includeShadowCasters &&
						(flags & DRAW_PACKET_FLAG_SHADOW_CASTER) !== 0) ||
					(includeShadowTransmitters &&
						(flags & DRAW_PACKET_FLAG_SHADOW_TRANSMITTER) !== 0);
				const matchesReflective =
					includeReflective &&
					(flags & DRAW_PACKET_FLAG_REFLECTIVE) !== 0;
				if (matchesMain || matchesShadow || matchesReflective) {
					packets.push(packet);
				}
			}
		}

		packets.sort(compareParticleMeshPackets);
		return packets;
	}

	public async renderParticles(
		encoder: ICommandEncoder,
		context: FrameContext,
		targets: WebGPUParticlePassTargets,
		frameResources: WebGPUPreparedFrameResources,
		mode: WebGPUSceneTargetMode,
		options: WebGPUParticleRenderOptions = {}
	): Promise<number> {
		const prepared = this._requirePreparedFrameResources(
			frameResources,
			"renderParticles"
		);
		const includeBlendModes = options.includeBlendModes ?? null;
		const pipelineMode = options.pipelineMode ?? "legacy";
		const sampleCount = this._resolveSampleCount(
			mode,
			options.sampleCountOverride
		);
		const particlePipelineKey = this._createParticlePipelineCacheKey(
			mode,
			sampleCount
		);
		const gpuBatches = context.transient.get(WEBGPU_PARTICLE_DRAW_BATCHES_KEY);
		if (gpuBatches && gpuBatches.length > 0) {
			const drawBatches = gpuBatches.filter((batch) => {
				if (batch.instanceCount <= 0) {
					return false;
				}
				if (!includeBlendModes || includeBlendModes.length === 0) {
					return true;
				}
				return includeBlendModes.includes(batch.blendMode);
			});
			if (drawBatches.length > 0) {
				return this._renderParticlesFromGPUBatches(
					encoder,
					context,
					targets,
					prepared,
					mode,
					pipelineMode,
					options.sampleCountOverride,
					drawBatches
				);
			}
		}
		const batches = context.transient.get(PARTICLE_TRANSIENT_BATCHES_KEY);
		if (!batches || batches.length === 0) return 0;

		const drawBatches = batches.filter((batch) => {
			if (batch.particles.length <= 0) {
				return false;
			}
			if (!includeBlendModes || includeBlendModes.length === 0) {
				return true;
			}
			return includeBlendModes.includes(batch.blendMode);
		});
		if (drawBatches.length === 0) return 0;

		const totalParticles = drawBatches.reduce(
			(sum, batch) => sum + batch.particles.length,
			0
		);
		if (totalParticles <= 0) return 0;

		await this._ensureParticleResources(
			mode,
			totalParticles,
			pipelineMode,
			options.sampleCountOverride
		);
		if (!this._particleInstanceBuffer || !this._particleQuadBuffer) return 0;

		const floatsPerInstance = WEBGPU_PARTICLE_INSTANCE_FLOATS;
		const instanceData = new Float32Array(totalParticles * floatsPerInstance);
		const drawRanges: Array<{
			batch: ParticleRenderBatch;
			firstInstance: number;
			instanceCount: number;
		}> = [];
		const activeCacheKeys = new Set<string>();

		let particleOffset = 0;
		for (const batch of drawBatches) {
			const firstInstance = particleOffset;
			for (const particle of batch.particles) {
				const offset = particleOffset * floatsPerInstance;
				instanceData[offset] = particle.position.x;
				instanceData[offset + 1] = particle.position.y;
				instanceData[offset + 2] = particle.position.z;
				instanceData[offset + 3] = Math.max(0.001, particle.size);

				instanceData[offset + 4] = particle.color.r / 255;
				instanceData[offset + 5] = particle.color.g / 255;
				instanceData[offset + 6] = particle.color.b / 255;
				instanceData[offset + 7] = clamp(particle.color.a);

				instanceData[offset + 8] = particle.uvRect.u0;
				instanceData[offset + 9] = particle.uvRect.v0;
				instanceData[offset + 10] = particle.uvRect.u1;
				instanceData[offset + 11] = particle.uvRect.v1;

				instanceData[offset + 12] = particle.rotation;
				instanceData[offset + 13] = batch.receiveShadows ? 1 : 0;
				instanceData[offset + 14] = 0;
				instanceData[offset + 15] = 0;
				particleOffset++;
			}

			drawRanges.push({
				batch,
				firstInstance,
				instanceCount: batch.particles.length,
			});
		}

		this._backend.writeBuffer(this._particleInstanceBuffer, instanceData);
		const frameBinding = prepared.frameBinding;
		const alphaPipeline = this._particlePipelineAlpha.get(particlePipelineKey);
		const oitAlphaPipeline =
			this._particlePipelineOITAlpha.get(particlePipelineKey);
		const additivePipeline =
			this._particlePipelineAdditive.get(particlePipelineKey);
		if (pipelineMode === "oit") {
			if (!oitAlphaPipeline) {
				return 0;
			}
		} else if (!alphaPipeline || !additivePipeline) {
			return 0;
		}

		encoder.beginRenderPass({
			label: targets.label,
			colorAttachments: targets.colorAttachments,
			depthStencilAttachment: {
				view: targets.depth,
				depthLoadOp: "load",
				depthStoreOp: "store",
			},
		});
		encoder.setBindingGroup(0, frameBinding);
		encoder.setVertexBuffer(0, this._particleQuadBuffer);
		encoder.setVertexBuffer(1, this._particleInstanceBuffer);
		const targetView =
			targets.colorAttachments.find((attachment) => attachment.view)?.view ??
			targets.depth;
		const targetWidth = Math.max(
			1,
			Math.floor(
				typeof targetView?.width === "number" ?
					targetView.width
				:	context.attachments.width
			)
		);
		const targetHeight = Math.max(
			1,
			Math.floor(
				typeof targetView?.height === "number" ?
					targetView.height
				:	context.attachments.height
			)
		);
		const dirtyRects =
			context.incremental?.enabled &&
			!context.incremental.forceFullFrame &&
			(context.incremental.dirtyRects?.length ?? 0) > 0 ?
				context.incremental.dirtyRects
					.map((rect) => {
						const minX = Math.max(
							0,
							Math.floor(
								(rect.x * targetWidth) /
									Math.max(1, context.attachments.width)
							)
						);
						const minY = Math.max(
							0,
							Math.floor(
								(rect.y * targetHeight) /
									Math.max(1, context.attachments.height)
							)
						);
						const maxX = Math.min(
							targetWidth,
							Math.ceil(
								((rect.x + rect.width) * targetWidth) /
									Math.max(1, context.attachments.width)
							)
						);
						const maxY = Math.min(
							targetHeight,
							Math.ceil(
								((rect.y + rect.height) * targetHeight) /
									Math.max(1, context.attachments.height)
							)
						);
						return {
							x: minX,
							y: minY,
							width: maxX - minX,
							height: maxY - minY,
						};
					})
					.filter((rect) => rect.width > 0 && rect.height > 0)
			:	[{
					x: 0,
					y: 0,
					width: targetWidth,
					height: targetHeight,
				}];

		for (const range of drawRanges) {
			const texture = await this._textureRegistry.getTextureForSlotAsync(
				range.batch.texture,
				0
			);
			const sampler = this._textureRegistry.getSamplerForTexture(
				range.batch.texture
			);
			const cacheKey = `particle_${range.batch.systemId}`;
			activeCacheKeys.add(cacheKey);
			const cachedBinding = this._particleBindingCache.get(cacheKey);
			const uvTransformBuffer =
				cachedBinding?.uvTransformBuffer ??
				this._backend.createBuffer({
					size: WEBGPU_PARTICLE_UV_UNIFORM_SIZE,
					usage: BufferUsage.Uniform | BufferUsage.CopyDst,
					label: `ParticleUVTransform_${range.batch.systemId}`,
				});
			let particleBinding: IBindingGroup;
			if (
				cachedBinding &&
				cachedBinding.texture === texture &&
				cachedBinding.sampler === sampler
			) {
				particleBinding = cachedBinding.group;
				cachedBinding.lastUsedFrame = this._frameId;
			} else {
				this._destroyBindingGroup(cachedBinding?.group ?? null);
				particleBinding = this._backend.createBindingGroup({
					layout: this._layouts.particleBindGroupLayout,
					entries: [
						{
							binding: WEBGPU_PARTICLE_BINDING_TEXTURE,
							resource: texture,
						},
						{
							binding: WEBGPU_PARTICLE_BINDING_SAMPLER,
							resource: sampler,
						},
						{
							binding: WEBGPU_PARTICLE_BINDING_UV_TRANSFORM,
							resource: uvTransformBuffer,
						},
					],
					label: `ParticleBinding_${range.batch.systemId}`,
				});
				this._particleBindingCache.set(cacheKey, {
					group: particleBinding,
					texture,
					sampler,
					uvTransformBuffer,
					lastUsedFrame: this._frameId,
				});
			}
			const uvTransformData = this._createParticleUVTransformData(
				range.batch.texture
			);
			this._backend.writeBuffer(uvTransformBuffer, uvTransformData);
			let pipeline: IRenderPipeline;
			if (pipelineMode === "oit") {
				pipeline = oitAlphaPipeline!;
			} else {
				pipeline =
					range.batch.blendMode === ParticleBlendMode.Additive ?
						additivePipeline!
					:	alphaPipeline!;
			}
			encoder.setPipeline(pipeline);
			encoder.setBindingGroup(1, particleBinding);
			for (const rect of dirtyRects) {
				encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
				encoder.draw(6, range.instanceCount, 0, range.firstInstance);
			}
		}

		encoder.endRenderPass();
		this._evictParticleBindings(activeCacheKeys);
		return totalParticles;
	}

	private _createParticleMeshPacket(
		batch: ParticleMeshRenderBatch,
		particle: ParticleMeshRenderItem,
		particleIndex: number
	): DrawPacket {
		const material = batch.material;
		const isTransparent = isMaterialTransparentPass(material);
		const isReflective =
			material.reflectivity > 0 && material.mirrorPlane !== null;
		const supportsShadowCasting =
			(batch.primitive.topology ?? DEFAULT_PRIMITIVE_DRAW_TOPOLOGY) ===
			DEFAULT_PRIMITIVE_DRAW_TOPOLOGY;
		const currentWorldMatrix = createParticleMeshWorldMatrix(
			particle.position,
			particle.rotation,
			particle.size
		);
		const previousWorldMatrix = createParticleMeshWorldMatrix(
			particle.previousPosition,
			particle.previousRotation,
			particle.size
		);
		const normalMatrix = Matrix4.normalMatrix(
			currentWorldMatrix
		) as Matrix3Arr;
		const worldCenter = Matrix4.transformPoint(
			currentWorldMatrix,
			batch.primitive.boundingSphere.center
		);
		const meshInstance = createParticleMeshInstance(
			batch,
			currentWorldMatrix
		);

		let passFlags = 0;
		if (isTransparent) {
			passFlags |= DRAW_PACKET_FLAG_TRANSPARENT;
			if (batch.castShadows && supportsShadowCasting) {
				passFlags |= DRAW_PACKET_FLAG_SHADOW_TRANSMITTER;
			}
		} else if (batch.castShadows && supportsShadowCasting) {
			passFlags |= DRAW_PACKET_FLAG_SHADOW_CASTER;
		}
		if (isReflective) {
			passFlags |= DRAW_PACKET_FLAG_REFLECTIVE;
		}

		return {
			id: [
				"particleMesh",
				batch.systemId,
				batch.templateIndex,
				batch.primitive.id,
				particleIndex,
			].join(":"),
			meshInstance,
			mesh: batch.mesh,
			primitive: batch.primitive,
			material,
			geometry: batch.primitive.geometry,
			worldMatrix: currentWorldMatrix,
			previousWorldMatrix,
			normalMatrix,
			worldBounds: {
				center: {
					x: worldCenter.x,
					y: worldCenter.y,
					z: worldCenter.z,
				},
				radius: batch.primitive.boundingSphere.radius *
					Math.max(0.001, particle.size),
			},
			sortDepth: particle.depth,
			pipelineKey: [
				material.type,
				material.shading,
				material.alphaMode ?? AlphaMode.Opaque,
				material.doubleSided ? "double" : "single",
				material.depthWrite ? "depth-write" : "depth-read",
			].join(":"),
			passFlags,
		};
	}

	private async _renderParticlesFromGPUBatches(
		encoder: ICommandEncoder,
		context: FrameContext,
		targets: WebGPUParticlePassTargets,
		frameResources: WebGPUPreparedFrameResources,
		mode: WebGPUSceneTargetMode,
		pipelineMode: "legacy" | "oit",
		sampleCountOverride: number | undefined,
		drawBatches: readonly WebGPUParticleDrawBatch[]
	): Promise<number> {
		const totalParticles = drawBatches.reduce(
			(sum, batch) => sum + batch.instanceCount,
			0
		);
		if (totalParticles <= 0) {
			return 0;
		}
		await this._ensureParticleResources(
			mode,
			0,
			pipelineMode,
			sampleCountOverride
		);
		if (!this._particleQuadBuffer) {
			return 0;
		}
		const sampleCount = this._resolveSampleCount(mode, sampleCountOverride);
		const particlePipelineKey = this._createParticlePipelineCacheKey(
			mode,
			sampleCount
		);
		const frameBinding = frameResources.frameBinding;
		const alphaPipeline = this._particlePipelineAlpha.get(particlePipelineKey);
		const oitAlphaPipeline =
			this._particlePipelineOITAlpha.get(particlePipelineKey);
		const additivePipeline =
			this._particlePipelineAdditive.get(particlePipelineKey);
		if (pipelineMode === "oit") {
			if (!oitAlphaPipeline) {
				return 0;
			}
		} else if (!alphaPipeline || !additivePipeline) {
			return 0;
		}

		encoder.beginRenderPass({
			label: targets.label,
			colorAttachments: targets.colorAttachments,
			depthStencilAttachment: {
				view: targets.depth,
				depthLoadOp: "load",
				depthStoreOp: "store",
			},
		});
		encoder.setBindingGroup(0, frameBinding);
		encoder.setVertexBuffer(0, this._particleQuadBuffer);
		const dirtyRects = this._resolveParticleDirtyRects(context, targets);
		const activeCacheKeys = new Set<string>();

		for (const batch of drawBatches) {
			const texture = await this._textureRegistry.getTextureForSlotAsync(
				batch.texture,
				0
			);
			const sampler = this._textureRegistry.getSamplerForTexture(batch.texture);
			const cacheKey = `particle_${batch.systemId}`;
			activeCacheKeys.add(cacheKey);
			const cachedBinding = this._particleBindingCache.get(cacheKey);
			const uvTransformBuffer =
				cachedBinding?.uvTransformBuffer ??
				this._backend.createBuffer({
					size: WEBGPU_PARTICLE_UV_UNIFORM_SIZE,
					usage: BufferUsage.Uniform | BufferUsage.CopyDst,
					label: `ParticleUVTransform_${batch.systemId}`,
				});
			let particleBinding: IBindingGroup;
			if (
				cachedBinding &&
				cachedBinding.texture === texture &&
				cachedBinding.sampler === sampler
			) {
				particleBinding = cachedBinding.group;
				cachedBinding.lastUsedFrame = this._frameId;
			} else {
				this._destroyBindingGroup(cachedBinding?.group ?? null);
				particleBinding = this._backend.createBindingGroup({
					layout: this._layouts.particleBindGroupLayout,
					entries: [
						{
							binding: WEBGPU_PARTICLE_BINDING_TEXTURE,
							resource: texture,
						},
						{
							binding: WEBGPU_PARTICLE_BINDING_SAMPLER,
							resource: sampler,
						},
						{
							binding: WEBGPU_PARTICLE_BINDING_UV_TRANSFORM,
							resource: uvTransformBuffer,
						},
					],
					label: `ParticleBinding_${batch.systemId}`,
				});
				this._particleBindingCache.set(cacheKey, {
					group: particleBinding,
					texture,
					sampler,
					uvTransformBuffer,
					lastUsedFrame: this._frameId,
				});
			}

			const uvTransformData = this._createParticleUVTransformData(batch.texture);
			this._backend.writeBuffer(uvTransformBuffer, uvTransformData);
			let pipeline: IRenderPipeline;
			if (pipelineMode === "oit") {
				pipeline = oitAlphaPipeline!;
			} else {
				pipeline =
					batch.blendMode === ParticleBlendMode.Additive ?
						additivePipeline!
					:	alphaPipeline!;
			}

			encoder.setPipeline(pipeline);
			encoder.setBindingGroup(1, particleBinding);
			encoder.setVertexBuffer(1, batch.instanceBuffer);
			for (const rect of dirtyRects) {
				encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
				if (typeof encoder.drawIndirect === "function") {
					encoder.drawIndirect(batch.indirectBuffer, batch.indirectOffset);
				} else {
					encoder.draw(6, batch.instanceCount, 0, 0);
				}
			}
		}

		encoder.endRenderPass();
		this._evictParticleBindings(activeCacheKeys);
		return totalParticles;
	}

	private _resolveParticleDirtyRects(
		context: FrameContext,
		targets: WebGPUParticlePassTargets
	): Array<{ x: number; y: number; width: number; height: number }> {
		const targetView =
			targets.colorAttachments.find((attachment) => attachment.view)?.view ??
			targets.depth;
		const targetWidth = Math.max(
			1,
			Math.floor(
				typeof targetView?.width === "number" ?
					targetView.width
				:	context.attachments.width
			)
		);
		const targetHeight = Math.max(
			1,
			Math.floor(
				typeof targetView?.height === "number" ?
					targetView.height
				:	context.attachments.height
			)
		);
		const hasIncrementalRects =
			context.incremental?.enabled &&
			!context.incremental.forceFullFrame &&
			(context.incremental.dirtyRects?.length ?? 0) > 0;
		if (!hasIncrementalRects) {
			return [
				{
					x: 0,
					y: 0,
					width: targetWidth,
					height: targetHeight,
				},
			];
		}
		return context.incremental.dirtyRects
			.map((rect) => {
				const minX = Math.max(
					0,
					Math.floor(
						(rect.x * targetWidth) / Math.max(1, context.attachments.width)
					)
				);
				const minY = Math.max(
					0,
					Math.floor(
						(rect.y * targetHeight) / Math.max(1, context.attachments.height)
					)
				);
				const maxX = Math.min(
					targetWidth,
					Math.ceil(
						((rect.x + rect.width) * targetWidth) /
							Math.max(1, context.attachments.width)
					)
				);
				const maxY = Math.min(
					targetHeight,
					Math.ceil(
						((rect.y + rect.height) * targetHeight) /
							Math.max(1, context.attachments.height)
					)
				);
				return {
					x: minX,
					y: minY,
					width: maxX - minX,
					height: maxY - minY,
				};
			})
			.filter((rect) => rect.width > 0 && rect.height > 0);
	}

	private _evictParticleBindings(activeCacheKeys?: Set<string>): void {
		const staleFrameThreshold =
			this._frameId - WEBGPU_PARTICLE_BINDING_CACHE_MAX_AGE_FRAMES;
		for (const [cacheKey, entry] of this._particleBindingCache.entries()) {
			if (activeCacheKeys?.has(cacheKey)) {
				continue;
			}
			if (entry.lastUsedFrame > staleFrameThreshold) {
				continue;
			}
			this._destroyParticleBindingEntry(cacheKey, entry);
		}

		if (
			this._particleBindingCache.size <=
			WEBGPU_PARTICLE_BINDING_CACHE_MAX_ENTRIES
		) {
			return;
		}

		const evictionCandidates = Array.from(this._particleBindingCache.entries())
			.filter(([cacheKey]) => !activeCacheKeys?.has(cacheKey))
			.sort((left, right) => left[1].lastUsedFrame - right[1].lastUsedFrame);
		while (
			this._particleBindingCache.size >
				WEBGPU_PARTICLE_BINDING_CACHE_MAX_ENTRIES &&
			evictionCandidates.length > 0
		) {
			const [cacheKey, entry] = evictionCandidates.shift()!;
			this._destroyParticleBindingEntry(cacheKey, entry);
		}
	}

	private _clearParticleBindingCache(): void {
		for (const [cacheKey, entry] of this._particleBindingCache.entries()) {
			this._destroyParticleBindingEntry(cacheKey, entry);
		}
		this._particleBindingCache.clear();
	}

	private _destroyParticleBindingEntry(
		cacheKey: string,
		entry: WebGPUParticleBindingCacheEntry
	): void {
		this._destroyBindingGroup(entry.group);
		entry.uvTransformBuffer.destroy();
		this._particleBindingCache.delete(cacheKey);
	}

	private _destroyBindingGroup(group: IBindingGroup | null): void {
		const destroyFn = (group as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(group);
		}
	}

	private async _ensureParticleResources(
		mode: WebGPUSceneTargetMode,
		totalParticles: number,
		pipelineMode: "legacy" | "oit",
		sampleCountOverride?: number
	): Promise<void> {
		if (!this._particleShaderModule) {
			const shader = await ShaderSource.load("webgpu.particle.composite");
			this._particleShaderModule = await this._backend.createShaderModule({
				label: "WebGPUParticleShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "unknown",
				sourceKind: "particle",
			});
		}

		if (!this._particleQuadBuffer) {
			this._particleQuadBuffer = this._backend.createBuffer({
				size: WEBGPU_PARTICLE_QUAD_VERTICES.byteLength,
				usage: BufferUsage.Vertex | BufferUsage.CopyDst,
				label: "WebGPUParticleQuad",
			});
			this._backend.writeBuffer(
				this._particleQuadBuffer,
				WEBGPU_PARTICLE_QUAD_VERTICES
			);
		}

		this._ensureParticleInstanceBuffer(totalParticles);
		if (pipelineMode === "oit") {
			await this._ensureParticlePipeline(mode, "oit-alpha", sampleCountOverride);
		} else {
			await this._ensureParticlePipeline(mode, "alpha", sampleCountOverride);
			await this._ensureParticlePipeline(mode, "additive", sampleCountOverride);
		}
	}

	private _ensureParticleInstanceBuffer(totalParticles: number): void {
		if (totalParticles <= this._particleInstanceCapacity) return;

		const resolvedParticles =
			Number.isFinite(totalParticles) ?
				Math.max(1, Math.floor(totalParticles))
			:	1;
		const maxCapacity = Math.max(
			256,
			Math.floor(Number.MAX_SAFE_INTEGER / WEBGPU_PARTICLE_INSTANCE_STRIDE)
		);
		if (resolvedParticles > maxCapacity) {
			throw new Error(
				`Particle instance request ${resolvedParticles} exceeds max capacity ${maxCapacity}.`
			);
		}
		const exponent = Math.ceil(Math.log2(resolvedParticles));
		const pow2Capacity = Math.pow(2, exponent);
		const nextCapacity = Math.max(
			256,
			Math.min(maxCapacity, Number.isFinite(pow2Capacity) ? pow2Capacity : 256)
		);
		this._particleInstanceBuffer?.destroy();
		this._particleInstanceBuffer = this._backend.createBuffer({
			size: nextCapacity * WEBGPU_PARTICLE_INSTANCE_STRIDE,
			usage: BufferUsage.Vertex | BufferUsage.CopyDst,
			label: "WebGPUParticleInstances",
		});
		this._particleInstanceCapacity = nextCapacity;
	}

	private _ensureParticlePipeline(
		mode: WebGPUSceneTargetMode,
		pipelineType: "alpha" | "additive" | "oit-alpha",
		sampleCountOverride?: number
	): Promise<void> {
		const cache =
			pipelineType === "additive" ? this._particlePipelineAdditive
			: pipelineType === "oit-alpha" ? this._particlePipelineOITAlpha
			: this._particlePipelineAlpha;
		const sampleCount = this._resolveSampleCount(mode, sampleCountOverride);
		const cacheKey = this._createParticlePipelineCacheKey(mode, sampleCount);
		if (cache.has(cacheKey) || !this._particleShaderModule) return Promise.resolve();

		const blend =
			pipelineType === "additive" ?
				{
					color: {
						srcFactor: "src-alpha",
						dstFactor: "one",
						operation: "add",
					},
					alpha: {
						srcFactor: "one",
						dstFactor: "one",
						operation: "add",
					},
				}
			:	{
					color: {
						srcFactor: "src-alpha",
						dstFactor: "one-minus-src-alpha",
						operation: "add",
					},
					alpha: {
						srcFactor: "one",
						dstFactor: "one-minus-src-alpha",
						operation: "add",
					},
				};
		const colorFormat =
			mode === "mrt" || mode === "color" ?
				TextureFormat.RGBA16Float
			:	(this._backend.canvasFormat as any);
		const depthFormat =
			mode === "mrt" || mode === "color" ?
				TextureFormat.Depth32Float
			:	this._resolveSinglePassDepthFormat();
		const fragmentEntryPoint =
			pipelineType === "oit-alpha" ? "fsMainOIT" : "fsMain";
		const fragmentTargets =
			pipelineType === "oit-alpha" ?
				[
					{
						format: TextureFormat.RGBA16Float,
						blend: {
							color: {
								srcFactor: "one",
								dstFactor: "one",
								operation: "add",
							},
							alpha: {
								srcFactor: "one",
								dstFactor: "one",
								operation: "add",
							},
						},
					},
					{
						format: TextureFormat.R8Unorm,
						blend: {
							color: {
								srcFactor: "zero",
								dstFactor: "one-minus-src",
								operation: "add",
							},
							alpha: {
								srcFactor: "zero",
								dstFactor: "one-minus-src",
								operation: "add",
							},
						},
					},
				]
			:	[
					{
						format: colorFormat,
						blend,
					},
				];

		return this._backend.createPipeline({
			layout: this._layouts.particlePipelineLayout,
			label: `WebGPUParticlePipeline_${pipelineType}_${mode}`,
			vertex: {
				module: this._particleShaderModule,
				entryPoint: "vsMain",
				buffers: WEBGPU_PARTICLE_VERTEX_LAYOUTS,
			},
			fragment: {
				module: this._particleShaderModule,
				entryPoint: fragmentEntryPoint,
				targets: fragmentTargets,
			},
			primitive: {
				topology: "triangle-list" as any,
				cullMode: "none",
				frontFace: "ccw",
			},
			depthStencil: {
				format: depthFormat,
				depthWriteEnabled: false,
				depthCompare: "less",
			},
			sampleCount,
		} as any).then((pipeline) => {
			cache.set(cacheKey, pipeline);
		});
	}

	private _createParticlePipelineCacheKey(
		mode: WebGPUSceneTargetMode,
		sampleCount: number
	): string {
		return `${mode}|msaa:${sampleCount}`;
	}

	private _resolveSampleCount(
		mode: WebGPUSceneTargetMode,
		sampleCountOverride?: number
	): number {
		if (mode !== "mrt" && mode !== "color") {
			return 1;
		}
		if (Number.isFinite(sampleCountOverride)) {
			return Math.max(1, Math.floor(sampleCountOverride as number));
		}
		const getter = (this._backend as { getMSAASampleCount?: () => number })
			.getMSAASampleCount;
		if (typeof getter !== "function") {
			return 1;
		}
		const sampleCount = getter.call(this._backend);
		if (!Number.isFinite(sampleCount)) {
			return 1;
		}
		return Math.max(1, Math.floor(sampleCount));
	}

	private _resolveSinglePassDepthFormat(): TextureFormat {
		const backend = this._backend as {
			canvasDepthFormat?: TextureFormat;
		};
		return backend.canvasDepthFormat ?? TextureFormat.Depth24Plus;
	}

	private _createParticleUVTransformData(
		texture: ParticleRenderBatch["texture"]
	) {
		const repeatX = texture?.repeat.x ?? 1;
		const repeatY = texture?.repeat.y ?? 1;
		const offsetX = texture?.offset.x ?? 0;
		const offsetY = texture?.offset.y ?? 0;
		const rotation = texture?.rotation ?? 0;
		const cosRotation = Math.cos(rotation);
		const sinRotation = Math.sin(rotation);
		return new Float32Array([
			repeatX,
			repeatY,
			offsetX,
			offsetY,
			cosRotation,
			sinRotation,
			0,
			0,
		]);
	}

	private _resolveAnimationState(
		packet: DrawPacket,
		geometry: WebGPUGeometryHandle,
		frameResources: WebGPUPreparedFrameResources
	): WebGPUModelAnimationBindingState {
		const runtimeJoint =
			frameResources.jointMatrixMap?.get(packet.meshInstance.id) ?? null;
		let jointMatrices: Float32Array | null = null;
		if (runtimeJoint?.skeleton) {
			runtimeJoint.skeleton.updateJointMatrices(
				packet.meshInstance.worldMatrix
			);
			jointMatrices = runtimeJoint.skeleton.toFloat32Array(
				runtimeJoint.matrices
			);
		} else if (packet.meshInstance.skeleton) {
			packet.meshInstance.skeleton.updateJointMatrices(
				packet.meshInstance.worldMatrix
			);
			jointMatrices = packet.meshInstance.skeleton.toFloat32Array();
		}

		const runtimeMorph =
			frameResources.morphWeightMap?.get(packet.primitive.id) ?? null;
		let morphTargetCount = Math.max(0, runtimeMorph?.targetCount ?? 0);
		let sourceMorphWeights: Float32Array | null = runtimeMorph?.weights ?? null;
		if (!sourceMorphWeights || morphTargetCount <= 0) {
			const primitiveIndex = packet.mesh.primitives.indexOf(packet.primitive);
			const instanceWeights =
				primitiveIndex >= 0 ?
					packet.meshInstance.morphWeights[primitiveIndex]
				:	null;
			sourceMorphWeights = instanceWeights ?? null;
			morphTargetCount = sourceMorphWeights?.length ?? 0;
		}

		morphTargetCount = Math.min(
			Math.max(0, morphTargetCount),
			geometry.morphTargetCount
		);
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

function createParticleMeshWorldMatrix(
	position: { x: number; y: number; z: number },
	rotation: number,
	size: number
): Matrix4 {
	const scale = Math.max(0.001, size);
	return Matrix4.compose(
		position,
		Quaternion.fromEuler(0, 0, rotation),
		{ x: scale, y: scale, z: scale }
	);
}

function createParticleMeshInstance(
	batch: ParticleMeshRenderBatch,
	worldMatrix: Matrix4
): MeshInstance {
	return {
		id: `particleMeshInstance:${batch.systemId}:${batch.templateIndex}`,
		mesh: batch.mesh,
		skeleton: null,
		morphWeights: batch.mesh.defaultMorphWeights,
		renderLayers: 1,
		worldMatrix,
	} as MeshInstance;
}

function compareParticleMeshPackets(left: DrawPacket, right: DrawPacket): number {
	const leftTransparent =
		(left.passFlags & DRAW_PACKET_FLAG_TRANSPARENT) !== 0;
	const rightTransparent =
		(right.passFlags & DRAW_PACKET_FLAG_TRANSPARENT) !== 0;
	if (leftTransparent !== rightTransparent) {
		return leftTransparent ? 1 : -1;
	}
	if (leftTransparent && left.sortDepth !== right.sortDepth) {
		return right.sortDepth - left.sortDepth;
	}
	if (!leftTransparent) {
		const keyCompare = left.pipelineKey.localeCompare(right.pipelineKey);
		if (keyCompare !== 0) return keyCompare;
		if (left.material !== right.material) {
			return left.material.name.localeCompare(right.material.name);
		}
	}
	return left.id.localeCompare(right.id);
}
