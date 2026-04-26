import type { RendererBackendBridge } from "../IRenderBackend";
import type { Texture } from "../../core/Texture";
import { PARTICLE_TRANSIENT_BATCHES_KEY } from "../../pipeline/types";
import type {
	DrawPacket,
	FrameContext,
	ParticleRenderBatch,
	PreparedScene,
} from "../../pipeline/types";
import { ParticleBlendMode } from "../../particles";
import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../../core/types";
import type { ResolvedFeatureState } from "../../pipeline/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import {
	BufferUsage,
	TextureFormat,
	type IBindingGroup,
	type IRenderBuffer,
	type IRenderPipeline,
	type IRenderTexture,
	type IShaderModule,
} from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
import { resolveWebGPUComputeFacade } from "./ComputeFacade";
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
import { WebGPUFrameBindingCache } from "./WebGPUFrameBindingCache";
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
	WebGPUSceneTargetMode,
	WebGPUTransparentPipelineMode,
} from "./WebGPUPipelineLibrary";
import { WebGPUShadowAtlasAllocator } from "./WebGPUShadowAtlasAllocator";
import { WebGPUShadowPass } from "./WebGPUShadowPass";
import {
	resolveShadowCasterBounds,
	syncShadowMapRegistry,
	updateShadowMapMetadata,
} from "../../pipeline/ShadowMetadata";
import {
	selectCSMDirectionalLights,
	type ShadowBackendCapabilities,
} from "../../pipeline/ShadowStrategyRegistry";
import { isShadowCastingLight, type ShadowCastingLight } from "../../lights";
import type { ShadowRenderSet } from "../../lights/ShadowMapping";
import { WebGPUTextureRegistry } from "./WebGPUTextureRegistry";
import { getWebGPUParticleShaderComposite } from "../../shaders/webgpu/particleShader";
import { clamp } from "../../maths/Common";
import {
	WEBGPU_PARTICLE_BINDING_SAMPLER,
	WEBGPU_PARTICLE_BINDING_TEXTURE,
	WEBGPU_PARTICLE_BINDING_UV_TRANSFORM,
	WEBGPU_PARTICLE_INSTANCE_FLOATS,
	WEBGPU_PARTICLE_INSTANCE_STRIDE,
	WEBGPU_PARTICLE_QUAD_VERTICES,
	WEBGPU_PARTICLE_UV_UNIFORM_SIZE,
	WEBGPU_PARTICLE_VERTEX_LAYOUTS,
} from "./particleLayout";
import type {
	WarmupPhaseCounters,
	WarmupPlan,
} from "../../pipeline/WarmupPlanner";
import { toShaderCompileError } from "../../pipeline/WarmupPlanner";
import type { ShaderCompileError } from "../../shaders/runtime";
import { Logger } from "../../foundation/Logger";

const WEBGPU_SHADOW_CAPABILITIES: ShadowBackendCapabilities = {
	backendKey: "webgpu",
	supportsSingleMap: true,
	supportsDirectionalCSM: true,
	maxCsmDirectionalLights: 1,
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

export interface WebGPUSkyboxDrawResources {
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

interface WebGPUDrawResourceOptions {
	transparentPipelineMode?: WebGPUTransparentPipelineMode;
	sceneTargetMode?: WebGPUSceneTargetMode;
}

interface WebGPUParticleRenderOptions {
	includeBlendModes?: readonly ParticleBlendMode[];
	pipelineMode?: "legacy" | "oit";
}

export class WebGPURenderResources {
	private _backend: WebGPUBackend;
	private _layouts: ReturnType<typeof createWebGPUPipelineLayouts>;
	private _geometryRegistry: WebGPUGeometryRegistry;
	private _textureRegistry: WebGPUTextureRegistry;
	private _shadowAtlases: WebGPUShadowAtlasAllocator;
	private _pipelineLibrary: WebGPUPipelineLibrary;
	private _frameBindings: WebGPUFrameBindingCache;
	private _clusteredLighting: WebGPUClusteredLightingRuntime;
	private _materialBindings: WebGPUMaterialBindingCache;
	private _shadowPass: WebGPUShadowPass;
	private _lightingState: WebGPULightingState | null = null;
	private _featureState: WebGPUFeatureState | null = null;
	private _environmentState: WebGPUEnvironmentState | null = null;
	private _jointMatrixMap: JointMatrixMap | null = null;
	private _morphWeightMap: MorphWeightMap | null = null;
	private _sceneTargetMode: WebGPUSceneTargetMode = "mrt";
	private _particleShaderModule: IShaderModule | null = null;
	private _particleQuadBuffer: IRenderBuffer | null = null;
	private _particleInstanceBuffer: IRenderBuffer | null = null;
	private _particleInstanceCapacity = 0;
	private _particlePipelineAlpha = new Map<
		WebGPUSceneTargetMode,
		IRenderPipeline
	>();
	private _particlePipelineOITAlpha = new Map<
		WebGPUSceneTargetMode,
		IRenderPipeline
	>();
	private _particlePipelineAdditive = new Map<
		WebGPUSceneTargetMode,
		IRenderPipeline
	>();
	private _particleBindingCache = new Map<
		string,
		WebGPUParticleBindingCacheEntry
	>();
	private _frameId = 0;
	private _destroyed = false;
	private _disposeShaderRuntimeListener: (() => void) | null = null;

	constructor(backend: WebGPUBackend);
	constructor(_renderer: RendererBackendBridge, backend: WebGPUBackend);
	constructor(
		backendOrRenderer: WebGPUBackend | RendererBackendBridge,
		backendMaybe?: WebGPUBackend
	) {
		const backend =
			backendMaybe ??
			(backendOrRenderer as WebGPUBackend);
		this._backend = backend;
		const computeFacade = resolveWebGPUComputeFacade(backend);
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
		this._pipelineLibrary = new WebGPUPipelineLibrary(backend, this._layouts);
		this._frameBindings = new WebGPUFrameBindingCache(
			backend,
			this._layouts,
			this._textureRegistry,
			this._shadowAtlases
		);
		this._clusteredLighting = new WebGPUClusteredLightingRuntime(
			computeFacade,
			this._layouts.clusteredSceneBindGroupLayout,
			this._layouts.sceneFrameBindGroupLayout,
			(key, message) =>
				Logger.warn(`[${key}] ${message}`, {
					scope: "WebGPUClusteredLightingRuntime",
				})
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
		const shaderRuntime = (
			this._backend as unknown as {
				shaderRuntime?: {
					onDidChange?: (listener: () => void) => () => void;
				};
			}
		).shaderRuntime;
		if (shaderRuntime && typeof shaderRuntime.onDidChange === "function") {
			this._disposeShaderRuntimeListener = shaderRuntime.onDidChange(() =>
				this.onShaderRuntimeChanged()
			);
		}
	}

	public async init(): Promise<void> {
		await this._pipelineLibrary.init();
	}

	public async warmup(
		context: FrameContext,
		plan: WarmupPlan
	): Promise<WarmupPhaseCounters> {
		let total = 0;
		let compiled = 0;
		let skipped = 0;
		let failed = 0;
		const errors: ShaderCompileError[] = [];

		try {
			this.setSceneTargetMode(plan.sceneTargetMode);
			this.prepareFrame(context);
		} catch (error) {
			failed++;
			errors.push(toShaderCompileError(error, "webgpu", "WebGPUPrepareFrame"));
		}

		if (plan.enableSkybox) {
			total++;
			try {
				await this.getSkyboxResources();
				compiled++;
			} catch (error) {
				failed++;
				errors.push(
					toShaderCompileError(error, "webgpu", "WebGPUSkyboxWarmup")
				);
			}
		}

		const drawPackets = [
			...context.scene.opaquePackets,
			...context.scene.transparentPackets,
		];
		for (const packet of drawPackets) {
			total++;
			try {
				const resources = await this.getDrawResources(packet);
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
		}

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
		await this._shadowPass.render(context, encoder);
	}

	public setSceneTargetMode(mode: WebGPUSceneTargetMode): void {
		this._sceneTargetMode = mode;
	}

	public prepareFrame(context: FrameContext): void;
	public prepareFrame(
		scene: PreparedScene,
		features: ResolvedFeatureState
	): void;
	public prepareFrame(
		contextOrScene: FrameContext | PreparedScene,
		featuresArg?: ResolvedFeatureState
	): void {
		if (this._isFrameContext(contextOrScene)) {
			this._jointMatrixMap =
				contextOrScene.transient.get(ANIMATION_WEBGPU_JOINT_MATRICES_KEY) ??
				null;
			this._morphWeightMap =
				contextOrScene.transient.get(ANIMATION_WEBGPU_MORPH_WEIGHTS_KEY) ??
				null;
		} else {
			this._jointMatrixMap = null;
			this._morphWeightMap = null;
		}

		const { scene, features, shAmbientCoeffs, renderWidth, renderHeight } =
			this._resolveFrameInputs(contextOrScene, featuresArg);
		this._frameId++;
		const featureState: WebGPUFeatureState = {
			enableLighting: features.enableLighting,
			enableGamma: features.enableGamma,
			enableSH: features.enableSH,
			enableShadows: features.enableShadows,
			enableReflection: features.enableReflection,
			enableSkybox: features.enableSkybox,
			enableOIT: features.enableOIT,
			enableSSAO: features.enableSSAO,
			enableSSGI: features.enableSSGI,
			enableTAA: features.enableTAA,
			enableSSR: features.enableSSR,
			enableVolumetric: features.enableVolumetric,
			enableFog: features.enableFog,
			enableBloom: features.enableBloom,
			enableClusteredLighting: features.enableClusteredLighting,
			taaOptions: features.taaOptions,
			fogOptions: features.fogOptions,
			bloomOptions: features.bloomOptions,
			clusteredLightingOptions: features.clusteredLightingOptions,
			warnings: [],
		};
		this._featureState = featureState;

		const shadowLights = scene.lights.filter(isShadowCastingLight);
		syncShadowMapRegistry(scene.shadowMaps, shadowLights);
		const shadowCasterBounds = resolveShadowCasterBounds(
			scene.shadowCasterPackets,
			scene.sceneBounds,
			scene.camera
		);
		const selectedCSMLights = selectCSMDirectionalLights(
			shadowLights,
			WEBGPU_SHADOW_CAPABILITIES.maxCsmDirectionalLights
		);
		if (features.enableShadows) {
			for (const light of shadowLights) {
				const shadowRenderSet = scene.shadowMaps.get(light);
				if (shadowRenderSet) {
					updateShadowMapMetadata(
						shadowRenderSet,
						light,
						shadowCasterBounds,
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

		this._lightingState = collectWebGPULighting(
			scene.lights,
			features.enableLighting,
			features.enableSH,
			features.enableShadows,
			scene.shadowMaps,
			featureState.enableClusteredLighting
		);
		for (const warning of this._lightingState.warnings) {
			Logger.warn(`[${warning.key}] ${warning.message}`, {
				scope: "WebGPURenderResources",
			});
		}

		this._environmentState = collectWebGPUEnvironment(
			scene,
			featureState.enableSH,
			shAmbientCoeffs
		);
		for (const warning of this._environmentState.warnings) {
			Logger.warn(`[${warning.key}] ${warning.message}`, {
				scope: "WebGPURenderResources",
			});
		}

		this._shadowAtlases.prepare(
			this._lightingState,
			this._resolveShadowAtlasTileSize(scene.shadowMaps, features.enableShadows)
		);
		this._frameBindings.prepare(
			scene,
			this._lightingState,
			this._environmentState,
			featureState,
			renderWidth,
			renderHeight,
			this._sceneTargetMode
		);
		this._clusteredLighting.prepareFrame(
			scene,
			featureState,
			this._lightingState,
			renderWidth,
			renderHeight
		);
		this._materialBindings.beginFrame();
		this._evictParticleBindings();
	}

	public getFrameBinding(): IBindingGroup {
		return this._frameBindings.getSceneBinding();
	}

	public getClusteredSceneBinding(): IBindingGroup {
		return this._clusteredLighting.getSceneBinding();
	}

	public async buildClusteredLighting(encoder: ICommandEncoder): Promise<void> {
		await this._clusteredLighting.build(
			encoder,
			this._frameBindings.getSceneBinding()
		);
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
		this._clusteredLighting.onShaderRuntimeChanged();
		this._shadowPass.onShaderRuntimeChanged();
	}

	public destroy(): void {
		if (this._destroyed) {
			return;
		}
		this._destroyed = true;
		this._disposeShaderRuntimeListener?.();
		this._disposeShaderRuntimeListener = null;
		this._clearParticleBindingCache();
		this._particleShaderModule = null;
		this._particlePipelineAlpha.clear();
		this._particlePipelineOITAlpha.clear();
		this._particlePipelineAdditive.clear();
		this._particleQuadBuffer?.destroy();
		this._particleQuadBuffer = null;
		this._particleInstanceBuffer?.destroy();
		this._particleInstanceBuffer = null;
		this._particleInstanceCapacity = 0;
		this._frameBindings.destroy();
		this._clusteredLighting.destroy();
		this._materialBindings.destroy();
		this._shadowPass.destroy();
		this._pipelineLibrary.destroy();
		this._shadowAtlases.destroy();
		this._textureRegistry.destroy();
		this._geometryRegistry.destroy();
		this._lightingState = null;
		this._featureState = null;
		this._environmentState = null;
		this._jointMatrixMap = null;
		this._morphWeightMap = null;
	}

	public getLightingState(): WebGPULightingState | null {
		return this._lightingState;
	}

	public getTextureForSlot(
		texture: Texture | null,
		slotIndex: number
	): IRenderTexture {
		return this._textureRegistry.getTextureForSlot(texture, slotIndex);
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

	private _resolveFrameInputs(
		contextOrScene: FrameContext | PreparedScene,
		featuresArg?: ResolvedFeatureState
	): {
		scene: PreparedScene;
		features: ResolvedFeatureState;
		shAmbientCoeffs: FrameContext["shAmbientCoeffs"] | null;
		renderWidth: number;
		renderHeight: number;
	} {
		if (this._isFrameContext(contextOrScene)) {
			return {
				scene: contextOrScene.scene,
				features: contextOrScene.features,
				shAmbientCoeffs: contextOrScene.shAmbientCoeffs,
				renderWidth: Math.max(1, contextOrScene.attachments.width || 1),
				renderHeight: Math.max(1, contextOrScene.attachments.height || 1),
			};
		}

		if (!featuresArg) {
			throw new Error(
				"WebGPURenderResources.prepareFrame() requires a resolved feature state."
			);
		}

		return {
			scene: contextOrScene,
			features: featuresArg,
			shAmbientCoeffs: null,
			renderWidth: 1,
			renderHeight: 1,
		};
	}

	private _isFrameContext(
		value: FrameContext | PreparedScene
	): value is FrameContext {
		return (
			"scene" in value &&
			"features" in value &&
			"attachments" in value &&
			"transient" in value
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
		options: WebGPUDrawResourceOptions = {}
	): Promise<WebGPUDrawResources[] | null> {
		const transparentPipelineMode =
			options.transparentPipelineMode ?? "default";
		const sceneTargetMode = options.sceneTargetMode ?? this._sceneTargetMode;
		const results: WebGPUDrawResources[] = [];
		const geometry = this._geometryRegistry.getGeometry(packet.primitive);
		const topology = geometry.topology;
		const frameBinding = this._frameBindings.getSceneBinding();
		const clusteredBinding = this._clusteredLighting.getSceneBinding();
		const animationState = this._resolveAnimationState(packet, geometry);

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

		const solidPipeline = await this._pipelineLibrary.getPipeline(
			packet.material,
			sceneTargetMode,
			false,
			topology,
			transparentPipelineMode
		);
		const solidTextures = solidMaterialData.textureSlots.map((slot, index) =>
			this._textureRegistry.getTextureForSlot(slot.map, index)
		);
		const solidSamplers = solidMaterialData.textureSlots.map((slot) =>
			this._textureRegistry.getSamplerForTexture(slot.map)
		);
		const solidModelBinding = this._materialBindings.getBinding(
			packet,
			solidPipeline,
			solidMaterialData,
			solidTextures,
			solidSamplers,
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
				transparentPipelineMode
			);
			const wireTextures = wireMaterialData.textureSlots.map((slot, index) =>
				this._textureRegistry.getTextureForSlot(slot.map, index)
			);
			const wireSamplers = wireMaterialData.textureSlots.map((slot) =>
				this._textureRegistry.getSamplerForTexture(slot.map)
			);
			const wireModelBinding = this._materialBindings.getBinding(
				packet,
				wirePipeline,
				wireMaterialData,
				wireTextures,
				wireSamplers,
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

	public async getSkyboxResources(
		sceneTargetMode: WebGPUSceneTargetMode = this._sceneTargetMode
	): Promise<WebGPUSkyboxDrawResources | null> {
		if (
			!this._featureState?.enableSkybox ||
			!this._environmentState?.skyboxTexture
		) {
			return null;
		}

		const pipeline = await this._pipelineLibrary.getSkyboxPipeline(
			sceneTargetMode
		);
		const frameBinding = this._frameBindings.getSkyboxBinding();

		return {
			pipeline,
			frameBinding,
		};
	}

	public async renderParticles(
		encoder: ICommandEncoder,
		context: FrameContext,
		targets: WebGPUParticlePassTargets,
		mode: WebGPUSceneTargetMode,
		options: WebGPUParticleRenderOptions = {}
	): Promise<number> {
		const includeBlendModes = options.includeBlendModes ?? null;
		const pipelineMode = options.pipelineMode ?? "legacy";
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

		await this._ensureParticleResources(mode, totalParticles, pipelineMode);
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
		const frameBinding = this._frameBindings.getSceneBinding();
		const alphaPipeline = this._particlePipelineAlpha.get(mode);
		const oitAlphaPipeline = this._particlePipelineOITAlpha.get(mode);
		const additivePipeline = this._particlePipelineAdditive.get(mode);
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
		const dirtyRects =
			context.incremental?.enabled &&
			!context.incremental.forceFullFrame &&
			(context.incremental.dirtyRects?.length ?? 0) > 0 ?
				context.incremental.dirtyRects
			:	[{
					x: 0,
					y: 0,
					width: Math.max(1, context.attachments.width),
					height: Math.max(1, context.attachments.height),
				}];

		for (const range of drawRanges) {
			const texture = this._textureRegistry.getTextureForSlot(
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
		pipelineMode: "legacy" | "oit"
	): Promise<void> {
		if (!this._particleShaderModule) {
			const shader = await getWebGPUParticleShaderComposite();
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
			this._ensureParticlePipeline(mode, "oit-alpha");
		} else {
			this._ensureParticlePipeline(mode, "alpha");
			this._ensureParticlePipeline(mode, "additive");
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
		pipelineType: "alpha" | "additive" | "oit-alpha"
	): void {
		const cache =
			pipelineType === "additive" ? this._particlePipelineAdditive
			: pipelineType === "oit-alpha" ? this._particlePipelineOITAlpha
			: this._particlePipelineAlpha;
		if (cache.has(mode) || !this._particleShaderModule) return;

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
			mode === "mrt" ?
				TextureFormat.RGBA16Float
			:	(this._backend.canvasFormat as any);
		const depthFormat =
			mode === "mrt" ?
				TextureFormat.Depth32Float
			:	this._resolveSinglePassDepthFormat();
		let sampleCount = 1;
		if (mode === "mrt") {
			const getter = (this._backend as { getMSAASampleCount?: () => number })
				.getMSAASampleCount;
			if (typeof getter === "function") {
				const resolved = getter.call(this._backend);
				if (Number.isFinite(resolved)) {
					sampleCount = Math.max(1, Math.floor(resolved));
				}
			}
		}

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

		const pipeline = this._backend.createPipeline({
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
		} as any);
		cache.set(mode, pipeline);
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
		geometry: WebGPUGeometryHandle
	): WebGPUModelAnimationBindingState {
		const runtimeJoint =
			this._jointMatrixMap?.get(packet.meshInstance.id) ?? null;
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

		const runtimeMorph = this._morphWeightMap?.get(packet.primitive.id) ?? null;
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
