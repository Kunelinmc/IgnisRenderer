import type { RendererBackendBridge } from "../IRenderBackend";
import { PARTICLE_TRANSIENT_BATCHES_KEY } from "../../pipeline/types";
import type {
	DrawPacket,
	FrameContext,
	ParticleRenderBatch,
	PreparedScene,
} from "../../pipeline/types";
import { AlphaMode } from "../../materials/Material";
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
	type IShaderModule,
} from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
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
import {
	WebGPUGeometryRegistry,
	type WebGPUGeometryHandle,
} from "./WebGPUGeometryRegistry";
import {
	WebGPUMaterialBindingCache,
	type WebGPUModelAnimationBindingState,
} from "./WebGPUMaterialBindingCache";
import { WebGPUPipelineLibrary } from "./WebGPUPipelineLibrary";
import type { WebGPUSceneTargetMode } from "./WebGPUPipelineLibrary";
import { WebGPUShadowAtlasAllocator } from "./WebGPUShadowAtlasAllocator";
import { WebGPUShadowPass } from "./WebGPUShadowPass";
import {
	resolveShadowCasterBounds,
	syncShadowMapRegistry,
	updateShadowMapMetadata,
} from "../../pipeline/ShadowMetadata";
import { isShadowCastingLight } from "../../lights";
import { WebGPUTextureRegistry } from "./WebGPUTextureRegistry";
import { getWebGPUParticleShader } from "../../shaders/webgpu/particleShader";
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

export interface WebGPUDrawResources {
	pipeline: any;
	frameBinding: any;
	modelBinding: any;
	vertexBuffer: any;
	indexBuffer: any;
	indexCount: number;
}

export interface WebGPUSkyboxDrawResources {
	pipeline: any;
	frameBinding: any;
}

export interface WebGPUParticlePassTargets {
	color: any;
	colorResolve?: any;
	depth: any;
}

export class WebGPURenderResources {
	private _renderer: RendererBackendBridge;
	private _backend: WebGPUBackend;
	private _layouts: ReturnType<typeof createWebGPUPipelineLayouts>;
	private _geometryRegistry: WebGPUGeometryRegistry;
	private _textureRegistry: WebGPUTextureRegistry;
	private _shadowAtlases: WebGPUShadowAtlasAllocator;
	private _pipelineLibrary: WebGPUPipelineLibrary;
	private _frameBindings: WebGPUFrameBindingCache;
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
	private _particlePipelineAdditive = new Map<
		WebGPUSceneTargetMode,
		IRenderPipeline
	>();
	private _particleBindingCache = new Map<
		string,
		{
			group: IBindingGroup;
			texture: any;
			sampler: any;
			uvTransformBuffer: IRenderBuffer;
		}
	>();

	constructor(renderer: RendererBackendBridge, backend: WebGPUBackend) {
		this._renderer = renderer;
		this._backend = backend;
		this._layouts = createWebGPUPipelineLayouts(backend.device);
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
		this._materialBindings = new WebGPUMaterialBindingCache(
			backend,
			this._layouts
		);
		this._shadowPass = new WebGPUShadowPass(
			backend,
			this._geometryRegistry,
			this._shadowAtlases
		);
	}

	public async init(): Promise<void> {
		await this._pipelineLibrary.init();
	}

	public renderShadows(context: FrameContext): void {
		this._shadowPass.render(context);
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
				(contextOrScene.transient.get(ANIMATION_WEBGPU_JOINT_MATRICES_KEY) as
					| JointMatrixMap
					| undefined) ?? null;
			this._morphWeightMap =
				(contextOrScene.transient.get(ANIMATION_WEBGPU_MORPH_WEIGHTS_KEY) as
					| MorphWeightMap
					| undefined) ?? null;
		} else {
			this._jointMatrixMap = null;
			this._morphWeightMap = null;
		}

		const { scene, features, shAmbientCoeffs, renderWidth, renderHeight } =
			this._resolveFrameInputs(contextOrScene, featuresArg);
		const featureState: WebGPUFeatureState = {
			enableLighting: features.enableLighting,
			enableGamma: features.enableGamma,
			enableSH: features.enableSH,
			enableShadows: features.enableShadows,
			enableReflection: features.enableReflection,
			enableSkybox: features.enableSkybox,
			enableSSAO: features.enableSSAO,
			enableTAA: features.enableTAA,
			enableSSR: features.enableSSR,
			enableVolumetric: features.enableVolumetric,
			taaOptions: features.taaOptions,
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
		if (features.enableShadows) {
			for (const light of shadowLights) {
				const shadowMap = scene.shadowMaps.get(light);
				if (shadowMap) {
					updateShadowMapMetadata(shadowMap, light, shadowCasterBounds);
				}
			}
		}

		this._lightingState = collectWebGPULighting(
			scene.lights,
			features.enableLighting,
			features.enableSH,
			features.enableShadows,
			scene.shadowMaps
		);
		for (const warning of this._lightingState.warnings) {
			this._renderer.warnOnce(warning.key, warning.message);
		}

		this._environmentState = collectWebGPUEnvironment(
			scene,
			featureState.enableSH,
			shAmbientCoeffs
		);
		for (const warning of this._environmentState.warnings) {
			this._renderer.warnOnce(warning.key, warning.message);
		}

		this._shadowAtlases.prepare(this._lightingState);
		this._frameBindings.prepare(
			scene,
			this._lightingState,
			this._environmentState,
			featureState,
			renderWidth,
			renderHeight
		);
		this._materialBindings.beginFrame();
	}

	public getFrameBinding(): IBindingGroup {
		return this._frameBindings.getSceneBinding();
	}

	public getLightingState(): WebGPULightingState | null {
		return this._lightingState;
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

	public async getDrawResources(
		packet: DrawPacket
	): Promise<WebGPUDrawResources[] | null> {
		if (packet.material.alphaMode === AlphaMode.Blend) {
			this._renderer.warnOnce(
				`webgpu-material-blend:${packet.material.type}:${packet.material.name}`,
				`WebGPU backend does not support alpha blend materials yet; skipping ${packet.material.name}`
			);
			return null;
		}

		const results: WebGPUDrawResources[] = [];
		const geometry = this._geometryRegistry.getGeometry(packet.primitive);
		const topology = geometry.topology;
		const frameBinding = this._frameBindings.getSceneBinding();
		const animationState = this._resolveAnimationState(packet, geometry);

		// ----- SOLID OBJECT -----
		const solidMaterialData = createWebGPUMaterialUniformData(
			packet.material,
			false
		);
		for (const warning of solidMaterialData.warnings) {
			this._renderer.warnOnce(warning.key, warning.message);
		}

		const solidPipeline = await this._pipelineLibrary.getPipeline(
			packet.material,
			this._sceneTargetMode,
			false,
			topology
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
				this._sceneTargetMode,
				true,
				topology
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
				vertexBuffer: geometry.vertexBuffer,
				indexBuffer: geometry.wireframeIndexBuffer,
				indexCount: geometry.wireframeIndexCount,
			});
		}

		return results;
	}

	public async getSkyboxResources(): Promise<WebGPUSkyboxDrawResources | null> {
		if (
			!this._featureState?.enableSkybox ||
			!this._environmentState?.skyboxTexture
		) {
			return null;
		}

		const pipeline = await this._pipelineLibrary.getSkyboxPipeline(
			this._sceneTargetMode
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
		mode: WebGPUSceneTargetMode
	): Promise<void> {
		const batches = context.transient.get(PARTICLE_TRANSIENT_BATCHES_KEY) as
			| ParticleRenderBatch[]
			| undefined;
		if (!batches || batches.length === 0) return;

		const drawBatches = batches.filter((batch) => batch.particles.length > 0);
		if (drawBatches.length === 0) return;

		const totalParticles = drawBatches.reduce(
			(sum, batch) => sum + batch.particles.length,
			0
		);
		if (totalParticles <= 0) return;

		await this._ensureParticleResources(mode, totalParticles);
		if (!this._particleInstanceBuffer || !this._particleQuadBuffer) return;

		const floatsPerInstance = WEBGPU_PARTICLE_INSTANCE_FLOATS;
		const instanceData = new Float32Array(totalParticles * floatsPerInstance);
		const drawRanges: Array<{
			batch: ParticleRenderBatch;
			firstInstance: number;
			instanceCount: number;
		}> = [];

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
		const additivePipeline = this._particlePipelineAdditive.get(mode);
		if (!alphaPipeline || !additivePipeline) return;

		encoder.beginRenderPass({
			label: mode === "mrt" ? "WebGPUParticlesMRT" : "WebGPUParticlesSingle",
			colorAttachments: [
				{
					view: targets.color,
					resolveTarget: targets.colorResolve,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: "load",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: targets.depth,
				depthLoadOp: "load",
				depthStoreOp: "store",
			},
		});
		encoder.setBindingGroup(0, frameBinding);
		encoder.setVertexBuffer(0, this._particleQuadBuffer);
		encoder.setVertexBuffer(1, this._particleInstanceBuffer);

		for (const range of drawRanges) {
			const texture = this._textureRegistry.getTextureForSlot(
				range.batch.texture,
				0
			);
			const sampler = this._textureRegistry.getSamplerForTexture(
				range.batch.texture
			);
			const cacheKey = `particle_${range.batch.systemId}`;
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
			} else {
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
				});
			}
			const uvTransformData = this._createParticleUVTransformData(
				range.batch.texture
			);
			this._backend.writeBuffer(uvTransformBuffer, uvTransformData);
			const pipeline =
				range.batch.blendMode === ParticleBlendMode.Additive ?
					additivePipeline
				:	alphaPipeline;
			encoder.setPipeline(pipeline);
			encoder.setBindingGroup(1, particleBinding);
			encoder.draw(6, range.instanceCount, 0, range.firstInstance);
		}

		encoder.endRenderPass();
	}

	private async _ensureParticleResources(
		mode: WebGPUSceneTargetMode,
		totalParticles: number
	): Promise<void> {
		if (!this._particleShaderModule) {
			const shaderCode = await getWebGPUParticleShader();
			this._particleShaderModule = await this._backend.createShaderModule({
				label: "WebGPUParticleShader",
				code: shaderCode,
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
		this._ensureParticlePipeline(mode, ParticleBlendMode.Alpha);
		this._ensureParticlePipeline(mode, ParticleBlendMode.Additive);
	}

	private _ensureParticleInstanceBuffer(totalParticles: number): void {
		if (totalParticles <= this._particleInstanceCapacity) return;

		const nextCapacity = Math.max(
			256,
			1 << Math.ceil(Math.log2(Math.max(1, totalParticles)))
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
		blendMode: ParticleBlendMode
	): void {
		const cache =
			blendMode === ParticleBlendMode.Additive ?
				this._particlePipelineAdditive
			:	this._particlePipelineAlpha;
		if (cache.has(mode) || !this._particleShaderModule) return;

		const blend =
			blendMode === ParticleBlendMode.Additive ?
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
			mode === "mrt" ? TextureFormat.Depth32Float : TextureFormat.Depth24Plus;
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

		const pipeline = this._backend.createPipeline({
			layout: this._layouts.particlePipelineLayout,
			label: `WebGPUParticlePipeline_${blendMode}_${mode}`,
			vertex: {
				module: this._particleShaderModule,
				entryPoint: "vsMain",
				buffers: WEBGPU_PARTICLE_VERTEX_LAYOUTS,
			},
			fragment: {
				module: this._particleShaderModule,
				entryPoint: "fsMain",
				targets: [
					{
						format: colorFormat,
						blend,
					},
				],
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
