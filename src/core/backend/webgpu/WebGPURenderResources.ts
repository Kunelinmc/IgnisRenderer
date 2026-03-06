import type { Renderer } from "../../Renderer";
import { PARTICLE_TRANSIENT_BATCHES_KEY } from "../../pipeline/types";
import type {
	DrawPacket,
	FrameContext,
	ParticleRenderBatch,
	PreparedScene,
} from "../../pipeline/types";
import { AlphaMode } from "../../../materials/Material";
import { ParticleBlendMode } from "../../../particles";
import type { ResolvedFeatureState } from "../../pipeline/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import {
	BufferUsage,
	TextureFormat,
	type IRenderBuffer,
	type IRenderPipeline,
	type IShaderModule,
} from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
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
import { WebGPUGeometryRegistry } from "./WebGPUGeometryRegistry";
import { WebGPUMaterialBindingCache } from "./WebGPUMaterialBindingCache";
import { WebGPUPipelineLibrary } from "./WebGPUPipelineLibrary";
import type { WebGPUSceneTargetMode } from "./WebGPUPipelineLibrary";
import { WebGPUShadowAtlasAllocator } from "./WebGPUShadowAtlasAllocator";
import { WebGPUShadowPass } from "./WebGPUShadowPass";
import { WebGPUTextureRegistry } from "./WebGPUTextureRegistry";
import { getWebGPUParticleShader } from "../../../shaders/webgpu/particleShader";

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
	depth: any;
}

export class WebGPURenderResources {
	private _renderer: Renderer;
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

	constructor(renderer: Renderer, backend: WebGPUBackend) {
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
	): Promise<WebGPUDrawResources | null> {
		if (packet.material.alphaMode === AlphaMode.Blend) {
			this._renderer.warnOnce(
				`webgpu-material-blend:${packet.material.type}:${packet.material.name}`,
				`WebGPU backend does not support alpha blend materials yet; skipping ${packet.material.name}`
			);
			return null;
		}

		const materialData = createWebGPUMaterialUniformData(packet.material);
		for (const warning of materialData.warnings) {
			this._renderer.warnOnce(warning.key, warning.message);
		}

		const geometry = this._geometryRegistry.getGeometry(packet.primitive);
		const pipeline = await this._pipelineLibrary.getPipeline(
			packet.material,
			this._sceneTargetMode
		);
		const textures = materialData.textureSlots.map((slot, index) =>
			this._textureRegistry.getTextureForSlot(slot.map, index)
		);
		const samplers = materialData.textureSlots.map((slot) =>
			this._textureRegistry.getSamplerForTexture(slot.map)
		);
		const frameBinding = this._frameBindings.getSceneBinding();
		const modelBinding = this._materialBindings.getBinding(
			packet,
			pipeline,
			materialData,
			textures,
			samplers
		);

		return {
			pipeline,
			frameBinding,
			modelBinding,
			vertexBuffer: geometry.vertexBuffer,
			indexBuffer: geometry.indexBuffer,
			indexCount: geometry.indexCount,
		};
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

		const floatsPerInstance = 16;
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
				instanceData[offset + 7] = Math.max(0, Math.min(1, particle.color.a));

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
			const particleBinding = this._backend.createBindingGroup({
				layout: this._layouts.particleBindGroupLayout,
				entries: [
					{ binding: 0, resource: texture },
					{ binding: 1, resource: sampler },
				],
				label: `ParticleBinding_${range.batch.systemId}`,
			});
			const pipeline =
				range.batch.blendMode === ParticleBlendMode.Additive
					? additivePipeline
					: alphaPipeline;
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
			const quadVertices = new Float32Array([
				-0.5, -0.5, 0, 1, 0.5, -0.5, 1, 1, 0.5, 0.5, 1, 0, -0.5, -0.5, 0, 1,
				0.5, 0.5, 1, 0, -0.5, 0.5, 0, 0,
			]);
			this._particleQuadBuffer = this._backend.createBuffer({
				size: quadVertices.byteLength,
				usage: BufferUsage.Vertex | BufferUsage.CopyDst,
				label: "WebGPUParticleQuad",
			});
			this._backend.writeBuffer(this._particleQuadBuffer, quadVertices);
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
			size: nextCapacity * 16 * 4,
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
			blendMode === ParticleBlendMode.Additive
				? this._particlePipelineAdditive
				: this._particlePipelineAlpha;
		if (cache.has(mode) || !this._particleShaderModule) return;

		const blend =
			blendMode === ParticleBlendMode.Additive
				? {
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
				: {
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
			mode === "mrt"
				? TextureFormat.RGBA16Float
				: (this._backend.canvasFormat as any);
		const depthFormat =
			mode === "mrt" ? TextureFormat.Depth32Float : TextureFormat.Depth24Plus;

		const pipeline = this._backend.createPipeline({
			layout: this._layouts.particlePipelineLayout,
			label: `WebGPUParticlePipeline_${blendMode}_${mode}`,
			vertex: {
				module: this._particleShaderModule,
				entryPoint: "vsMain",
				buffers: [
					{
						arrayStride: 16,
						stepMode: "vertex",
						attributes: [
							{ shaderLocation: 0, offset: 0, format: "float32x2" },
							{ shaderLocation: 1, offset: 8, format: "float32x2" },
						],
					},
					{
						arrayStride: 64,
						stepMode: "instance",
						attributes: [
							{ shaderLocation: 2, offset: 0, format: "float32x4" },
							{ shaderLocation: 3, offset: 16, format: "float32x4" },
							{ shaderLocation: 4, offset: 32, format: "float32x4" },
							{ shaderLocation: 5, offset: 48, format: "float32" },
							{ shaderLocation: 6, offset: 52, format: "float32" },
						],
					},
				],
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
		} as any);
		cache.set(mode, pipeline);
	}
}
