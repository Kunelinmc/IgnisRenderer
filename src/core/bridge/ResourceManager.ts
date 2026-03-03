import { Matrix4 } from "../../maths/Matrix4";
import { Material } from "../../materials/Material";
import { getModelMatrix } from "../modelMatrix";
import type { IModel } from "../types";
import type { Renderer } from "../Renderer";
import type { Texture } from "../Texture";
import { IDevice } from "../ral/IDevice";
import {
	AddressMode,
	BufferUsage,
	FilterMode,
	type BindingEntry,
	type IBindingGroup,
	type IRenderBuffer,
	type IRenderPipeline,
	type IRenderTexture,
	type ISampler,
	TextureFormat,
	TextureUsage,
} from "../ral/types";
import { WEBGPU_SCENE_SHADER } from "../../shaders/webgpu/sceneShader";

import {
	alignTo,
	collectWebGPULighting,
	createTextureUploadData,
	createWebGPUMaterialUniformData,
	packFrameUniformData,
	packModelUniformData,
	resolveWebGPUFeatureState,
	WEBGPU_FRAME_UNIFORM_FLOATS,
	WEBGPU_MODEL_UNIFORM_FLOATS,
	WEBGPU_TEXTURE_SLOT,
	type WebGPUFeatureState,
	type WebGPULightingState,
	type WebGPUMaterialUniformData,
} from "./webgpuUtils";
import type { ShadowMap } from "../../utils/ShadowMapping";

interface GeometryCacheEntry {
	vertexBuffer: IRenderBuffer;
	indexBuffer: IRenderBuffer;
	indexCount: number;
}

interface WebGPUPipelineCacheEntry {
	key: string;
	pipeline: IRenderPipeline;
}

interface WebGPUModelBindingCacheEntry {
	uniformBuffer: IRenderBuffer;
	bindingGroup: IBindingGroup | null;
	pipeline: IRenderPipeline | null;
	textures: IRenderTexture[];
	samplers: ISampler[];
	material: Material | null;
}

export interface WebGPUDrawResources {
	pipeline: IRenderPipeline;
	frameBinding: IBindingGroup;
	modelBinding: IBindingGroup;
	vertexBuffer: IRenderBuffer;
	indexBuffer: IRenderBuffer;
	indexCount: number;
}

/**
 * ResourceManager: The bridge between scene data and the RAL.
 */
export class ResourceManager {
	private _device: IDevice;
	private _defaultMaterial: Material = new Material();

	private _geometryCache = new WeakMap<IModel, GeometryCacheEntry>();
	private _softwarePipelineCache = new WeakMap<Material, IRenderPipeline>();
	private _webgpuMaterialPipelineCache = new WeakMap<
		Material,
		WebGPUPipelineCacheEntry
	>();
	private _webgpuPipelineCache = new Map<string, IRenderPipeline>();
	private _frameBindingCache = new Map<IRenderPipeline, IBindingGroup>();
	private _modelBindingCache = new WeakMap<
		IModel,
		WebGPUModelBindingCacheEntry
	>();
	private _textureCache = new WeakMap<Texture, IRenderTexture>();
	private _samplerCache = new WeakMap<Texture, ISampler>();

	private _frameUniformBuffer: IRenderBuffer | null = null;
	private _sceneShaderModule: any = null;
	private _whiteTexture: IRenderTexture | null = null;
	private _neutralNormalTexture: IRenderTexture | null = null;
	private _whiteSampler: ISampler | null = null;
	private _directionalShadowAtlas: {
		tileSize: number;
		texture: IRenderTexture;
	} | null = null;
	private _spotShadowAtlas: {
		tileSize: number;
		texture: IRenderTexture;
	} | null = null;

	constructor(device: IDevice) {
		this._device = device;
	}

	public async init(): Promise<void> {
		if (this._device.type === "webgpu") {
			await this._getSceneShaderModule();
		}
	}

	public getGeometry(model: IModel): GeometryCacheEntry {
		let cached = this._geometryCache.get(model);
		if (!cached) {
			cached = this._uploadModel(model);
			this._geometryCache.set(model, cached);
		}
		return cached;
	}

	public async getPipeline(material: Material): Promise<IRenderPipeline> {
		if (this._device.type === "webgpu") {
			return await this._getWebGPUPipeline(
				material,
				createWebGPUMaterialUniformData(material)
			);
		}

		let cached = this._softwarePipelineCache.get(material);
		if (!cached) {
			cached = await this._createSoftwarePipeline(material);
			this._softwarePipelineCache.set(material, cached);
		}
		return cached;
	}

	public getMaterial(model: IModel): Material {
		return model.faces[0]?.material ?? this._defaultMaterial;
	}

	public getSoftwareBinding(model: IModel): IBindingGroup | null {
		if (this._device.type !== "software") {
			return null;
		}

		const modelMatrix = getModelMatrix(model);
		return {
			label: `SoftwareBinding_${model.id}`,
			_material: this.getMaterial(model),
			_modelMatrix: modelMatrix,
			_normalMatrix: Matrix4.normalMatrix(modelMatrix),
		} as any;
	}

	public prepareWebGPUFrame(
		renderer: Renderer,
		featureState: WebGPUFeatureState
	): void {
		if (this._device.type !== "webgpu") {
			return;
		}

		this._emitWarnings(renderer, featureState.warnings);

		const lightingState = collectWebGPULighting(
			renderer.scene.lights as any,
			featureState.enableLighting,
			featureState.enableShadows,
			renderer.shadowMaps as ReadonlyMap<any, ShadowMap>
		);
		this._emitWarnings(renderer, lightingState.warnings);
		this._prepareShadowAtlases(lightingState);

		const frameUniform = this._getFrameUniformBuffer();
		const frameData = packFrameUniformData({
			viewProjectionMatrix: renderer.camera.viewProjectionMatrix,
			cameraPosition: renderer.camera.position,
			ambientColor: lightingState.ambientColor,
			directionalLights: lightingState.directionalLights,
			directionalShadows: lightingState.directionalShadows,
			pointLights: lightingState.pointLights,
			spotLights: lightingState.spotLights,
			spotShadows: lightingState.spotShadows,
			enableLighting: featureState.enableLighting,
			enableGamma: featureState.enableGamma,
			enableShadows: featureState.enableShadows,
		});

		this._device.writeBuffer(frameUniform, this._toArrayBuffer(frameData));
		this._frameBindingCache.clear();
	}

	public async getWebGPUDrawResources(
		model: IModel,
		renderer: Renderer
	): Promise<WebGPUDrawResources | null> {
		if (this._device.type !== "webgpu") {
			return null;
		}

		const material = this.getMaterial(model);
		if (this._hasMixedMaterials(model, material)) {
			renderer.warnOnce(
				`webgpu-model-material:${model.id}`,
				`WebGPU backend currently supports one material per model; using the first material on model ${model.id}`
			);
		}

		if (material.alphaMode === "BLEND") {
			renderer.warnOnce(
				`webgpu-material-blend:${material.type}:${material.name}`,
				`WebGPU backend does not support alpha blend materials yet; skipping ${material.name}`
			);
			return null;
		}

		const materialData = createWebGPUMaterialUniformData(material);
		this._emitWarnings(renderer, materialData.warnings);

		const geometry = this.getGeometry(model);
		const pipeline = await this._getWebGPUPipeline(material, materialData);
		const frameBinding = this._getFrameBinding(pipeline);
		const textures = materialData.textureSlots.map((slot, index) =>
			this._getTextureForSlot(slot.map, index)
		);
		const samplers = materialData.textureSlots.map((slot) =>
			this._getSamplerForTexture(slot.map)
		);
		const modelBinding = this._getModelBinding(
			model,
			material,
			materialData,
			pipeline,
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

	public disposeModel(model: IModel): void {
		const geometry = this._geometryCache.get(model);
		if (geometry) {
			geometry.vertexBuffer.destroy();
			geometry.indexBuffer.destroy();
			this._geometryCache.delete(model);
		}

		const modelBinding = this._modelBindingCache.get(model);
		if (modelBinding) {
			modelBinding.uniformBuffer.destroy();
			this._modelBindingCache.delete(model);
		}
	}

	public disposeMaterial(material: Material): void {
		this._softwarePipelineCache.delete(material);
		this._webgpuMaterialPipelineCache.delete(material);
	}

	private _uploadModel(model: IModel): GeometryCacheEntry {
		const vertexData: number[] = [];
		const indexData: number[] = [];
		let vertexCount = 0;

		for (const face of model.faces) {
			const base = vertexCount;
			for (const vertex of face.vertices) {
				vertexData.push(vertex.x, vertex.y, vertex.z);
				vertexData.push(
					vertex.normal?.x ?? 0,
					vertex.normal?.y ?? 0,
					vertex.normal?.z ?? 0
				);
				vertexData.push(vertex.u ?? 0, vertex.v ?? 0);
				vertexData.push(
					vertex.tangent?.x ?? 0,
					vertex.tangent?.y ?? 0,
					vertex.tangent?.z ?? 0,
					vertex.tangent?.w ?? 0
				);
				vertexData.push(vertex.u2 ?? 0, vertex.v2 ?? 0);
				vertexCount++;
			}

			for (let i = 1; i < face.vertices.length - 1; i++) {
				indexData.push(base, base + i, base + i + 1);
			}
		}

		const vertexBuffer = this._device.createBuffer({
			size: vertexData.length * 4,
			usage: BufferUsage.Vertex | BufferUsage.CopyDst,
			label: `VertexBuffer_${model.id}`,
		});

		const indexBuffer = this._device.createBuffer({
			size: indexData.length * 4,
			usage: BufferUsage.Index | BufferUsage.CopyDst,
			label: `IndexBuffer_${model.id}`,
		});

		this._device.writeBuffer(
			vertexBuffer,
			this._toArrayBuffer(new Float32Array(vertexData))
		);
		this._device.writeBuffer(
			indexBuffer,
			this._toArrayBuffer(new Uint32Array(indexData))
		);

		return {
			vertexBuffer,
			indexBuffer,
			indexCount: indexData.length,
		};
	}

	private async _createSoftwarePipeline(
		material: Material
	): Promise<IRenderPipeline> {
		const vertexModule = await this._device.createShaderModule({
			code: material.vertexCode || "",
			softwareDelegate: material.vertexJS,
		});

		const fragmentModule = await this._device.createShaderModule({
			code: material.fragmentCode || "",
			softwareDelegate: material.fragmentJS,
		});

		const pipeline = this._device.createPipeline({
			label: `Pipeline_${material.name || material.type}`,
			vertex: {
				module: vertexModule,
				entryPoint: "main",
				buffers: [
					{
						arrayStride: 56,
						attributes: [
							{ format: "float32x3", offset: 0, shaderLocation: 0 },
							{ format: "float32x2", offset: 24, shaderLocation: 1 },
							{ format: "float32x3", offset: 12, shaderLocation: 2 },
							{ format: "float32x4", offset: 32, shaderLocation: 3 },
							{ format: "float32x2", offset: 48, shaderLocation: 4 },
						],
					},
				],
			},
			fragment: {
				module: fragmentModule,
				entryPoint: "main",
				targets: [{ format: this._device.canvasFormat as any }],
			},
			primitive: {
				topology: "triangle-list",
				cullMode: material.doubleSided ? "none" : "back",
				frontFace: "ccw",
			},
			depthStencil: {
				format: "depth24plus" as any,
				depthWriteEnabled: true,
				depthCompare: "less",
			},
		} as any);

		(pipeline as any)._material = material;
		return pipeline;
	}

	private async _getWebGPUPipeline(
		material: Material,
		materialData: WebGPUMaterialUniformData
	): Promise<IRenderPipeline> {
		const cached = this._webgpuMaterialPipelineCache.get(material);
		if (cached && cached.key === materialData.pipelineKey) {
			return cached.pipeline;
		}

		let pipeline = this._webgpuPipelineCache.get(materialData.pipelineKey);
		if (!pipeline) {
			pipeline = await this._createWebGPUPipeline(
				material,
				materialData.pipelineKey
			);
			this._webgpuPipelineCache.set(materialData.pipelineKey, pipeline);
		}

		this._webgpuMaterialPipelineCache.set(material, {
			key: materialData.pipelineKey,
			pipeline,
		});

		return pipeline;
	}

	private async _createWebGPUPipeline(
		material: Material,
		pipelineKey: string
	): Promise<IRenderPipeline> {
		const sceneShaderModule = await this._getSceneShaderModule();

		return this._device.createPipeline({
			label: `WebGPUScenePipeline_${pipelineKey}`,
			vertex: {
				module: sceneShaderModule,
				entryPoint: "vsMain",
				buffers: [
					{
						arrayStride: 56,
						attributes: [
							{ format: "float32x3", offset: 0, shaderLocation: 0 },
							{ format: "float32x2", offset: 24, shaderLocation: 1 },
							{ format: "float32x3", offset: 12, shaderLocation: 2 },
							{ format: "float32x4", offset: 32, shaderLocation: 3 },
							{ format: "float32x2", offset: 48, shaderLocation: 4 },
						],
					},
				],
			},
			fragment: {
				module: sceneShaderModule,
				entryPoint: "fsMain",
				targets: [{ format: this._device.canvasFormat as any }],
			},
			primitive: {
				topology: "triangle-list",
				cullMode: material.doubleSided ? "none" : "back",
				frontFace: "ccw",
			},
			depthStencil: {
				format: TextureFormat.Depth24Plus,
				depthWriteEnabled: true,
				depthCompare: "less",
			},
		} as any);
	}

	private async _getSceneShaderModule() {
		if (!this._sceneShaderModule) {
			this._sceneShaderModule = await this._device.createShaderModule({
				code: WEBGPU_SCENE_SHADER,
				label: "WebGPUSceneShader",
			});
		}

		return this._sceneShaderModule;
	}

	private _getFrameUniformBuffer(): IRenderBuffer {
		if (!this._frameUniformBuffer) {
			this._frameUniformBuffer = this._device.createBuffer({
				size: WEBGPU_FRAME_UNIFORM_FLOATS * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: "WebGPUFrameUniforms",
			});
		}

		return this._frameUniformBuffer;
	}

	private _getFrameBinding(pipeline: IRenderPipeline): IBindingGroup {
		let cached = this._frameBindingCache.get(pipeline);
		if (!cached) {
			const entries: BindingEntry[] = [
				{
					binding: 0,
					resource: this._getFrameUniformBuffer(),
				},
			];

			entries.push({
				binding: 1,
				resource: this._directionalShadowAtlas?.texture ?? this._getWhiteTexture(),
			});
			entries.push({
				binding: 2,
				resource: this._spotShadowAtlas?.texture ?? this._getWhiteTexture(),
			});

			cached = this._device.createBindingGroup({
				label: `FrameBinding_${pipeline.label ?? "scene"}`,
				pipeline,
				layoutIndex: 0,
				entries,
			});
			this._frameBindingCache.set(pipeline, cached);
		}

		return cached;
	}

	private _prepareShadowAtlases(lightingState: WebGPULightingState): void {
		this._directionalShadowAtlas = this._prepareShadowAtlas(
			lightingState.directionalShadows,
			this._directionalShadowAtlas,
			'WebGPUDirectionalShadowAtlas'
		);
		this._spotShadowAtlas = this._prepareShadowAtlas(
			lightingState.spotShadows,
			this._spotShadowAtlas,
			'WebGPUSpotShadowAtlas'
		);
	}

	private _prepareShadowAtlas(
		shadows: Array<{
			enabled: boolean;
			shadowMap: ShadowMap | null;
			atlasTileSize: number;
		}>,
		current: { tileSize: number; texture: IRenderTexture } | null,
		label: string
	): { tileSize: number; texture: IRenderTexture } | null {
		let tileSize = 0;
		for (const shadow of shadows) {
			if (!shadow?.enabled || !shadow.shadowMap) continue;
			tileSize = Math.max(tileSize, shadow.shadowMap.size | 0);
		}

		for (const shadow of shadows) {
			if (!shadow) continue;
			shadow.atlasTileSize = tileSize;
		}

		if (tileSize <= 0) {
			return null;
		}

		let atlas = current;
		if (!atlas || atlas.tileSize !== tileSize) {
			atlas?.texture.destroy();
			atlas = {
				tileSize,
				texture: this._device.createTexture({
					width: tileSize * 2,
					height: tileSize * 2,
					format: TextureFormat.RGBA8Unorm,
					usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
					label,
				}),
			};
		}

		const upload = this._createShadowAtlasUploadData(shadows, tileSize);
		this._device.writeTexture(
			atlas.texture,
			upload.data,
			{
				bytesPerRow: upload.bytesPerRow,
				rowsPerImage: upload.height,
			},
			{
				width: upload.width,
				height: upload.height,
				depthOrArrayLayers: 1,
			}
		);

		return atlas;
	}

	private _createShadowAtlasUploadData(
		shadows: Array<{ enabled: boolean; shadowMap: ShadowMap | null }>,
		tileSize: number
	): {
		data: Uint8Array
		bytesPerRow: number
		width: number
		height: number
	} {
		const atlasWidth = tileSize * 2;
		const atlasHeight = tileSize * 2;
		const bytesPerRow = alignTo(atlasWidth * 4, 256);
		const data = new Uint8Array(bytesPerRow * atlasHeight).fill(255);

		for (let shadowIndex = 0; shadowIndex < shadows.length; shadowIndex++) {
			const shadow = shadows[shadowIndex];
			if (!shadow?.enabled || !shadow.shadowMap) continue;

			const { size, buffer } = shadow.shadowMap;
			const tileX = shadowIndex % 2;
			const tileY = (shadowIndex / 2) | 0;
			const originX = tileX * tileSize;
			const originY = tileY * tileSize;

			for (let y = 0; y < size; y++) {
				const shadowRow = y * size;
				const atlasRowOffset = (originY + y) * bytesPerRow;
				for (let x = 0; x < size; x++) {
					const depth = buffer[shadowRow + x];
					const normalized = Number.isFinite(depth)
						? clamp(depth * 0.5 + 0.5, 0, 1)
						: 1;
					const encoded = Math.round(normalized * 0xffffffff);
					const pixelOffset = atlasRowOffset + (originX + x) * 4;
					data[pixelOffset] = encoded >>> 24;
					data[pixelOffset + 1] = (encoded >>> 16) & 0xff;
					data[pixelOffset + 2] = (encoded >>> 8) & 0xff;
					data[pixelOffset + 3] = encoded & 0xff;
				}
			}
		}

		return {
			data,
			bytesPerRow,
			width: atlasWidth,
			height: atlasHeight,
		};
	}

	private _getModelBinding(
		model: IModel,
		material: Material,
		materialData: WebGPUMaterialUniformData,
		pipeline: IRenderPipeline,
		textures: IRenderTexture[],
		samplers: ISampler[]
	): IBindingGroup {
		let cached = this._modelBindingCache.get(model);
		if (!cached) {
			cached = {
				uniformBuffer: this._device.createBuffer({
					size: WEBGPU_MODEL_UNIFORM_FLOATS * 4,
					usage: BufferUsage.Uniform | BufferUsage.CopyDst,
					label: `ModelUniform_${model.id}`,
				}),
				bindingGroup: null,
				pipeline: null,
				textures: [],
				samplers: [],
				material: null,
			};
			this._modelBindingCache.set(model, cached);
		}

		const modelMatrix = getModelMatrix(model);
		const normalMatrix = Matrix4.normalMatrix(modelMatrix);
		const modelUniformData = packModelUniformData(
			modelMatrix,
			normalMatrix as any,
			materialData
		);
		this._device.writeBuffer(
			cached.uniformBuffer,
			this._toArrayBuffer(modelUniformData)
		);

		if (
			!cached.bindingGroup ||
			cached.pipeline !== pipeline ||
			!this._areTextureArraysEqual(cached.textures, textures) ||
			!this._areSamplerArraysEqual(cached.samplers, samplers) ||
			cached.material !== material
		) {
			const entries: BindingEntry[] = [
				{
					binding: 0,
					resource: cached.uniformBuffer,
				},
			];

			for (let i = 0; i < textures.length; i++) {
				entries.push({
					binding: 1 + i * 2,
					resource: textures[i],
				});
				entries.push({
					binding: 2 + i * 2,
					resource: samplers[i],
				});
			}

			cached.bindingGroup = this._device.createBindingGroup({
				label: `ModelBinding_${model.id}`,
				pipeline,
				layoutIndex: 1,
				entries,
			});
			cached.pipeline = pipeline;
			cached.textures = textures.slice();
			cached.samplers = samplers.slice();
			cached.material = material;
		}

		return cached.bindingGroup;
	}

	private _getTextureForSlot(
		texture: Texture | null,
		slotIndex: number
	): IRenderTexture {
		if (!texture?.data || texture.width <= 0 || texture.height <= 0) {
			return this._isNormalSlot(slotIndex)
				? this._getNeutralNormalTexture()
				: this._getWhiteTexture();
		}

		let cached = this._textureCache.get(texture);
		if (!cached) {
			cached = this._device.createTexture({
				width: texture.width,
				height: texture.height,
				format: TextureFormat.RGBA8Unorm,
				usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
				label: `Texture_${slotIndex}_${texture.width}x${texture.height}`,
			});

			const upload = createTextureUploadData(texture);
			this._device.writeTexture(
				cached,
				new Uint8Array(upload.data),
				{
					bytesPerRow: upload.bytesPerRow,
					rowsPerImage: upload.height,
				},
				{
					width: upload.width,
					height: upload.height,
					depthOrArrayLayers: 1,
				}
			);
			this._textureCache.set(texture, cached);
		}

		return cached;
	}

	private _getSamplerForTexture(texture: Texture | null): ISampler {
		if (!texture) {
			return this._getWhiteSampler();
		}

		let cached = this._samplerCache.get(texture);
		if (!cached) {
			cached = this._device.createSampler({
				addressModeU: this._mapWrapMode(texture.wrapS),
				addressModeV: this._mapWrapMode(texture.wrapT),
				magFilter: this._mapFilterMode(texture.magFilter),
				minFilter: this._mapFilterMode(texture.minFilter),
				mipmapFilter: this._mapFilterMode(texture.minFilter),
				label: `Sampler_${texture.width}x${texture.height}`,
			});
			this._samplerCache.set(texture, cached);
		}

		return cached;
	}

	private _getWhiteTexture(): IRenderTexture {
		if (!this._whiteTexture) {
			this._whiteTexture = this._device.createTexture({
				width: 1,
				height: 1,
				format: TextureFormat.RGBA8Unorm,
				usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
				label: "WebGPUWhiteTexture",
			});
			this._device.writeTexture(
				this._whiteTexture,
				new Uint8Array(256).fill(255),
				{
					bytesPerRow: 256,
					rowsPerImage: 1,
				},
				{
					width: 1,
					height: 1,
					depthOrArrayLayers: 1,
				}
			);
		}

		return this._whiteTexture;
	}

	private _getNeutralNormalTexture(): IRenderTexture {
		if (!this._neutralNormalTexture) {
			this._neutralNormalTexture = this._device.createTexture({
				width: 1,
				height: 1,
				format: TextureFormat.RGBA8Unorm,
				usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
				label: "WebGPUNeutralNormalTexture",
			});

			const data = new Uint8Array(256);
			data[0] = 128;
			data[1] = 128;
			data[2] = 255;
			data[3] = 255;

			this._device.writeTexture(
				this._neutralNormalTexture,
				data,
				{
					bytesPerRow: 256,
					rowsPerImage: 1,
				},
				{
					width: 1,
					height: 1,
					depthOrArrayLayers: 1,
				}
			);
		}

		return this._neutralNormalTexture;
	}

	private _getWhiteSampler(): ISampler {
		if (!this._whiteSampler) {
			this._whiteSampler = this._device.createSampler({
				addressModeU: AddressMode.Repeat,
				addressModeV: AddressMode.Repeat,
				magFilter: FilterMode.Linear,
				minFilter: FilterMode.Linear,
				mipmapFilter: FilterMode.Linear,
				label: "WebGPUWhiteSampler",
			});
		}

		return this._whiteSampler;
	}

	private _mapWrapMode(value?: string): AddressMode {
		switch (value) {
			case "Clamp":
				return AddressMode.ClampToEdge;
			case "MirroredRepeat":
				return AddressMode.MirrorRepeat;
			default:
				return AddressMode.Repeat;
		}
	}

	private _mapFilterMode(value?: string): FilterMode {
		return value === "Nearest" || value === "NearestMipmapNearest"
			? FilterMode.Nearest
			: FilterMode.Linear;
	}

	private _isNormalSlot(slotIndex: number): boolean {
		return (
			slotIndex === WEBGPU_TEXTURE_SLOT.NORMAL ||
			slotIndex === WEBGPU_TEXTURE_SLOT.CLEARCOAT_NORMAL
		);
	}

	private _areTextureArraysEqual(
		left: IRenderTexture[],
		right: IRenderTexture[]
	): boolean {
		if (left.length !== right.length) {
			return false;
		}

		for (let i = 0; i < left.length; i++) {
			if (left[i] !== right[i]) {
				return false;
			}
		}

		return true;
	}

	private _areSamplerArraysEqual(left: ISampler[], right: ISampler[]): boolean {
		if (left.length !== right.length) {
			return false;
		}

		for (let i = 0; i < left.length; i++) {
			if (left[i] !== right[i]) {
				return false;
			}
		}

		return true;
	}

	private _hasMixedMaterials(model: IModel, baseMaterial: Material): boolean {
		for (const face of model.faces) {
			if ((face.material ?? baseMaterial) !== baseMaterial) {
				return true;
			}
		}

		return false;
	}

	private _emitWarnings(
		renderer: Renderer,
		warnings: { key: string; message: string }[]
	): void {
		for (const warning of warnings) {
			renderer.warnOnce(warning.key, warning.message);
		}
	}

	private _toArrayBuffer(data: ArrayBufferView): ArrayBuffer {
		const copy = new Uint8Array(data.byteLength);
		copy.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
		return copy.buffer;
	}
}

export { resolveWebGPUFeatureState };

function clamp(value: number, min: number, max: number): number {
	return Math.max(min, Math.min(max, value));
}
