import type { Renderer } from "../Renderer";
import type { DrawPacket, PreparedScene } from "../pipeline/types";
import type { ResolvedFeatureState } from "../pipeline/types";
import type { WebGPUBackend } from "../backend/WebGPUBackend";
import {
	collectWebGPULighting,
	createWebGPUMaterialUniformData,
	type WebGPUFeatureState,
	type WebGPULightingState,
} from "../bridge/webgpu";
import { createWebGPUPipelineLayouts } from "../backend/webgpu/WebGPUPipelineLayouts";
import { FrameBindingCache } from "./FrameBindingCache";
import { GeometryRegistry } from "./GeometryRegistry";
import { MaterialBindingCache } from "./MaterialBindingCache";
import { PipelineLibrary } from "./PipelineLibrary";
import { ShadowAtlasAllocator } from "./ShadowAtlasAllocator";
import { TextureRegistry } from "./TextureRegistry";

export interface WebGPUDrawResources {
	pipeline: any;
	frameBinding: any;
	modelBinding: any;
	vertexBuffer: any;
	indexBuffer: any;
	indexCount: number;
}

export class RenderResources {
	private _renderer: Renderer;
	private _backend: WebGPUBackend;
	private _layouts: ReturnType<typeof createWebGPUPipelineLayouts>;
	private _geometryRegistry: GeometryRegistry;
	private _textureRegistry: TextureRegistry;
	private _shadowAtlases: ShadowAtlasAllocator;
	private _pipelineLibrary: PipelineLibrary;
	private _frameBindings: FrameBindingCache;
	private _materialBindings: MaterialBindingCache;
	private _lightingState: WebGPULightingState | null = null;

	constructor(renderer: Renderer, backend: WebGPUBackend) {
		this._renderer = renderer;
		this._backend = backend;
		this._layouts = createWebGPUPipelineLayouts(backend.device);
		this._geometryRegistry = new GeometryRegistry(backend);
		this._textureRegistry = new TextureRegistry(backend);
		this._shadowAtlases = new ShadowAtlasAllocator(backend);
		this._pipelineLibrary = new PipelineLibrary(backend, this._layouts);
		this._frameBindings = new FrameBindingCache(
			backend,
			this._layouts,
			this._textureRegistry,
			this._shadowAtlases
		);
		this._materialBindings = new MaterialBindingCache(backend, this._layouts);
	}

	public async init(): Promise<void> {
		await this._pipelineLibrary.init();
	}

	public prepareFrame(
		frame: PreparedScene,
		features: ResolvedFeatureState
	): void {
		const featureState: WebGPUFeatureState = {
			enableLighting: features.enableLighting,
			enableGamma: features.enableGamma,
			enableSH: false,
			enableShadows: features.enableShadows,
			enableReflection: false,
			enableSkybox: false,
			enableSSAO: false,
			enableVolumetric: false,
			warnings: [],
		};

		this._lightingState = collectWebGPULighting(
			frame.lights,
			features.enableLighting,
			features.enableShadows,
			frame.shadowMaps
		);
		for (const warning of this._lightingState.warnings) {
			this._renderer.warnOnce(warning.key, warning.message);
		}

		this._shadowAtlases.prepare(this._lightingState);
		this._frameBindings.prepare(frame, this._lightingState, featureState);
	}

	public async getDrawResources(
		packet: DrawPacket
	): Promise<WebGPUDrawResources | null> {
		if (packet.material.alphaMode === "BLEND") {
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
		const pipeline = await this._pipelineLibrary.getPipeline(packet.material);
		const textures = materialData.textureSlots.map((slot, index) =>
			this._textureRegistry.getTextureForSlot(slot.map, index)
		);
		const samplers = materialData.textureSlots.map((slot) =>
			this._textureRegistry.getSamplerForTexture(slot.map)
		);
		const frameBinding = this._frameBindings.getBinding(pipeline);
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
}
