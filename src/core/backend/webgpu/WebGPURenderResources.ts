import type { Renderer } from "../../Renderer";
import type {
	DrawPacket,
	FrameContext,
	PreparedScene,
} from "../../pipeline/types";
import { AlphaMode } from "../../../materials/Material";
import type { ResolvedFeatureState } from "../../pipeline/types";
import type { WebGPUBackend } from "../WebGPUBackend";
import {
	collectWebGPULighting,
	createWebGPUMaterialUniformData,
	type WebGPUFeatureState,
	type WebGPULightingState,
} from "./";
import { createWebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";
import { WebGPUFrameBindingCache } from "./WebGPUFrameBindingCache";
import { WebGPUGeometryRegistry } from "./WebGPUGeometryRegistry";
import { WebGPUMaterialBindingCache } from "./WebGPUMaterialBindingCache";
import { WebGPUPipelineLibrary } from "./WebGPUPipelineLibrary";
import { WebGPUShadowAtlasAllocator } from "./WebGPUShadowAtlasAllocator";
import { WebGPUTextureRegistry } from "./WebGPUTextureRegistry";

export interface WebGPUDrawResources {
	pipeline: any;
	frameBinding: any;
	modelBinding: any;
	vertexBuffer: any;
	indexBuffer: any;
	indexCount: number;
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
	private _lightingState: WebGPULightingState | null = null;

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
	}

	public async init(): Promise<void> {
		await this._pipelineLibrary.init();
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
		const { scene, features } = this._resolveFrameInputs(
			contextOrScene,
			featuresArg
		);
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
			scene.lights,
			features.enableLighting,
			features.enableShadows,
			scene.shadowMaps
		);
		for (const warning of this._lightingState.warnings) {
			this._renderer.warnOnce(warning.key, warning.message);
		}

		this._shadowAtlases.prepare(this._lightingState);
		this._frameBindings.prepare(scene, this._lightingState, featureState);
		this._materialBindings.beginFrame();
	}

	private _resolveFrameInputs(
		contextOrScene: FrameContext | PreparedScene,
		featuresArg?: ResolvedFeatureState
	): {
		scene: PreparedScene;
		features: ResolvedFeatureState;
	} {
		if (this._isFrameContext(contextOrScene)) {
			return {
				scene: contextOrScene.scene,
				features: contextOrScene.features,
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
