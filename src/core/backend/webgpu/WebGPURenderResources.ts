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

		const pipeline = await this._pipelineLibrary.getSkyboxPipeline();
		const frameBinding = this._frameBindings.getSkyboxBinding();

		return {
			pipeline,
			frameBinding,
		};
	}
}
