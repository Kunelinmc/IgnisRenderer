import { createInlineCompositeShaderSource } from "../../shaders/runtime";
import { ShaderSource } from "../../shaders/ShaderSource";
import { createWebGPUMaterialUniformData } from "./";
import { createWebGPUSceneVertexBufferLayout } from "./bufferLayouts";
import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../../core/types";
import { TextureFormat, type ColorTargetState } from "../types";
import type { PrimitiveDrawTopology } from "../../core/types";
import {
	AlphaMode,
	type Material,
} from "../../materials/Material";
import {
	isMaterialTransparentPass,
	materialUsesTransmission,
} from "../../materials/transparency";
import {
	ShaderMaterial,
} from "../../materials/ShaderMaterial";
import type { IRenderPipeline, IShaderModule } from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
import {
	SINGLE_SAMPLE_WEBGPU_MSAA_CONTEXT,
	type WebGPUMSAAContext,
} from "./WebGPUMSAAController";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";
import { Logger } from "../../foundation/Logger";
import {
	resolveWebGPUScenePassDescriptor,
	type WebGPUScenePassDescriptor,
	type WebGPUScenePipelineDrawMode,
	type WebGPUSceneTargetMode,
	type WebGPUTransparentPipelineMode,
} from "./WebGPUScenePassDescriptors";

export type {
	WebGPUScenePipelineDrawMode,
	WebGPUSceneTargetMode,
	WebGPUTransparentPipelineMode,
} from "./WebGPUScenePassDescriptors";
const COLOR_WRITE_NONE = 0;
const ALPHA_BLEND_STATE = {
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
const PLANAR_REFLECTION_BLEND_STATE = {
	color: {
		srcFactor: "src-alpha",
		dstFactor: "one-minus-src-alpha",
		operation: "add",
	},
	alpha: {
		srcFactor: "zero",
		dstFactor: "one",
		operation: "add",
	},
};
// Transmission shading currently outputs straight (non-premultiplied) color.
// Use regular alpha blending to avoid over-bright transmission composition.
const TRANSMISSION_BLEND_STATE = ALPHA_BLEND_STATE;
const OIT_ACCUM_BLEND_STATE = {
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
};
const OIT_REVEAL_BLEND_STATE = {
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
};

interface CachedPipelineEntry {
	key: string;
	descriptorKey: string;
	shaderKey: string;
	depthFormat: TextureFormat;
	sampleCount: number;
	topology: PrimitiveDrawTopology;
	pipeline: IRenderPipeline;
}

interface WebGPUSceneProgram {
	vertexModule: IShaderModule;
	fragmentModule: IShaderModule;
	vertexEntryPoint: string;
	fragmentEntryPoint: string;
}

interface WebGPUPipelineLibraryOptions {
	listenToShaderRuntime?: boolean;
	msaaContext?: WebGPUMSAAContext;
}

export class WebGPUPipelineLibrary {
	private _backend: WebGPUBackend;
	private _layouts: WebGPUPipelineLayouts;
	private _msaa: WebGPUMSAAContext;
	private _disposeShaderRuntimeListener: (() => void) | null = null;
	private _sceneShaderModule: IShaderModule | null = null;
	private _sceneShaderDirectiveTag = "";
	private _environmentShaderModule: IShaderModule | null = null;
	private _environmentShaderDirectiveTag = "";
	private _deferredLightingShaderModule: IShaderModule | null = null;
	private _deferredLightingShaderDirectiveTag = "";
	private _planarReflectionCompositeShaderModule: IShaderModule | null = null;
	private _planarReflectionCompositeDirectiveTag = "";
	private _deferredLightingPipeline: IRenderPipeline | null = null;
	private _customShaderModuleCache = new Map<string, IShaderModule>();
	private _environmentPipelines = new Map<string, IRenderPipeline>();
	private _materialPipelineCache = new WeakMap<Material, CachedPipelineEntry>();
	private _pipelineCache = new Map<string, IRenderPipeline>();
	private _earlyZPrepassCache = new Map<string, IRenderPipeline>();

	constructor(
		backend: WebGPUBackend,
		layouts: WebGPUPipelineLayouts,
		options: WebGPUPipelineLibraryOptions = {}
	) {
		this._backend = backend;
		this._layouts = layouts;
		this._msaa = options.msaaContext ?? SINGLE_SAMPLE_WEBGPU_MSAA_CONTEXT;
		const shaderRuntime = this._getShaderRuntime();
		const listenToShaderRuntime = options.listenToShaderRuntime !== false;
		if (
			listenToShaderRuntime &&
			shaderRuntime &&
			typeof shaderRuntime.onDidChange === "function"
		) {
			this._disposeShaderRuntimeListener = shaderRuntime.onDidChange(() =>
				this.invalidateShaderRuntimeCaches()
			);
		}
	}

	public destroy(): void {
		this._disposeShaderRuntimeListener?.();
		this._disposeShaderRuntimeListener = null;
		this.invalidateShaderRuntimeCaches();
	}

	public invalidateShaderRuntimeCaches(): void {
		this._sceneShaderModule = null;
		this._sceneShaderDirectiveTag = "";
		this._environmentShaderModule = null;
		this._environmentShaderDirectiveTag = "";
		this._deferredLightingShaderModule = null;
		this._deferredLightingShaderDirectiveTag = "";
		this._planarReflectionCompositeShaderModule = null;
		this._planarReflectionCompositeDirectiveTag = "";
		this._deferredLightingPipeline = null;
		this._customShaderModuleCache.clear();
		this._environmentPipelines.clear();
		this._materialPipelineCache = new WeakMap<Material, CachedPipelineEntry>();
		this._pipelineCache.clear();
		this._earlyZPrepassCache.clear();
	}

	public async init(): Promise<void> {
		await Promise.all([
			this._getSceneShaderModule(),
			this._getEnvironmentShaderModule(),
			this._getDeferredLightingShaderModule(),
		]);
	}

	public async getPipeline(
		material: Material,
		mode: WebGPUSceneTargetMode = "single",
		isWireframe = false,
		topology: PrimitiveDrawTopology = DEFAULT_PRIMITIVE_DRAW_TOPOLOGY,
		transparentMode: WebGPUTransparentPipelineMode = "default",
		drawMode: WebGPUScenePipelineDrawMode = "default",
		sampleCountOverride?: number
	): Promise<IRenderPipeline> {
		const descriptor = resolveWebGPUScenePassDescriptor(
			mode,
			transparentMode,
			drawMode
		);
		const sampleCount = this._resolveSampleCount(
			descriptor.sceneTargetMode,
			sampleCountOverride
		);
		const depthFormat = this._resolveSceneDepthFormat(
			descriptor.sceneTargetMode
		);
		const { pipelineKey } = createWebGPUMaterialUniformData(
			material,
			isWireframe
		);
		const initialShaderKey = this._getShaderCacheKey(material);
		const cached = this._materialPipelineCache.get(material);
		if (
			cached &&
			cached.key === pipelineKey &&
			cached.descriptorKey === descriptor.pipelineKeyPart &&
			cached.shaderKey === initialShaderKey &&
			cached.depthFormat === depthFormat &&
			cached.sampleCount === sampleCount &&
			cached.topology === topology
		) {
			return cached.pipeline;
		}

		const initialCacheKey =
			`${pipelineKey}|${descriptor.pipelineKeyPart}|${initialShaderKey}` +
			`|topology:${topology}|depth:${depthFormat}|msaa:${sampleCount}`;
		let pipeline = this._pipelineCache.get(initialCacheKey);
		if (!pipeline) {
			pipeline = await this._createPipeline(
				material,
				descriptor,
				isWireframe,
				topology,
				sampleCountOverride
			);
		}
		const finalShaderKey = this._getShaderCacheKey(material);
		const finalCacheKey =
			`${pipelineKey}|${descriptor.pipelineKeyPart}|${finalShaderKey}` +
			`|topology:${topology}|depth:${depthFormat}|msaa:${sampleCount}`;
		const cachedFinalPipeline = this._pipelineCache.get(finalCacheKey);
		if (cachedFinalPipeline) {
			pipeline = cachedFinalPipeline;
		} else {
			this._pipelineCache.set(finalCacheKey, pipeline);
		}
		if (finalCacheKey !== initialCacheKey) {
			this._pipelineCache.delete(initialCacheKey);
		}

		this._materialPipelineCache.set(material, {
			key: pipelineKey,
			descriptorKey: descriptor.pipelineKeyPart,
			shaderKey: finalShaderKey,
			depthFormat,
			sampleCount,
			topology,
			pipeline,
		});

		return pipeline;
	}

	private async _createPipeline(
		material: Material,
		descriptor: WebGPUScenePassDescriptor,
		isWireframe: boolean,
		topology: PrimitiveDrawTopology,
		sampleCountOverride?: number
	): Promise<IRenderPipeline> {
		const mode = descriptor.sceneTargetMode;
		if (descriptor.drawMode === "planar-reflection-composite") {
			return this._createPlanarReflectionCompositePipeline(
				material,
				descriptor,
				isWireframe,
				topology,
				sampleCountOverride
			);
		}
		const sampleCount = this._resolveSampleCount(mode, sampleCountOverride);
		const { pipelineKey } = createWebGPUMaterialUniformData(
			material,
			isWireframe
		);
		const sceneProgram = await this._resolveSceneProgram(
			material,
			descriptor
		);
		const isTransparent = isMaterialTransparentPass(material);
		const usesTransmission = materialUsesTransmission(material);
		const isTransmissionCapture =
			descriptor.transparentMode === "transmission-capture";
		const fragmentTargets = this._createSceneFragmentTargets(
			descriptor,
			isTransparent,
			usesTransmission
		);
		const depthFormat = this._resolveSceneDepthFormat(mode);
		const depthWrite = material.depthWrite;
		const isEarlyZColor =
			descriptor.depthStateMode === "early-z-color" &&
			!isTransparent &&
			depthWrite;

		const effectiveTopology = isWireframe ? "line-list" : topology;
		const triangleTopology =
			effectiveTopology === DEFAULT_PRIMITIVE_DRAW_TOPOLOGY;

		return await this._backend.createPipeline({
			layout: this._resolvePipelineLayout(descriptor),
			label:
				descriptor.drawMode === "default" ?
					`WebGPUScenePipeline_${pipelineKey}_${mode}`
				:	`WebGPUScenePipeline_${pipelineKey}_${mode}_${descriptor.drawMode}`,
			vertex: {
				module: sceneProgram.vertexModule,
				entryPoint: sceneProgram.vertexEntryPoint,
				buffers: [createWebGPUSceneVertexBufferLayout()],
			},
			fragment: {
				module: sceneProgram.fragmentModule,
				entryPoint: sceneProgram.fragmentEntryPoint,
				targets: fragmentTargets as any,
			},
			primitive: {
				topology: effectiveTopology as any,
				cullMode:
					isWireframe || !triangleTopology ? "none" : (material.cullMode as any),
				frontFace: descriptor.frontFace,
			},
			depthStencil: {
				format: depthFormat,
				depthWriteEnabled:
					isTransmissionCapture ? true
					: isEarlyZColor ? false
					: depthWrite && !isTransparent,
				depthCompare: isEarlyZColor ? "less-equal" : "less",
			},
			sampleCount,
		} as any);
	}

	private async _createPlanarReflectionCompositePipeline(
		material: Material,
		descriptor: WebGPUScenePassDescriptor,
		isWireframe: boolean,
		topology: PrimitiveDrawTopology,
		sampleCountOverride?: number
	): Promise<IRenderPipeline> {
		const mode = descriptor.sceneTargetMode;
		const sampleCount = this._resolveSampleCount(mode, sampleCountOverride);
		const depthFormat = this._resolveSceneDepthFormat(mode);
		const { pipelineKey } = createWebGPUMaterialUniformData(
			material,
			isWireframe
		);
		const shaderModule =
			await this._getPlanarReflectionCompositeShaderModule();
		const effectiveTopology = isWireframe ? "line-list" : topology;
		const triangleTopology =
			effectiveTopology === DEFAULT_PRIMITIVE_DRAW_TOPOLOGY;

		return await this._backend.createPipeline({
			layout: this._layouts.planarReflectionPipelineLayout,
			label: `WebGPUPlanarReflectionCompositePipeline_${pipelineKey}_${mode}`,
			vertex: {
				module: shaderModule,
				entryPoint: "vsMain",
				buffers: [createWebGPUSceneVertexBufferLayout()],
			},
			fragment: {
				module: shaderModule,
				entryPoint: "fsMain",
				targets: [
					{
						format:
							mode === "mrt" || mode === "gbuffer" || mode === "color" ?
								TextureFormat.RGBA16Float
							:	this._backend.canvasFormat,
						blend: PLANAR_REFLECTION_BLEND_STATE,
					},
					{
						format: TextureFormat.R8Unorm,
					},
				],
			},
			primitive: {
				topology: effectiveTopology as any,
				cullMode:
					isWireframe || !triangleTopology ? "none" : (material.cullMode as any),
				frontFace: descriptor.frontFace,
			},
			depthStencil: {
				format: depthFormat,
				depthWriteEnabled: false,
				depthCompare: "less-equal",
			},
			sampleCount,
		} as any);
	}

	public async getEarlyZPrepassPipeline(
		material: Material,
		mode: WebGPUSceneTargetMode = "single",
		isWireframe = false,
		topology: PrimitiveDrawTopology = DEFAULT_PRIMITIVE_DRAW_TOPOLOGY,
		sampleCountOverride?: number
	): Promise<IRenderPipeline | null> {
		const descriptor = resolveWebGPUScenePassDescriptor(
			mode,
			"default",
			"early-z-prepass"
		);
		if (isMaterialTransparentPass(material)) {
			return null;
		}
		if (!material.depthWrite) {
			return null;
		}
		const isMask = isMaterialMask(material);
		const sampleCount = this._resolveSampleCount(mode, sampleCountOverride);
		const depthFormat = this._resolveSceneDepthFormat(mode);
		const { pipelineKey } = createWebGPUMaterialUniformData(
			material,
			isWireframe
		);
		const shaderKey = this._getShaderCacheKey(material);
		const cacheKey =
			`earlyz|${pipelineKey}|${descriptor.pipelineKeyPart}|` +
			`mask:${isMask ? 1 : 0}|` +
			`wire:${isWireframe ? 1 : 0}|topology:${topology}|` +
			`depth:${depthFormat}|msaa:${sampleCount}|shader:${shaderKey}`;
		const cached = this._earlyZPrepassCache.get(cacheKey);
		if (cached) {
			return cached;
		}

		const effectiveTopology = isWireframe ? "line-list" : topology;
		const triangleTopology =
			effectiveTopology === DEFAULT_PRIMITIVE_DRAW_TOPOLOGY;
		const resolved =
			await this._resolveEarlyZPrepassProgram(material, descriptor, isMask);
		if (!resolved) {
			return null;
		}

		const desc: any = {
			layout: this._resolvePipelineLayout(descriptor),
			label: `WebGPUSceneEarlyZPipeline_${pipelineKey}_${mode}`,
			vertex: {
				module: resolved.vertexModule,
				entryPoint: resolved.vertexEntryPoint,
				buffers: [createWebGPUSceneVertexBufferLayout()],
			},
			primitive: {
				topology: effectiveTopology as any,
				cullMode:
					isWireframe || !triangleTopology ? "none" : (material.cullMode as any),
				frontFace: descriptor.frontFace,
			},
			depthStencil: {
				format: depthFormat,
				depthWriteEnabled: true,
				depthCompare: "less",
			},
			sampleCount,
		};
		if (resolved.fragmentModule && resolved.fragmentEntryPoint) {
			desc.fragment = {
				module: resolved.fragmentModule,
				entryPoint: resolved.fragmentEntryPoint,
				targets: [],
			};
		}

		const pipeline = await this._backend.createPipeline(desc);
		this._earlyZPrepassCache.set(cacheKey, pipeline);
		return pipeline;
	}

	private _createSceneFragmentTargets(
		descriptor: WebGPUScenePassDescriptor,
		isTransparent: boolean,
		usesTransmission: boolean
	): ColorTargetState[] {
		const { fragmentTargetKind, transparentMode } = descriptor;
		const useTransmissionBlend =
			transparentMode === "transmission" ||
			(transparentMode === "default" && usesTransmission);
		const colorBlend =
			!isTransparent ? undefined
			: useTransmissionBlend ? TRANSMISSION_BLEND_STATE
			: ALPHA_BLEND_STATE;
		const motionBlend = !isTransparent ? undefined : ALPHA_BLEND_STATE;

		if (fragmentTargetKind === "gbuffer") {
			return [
				{ format: TextureFormat.RGBA8Unorm },
				{ format: TextureFormat.RGBA16Float },
				{ format: TextureFormat.RGBA16Float },
				{ format: TextureFormat.RGBA16Float },
				{ format: TextureFormat.RGBA16Float },
				{ format: TextureFormat.RGBA16Float },
				{ format: TextureFormat.RGBA16Float },
			];
		}

		if (fragmentTargetKind === "single") {
			return [
				{
					format:
						descriptor.sceneTargetMode === "color" ?
							TextureFormat.RGBA16Float
						:	this._backend.canvasFormat,
					blend: colorBlend,
				},
			];
		}

		if (isTransparent && fragmentTargetKind === "oit") {
			return [
				{
					format: TextureFormat.RGBA16Float,
					blend: OIT_ACCUM_BLEND_STATE,
				},
				{
					format: TextureFormat.R8Unorm,
					blend: OIT_REVEAL_BLEND_STATE,
				},
			];
		}

		if (fragmentTargetKind === "transmission-capture") {
			return [
				{ format: TextureFormat.RGBA16Float },
				{ format: TextureFormat.RGBA16Float },
				{ format: TextureFormat.RGBA16Float },
				{ format: TextureFormat.RGBA16Float },
			];
		}

		return [
			{
				format: TextureFormat.RGBA16Float,
				blend: colorBlend,
			},
			{
				format: TextureFormat.RGBA8Unorm,
				writeMask: isTransparent ? COLOR_WRITE_NONE : undefined,
			},
			{
				format: TextureFormat.RGBA16Float,
				writeMask: isTransparent ? COLOR_WRITE_NONE : undefined,
			},
			{
				format: TextureFormat.RGBA16Float,
				writeMask: isTransparent ? COLOR_WRITE_NONE : undefined,
			},
			{
				format: TextureFormat.RGBA16Float,
				blend: motionBlend,
			},
		];
	}

	private async _resolveSceneProgram(
		material: Material,
		descriptor: WebGPUScenePassDescriptor
	): Promise<WebGPUSceneProgram> {
		const mode = descriptor.sceneTargetMode;
		if (!(material instanceof ShaderMaterial)) {
			const shaderModule = await this._getSceneShaderModule();
			return {
				vertexModule: shaderModule,
				fragmentModule: shaderModule,
				vertexEntryPoint: "vsMain",
				fragmentEntryPoint:
					descriptor.shaderEntryMode === "gbuffer" ? "fsMainGBuffer"
					: descriptor.shaderEntryMode === "oit" ? "fsMainOIT"
					: descriptor.shaderEntryMode === "transmission-capture" ?
						"fsMainTransmissionCapture"
					: descriptor.shaderEntryMode === "mrt" ? "fsMain"
					:	"fsMainSingle",
			};
		}

		try {
			const shaderMode = this._resolveShaderMaterialMode(mode);
			const program = material.resolveWebGPUProgram(shaderMode, {
				enableRuntimeInjects: this._supportsRuntimeInjects(),
			});
			const shaderCacheKey = material.getWebGPUCacheKey();
			const vertexModule = await this._getCustomShaderModule(
				`${shaderCacheKey}:${mode}:vertex`,
				program.vertexCode,
				`WebGPUShaderMaterialVertex_${shaderCacheKey}`,
				"vertex",
				program.vertexEntryPoint
			);
			const fragmentModule = await this._getCustomShaderModule(
				`${shaderCacheKey}:${mode}:fragment`,
				program.fragmentCode,
				`WebGPUShaderMaterialFragment_${shaderCacheKey}_${mode}`,
				"fragment",
				program.fragmentEntryPoint
			);

			return {
				vertexModule,
				fragmentModule,
				vertexEntryPoint: program.vertexEntryPoint,
				fragmentEntryPoint: program.fragmentEntryPoint,
			};
		} catch (error) {
			if (!this._isWarnMode()) {
				throw error;
			}
			const key = `webgpu-shader-material-compile-failed-${material.shaderId}`;
			Logger.warn(
				`[${key}] ShaderMaterial ${material.name} custom WebGPU shader compile failed; using built-in scene shader. ${String(error)}`,
				{ scope: "WebGPUPipelineLibrary", onceKey: key }
			);
			const shaderModule = await this._getSceneShaderModule();
			return {
				vertexModule: shaderModule,
				fragmentModule: shaderModule,
				vertexEntryPoint: "vsMain",
				fragmentEntryPoint:
					descriptor.shaderEntryMode === "gbuffer" ? "fsMainGBuffer"
					: descriptor.shaderEntryMode === "oit" ? "fsMainOIT"
					: descriptor.shaderEntryMode === "transmission-capture" ?
						"fsMainTransmissionCapture"
					: descriptor.shaderEntryMode === "mrt" ? "fsMain"
					:	"fsMainSingle",
			};
		}
	}

	private async _resolveEarlyZPrepassProgram(
		material: Material,
		descriptor: WebGPUScenePassDescriptor,
		isMask: boolean
	): Promise<{
		vertexModule: IShaderModule;
		vertexEntryPoint: string;
		fragmentModule: IShaderModule | null;
		fragmentEntryPoint: string | null;
	} | null> {
		if (!(material instanceof ShaderMaterial)) {
			const shaderModule = await this._getSceneShaderModule();
			return {
				vertexModule: shaderModule,
				vertexEntryPoint: "vsMain",
				fragmentModule: isMask ? shaderModule : null,
				fragmentEntryPoint: isMask ? "fsMainDepthMask" : null,
			};
		}

		try {
			const mode = descriptor.sceneTargetMode;
			const shaderCacheKey = material.getWebGPUCacheKey();
			const shaderMode = this._resolveShaderMaterialMode(mode);
			if (isMask) {
				const depthProgram = material.resolveWebGPUDepthPrepassProgram(shaderMode, {
					enableRuntimeInjects: this._supportsRuntimeInjects(),
				});
				if (!depthProgram) {
					this._warnShaderMaterialDepthPrepassSkipped(
						material,
						"missing depth pre-pass fragment contract."
					);
					return null;
				}
				const vertexModule = await this._getCustomShaderModule(
					`${shaderCacheKey}:${mode}:depth-prepass:vertex`,
					depthProgram.vertexCode,
					`WebGPUShaderMaterialDepthVertex_${shaderCacheKey}_${mode}`,
					"vertex",
					depthProgram.vertexEntryPoint
				);
				const fragmentModule = await this._getCustomShaderModule(
					`${shaderCacheKey}:${mode}:depth-prepass:fragment`,
					depthProgram.fragmentCode,
					`WebGPUShaderMaterialDepthFragment_${shaderCacheKey}_${mode}`,
					"fragment",
					depthProgram.fragmentEntryPoint
				);
				return {
					vertexModule,
					vertexEntryPoint: depthProgram.vertexEntryPoint,
					fragmentModule,
					fragmentEntryPoint: depthProgram.fragmentEntryPoint,
				};
			}

			const regularProgram = material.resolveWebGPUProgram(shaderMode, {
				enableRuntimeInjects: this._supportsRuntimeInjects(),
			});
			const vertexModule = await this._getCustomShaderModule(
				`${shaderCacheKey}:${mode}:depth-prepass:vertex`,
				regularProgram.vertexCode,
				`WebGPUShaderMaterialDepthVertex_${shaderCacheKey}_${mode}`,
				"vertex",
				regularProgram.vertexEntryPoint
			);
			return {
				vertexModule,
				vertexEntryPoint: regularProgram.vertexEntryPoint,
				fragmentModule: null,
				fragmentEntryPoint: null,
			};
		} catch (error) {
			this._warnShaderMaterialDepthPrepassSkipped(material, String(error));
			return null;
		}
	}

	private _warnShaderMaterialDepthPrepassSkipped(
		material: ShaderMaterial,
		reason: string
	): void {
		const key = `webgpu-earlyz-shader-material-skip-${material.shaderId}`;
		Logger.warn(
			`[${key}] ShaderMaterial ${material.name} early-z pre-pass is skipped: ${reason}`,
			{ scope: "WebGPUPipelineLibrary", onceKey: key }
		);
	}

	private async _getCustomShaderModule(
		key: string,
		code: string,
		label: string,
		stage: "vertex" | "fragment",
		entryPoint: string
	): Promise<IShaderModule> {
		let module = this._customShaderModuleCache.get(key);
		if (!module) {
			const composite = createInlineCompositeShaderSource(
				code,
				`<shader-material:${key}>`,
				"source"
			);
			module = await this._backend.createShaderModule({
				code: composite.code,
				sourceMap: composite.sourceMap,
				label,
				language: "wgsl",
				stage,
				entryPoint,
				sourceKind: "custom-material",
			});
			this._customShaderModuleCache.set(key, module);
		}
		return module;
	}

	private _getShaderCacheKey(material: Material): string {
		const directiveTag = this._getDirectiveCacheTag();
		if (material instanceof ShaderMaterial) {
			return (
				`shader:${material.getWebGPUCacheKey()}` +
				`|runtime:${this._getShaderRuntimeRevision()}` +
				`|directive:${directiveTag}`
			);
		}
		return (
			`builtin-scene|runtime:${this._getShaderRuntimeRevision()}` +
			`|directive:${directiveTag}`
		);
	}

	private _resolvePipelineLayout(
		descriptor: WebGPUScenePassDescriptor
	): unknown {
		switch (descriptor.pipelineLayoutKind) {
			case "scene-gbuffer":
				return this._layouts.sceneGBufferPipelineLayout;
			case "scene-depth-prepass":
				return this._layouts.sceneDepthPrepassPipelineLayout;
			case "planar-reflection":
				return this._layouts.planarReflectionPipelineLayout;
			case "scene":
			default:
				return this._layouts.scenePipelineLayout;
		}
	}

	public async getEnvironmentPipeline(
		mode: WebGPUSceneTargetMode = "single",
		sampleCountOverride?: number
	): Promise<IRenderPipeline> {
		const sampleCount = this._resolveSampleCount(mode, sampleCountOverride);
		const depthFormat = this._resolveSceneDepthFormat(mode);
		const cacheKey = `${mode}|depth:${depthFormat}|msaa:${sampleCount}`;
		const cached = this._environmentPipelines.get(cacheKey);
		if (cached) {
			return cached;
		}

		const shaderModule = await this._getEnvironmentShaderModule();
		const targetFormat =
			mode === "mrt" || mode === "gbuffer" || mode === "color" ?
				TextureFormat.RGBA16Float
			:	this._backend.canvasFormat;
		const pipeline = await this._backend.createPipeline({
			layout: this._layouts.environmentPipelineLayout,
			label: `WebGPUEnvironmentPipeline_${mode}`,
			vertex: {
				module: shaderModule,
				entryPoint: "vsMain",
			},
			fragment: {
				module: shaderModule,
				entryPoint: "fsMain",
				targets: [{ format: targetFormat }],
			},
			primitive: {
				topology: "triangle-list" as any,
				cullMode: "none",
				frontFace: "ccw",
			},
			depthStencil: {
				format: depthFormat,
				depthWriteEnabled: false,
				depthCompare: "always",
			},
			sampleCount,
		} as any);
		this._environmentPipelines.set(cacheKey, pipeline);
		return pipeline;
	}

	/**
	 * Returns the fullscreen pipeline that resolves WebGPU deferred lighting.
	 *
	 * @returns A cached render pipeline for `fsMainDeferredLighting`.
	 * @sideEffects May compile the deferred lighting shader module and pipeline.
	 */
	public async getDeferredLightingPipeline(): Promise<IRenderPipeline> {
		if (this._deferredLightingPipeline) {
			return this._deferredLightingPipeline;
		}
		const shaderModule = await this._getDeferredLightingShaderModule();
		this._deferredLightingPipeline = await this._backend.createPipeline({
			layout: this._layouts.deferredLightingPipelineLayout,
			label: "WebGPUDeferredLightingPipeline",
			vertex: {
				module: shaderModule,
				entryPoint: "vsMainDeferredLighting",
			},
			fragment: {
				module: shaderModule,
				entryPoint: "fsMainDeferredLighting",
				targets: [{ format: TextureFormat.RGBA16Float }],
			},
			primitive: {
				topology: "triangle-list" as any,
				cullMode: "none",
				frontFace: "ccw",
			},
			sampleCount: 1,
		} as any);
		return this._deferredLightingPipeline;
	}

	private _resolveSceneDepthFormat(mode: WebGPUSceneTargetMode): TextureFormat {
		if (mode === "mrt" || mode === "gbuffer" || mode === "color") {
			return TextureFormat.Depth32Float;
		}
		const backend = this._backend as {
			canvasDepthFormat?: TextureFormat;
		};
		return backend.canvasDepthFormat ?? TextureFormat.Depth24Plus;
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
		return this._msaa.sampleCount;
	}

	private _resolveShaderMaterialMode(
		mode: WebGPUSceneTargetMode
	): "single" | "mrt" | "deferred" {
		if (mode === "gbuffer") {
			return "deferred";
		}
		return mode === "color" ? "single" : mode;
	}

	private async _getSceneShaderModule(): Promise<IShaderModule> {
		const directiveTag = this._getDirectiveCacheTag();
		if (
			this._sceneShaderModule &&
			this._sceneShaderDirectiveTag === directiveTag
		) {
			return this._sceneShaderModule;
		}
		if (!this._sceneShaderModule || this._sceneShaderDirectiveTag !== directiveTag) {
			const shader = await ShaderSource.load("webgpu.scene.composite");
			this._sceneShaderModule = await this._backend.createShaderModule({
				code: shader.code,
				sourceMap: shader.sourceMap,
				label: "WebGPUSceneShader",
				language: "wgsl",
				stage: "unknown",
				sourceKind: "builtin-scene",
			});
			this._sceneShaderDirectiveTag = this._getDirectiveCacheTag();
		}

		return this._sceneShaderModule;
	}

	private async _getEnvironmentShaderModule(): Promise<IShaderModule> {
		const directiveTag = this._getDirectiveCacheTag();
		if (
			this._environmentShaderModule &&
			this._environmentShaderDirectiveTag === directiveTag
		) {
			return this._environmentShaderModule;
		}
		if (!this._environmentShaderModule || this._environmentShaderDirectiveTag !== directiveTag) {
			const shader = await ShaderSource.load("webgpu.environment.composite");
			this._environmentShaderModule = await this._backend.createShaderModule({
				code: shader.code,
				sourceMap: shader.sourceMap,
				label: "WebGPUEnvironmentShader",
				language: "wgsl",
				stage: "unknown",
				sourceKind: "builtin-environment",
			});
			this._environmentShaderDirectiveTag = this._getDirectiveCacheTag();
		}

		return this._environmentShaderModule;
	}

	private async _getDeferredLightingShaderModule(): Promise<IShaderModule> {
		const directiveTag = this._getDirectiveCacheTag();
		if (
			this._deferredLightingShaderModule &&
			this._deferredLightingShaderDirectiveTag === directiveTag
		) {
			return this._deferredLightingShaderModule;
		}
		if (
			!this._deferredLightingShaderModule ||
			this._deferredLightingShaderDirectiveTag !== directiveTag
		) {
			const shader = await ShaderSource.load("webgpu.deferredLighting.composite");
			this._deferredLightingShaderModule =
				await this._backend.createShaderModule({
					code: shader.code,
					sourceMap: shader.sourceMap,
					label: "WebGPUDeferredLightingShader",
					language: "wgsl",
					stage: "unknown",
					sourceKind: "builtin-scene",
				});
			this._deferredLightingShaderDirectiveTag =
				this._getDirectiveCacheTag();
		}

		return this._deferredLightingShaderModule;
	}

	private async _getPlanarReflectionCompositeShaderModule():
		Promise<IShaderModule> {
		const directiveTag = this._getDirectiveCacheTag();
		if (
			this._planarReflectionCompositeShaderModule &&
			this._planarReflectionCompositeDirectiveTag === directiveTag
		) {
			return this._planarReflectionCompositeShaderModule;
		}
		if (
			!this._planarReflectionCompositeShaderModule ||
			this._planarReflectionCompositeDirectiveTag !== directiveTag
		) {
			const shader = await ShaderSource.load(
				"webgpu.utility.planarReflectionComposite.composite"
			);
			this._planarReflectionCompositeShaderModule =
				await this._backend.createShaderModule({
					code: shader.code,
					sourceMap: shader.sourceMap,
					label: "WebGPUPlanarReflectionCompositeShader",
					language: "wgsl",
					stage: "unknown",
					sourceKind: "builtin-scene",
				});
			this._planarReflectionCompositeDirectiveTag =
				this._getDirectiveCacheTag();
		}

		return this._planarReflectionCompositeShaderModule;
	}

	private _isWarnMode(): boolean {
		const shaderRuntime = this._getShaderRuntime();
		if (!shaderRuntime || typeof shaderRuntime.getMode !== "function") {
			return false;
		}
		return shaderRuntime.getMode() === "warn";
	}

	private _getShaderRuntimeRevision(): number {
		const shaderRuntime = this._getShaderRuntime();
		if (!shaderRuntime || typeof shaderRuntime.revision !== "number") {
			return 0;
		}
		return shaderRuntime.revision;
	}

	private _getShaderRuntime():
		| {
				revision?: number;
				getMode?: () => "strict" | "warn" | "silent";
				onDidChange?: (listener: () => void) => () => void;
		  }
		| null {
		const backend = this._backend as unknown as {
			shaderRuntime?: {
				revision?: number;
				getMode?: () => "strict" | "warn" | "silent";
				onDidChange?: (listener: () => void) => () => void;
			};
		};
		return backend.shaderRuntime ?? null;
	}

	private _getDirectiveCacheTag(): string {
		const backend = this._backend as unknown as {
			getShaderDirectiveCacheTag?: () => string;
		};
		if (typeof backend.getShaderDirectiveCacheTag === "function") {
			return backend.getShaderDirectiveCacheTag();
		}
		return "none";
	}

	private _supportsRuntimeInjects(): boolean {
		return this._getDirectiveCacheTag() !== "none";
	}
}

function isMaterialMask(material: Material): boolean {
	return (material.alphaMode ?? AlphaMode.Opaque) === AlphaMode.Mask;
}
