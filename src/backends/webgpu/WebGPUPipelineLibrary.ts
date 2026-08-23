import { createInlineCompositeShaderSource } from "../../shaders/runtime";
import { ShaderSource } from "../../shaders/ShaderSource";
import { createWebGPUMaterialUniformData } from "./";
import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../../core/types";
import { TextureFormat, type ColorTargetState } from "../types";
import type { PrimitiveDrawTopology } from "../../core/types";
import { AlphaMode, type Material } from "../../materials/Material";
import type { IRenderPipeline, IShaderModule } from "../types";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";
import type { WebGPUGeometryHandle } from "./WebGPUGeometryRegistry";
import type { WebGPUMaterialUniformData } from "./types";
import { packWebGPUVertexGeometry } from "./WebGPUGeometryPacking";
import { Logger } from "../../foundation/Logger";
import { GBufferSlot, WEBGPU_MRT_COLOR_FORMATS } from "./constants";
import {
	resolveWebGPUScenePassDescriptor,
	type WebGPUScenePassDescriptor,
	type WebGPUScenePipelineDrawMode,
	type WebGPUSceneTargetMode,
	type WebGPUTransparentPipelineMode,
} from "./WebGPUScenePassDescriptors";
import {
	WebGPUMaterialPipelineResolver,
	readWebGPUShaderRuntimeView,
	type WebGPUMaterialPipelineState,
	type WebGPUShaderRuntimeView,
} from "./WebGPUMaterialPipelineResolver";
import { destroyUniqueWebGPUHandles } from "./WebGPUManagedResourceUtils";
import type {
	WebGPUDrawPipelineProvider,
	WebGPUDrawPipelineRequest,
} from "./WebGPUResourceContracts";

export type {
	WebGPUScenePipelineDrawMode,
	WebGPUSceneTargetMode,
	WebGPUTransparentPipelineMode,
} from "./WebGPUScenePassDescriptors";
const COLOR_WRITE_NONE = 0;
const FALLBACK_GEOMETRY_LAYOUT = packWebGPUVertexGeometry(
	{
		positions: new Float32Array(3),
		indices: new Uint32Array(0),
	},
	1
);
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

interface WebGPUSceneProgram {
	vertexModule: IShaderModule;
	fragmentModule: IShaderModule;
	vertexEntryPoint: string;
	fragmentEntryPoint: string;
	fragmentTargetMode: "single" | "mrt" | "deferred" | null;
}

interface GeometryLayout {
	readonly layoutKey: string;
	readonly sceneVertexLayouts: WebGPUGeometryHandle["sceneVertexLayouts"];
}

interface WebGPUPipelineLibraryOptions {
	listenToShaderRuntime?: boolean;
}

export class WebGPUPipelineLibrary implements WebGPUDrawPipelineProvider {
	private _backend: WebGPUDeviceResourceHost;
	private _layouts: WebGPUPipelineLayouts;
	private _disposeShaderRuntimeListener: (() => void) | null = null;
	private _sceneShaderModule: IShaderModule | null = null;
	private _sceneShaderDirectiveTag = "";
	private _customShaderModuleCache = new Map<string, IShaderModule>();
	private _materialPipelineCache = new WeakMap<Material, Map<string, IRenderPipeline>>();
	private _pipelineCache = new Map<string, IRenderPipeline>();
	private _earlyZPrepassCache = new Map<string, IRenderPipeline>();
	private _earlyZPrepassInFlight = new Map<
		string,
		Promise<IRenderPipeline | null>
	>();
	private _materialPipelineResolver = new WebGPUMaterialPipelineResolver();
	private _shaderCacheGeneration = 0;

	constructor(
		backend: WebGPUDeviceResourceHost,
		layouts: WebGPUPipelineLayouts,
		options: WebGPUPipelineLibraryOptions = {}
	) {
		this._backend = backend;
		this._layouts = layouts;
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
		this._shaderCacheGeneration++;
		this._materialPipelineResolver.clear();
		destroyUniqueWebGPUHandles([
			...this._pipelineCache.values(),
			...this._earlyZPrepassCache.values(),
		], "pipeline", "WebGPUPipelineLibrary");
		destroyUniqueWebGPUHandles([
			...this._customShaderModuleCache.values(),
			this._sceneShaderModule,
		], "shader module", "WebGPUPipelineLibrary");
		this._sceneShaderModule = null;
		this._sceneShaderDirectiveTag = "";
		this._customShaderModuleCache.clear();
		this._materialPipelineCache = new WeakMap<Material, Map<string, IRenderPipeline>>();
		this._pipelineCache.clear();
		this._earlyZPrepassCache.clear();
		this._earlyZPrepassInFlight.clear();
	}

	public async init(): Promise<void> {
		await this._getSceneShaderModule();
	}

	public resolvePipeline(
		request: WebGPUDrawPipelineRequest,
	): Promise<IRenderPipeline | null> {
		return request.pass.drawMode === "early-z-prepass"
			? this._resolveEarlyZPipeline(request)
			: this._resolveScenePipeline(request);
	}

	private async _resolveScenePipeline(
		request: WebGPUDrawPipelineRequest,
	): Promise<IRenderPipeline> {
		const { materialState, pass, topology, geometryLayout } = request;
		const sampleCount = this._resolveSampleCount(
			pass.sceneTargetMode,
			request.sampleCount,
		);
		const depthFormat = this._resolveSceneDepthFormat(pass.sceneTargetMode);
		const cacheKey =
			`${materialState.pipelineKey}|${pass.pipelineKeyPart}|` +
			`${materialState.shaderCacheKey}|topology:${topology}|` +
			`layout:${geometryLayout.layoutKey}|depth:${depthFormat}|msaa:${sampleCount}`;
		const cached = this._pipelineCache.get(cacheKey);
		if (cached) return cached;
		const pipeline = await this._createPipeline(
			materialState,
			pass,
			topology,
			geometryLayout,
			sampleCount,
		);
		const winner = this._pipelineCache.get(cacheKey);
		if (winner) {
			destroyUniqueWebGPUHandles(
				[pipeline],
				"pipeline",
				"WebGPUPipelineLibrary",
			);
			return winner;
		}
		this._pipelineCache.set(cacheKey, pipeline);
		return pipeline;
	}

	public async getPipeline(
		material: Material,
		mode: WebGPUSceneTargetMode = "single",
		isWireframe = false,
		topology: PrimitiveDrawTopology = DEFAULT_PRIMITIVE_DRAW_TOPOLOGY,
		transparentMode: WebGPUTransparentPipelineMode = "default",
		drawMode: WebGPUScenePipelineDrawMode = "default",
		sampleCount: number,
		deferredGBufferLayout: "base" | "extended" = "extended",
		geometry?: WebGPUGeometryHandle,
		resolvedMaterialData?: WebGPUMaterialUniformData,
	): Promise<IRenderPipeline> {
		if (!resolvedMaterialData) material.refreshRevision();
		const geometryLayout = resolveGeometryLayout(geometry);
		const descriptor = resolveWebGPUScenePassDescriptor(
			mode,
			transparentMode,
			drawMode,
			deferredGBufferLayout
		);
		const resolvedSampleCount = this._resolveSampleCount(
			descriptor.sceneTargetMode,
			sampleCount
		);
		const depthFormat = this._resolveSceneDepthFormat(
			descriptor.sceneTargetMode
		);
		const materialData =
			resolvedMaterialData ?? createWebGPUMaterialUniformData(material, isWireframe);
		const pipelineState = this._resolveMaterialPipelineState(
			material,
			materialData,
			isWireframe,
			descriptor,
			"scene",
		);
		const { pipelineKey } = pipelineState;
		const initialShaderKey = pipelineState.shaderCacheKey;

		const initialCacheKey =
			`${pipelineKey}|${descriptor.pipelineKeyPart}|${initialShaderKey}` +
			`|topology:${topology}|layout:${geometryLayout.layoutKey}|` +
			`depth:${depthFormat}|msaa:${resolvedSampleCount}`;
		let materialVariants = this._materialPipelineCache.get(material);
		if (!materialVariants) {
			materialVariants = new Map();
			this._materialPipelineCache.set(material, materialVariants);
		}
		const materialCached = materialVariants.get(initialCacheKey);
		if (materialCached) {
			materialVariants.delete(initialCacheKey);
			materialVariants.set(initialCacheKey, materialCached);
			return materialCached;
		}
		let pipeline = this._pipelineCache.get(initialCacheKey);
		if (!pipeline) {
			pipeline = await this._createPipeline(
				pipelineState,
				descriptor,
				topology,
				geometryLayout,
				resolvedSampleCount,
			);
		}
		const finalShaderKey = pipelineState.shaderCacheKey;
		const finalCacheKey =
			`${pipelineKey}|${descriptor.pipelineKeyPart}|${finalShaderKey}` +
			`|topology:${topology}|layout:${geometryLayout.layoutKey}|` +
			`depth:${depthFormat}|msaa:${resolvedSampleCount}`;
		const cachedFinalPipeline = this._pipelineCache.get(finalCacheKey);
		if (cachedFinalPipeline) {
			if (pipeline !== cachedFinalPipeline) {
				destroyUniqueWebGPUHandles(
					[pipeline],
					"pipeline",
					"WebGPUPipelineLibrary",
				);
			}
			pipeline = cachedFinalPipeline;
		} else {
			this._pipelineCache.set(finalCacheKey, pipeline);
		}
		if (finalCacheKey !== initialCacheKey) {
			this._pipelineCache.delete(initialCacheKey);
		}

		materialVariants.delete(initialCacheKey);
		materialVariants.set(finalCacheKey, pipeline);
		while (materialVariants.size > 32) {
			const oldest = materialVariants.keys().next().value as string | undefined;
			if (!oldest) break;
			materialVariants.delete(oldest);
		}

		return pipeline;
	}

	private async _createPipeline(
		pipelineState: WebGPUMaterialPipelineState,
		descriptor: WebGPUScenePassDescriptor,
		topology: PrimitiveDrawTopology,
		geometry: GeometryLayout,
		sampleCount: number,
	): Promise<IRenderPipeline> {
		const mode = descriptor.sceneTargetMode;
		const resolvedSampleCount = this._resolveSampleCount(mode, sampleCount);
		const { pipelineKey } = pipelineState;
		const sceneProgram = await this._resolveSceneProgram(
			pipelineState,
			descriptor
		);
		const isTransparent = pipelineState.transparent;
		const usesTransmission = pipelineState.usesTransmission;
		const isTransmissionCapture =
			descriptor.transparentMode === "transmission-capture";
		const fragmentTargets = this._createSceneFragmentTargets(
			descriptor,
			isTransparent,
			usesTransmission
		);
		this._disableUnwrittenFragmentTargets(
			fragmentTargets,
			sceneProgram.fragmentTargetMode
		);
		const depthFormat = this._resolveSceneDepthFormat(mode);
		const depthWrite = pipelineState.depthWrite;
		const isEarlyZColor =
			descriptor.depthStateMode === "early-z-color" &&
			!isTransparent &&
			depthWrite;

		const effectiveTopology = pipelineState.wireframe ? "line-list" : topology;
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
				buffers: [...geometry.sceneVertexLayouts],
			},
			fragment: {
				module: sceneProgram.fragmentModule,
				entryPoint: sceneProgram.fragmentEntryPoint,
				targets: fragmentTargets as any,
			},
			primitive: {
				topology: effectiveTopology as any,
				cullMode:
					pipelineState.wireframe || !triangleTopology ?
						"none" : (pipelineState.cullMode as any),
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
			sampleCount: resolvedSampleCount,
		} as any);
	}

	public async getEarlyZPrepassPipeline(
		material: Material,
		mode: WebGPUSceneTargetMode = "single",
		isWireframe = false,
		topology: PrimitiveDrawTopology = DEFAULT_PRIMITIVE_DRAW_TOPOLOGY,
		sampleCount: number,
		geometry?: WebGPUGeometryHandle,
		resolvedMaterialData?: WebGPUMaterialUniformData,
	): Promise<IRenderPipeline | null> {
		if (!resolvedMaterialData) material.refreshRevision();
		const geometryLayout = resolveGeometryLayout(geometry);
		const descriptor = resolveWebGPUScenePassDescriptor(
			mode,
			"default",
			"early-z-prepass"
		);
		const materialData =
			resolvedMaterialData ?? createWebGPUMaterialUniformData(material, isWireframe);
		let pipelineState: WebGPUMaterialPipelineState;
		try {
			pipelineState = this._resolveMaterialPipelineState(
				material,
				materialData,
				isWireframe,
				descriptor,
				"early-z",
			);
		} catch (error) {
			this._warnShaderMaterialDepthPrepassSkipped(
				material.name,
				material instanceof Object && "shaderId" in material
					? Number((material as { shaderId: number }).shaderId)
					: null,
				String(error),
			);
			return null;
		}
		return this._resolveEarlyZPipeline({
			materialState: pipelineState,
			pass: descriptor,
			topology,
			geometryLayout,
			sampleCount,
		});
	}

	private _resolveEarlyZPipeline(
		request: WebGPUDrawPipelineRequest,
	): Promise<IRenderPipeline | null> {
		const { materialState, pass, topology, geometryLayout } = request;
		if (materialState.transparent || !materialState.depthWrite) {
			return Promise.resolve(null);
		}
		const isMask = materialState.alphaMode === AlphaMode.Mask;
		const sampleCount = this._resolveSampleCount(
			pass.sceneTargetMode,
			request.sampleCount,
		);
		const depthFormat = this._resolveSceneDepthFormat(pass.sceneTargetMode);
		const cacheKey =
			`earlyz|${materialState.pipelineKey}|${pass.pipelineKeyPart}|` +
			`mask:${isMask ? 1 : 0}|wire:${materialState.wireframe ? 1 : 0}|` +
			`topology:${topology}|layout:${geometryLayout.layoutKey}|` +
			`depth:${depthFormat}|msaa:${sampleCount}|` +
			`shader:${materialState.shaderCacheKey}`;
		const cached = this._earlyZPrepassCache.get(cacheKey);
		if (cached) return Promise.resolve(cached);
		const pending = this._earlyZPrepassInFlight.get(cacheKey);
		if (pending) return pending;

		const generation = this._shaderCacheGeneration;
		const creationPromise = (async (): Promise<IRenderPipeline | null> => {
			const cachedAfterDispatch = this._earlyZPrepassCache.get(cacheKey);
			if (cachedAfterDispatch) return cachedAfterDispatch;
			const effectiveTopology = materialState.wireframe ? "line-list" : topology;
			const triangleTopology =
				effectiveTopology === DEFAULT_PRIMITIVE_DRAW_TOPOLOGY;
			const resolved = await this._resolveEarlyZPrepassProgram(
				materialState,
				isMask,
			);
			if (!resolved) return null;
			const desc: any = {
				layout: this._resolvePipelineLayout(pass),
				label:
					`WebGPUSceneEarlyZPipeline_${materialState.pipelineKey}_` +
					pass.sceneTargetMode,
				vertex: {
					module: resolved.vertexModule,
					entryPoint: resolved.vertexEntryPoint,
					buffers: [...geometryLayout.sceneVertexLayouts],
				},
				primitive: {
					topology: effectiveTopology as any,
					cullMode:
						materialState.wireframe || !triangleTopology
							? "none"
							: materialState.cullMode as any,
					frontFace: pass.frontFace,
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
			if (generation !== this._shaderCacheGeneration) {
				destroyUniqueWebGPUHandles(
					[pipeline],
					"pipeline",
					"WebGPUPipelineLibrary",
				);
				return this._resolveEarlyZPipeline(request);
			}
			const winner = this._earlyZPrepassCache.get(cacheKey);
			if (winner) {
				destroyUniqueWebGPUHandles(
					[pipeline],
					"pipeline",
					"WebGPUPipelineLibrary",
				);
				return winner;
			}
			this._earlyZPrepassCache.set(cacheKey, pipeline);
			return pipeline;
		})();
		this._earlyZPrepassInFlight.set(cacheKey, creationPromise);
		return creationPromise.finally(() => {
			if (this._earlyZPrepassInFlight.get(cacheKey) === creationPromise) {
				this._earlyZPrepassInFlight.delete(cacheKey);
			}
		});
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

		if (fragmentTargetKind === "gbuffer" || fragmentTargetKind === "gbuffer-base") {
			const targets: ColorTargetState[] = [];
			targets[GBufferSlot.AlbedoAlpha] = {
				format: TextureFormat.RGBA8Unorm,
			};
			targets[GBufferSlot.NormalRoughMetal] = {
				format: TextureFormat.RGBA8Unorm,
			};
			targets[GBufferSlot.EmissiveOcclusion] = {
				format: TextureFormat.RGBA16Float,
			};
			targets[GBufferSlot.MotionDepth] = {
				format: TextureFormat.RGBA16Float,
			};
			if (fragmentTargetKind === "gbuffer-base") {
				return targets;
			}
			targets[GBufferSlot.Specular] = {
				format: TextureFormat.RGBA16Float,
			};
			targets[GBufferSlot.CoatSheen] = {
				format: TextureFormat.RGBA16Float,
			};
			targets[GBufferSlot.SheenReflectance] = {
				format: TextureFormat.RGBA8Unorm,
			};
			return targets;
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
				format: WEBGPU_MRT_COLOR_FORMATS[0],
				blend: colorBlend,
			},
			{
				format: WEBGPU_MRT_COLOR_FORMATS[1],
				writeMask: isTransparent ? COLOR_WRITE_NONE : undefined,
			},
			{
				format: WEBGPU_MRT_COLOR_FORMATS[2],
				writeMask: isTransparent ? COLOR_WRITE_NONE : undefined,
			},
			{
				format: WEBGPU_MRT_COLOR_FORMATS[3],
				writeMask: isTransparent ? COLOR_WRITE_NONE : undefined,
			},
			{
				format: WEBGPU_MRT_COLOR_FORMATS[4],
				blend: motionBlend,
			},
		];
	}

	private _disableUnwrittenFragmentTargets(
		targets: ColorTargetState[],
		fragmentTargetMode: "single" | "mrt" | "deferred" | null
	): void {
		if (fragmentTargetMode !== "single") {
			return;
		}
		for (let index = 1; index < targets.length; index++) {
			targets[index].writeMask = COLOR_WRITE_NONE;
		}
	}

	private async _resolveSceneProgram(
		pipelineState: WebGPUMaterialPipelineState,
		descriptor: WebGPUScenePassDescriptor
	): Promise<WebGPUSceneProgram> {
		if (pipelineState.program.kind === "builtin") {
			this._warnMaterialPipelineFallback(pipelineState);
			const shaderModule = await this._getSceneShaderModule();
			return {
				vertexModule: shaderModule,
				fragmentModule: shaderModule,
				vertexEntryPoint: "vsMain",
				fragmentEntryPoint:
					descriptor.shaderEntryMode === "gbuffer" ? "fsMainGBuffer"
					: descriptor.shaderEntryMode === "gbuffer-base" ? "fsMainGBufferBase"
					: descriptor.shaderEntryMode === "oit" ? "fsMainOIT"
					: descriptor.shaderEntryMode === "transmission-capture" ?
						"fsMainTransmissionCapture"
					: descriptor.shaderEntryMode === "mrt" ? "fsMain"
					:	"fsMainSingle",
				fragmentTargetMode: null,
			};
		}

		try {
			const program = pipelineState.program.regularProgram;
			if (!program) {
				throw new Error("Custom scene shader program was not resolved.");
			}
			const shaderCacheKey = pipelineState.program.cacheKey;
			const mode = descriptor.sceneTargetMode;
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
				fragmentTargetMode: program.fragmentTargetMode,
			};
		} catch (error) {
			if (pipelineState.shaderRuntime.mode !== "warn") {
				throw error;
			}
			const shaderId = pipelineState.diagnostic.shaderId ?? "unknown";
			const key = `webgpu-shader-material-compile-failed-${shaderId}`;
			Logger.warn(
				`[${key}] ShaderMaterial ${pipelineState.diagnostic.materialName} custom WebGPU shader compile failed; using built-in scene shader. ${String(error)}`,
				{ scope: "WebGPUPipelineLibrary", onceKey: key }
			);
			const shaderModule = await this._getSceneShaderModule();
			return {
				vertexModule: shaderModule,
				fragmentModule: shaderModule,
				vertexEntryPoint: "vsMain",
				fragmentEntryPoint:
					descriptor.shaderEntryMode === "gbuffer" ? "fsMainGBuffer"
					: descriptor.shaderEntryMode === "gbuffer-base" ? "fsMainGBufferBase"
					: descriptor.shaderEntryMode === "oit" ? "fsMainOIT"
					: descriptor.shaderEntryMode === "transmission-capture" ?
						"fsMainTransmissionCapture"
					: descriptor.shaderEntryMode === "mrt" ? "fsMain"
					:	"fsMainSingle",
				fragmentTargetMode: null,
			};
		}
	}

	private _warnMaterialPipelineFallback(
		pipelineState: WebGPUMaterialPipelineState,
	): void {
		const reason = pipelineState.diagnostic.fallbackReason;
		if (!reason) return;
		const shaderId = pipelineState.diagnostic.shaderId ?? "unknown";
		const key = `webgpu-shader-material-compile-failed-${shaderId}`;
		Logger.warn(
			`[${key}] ShaderMaterial ${pipelineState.diagnostic.materialName} custom WebGPU shader compile failed; using built-in scene shader. ${reason}`,
			{ scope: "WebGPUPipelineLibrary", onceKey: key },
		);
	}

	private async _resolveEarlyZPrepassProgram(
		pipelineState: WebGPUMaterialPipelineState,
		isMask: boolean
	): Promise<{
		vertexModule: IShaderModule;
		vertexEntryPoint: string;
		fragmentModule: IShaderModule | null;
		fragmentEntryPoint: string | null;
	} | null> {
		if (pipelineState.program.kind === "builtin") {
			const shaderModule = await this._getSceneShaderModule();
			return {
				vertexModule: shaderModule,
				vertexEntryPoint: "vsMain",
				fragmentModule: isMask ? shaderModule : null,
				fragmentEntryPoint: isMask ? "fsMainDepthMask" : null,
			};
		}

		try {
			const shaderCacheKey = pipelineState.program.cacheKey;
			if (isMask) {
				const depthProgram = pipelineState.program.depthPrepassProgram;
				if (!depthProgram) {
					this._warnShaderMaterialDepthPrepassSkipped(
						pipelineState.diagnostic.materialName,
						pipelineState.diagnostic.shaderId,
						"missing depth pre-pass fragment contract."
					);
					return null;
				}
				const vertexModule = await this._getCustomShaderModule(
					`${shaderCacheKey}:depth-prepass:vertex`,
					depthProgram.vertexCode,
					`WebGPUShaderMaterialDepthVertex_${shaderCacheKey}`,
					"vertex",
					depthProgram.vertexEntryPoint
				);
				const fragmentModule = await this._getCustomShaderModule(
					`${shaderCacheKey}:depth-prepass:fragment`,
					depthProgram.fragmentCode,
					`WebGPUShaderMaterialDepthFragment_${shaderCacheKey}`,
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

			const regularProgram = pipelineState.program.regularProgram;
			if (!regularProgram) {
				throw new Error("Custom early-Z vertex program was not resolved.");
			}
			const vertexModule = await this._getCustomShaderModule(
				`${shaderCacheKey}:depth-prepass:vertex`,
				regularProgram.vertexCode,
				`WebGPUShaderMaterialDepthVertex_${shaderCacheKey}`,
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
			this._warnShaderMaterialDepthPrepassSkipped(
				pipelineState.diagnostic.materialName,
				pipelineState.diagnostic.shaderId,
				String(error),
			);
			return null;
		}
	}

	private _warnShaderMaterialDepthPrepassSkipped(
		materialName: string,
		shaderId: number | null,
		reason: string
	): void {
		const key = `webgpu-earlyz-shader-material-skip-${shaderId ?? "unknown"}`;
		Logger.warn(
			`[${key}] ShaderMaterial ${materialName} early-z pre-pass is skipped: ${reason}`,
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
		sampleCount: number
	): number {
		if (mode !== "mrt" && mode !== "color") {
			return 1;
		}
		if (!Number.isFinite(sampleCount)) {
			throw new Error("WebGPU scene pipeline sampleCount must be a finite number.");
		}
		return Math.max(1, Math.floor(sampleCount));
	}

	private _resolveShaderMaterialMode(
		mode: WebGPUSceneTargetMode
	): "single" | "mrt" | "deferred" {
		if (mode === "gbuffer") {
			return "deferred";
		}
		return mode === "color" ? "single" : mode;
	}

	private _resolveMaterialPipelineState(
		material: Material,
		materialData: WebGPUMaterialUniformData,
		wireframe: boolean,
		descriptor: WebGPUScenePassDescriptor,
		purpose: "scene" | "early-z",
	): WebGPUMaterialPipelineState {
		return this._materialPipelineResolver.resolve(
			material,
			materialData,
			wireframe,
			this._resolveShaderMaterialMode(descriptor.sceneTargetMode),
			purpose,
			this._getShaderRuntimeView(),
		);
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
			const shader = await ShaderSource.load("webgpu.scene");
			this._sceneShaderModule = await this._backend.createShaderModule({
				code: shader.source.code,
				sourceMap: shader.source.sourceMap,
				label: "WebGPUSceneShader",
				language: "wgsl",
				stage: "unknown",
				sourceKind: "builtin-scene",
			});
			this._sceneShaderDirectiveTag = this._getDirectiveCacheTag();
		}

		return this._sceneShaderModule;
	}

	private _getShaderRuntime():
		| {
				revision?: number;
				getMode?: () => "strict" | "warn" | "silent";
				onDidChange?: (listener: () => void) => () => void;
		  }
		| null {
		return this._backend.shaderRuntime ?? null;
	}

	private _getShaderRuntimeView(): WebGPUShaderRuntimeView {
		return readWebGPUShaderRuntimeView(this._backend);
	}

	private _getDirectiveCacheTag(): string {
		if (typeof this._backend.getShaderDirectiveCacheTag === "function") {
			return this._backend.getShaderDirectiveCacheTag();
		}
		return "none";
	}

}

function resolveGeometryLayout(
	geometry?: WebGPUGeometryHandle
): GeometryLayout {
	if (geometry) return geometry;
	return {
		layoutKey: FALLBACK_GEOMETRY_LAYOUT.layoutKey,
		sceneVertexLayouts: FALLBACK_GEOMETRY_LAYOUT.sceneLayouts,
	};
}
