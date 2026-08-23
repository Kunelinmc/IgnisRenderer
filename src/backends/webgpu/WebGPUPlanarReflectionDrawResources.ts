import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../../core/types";
import { ShaderSource } from "../../shaders/ShaderSource";
import {
	TextureFormat,
	type IRenderPipeline,
	type IShaderModule,
} from "../types";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type { DrawPacket } from "../../pipeline/types";
import type { WebGPUDrawResourceAssembler } from "./WebGPUDrawResourceAssembler";
import { destroyUniqueWebGPUHandles } from "./WebGPUManagedResourceUtils";
import { readWebGPUShaderRuntimeView } from "./WebGPUMaterialPipelineResolver";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";
import type {
	WebGPUDrawPipelineProvider,
	WebGPUDrawPipelineRequest,
	WebGPUDrawResourceOptions,
	WebGPUDrawResources,
	WebGPUPreparedFrameResources,
} from "./WebGPUResourceContracts";

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

/** @internal Feature-owned planar-reflection composite draw pipelines. */
export class WebGPUPlanarReflectionDrawResources
	implements WebGPUDrawPipelineProvider {
	private _shaderModule: IShaderModule | null = null;
	private _shaderDirectiveTag = "";
	private _pipelines = new Map<string, IRenderPipeline>();

	public constructor(
		private readonly _backend: WebGPUDeviceResourceHost,
		private readonly _layouts: Pick<WebGPUPipelineLayouts, "planarReflectionPipelineLayout">,
		private readonly _draws: WebGPUDrawResourceAssembler,
	) {}

	public getDrawResources(
		packet: DrawPacket,
		frameResources: WebGPUPreparedFrameResources,
		options: WebGPUDrawResourceOptions,
	): Promise<WebGPUDrawResources[] | null> {
		return this._draws.getDrawResources(packet, frameResources, options, this);
	}

	public async resolvePipeline(
		request: WebGPUDrawPipelineRequest,
	): Promise<IRenderPipeline> {
		if (request.pass.drawMode !== "planar-reflection-composite") {
			throw new Error(
				"WebGPU planar-reflection resources require the composite draw mode.",
			);
		}
		const sampleCount = resolveSampleCount(request.sampleCount);
		const state = request.materialState;
		const cacheKey = [
			state.pipelineKey,
			state.shaderCacheKey,
			request.pass.pipelineKeyPart,
			request.topology,
			request.geometryLayout.layoutKey,
			TextureFormat.Depth32Float,
			sampleCount,
		].join("|");
		const cached = this._pipelines.get(cacheKey);
		if (cached) return cached;

		const shaderModule = await this._getShaderModule();
		const effectiveTopology = state.wireframe
			? "line-list"
			: request.topology;
		const triangleTopology =
			effectiveTopology === DEFAULT_PRIMITIVE_DRAW_TOPOLOGY;
		const pipeline = await this._backend.createPipeline({
			layout: this._layouts.planarReflectionPipelineLayout,
			label:
				`WebGPUPlanarReflectionCompositePipeline_${state.pipelineKey}_` +
				request.pass.sceneTargetMode,
			vertex: {
				module: shaderModule,
				entryPoint: "vsMain",
				buffers: [...request.geometryLayout.sceneVertexLayouts],
			},
			fragment: {
				module: shaderModule,
				entryPoint: "fsMain",
				targets: [
					{
						format: TextureFormat.RGBA16Float,
						blend: PLANAR_REFLECTION_BLEND_STATE,
					},
					{ format: TextureFormat.R8Unorm },
				],
			},
			primitive: {
				topology: effectiveTopology as any,
				cullMode:
					state.wireframe || !triangleTopology
						? "none"
						: state.cullMode as any,
				frontFace: request.pass.frontFace,
			},
			depthStencil: {
				format: TextureFormat.Depth32Float,
				depthWriteEnabled: false,
				depthCompare: "less-equal",
			},
			sampleCount,
		} as any);
		const winner = this._pipelines.get(cacheKey);
		if (winner) {
			destroyUniqueWebGPUHandles(
				[pipeline],
				"pipeline",
				"WebGPUPlanarReflectionDrawResources",
			);
			return winner;
		}
		this._pipelines.set(cacheKey, pipeline);
		return pipeline;
	}

	public onShaderRuntimeChanged(): void {
		destroyUniqueWebGPUHandles(
			[...this._pipelines.values()],
			"pipeline",
			"WebGPUPlanarReflectionDrawResources",
		);
		destroyUniqueWebGPUHandles(
			[this._shaderModule],
			"shader module",
			"WebGPUPlanarReflectionDrawResources",
		);
		this._pipelines.clear();
		this._shaderModule = null;
		this._shaderDirectiveTag = "";
	}

	public destroy(): void {
		this.onShaderRuntimeChanged();
	}

	private async _getShaderModule(): Promise<IShaderModule> {
		const directiveTag =
			readWebGPUShaderRuntimeView(this._backend).directiveCacheTag;
		if (this._shaderModule && this._shaderDirectiveTag === directiveTag) {
			return this._shaderModule;
		}
		const shader = await ShaderSource.load(
			"webgpu.utility.planarReflectionComposite",
		);
		const module = await this._backend.createShaderModule({
			code: shader.source.code,
			sourceMap: shader.source.sourceMap,
			label: "WebGPUPlanarReflectionCompositeShader",
			language: "wgsl",
			stage: "unknown",
			sourceKind: "builtin-scene",
		});
		if (this._shaderModule) {
			destroyUniqueWebGPUHandles(
				[this._shaderModule],
				"shader module",
				"WebGPUPlanarReflectionDrawResources",
			);
		}
		this._shaderModule = module;
		this._shaderDirectiveTag = directiveTag;
		return module;
	}
}

function resolveSampleCount(sampleCount: number): number {
	if (!Number.isFinite(sampleCount)) {
		throw new Error("WebGPU planar-reflection sampleCount must be finite.");
	}
	return Math.max(1, Math.floor(sampleCount));
}
