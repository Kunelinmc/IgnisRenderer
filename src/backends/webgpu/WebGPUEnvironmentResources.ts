import { ShaderSource } from "../../shaders/ShaderSource";
import { TextureFormat } from "../../core/TextureFormat";
import {
	type IRenderPipeline,
	type IShaderModule,
} from "../types";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import { destroyUniqueWebGPUHandles } from "./WebGPUManagedResourceUtils";
import { readWebGPUShaderRuntimeView } from "./WebGPUMaterialPipelineResolver";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";
import type { WebGPUSceneTargetMode } from "./WebGPUScenePassDescriptors";
import {
	toShaderCompileError,
	type WarmupPhaseCounters,
} from "../../pipeline/WarmupPlanner";
import type { WebGPUFeatureWarmupContributor } from "./WebGPUFeatureWarmup";

export interface WebGPUEnvironmentWarmupRequest {
	readonly modes: readonly WebGPUSceneTargetMode[];
	readonly sampleCount: number;
	yieldIfNeeded(): Promise<void>;
}

/** @internal Shared environment pipeline resources used by scene capture paths. */
export class WebGPUEnvironmentResources
	implements WebGPUFeatureWarmupContributor<WebGPUEnvironmentWarmupRequest> {
	private _shaderModule: IShaderModule | null = null;
	private _shaderDirectiveTag = "";
	private _pipelines = new Map<string, IRenderPipeline>();

	public constructor(
		private readonly _backend: WebGPUDeviceResourceHost,
		private readonly _layouts: Pick<WebGPUPipelineLayouts, "environmentPipelineLayout">,
	) {}

	public async getPipeline(
		mode: WebGPUSceneTargetMode,
		sampleCount: number,
	): Promise<IRenderPipeline> {
		const resolvedSampleCount = resolveSampleCount(mode, sampleCount);
		const depthFormat = resolveDepthFormat(this._backend, mode);
		const cacheKey = `${mode}|depth:${depthFormat}|msaa:${resolvedSampleCount}`;
		const cached = this._pipelines.get(cacheKey);
		if (cached) return cached;

		const shaderModule = await this._getShaderModule();
		const targetFormat =
			mode === "mrt" || mode === "gbuffer" || mode === "color"
				? TextureFormat.RGBA16Float
				: this._backend.canvasFormat;
		const pipeline = await this._backend.createPipeline({
			layout: this._layouts.environmentPipelineLayout,
			label: `WebGPUEnvironmentPipeline_${mode}`,
			vertex: { module: shaderModule, entryPoint: "vsMain" },
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
			sampleCount: resolvedSampleCount,
		} as any);
		const winner = this._pipelines.get(cacheKey);
		if (winner) {
			destroyUniqueWebGPUHandles([pipeline], "pipeline", "WebGPUEnvironmentResources");
			return winner;
		}
		this._pipelines.set(cacheKey, pipeline);
		return pipeline;
	}

	public async warmup(
		request: WebGPUEnvironmentWarmupRequest,
	): Promise<WarmupPhaseCounters> {
		let compiled = 0;
		let failed = 0;
		const errors = [];
		const uniqueModes = [...new Set(request.modes)];
		for (const mode of uniqueModes) {
			try {
				await this.getPipeline(mode, request.sampleCount);
				compiled++;
			} catch (error) {
				failed++;
				errors.push(toShaderCompileError(
					error,
					"webgpu",
					`WebGPUEnvironmentWarmup:${mode}`,
				));
			}
			await request.yieldIfNeeded();
		}
		return {
			phase: "webgpu-environment",
			total: uniqueModes.length,
			compiled,
			skipped: 0,
			failed,
			errors,
		};
	}

	public onShaderRuntimeChanged(): void {
		destroyUniqueWebGPUHandles(
			[...this._pipelines.values()],
			"pipeline",
			"WebGPUEnvironmentResources",
		);
		destroyUniqueWebGPUHandles(
			[this._shaderModule],
			"shader module",
			"WebGPUEnvironmentResources",
		);
		this._pipelines.clear();
		this._shaderModule = null;
		this._shaderDirectiveTag = "";
	}

	public destroy(): void {
		this.onShaderRuntimeChanged();
	}

	private async _getShaderModule(): Promise<IShaderModule> {
		const directiveTag = readWebGPUShaderRuntimeView(this._backend).directiveCacheTag;
		if (this._shaderModule && this._shaderDirectiveTag === directiveTag) {
			return this._shaderModule;
		}
		const shader = await ShaderSource.load("webgpu.environment");
		const module = await this._backend.createShaderModule({
			code: shader.source.code,
			sourceMap: shader.source.sourceMap,
			label: "WebGPUEnvironmentShader",
			language: "wgsl",
			stage: "unknown",
			sourceKind: "builtin-environment",
		});
		if (this._shaderModule) {
			destroyUniqueWebGPUHandles(
				[this._shaderModule],
				"shader module",
				"WebGPUEnvironmentResources",
			);
		}
		this._shaderModule = module;
		this._shaderDirectiveTag = directiveTag;
		return module;
	}
}

function resolveDepthFormat(
	backend: WebGPUDeviceResourceHost,
	mode: WebGPUSceneTargetMode,
): TextureFormat {
	return mode === "mrt" || mode === "gbuffer" || mode === "color"
		? TextureFormat.Depth32Float
		: backend.canvasDepthFormat;
}

function resolveSampleCount(mode: WebGPUSceneTargetMode, sampleCount: number): number {
	if (mode !== "mrt" && mode !== "color") return 1;
	if (!Number.isFinite(sampleCount)) {
		throw new Error("WebGPU environment pipeline sampleCount must be finite.");
	}
	return Math.max(1, Math.floor(sampleCount));
}
