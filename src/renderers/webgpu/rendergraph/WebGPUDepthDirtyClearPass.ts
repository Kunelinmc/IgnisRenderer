import type {
	DirtyRect,
} from "../../../pipeline/incremental";
import { Logger } from "../../../foundation/Logger";
import { ShaderSource } from "../../../shaders/ShaderSource";
import type { ICommandEncoder } from "../../ICommandEncoder";
import {
	TextureFormat,
	type IRenderPipeline,
	type IRenderTexture,
	type IShaderModule,
} from "../../types";
import type { WebGPUBackend } from "../../WebGPUBackend";

/**
 * Records depth-only clears scoped to incremental dirty rectangles.
 */
export class WebGPUDepthDirtyClearPass {
	private readonly _backend: WebGPUBackend;
	private _shaderModule: IShaderModule | null = null;
	private readonly _pipelines = new Map<string, IRenderPipeline>();

	public constructor(backend: WebGPUBackend) {
		this._backend = backend;
	}

	/**
	 * Records a depth dirty-rect clear pass.
	 *
	 * @param encoder Active command encoder.
	 * @param depthAttachment Depth texture to clear inside dirty rectangles.
	 * @param depthFormat Depth attachment format.
	 * @param sampleCount Depth attachment sample count.
	 * @param dirtyRects Dirty rectangles in attachment coordinates.
	 * @returns `true` when the partial clear pass was recorded.
	 * @sideEffects May create and cache the depth clear shader pipeline.
	 */
	public async record(
		encoder: ICommandEncoder | null,
		depthAttachment: IRenderTexture,
		depthFormat: TextureFormat,
		sampleCount: number,
		dirtyRects: readonly DirtyRect[]
	): Promise<boolean> {
		if (!encoder || dirtyRects.length === 0) {
			return false;
		}
		try {
			const pipeline = await this._getPipeline(depthFormat, sampleCount);
			encoder.beginRenderPass({
				label: "WebGPUDepthDirtyClear",
				colorAttachments: [],
				depthStencilAttachment: {
					view: depthAttachment,
					depthLoadOp: "load",
					depthStoreOp: "store",
					depthClearValue: 1,
				},
			});
			encoder.setPipeline(pipeline);
			for (const rect of dirtyRects) {
				encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
				encoder.draw(3);
			}
			encoder.endRenderPass();
			return true;
		} catch (error) {
			const key = "webgpu-depth-partial-reuse-fallback";
			Logger.warn(
				`[${key}] WebGPU partial depth reuse unavailable; falling back to full depth clear. ${String(error)}`,
				{ scope: "WebGPUFrameExecutor", onceKey: key }
			);
			return false;
		}
	}

	/**
	 * Invalidates shader and pipeline resources.
	 *
	 * @sideEffects Destroys cached managed GPU resources.
	 */
	public onShaderRuntimeChanged(): void {
		this._destroyManagedResource(this._shaderModule);
		for (const pipeline of this._pipelines.values()) {
			this._destroyManagedResource(pipeline);
		}
		this._shaderModule = null;
		this._pipelines.clear();
	}

	/**
	 * Releases all managed GPU resources.
	 *
	 * @sideEffects Destroys cached shader and pipeline objects.
	 */
	public destroy(): void {
		this.onShaderRuntimeChanged();
	}

	private async _getPipeline(
		depthFormat: TextureFormat,
		sampleCount: number
	): Promise<IRenderPipeline> {
		const resolvedSampleCount = Math.max(1, Math.floor(sampleCount || 1));
		const cacheKey = `${depthFormat}|${resolvedSampleCount}`;
		const cached = this._pipelines.get(cacheKey);
		if (cached) {
			return cached;
		}

		if (!this._shaderModule) {
			const composite =
				await ShaderSource.load("webgpu.utility.depthDirtyClear.composite");
			this._shaderModule = await this._backend.createShaderModule({
				label: "WebGPUDepthDirtyClearShader",
				code: composite.code,
				sourceMap: composite.sourceMap,
				language: "wgsl",
				stage: "unknown",
				sourceKind: "postprocess",
			});
		}

		const pipeline = await this._backend.createPipeline({
			label: `WebGPUDepthDirtyClearPipeline_${cacheKey}`,
			vertex: {
				module: this._shaderModule,
				entryPoint: "vsMain",
			},
			primitive: {
				topology: "triangle-list" as any,
				cullMode: "none",
				frontFace: "ccw",
			},
			depthStencil: {
				format: depthFormat,
				depthWriteEnabled: true,
				depthCompare: "always",
			},
			sampleCount: resolvedSampleCount,
		} as any);
		this._pipelines.set(cacheKey, pipeline);
		return pipeline;
	}

	private _destroyManagedResource(resource: unknown): void {
		const destroyFn = (resource as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(resource);
		}
	}
}
