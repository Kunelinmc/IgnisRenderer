import type { DirtyRect } from "../../../pipeline/incremental";
import type { PresentationAlphaMode } from "../../IRenderBackend";
import { ShaderSource } from "../../../shaders/ShaderSource";
import type { ICommandEncoder } from "../../ICommandEncoder";
import type { TextureFormat } from "../../../core/TextureFormat";
import type {
	IRenderPipeline,
	IRenderTexture,
	IShaderModule,
} from "../../types";
import type { WebGPUFrameHost } from "./WebGPUFrameHost";

export type WebGPUColorDirtyClearMode =
	| "color"
	| "mrt"
	| "deferred"
	| "extended";

export interface WebGPUColorDirtyClearAttachment {
	readonly view: IRenderTexture;
	readonly resolveTarget?: IRenderTexture;
	readonly format: TextureFormat;
}

/** Records shader-based color clears scoped to incremental dirty rectangles. */
export class WebGPUColorDirtyClearPass {
	private readonly _host: WebGPUFrameHost;
	private _shaderModule: IShaderModule | null = null;
	private readonly _pipelines = new Map<string, IRenderPipeline>();

	public constructor(host: WebGPUFrameHost) {
		this._host = host;
	}

	/**
	 * Clears active scene-color and G-buffer attachments inside dirty rectangles.
	 *
	 * @internal Owned by WebGPU scene recording.
	 */
	public async record(
		encoder: ICommandEncoder,
		mode: WebGPUColorDirtyClearMode,
		attachments: readonly WebGPUColorDirtyClearAttachment[],
		sampleCount: number,
		dirtyRects: readonly DirtyRect[],
		presentationAlphaMode: PresentationAlphaMode = "opaque",
	): Promise<void> {
		if (dirtyRects.length === 0) return;
		this._validateAttachmentCount(mode, attachments.length);
		const pipeline = await this._getPipeline(
			mode,
			attachments,
			sampleCount,
			presentationAlphaMode,
		);
		encoder.beginRenderPass({
			label: "WebGPUColorDirtyClear",
			colorAttachments: attachments.map((attachment) => ({
				view: attachment.view,
				resolveTarget: attachment.resolveTarget,
				loadOp: "load" as const,
				storeOp: "store" as const,
			})),
		});
		encoder.setPipeline(pipeline);
		for (const rect of dirtyRects) {
			encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
			encoder.draw(3);
		}
		encoder.endRenderPass();
	}

	public onShaderRuntimeChanged(): void {
		this._destroyManagedResource(this._shaderModule);
		for (const pipeline of this._pipelines.values()) {
			this._destroyManagedResource(pipeline);
		}
		this._shaderModule = null;
		this._pipelines.clear();
	}

	public destroy(): void {
		this.onShaderRuntimeChanged();
	}

	private async _getPipeline(
		mode: WebGPUColorDirtyClearMode,
		attachments: readonly WebGPUColorDirtyClearAttachment[],
		sampleCount: number,
		presentationAlphaMode: PresentationAlphaMode,
	): Promise<IRenderPipeline> {
		const resolvedSampleCount = Math.max(1, Math.floor(sampleCount || 1));
		const cacheKey =
			`${mode}|${attachments.map((attachment) => attachment.format).join(",")}` +
			`|${resolvedSampleCount}|${presentationAlphaMode}`;
		const cached = this._pipelines.get(cacheKey);
		if (cached) return cached;

		if (!this._shaderModule) {
			const composite =
				await ShaderSource.load("webgpu.utility.colorDirtyClear");
			this._shaderModule = await this._host.createShaderModule({
				label: "WebGPUColorDirtyClearShader",
				code: composite.source.code,
				sourceMap: composite.source.sourceMap,
				language: "wgsl",
				stage: "unknown",
				sourceKind: "postprocess",
			});
		}

		const entryPointBase =
			mode === "color" ? "fsColor"
			: mode === "mrt" ? "fsMRT"
			: mode === "deferred" ? "fsDeferred"
			: "fsExtended";
		const entryPoint = presentationAlphaMode === "premultiplied" ?
			`${entryPointBase}Transparent`
		:	entryPointBase;
		const pipeline = await this._host.createPipeline({
			label: `WebGPUColorDirtyClearPipeline_${cacheKey}`,
			vertex: {
				module: this._shaderModule,
				entryPoint: "vsMain",
			},
			fragment: {
				module: this._shaderModule,
				entryPoint,
				targets: attachments.map((attachment) => ({
					format: attachment.format,
				})),
			},
			primitive: {
				topology: "triangle-list" as any,
				cullMode: "none",
				frontFace: "ccw",
			},
			sampleCount: resolvedSampleCount,
		});
		this._pipelines.set(cacheKey, pipeline);
		return pipeline;
	}

	private _validateAttachmentCount(
		mode: WebGPUColorDirtyClearMode,
		count: number
	): void {
		const expected =
			mode === "color" ? 1
			: mode === "mrt" || mode === "deferred" ? 5
			: 8;
		if (count !== expected) {
			throw new Error(
				`WebGPU ${mode} dirty clear requires ${expected} attachments; ` +
					`received ${count}.`
			);
		}
	}

	private _destroyManagedResource(resource: unknown): void {
		const destroyFn = (resource as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(resource);
		}
	}
}
