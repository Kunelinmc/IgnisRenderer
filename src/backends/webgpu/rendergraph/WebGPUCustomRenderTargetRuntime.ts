import type {
	FrameContext,
	FramePass,
} from "../../../pipeline/types";
import type { ICommandEncoder } from "../../ICommandEncoder";
import type {
	CustomRenderPassContext,
	CustomRenderPassResourceFacade,
	CustomRenderTargetExecutionTarget,
	RenderTargetDescriptor,
	RenderTargetReadbackOptions,
	RenderTargetReadbackResult,
} from "../../../rendering/CustomRenderTargets";
import {
	TextureFormat,
	TextureUsage,
	type IRenderTexture,
} from "../../types";
import type { WebGPUFrameHost } from "./WebGPUFrameHost";
import { ComputeRuntime } from "../ComputeRuntime";
import { Logger } from "../../../foundation/Logger";

interface WebGPUCustomRenderTarget {
	descriptor: RenderTargetDescriptor;
	width: number;
	height: number;
	sampleCount: number;
	color: IRenderTexture[];
	depth: IRenderTexture | null;
}

/**
 * Owns WebGPU custom render-target allocation, pass execution, and readback.
 */
export class WebGPUCustomRenderTargetRuntime {
	private readonly _host: WebGPUFrameHost;
	private _readbackRuntime: ComputeRuntime | null = null;
	private readonly _targets = new Map<string, WebGPUCustomRenderTarget>();
	private _lastSuccessfulFrame = false;

	public constructor(host: WebGPUFrameHost) {
		this._host = host;
	}

	public sync(context: FrameContext): void {
		if (!context.renderTargets) {
			return;
		}
		const descriptors = context.renderTargets.getAll();
		const activeIds = new Set(descriptors.map((descriptor) => descriptor.id));
		for (const id of Array.from(this._targets.keys())) {
			if (!activeIds.has(id)) {
				this._destroyTarget(id);
			}
		}
		for (const descriptor of descriptors) {
			const width = resolveTargetWidth(descriptor, context.attachments.width);
			const height = resolveTargetHeight(descriptor, context.attachments.height);
			const sampleCount = descriptor.sampleCount ?? 1;
			const current = this._targets.get(descriptor.id);
			if (
				current &&
				current.width === width &&
				current.height === height &&
				current.sampleCount === sampleCount &&
				targetDescriptorsEqual(current.descriptor, descriptor)
			) {
				continue;
			}
			if (sampleCount !== 1) {
				throw new Error(
					`WebGPU custom render target "${descriptor.id}" sampleCount must be 1.`
				);
			}
			const replacement = this._createTarget(descriptor, width, height, sampleCount);
			this._targets.set(descriptor.id, replacement);
			if (current) {
				this._destroyTargetResources(current);
			}
		}
	}

	public hasPass(pass: FramePass, context: FrameContext): boolean {
		return context.customRenderPasses?.has(pass.stage) === true;
	}

	public async executePass(
		pass: FramePass,
		context: FrameContext,
		encoder: ICommandEncoder
	): Promise<void> {
		const descriptor = context.customRenderPasses.get(pass.stage);
		if (!descriptor) {
			return;
		}
		const target = this._targets.get(descriptor.target);
		if (!target) {
			const key = `webgpu-custom-render-pass-target-missing-${pass.stage}`;
			Logger.warn(
				`[${key}] WebGPU custom render pass "${pass.stage}" target "${descriptor.target}" is unavailable; skipping.`,
				{ scope: "WebGPUCustomRenderTargetRuntime", onceKey: key }
			);
			return;
		}
		const executionTarget = toExecutionTarget(target);
		const passContext: CustomRenderPassContext = {
			backend: "webgpu",
			frameContext: context,
			encoder,
			target: executionTarget,
			width: target.width,
			height: target.height,
			resources: this._createResourceFacade(),
		};
		await descriptor.execute(passContext);
	}

	public markFrameCommitted(): void {
		this._lastSuccessfulFrame = true;
	}

	public markFrameAborted(): void {
		this._lastSuccessfulFrame = false;
	}

	public async readColor(
		id: string,
		attachmentIndex = 0,
		options: RenderTargetReadbackOptions = {}
	): Promise<RenderTargetReadbackResult> {
		if (!this._lastSuccessfulFrame) {
			throw new Error(
				`Render target "${id}" cannot be read before a successful frame completes.`
			);
		}
		const target = this._targets.get(id);
		if (!target) {
			throw new Error(`Render target "${id}" is unavailable.`);
		}
		const texture = target.color[attachmentIndex];
		if (!texture) {
			throw new Error(
				`Render target "${id}" color attachment ${attachmentIndex} is unavailable.`
			);
		}
		const width = resolveReadbackDimension(id, "width", options.width, target.width);
		const height = resolveReadbackDimension(id, "height", options.height, target.height);
		const format = target.descriptor.color[attachmentIndex]?.format ??
			TextureFormat.RGBA8Unorm;
		const result = await this._getReadbackRuntime().readTexture({
			texture,
			width,
			height,
			format,
		});
		return {
			...result,
			origin: "top-left",
		};
	}

	public destroy(): void {
		for (const id of Array.from(this._targets.keys())) {
			this._destroyTarget(id);
		}
		this._readbackRuntime?.destroy();
		this._readbackRuntime = null;
		this._lastSuccessfulFrame = false;
	}

	private _getReadbackRuntime(): ComputeRuntime {
		if (!this._readbackRuntime) {
			this._readbackRuntime = new ComputeRuntime(this._host.computeFacade);
		}
		return this._readbackRuntime;
	}

	private _createTarget(
		descriptor: RenderTargetDescriptor,
		width: number,
		height: number,
		sampleCount: number
	): WebGPUCustomRenderTarget {
		const color: IRenderTexture[] = [];
		let depth: IRenderTexture | null = null;
		try {
			for (let index = 0; index < descriptor.color.length; index++) {
				const attachment = descriptor.color[index];
				const texture = this._host.createTexture({
					width,
					height,
					format: attachment.format,
					sampleCount,
					usage:
						TextureUsage.RenderAttachment |
						TextureUsage.TextureBinding |
						TextureUsage.CopySrc |
						TextureUsage.CopyDst,
					label:
						attachment.label ??
						`WebGPUCustomRenderTarget_${descriptor.id}_Color${index}`,
				});
				color.push(texture);
				assertExactTextureFormat(descriptor.id, `color attachment ${index}`, texture,
					attachment.format);
			}
			if (descriptor.depth) {
				depth = this._host.createTexture({
					width,
					height,
					format: descriptor.depth.format,
					sampleCount,
					usage:
						TextureUsage.RenderAttachment |
						TextureUsage.TextureBinding |
						TextureUsage.CopySrc,
					label:
						descriptor.depth.label ??
						`WebGPUCustomRenderTarget_${descriptor.id}_Depth`,
				});
				assertExactTextureFormat(
					descriptor.id,
					"depth attachment",
					depth,
					descriptor.depth.format
				);
			}
			return {
				descriptor,
				width,
				height,
				sampleCount,
				color,
				depth,
			};
		} catch (error) {
			for (const texture of color) {
				texture.destroy();
			}
			depth?.destroy();
			throw error;
		}
	}

	private _destroyTarget(id: string): void {
		const target = this._targets.get(id);
		if (!target) {
			return;
		}
		this._destroyTargetResources(target);
		this._targets.delete(id);
	}

	private _destroyTargetResources(target: WebGPUCustomRenderTarget): void {
		for (const texture of target.color) {
			texture.destroy();
		}
		target.depth?.destroy();
	}

	private _createResourceFacade(): CustomRenderPassResourceFacade {
		const host = this._host;
		return {
			createBuffer: (desc) => host.createBuffer(desc),
			createTexture: (desc) => host.createTexture(desc),
			createSampler: (desc) => host.createSampler(desc),
			createShaderModule: (desc) => host.createShaderModule(desc) as any,
			createRenderPipeline: (desc) => host.createPipeline(desc),
			createBindingGroup: (desc) => host.createBindingGroup(desc),
		};
	}
}

function toExecutionTarget(
	target: WebGPUCustomRenderTarget
): CustomRenderTargetExecutionTarget {
	return {
		id: target.descriptor.id,
		width: target.width,
		height: target.height,
		sampleCount: target.sampleCount,
		color: target.color.map((texture, index) => ({
			texture,
			format: target.descriptor.color[index].format,
		})),
		depth: target.depth && target.descriptor.depth ? {
			texture: target.depth,
			format: target.descriptor.depth.format,
		} : null,
	};
}

function resolveTargetWidth(
	descriptor: RenderTargetDescriptor,
	canvasWidth: number
): number {
	if (descriptor.size.mode === "fixed") {
		return Math.max(1, Math.floor(descriptor.size.width));
	}
	return Math.max(
		1,
		Math.floor(canvasWidth * Math.max(0.0001, descriptor.size.scale ?? 1))
	);
}

function resolveTargetHeight(
	descriptor: RenderTargetDescriptor,
	canvasHeight: number
): number {
	if (descriptor.size.mode === "fixed") {
		return Math.max(1, Math.floor(descriptor.size.height));
	}
	return Math.max(
		1,
		Math.floor(canvasHeight * Math.max(0.0001, descriptor.size.scale ?? 1))
	);
}

function targetDescriptorsEqual(
	a: RenderTargetDescriptor,
	b: RenderTargetDescriptor
): boolean {
	return JSON.stringify(a) === JSON.stringify(b);
}

function assertExactTextureFormat(
	id: string,
	attachment: string,
	texture: IRenderTexture,
	requestedFormat: TextureFormat
): void {
	const actualFormat = texture.format ?? texture.requestedFormat;
	if (actualFormat !== requestedFormat || texture.formatFallbackReason) {
		throw new Error(
			`WebGPU custom render target "${id}" ${attachment} requested ` +
				`"${requestedFormat}" but received "${actualFormat ?? "unknown"}".`
		);
	}
}

function resolveReadbackDimension(
	id: string,
	name: "width" | "height",
	requested: number | undefined,
	limit: number
): number {
	const value = requested ?? limit;
	if (!Number.isInteger(value) || value <= 0 || value > limit) {
		throw new Error(
			`Render target "${id}" readback ${name} must be between 1 and ${limit}.`
		);
	}
	return value;
}
