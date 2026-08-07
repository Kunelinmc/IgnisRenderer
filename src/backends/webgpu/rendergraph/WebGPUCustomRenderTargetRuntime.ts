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
import type {
	WebGPUSampleCountResolver,
	WebGPUSampleCountSelection,
} from "../WebGPUSampleCountResolver";
import { getWebGPURenderTargetPixelByteCost } from "../WebGPUTextureFormatInfo";
import type {
	WebGPUFrameGraphContribution,
	WebGPUFrameGraphModule,
	WebGPUFrameModulePlanningInput,
} from "./WebGPUFrameGraphModule";
import { createWebGPUFrameGraphNode } from "./WebGPUFrameGraphPlanningUtils";

const WEBGPU_CUSTOM_TARGET_SAMPLE_COUNT_FALLBACK_KEY =
	"webgpu-custom-target-sample-count-runtime-fallback-1x";

interface WebGPUCustomRenderTargetColor {
	texture: IRenderTexture;
	resolveTexture: IRenderTexture | null;
}

interface WebGPUCustomRenderTarget {
	descriptor: RenderTargetDescriptor;
	width: number;
	height: number;
	sampleCount: number;
	selectionSignature: string;
	color: WebGPUCustomRenderTargetColor[];
	depth: IRenderTexture | null;
}

/**
 * Owns WebGPU custom render-target allocation, pass execution, and readback.
 */
export class WebGPUCustomRenderTargetRuntime implements WebGPUFrameGraphModule {
	public readonly id = "custom-render-target";
	public readonly executors = {};
	private readonly _host: WebGPUFrameHost;
	private _readbackRuntime: ComputeRuntime | null = null;
	private readonly _targets = new Map<string, WebGPUCustomRenderTarget>();
	private _lastSuccessfulFrame = false;

	constructor(
		host: WebGPUFrameHost,
		private readonly _sampleCountResolver: WebGPUSampleCountResolver,
	) {
		this._host = host;
	}

	public planStage(
		input: WebGPUFrameModulePlanningInput,
	): readonly WebGPUFrameGraphContribution[] {
		if (!this.hasPass(input.pass, input.context)) return [];
		return [{
			order: 100,
			nodes: [{
				...createWebGPUFrameGraphNode(
					input.pass,
					"opaque-external",
					`WebGPUOpaque:${input.pass.stage}`,
				),
				domain: "cpu",
				retention: "always",
				opaque: true,
			}],
		}];
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
			const selection = this._resolveSampleCount(descriptor);
			const current = this._targets.get(descriptor.id);
			if (
				current &&
				current.width === width &&
				current.height === height &&
				current.sampleCount === selection.sampleCount &&
				current.selectionSignature === selection.signature &&
				targetDescriptorsEqual(current.descriptor, descriptor)
			) {
				continue;
			}
			let replacement: WebGPUCustomRenderTarget;
			try {
				replacement = this._createTarget(descriptor, width, height, selection);
			} catch (error) {
				if (selection.sampleCount === 1) {
					throw error;
				}
				this._sampleCountResolver.fallbackToSingleSample(selection.signature);
				this._warnRuntimeFallback(descriptor.id, selection, error);
				const fallbackSelection = this._resolveSampleCount(descriptor);
				replacement = this._createTarget(descriptor, width, height, fallbackSelection);
			}
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
		encoder: ICommandEncoder,
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
				{ scope: "WebGPUCustomRenderTargetRuntime", onceKey: key },
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

	public commitFrameState(): void {
		this._lastSuccessfulFrame = true;
	}

	public abortFrameState(): void {
		this._lastSuccessfulFrame = false;
	}

	public async readColor(
		id: string,
		attachmentIndex = 0,
		options: RenderTargetReadbackOptions = {},
	): Promise<RenderTargetReadbackResult> {
		if (!this._lastSuccessfulFrame) {
			throw new Error(
				`Render target "${id}" cannot be read before a successful frame completes.`,
			);
		}
		const target = this._targets.get(id);
		if (!target) {
			throw new Error(`Render target "${id}" is unavailable.`);
		}
		const attachment = target.color[attachmentIndex];
		if (!attachment) {
			throw new Error(
				`Render target "${id}" color attachment ${attachmentIndex} is unavailable.`,
			);
		}
		const texture = attachment.resolveTexture ?? attachment.texture;
		const width = resolveReadbackDimension(id, "width", options.width, target.width);
		const height = resolveReadbackDimension(id, "height", options.height, target.height);
		const format = target.descriptor.color[attachmentIndex]?.format ?? TextureFormat.RGBA8Unorm;
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
		selection: WebGPUSampleCountSelection,
	): WebGPUCustomRenderTarget {
		const color: WebGPUCustomRenderTargetColor[] = [];
		let depth: IRenderTexture | null = null;
		try {
			for (let index = 0; index < descriptor.color.length; index++) {
				const attachment = descriptor.color[index];
				const texture = this._host.createTexture({
					width,
					height,
					format: attachment.format,
					sampleCount: selection.sampleCount,
					usage:
						selection.sampleCount > 1
							? TextureUsage.RenderAttachment
							: TextureUsage.RenderAttachment |
								TextureUsage.TextureBinding |
								TextureUsage.CopySrc |
								TextureUsage.CopyDst,
					label:
						attachment.label ??
						`WebGPUCustomRenderTarget_${descriptor.id}_Color${index}`,
				});
				const colorAttachment: WebGPUCustomRenderTargetColor = {
					texture,
					resolveTexture: null,
				};
				color.push(colorAttachment);
				assertExactTextureFormat(
					descriptor.id,
					`color attachment ${index}`,
					texture,
					attachment.format,
				);
				const resolveTexture =
					selection.sampleCount > 1
						? this._host.createTexture({
								width,
								height,
								format: attachment.format,
								sampleCount: 1,
								usage:
									TextureUsage.RenderAttachment |
									TextureUsage.TextureBinding |
									TextureUsage.CopySrc |
									TextureUsage.CopyDst,
								label: `${
									attachment.label ??
									`WebGPUCustomRenderTarget_${descriptor.id}_Color${index}`
								}_Resolve`,
							})
						: null;
				colorAttachment.resolveTexture = resolveTexture;
				if (resolveTexture) {
					assertExactTextureFormat(
						descriptor.id,
						`color attachment ${index} resolve texture`,
						resolveTexture,
						attachment.format,
					);
				}
			}
			if (descriptor.depth) {
				depth = this._host.createTexture({
					width,
					height,
					format: descriptor.depth.format,
					sampleCount: selection.sampleCount,
					usage:
						TextureUsage.RenderAttachment |
						TextureUsage.TextureBinding |
						(selection.sampleCount === 1 ? TextureUsage.CopySrc : 0),
					label:
						descriptor.depth.label ?? `WebGPUCustomRenderTarget_${descriptor.id}_Depth`,
				});
				assertExactTextureFormat(
					descriptor.id,
					"depth attachment",
					depth,
					descriptor.depth.format,
				);
			}
			return {
				descriptor,
				width,
				height,
				sampleCount: selection.sampleCount,
				selectionSignature: selection.signature,
				color,
				depth,
			};
		} catch (error) {
			for (const attachment of color) {
				attachment.texture.destroy();
				attachment.resolveTexture?.destroy();
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
		for (const attachment of target.color) {
			attachment.texture.destroy();
			attachment.resolveTexture?.destroy();
		}
		target.depth?.destroy();
	}

	private _resolveSampleCount(descriptor: RenderTargetDescriptor): WebGPUSampleCountSelection {
		const formats = [
			...descriptor.color.map((attachment) => attachment.format as GPUTextureFormat),
			...(descriptor.depth ? [descriptor.depth.format as GPUTextureFormat] : []),
		];
		return this._sampleCountResolver.resolveDomainSampleCount(
			`custom-target:${descriptor.id}`,
			descriptor.sampleCount ?? 1,
			formats,
			{
				colorAttachmentCount: descriptor.color.length,
				colorAttachmentBytesPerSample: descriptor.color.reduce(
					(total, attachment) =>
						total + getWebGPURenderTargetPixelByteCost(attachment.format),
					0,
				),
			},
		);
	}

	private _warnRuntimeFallback(
		id: string,
		selection: WebGPUSampleCountSelection,
		error: unknown,
	): void {
		Logger.warn(
			`[${WEBGPU_CUSTOM_TARGET_SAMPLE_COUNT_FALLBACK_KEY}] WebGPU custom ` +
				`render target "${id}" ${selection.sampleCount}x allocation failed; ` +
				`retrying at 1x. ${String(error)}`,
			{
				scope: "WebGPUCustomRenderTargetRuntime",
				onceKey: `${WEBGPU_CUSTOM_TARGET_SAMPLE_COUNT_FALLBACK_KEY}:${selection.signature}`,
			},
		);
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
		color: target.color.map((attachment, index) => ({
			texture: attachment.texture,
			format: target.descriptor.color[index].format,
			resolveTexture: attachment.resolveTexture,
		})),
		depth: target.depth && target.descriptor.depth ? {
			texture: target.depth,
			format: target.descriptor.depth.format,
			resolveTexture: null,
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
