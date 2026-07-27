import { EventEmitter } from "../core/EventEmitter";
import type { TextureReadbackResult } from "../backends/IComputeRuntime";
import { getTextureFormatInfo } from "../backends/TextureFormatInfo";
import type { ICommandEncoder } from "../backends/ICommandEncoder";
import type {
	BindingGroupDesc,
	BufferDesc,
	ComputePipelineDesc,
	IBindingGroup,
	IComputePipeline,
	IRenderBuffer,
	IRenderPipeline,
	IRenderTexture,
	ISampler,
	IShaderModule,
	PipelineDesc,
	SamplerDesc,
	ShaderModuleDesc,
	TextureDesc,
	TextureFormat,
} from "../backends/types";
import type { FrameContext, FramePassStage } from "../pipeline/types";
import type {
	RenderPipelineStageIncrementalOptions,
	RenderPipelineStagePredicate,
} from "../pipeline/RenderPipelineRegistry";
import type { RenderBackendType } from "../backends/IRenderBackend";

export type RenderTargetSizeDescriptor =
	| {
			readonly mode: "canvas-scale";
			readonly scale?: number;
	  }
	| {
			readonly mode: "fixed";
			readonly width: number;
			readonly height: number;
	  };

export interface RenderTargetColorAttachmentDescriptor {
	readonly format: TextureFormat;
	readonly label?: string;
}

export interface RenderTargetDepthAttachmentDescriptor {
	readonly format: TextureFormat;
	readonly label?: string;
}

export interface RenderTargetDescriptor {
	readonly id: string;
	readonly size: RenderTargetSizeDescriptor;
	readonly color: readonly RenderTargetColorAttachmentDescriptor[];
	readonly depth?: RenderTargetDepthAttachmentDescriptor | null;
	readonly sampleCount?: number;
	readonly label?: string;
}

export interface RenderTargetReadbackOptions {
	readonly width?: number;
	readonly height?: number;
}

export type RenderTargetReadbackOrigin = "top-left" | "bottom-left";

export interface RenderTargetReadbackResult extends TextureReadbackResult {
	readonly origin: RenderTargetReadbackOrigin;
}

export interface RenderTargetHandle {
	readonly id: string;
	readonly descriptor: RenderTargetDescriptor;
}

export interface CustomRenderTargetAttachment {
	readonly texture: IRenderTexture;
	readonly format: TextureFormat;
}

export interface CustomRenderTargetExecutionTarget {
	readonly id: string;
	readonly width: number;
	readonly height: number;
	readonly color: readonly CustomRenderTargetAttachment[];
	readonly depth?: CustomRenderTargetAttachment | null;
	readonly sampleCount: number;
}

export interface CustomRenderPassResourceFacade {
	createBuffer(desc: BufferDesc): IRenderBuffer;
	createTexture(desc: TextureDesc): IRenderTexture;
	createSampler(desc: SamplerDesc): ISampler;
	createShaderModule(desc: ShaderModuleDesc): Promise<IShaderModule> | IShaderModule;
	createRenderPipeline(desc: PipelineDesc): Promise<IRenderPipeline> | IRenderPipeline;
	createComputePipeline?(desc: ComputePipelineDesc): Promise<IComputePipeline> | IComputePipeline;
	createBindingGroup(desc: BindingGroupDesc): IBindingGroup;
}

export interface CustomRenderPassContext {
	readonly backend: RenderBackendType;
	readonly frameContext: FrameContext;
	readonly encoder: ICommandEncoder;
	readonly target: CustomRenderTargetExecutionTarget;
	readonly width: number;
	readonly height: number;
	readonly resources: CustomRenderPassResourceFacade;
}

export interface CustomRenderPassDescriptor {
	readonly id: FramePassStage;
	readonly target: string;
	readonly dependsOn?: readonly FramePassStage[];
	readonly shouldRun?: RenderPipelineStagePredicate;
	readonly incremental?: RenderPipelineStageIncrementalOptions;
	readonly label?: string;
	execute(context: CustomRenderPassContext): void | Promise<void>;
}

export interface RenderTargetRegistryChange {
	readonly id: string;
	readonly reason: "register" | "unregister" | "lifecycle";
}

export interface CustomRenderPassRegistryChange {
	readonly id: FramePassStage;
	readonly reason: "register" | "unregister" | "lifecycle";
}

export interface RenderTargetRegistryOptions {
	readonly readColor?: (
		id: string,
		attachmentIndex: number,
		options?: RenderTargetReadbackOptions
	) => Promise<RenderTargetReadbackResult>;
	/** @internal Renderer registry coordination hook. */
	readonly getReferencingPassId?: (id: string) => FramePassStage | null;
}

export class RenderTargetRegistry extends EventEmitter<{
	change: [RenderTargetRegistryChange];
}> {
	private readonly _targets = new Map<string, RenderTargetDescriptor>();
	private readonly _readColor?: RenderTargetRegistryOptions["readColor"];
	private readonly _getReferencingPassId?: RenderTargetRegistryOptions["getReferencingPassId"];

	public constructor(options: RenderTargetRegistryOptions = {}) {
		super();
		this._readColor = options.readColor;
		this._getReferencingPassId = options.getReferencingPassId;
	}

	public register(descriptor: RenderTargetDescriptor): RenderTargetHandle {
		const normalized = normalizeRenderTargetDescriptor(descriptor);
		if (this._targets.has(normalized.id)) {
			throw new Error(`Render target "${normalized.id}" is already registered.`);
		}
		this._targets.set(normalized.id, normalized);
		this.emit("change", { id: normalized.id, reason: "register" });
		return this._createHandle(normalized);
	}

	public unregister(id: string): void {
		if (!this._targets.has(id)) {
			return;
		}
		const passId = this._getReferencingPassId?.(id) ?? null;
		if (passId) {
			throw new Error(
				`Render target "${id}" is referenced by custom render pass "${passId}".`
			);
		}
		this._targets.delete(id);
		this.emit("change", { id, reason: "unregister" });
	}

	public get(id: string): RenderTargetHandle | null {
		const descriptor = this._targets.get(id);
		return descriptor ? this._createHandle(descriptor) : null;
	}

	public getDescriptors(): readonly RenderTargetDescriptor[] {
		return Array.from(this._targets.values()).map(cloneRenderTargetDescriptor);
	}

	public createSnapshot(): RenderTargetRegistrySnapshot {
		return new RenderTargetRegistrySnapshot(this.getDescriptors());
	}

	public readColor(
		id: string,
		attachmentIndex = 0,
		options?: RenderTargetReadbackOptions
	): Promise<RenderTargetReadbackResult> {
		if (!this._targets.has(id)) {
			return Promise.reject(new Error(`Render target "${id}" is not registered.`));
		}
		const descriptor = this._targets.get(id)!;
		if (
			!Number.isInteger(attachmentIndex) ||
			attachmentIndex < 0 ||
			attachmentIndex >= descriptor.color.length
		) {
			return Promise.reject(
				new Error(
					`Render target "${id}" color attachment ${attachmentIndex} is unavailable.`
				)
			);
		}
		if (!this._readColor) {
			return Promise.reject(
				new Error("Render target readback is unavailable for this renderer.")
			);
		}
		validateReadbackOptions(id, options);
		return this._readColor(id, attachmentIndex, options);
	}

	private _createHandle(descriptor: RenderTargetDescriptor): RenderTargetHandle {
		return {
			id: descriptor.id,
			descriptor: cloneRenderTargetDescriptor(descriptor),
		};
	}
}

export class RenderTargetRegistrySnapshot {
	private readonly _targets = new Map<string, RenderTargetDescriptor>();

	public constructor(descriptors: readonly RenderTargetDescriptor[] = []) {
		for (const descriptor of descriptors) {
			const normalized = normalizeRenderTargetDescriptor(descriptor);
			this._targets.set(normalized.id, normalized);
		}
	}

	public get(id: string): RenderTargetDescriptor | null {
		const descriptor = this._targets.get(id);
		return descriptor ? cloneRenderTargetDescriptor(descriptor) : null;
	}

	public has(id: string): boolean {
		return this._targets.has(id);
	}

	public getAll(): readonly RenderTargetDescriptor[] {
		return Array.from(this._targets.values()).map(cloneRenderTargetDescriptor);
	}
}

export interface CustomRenderPassRegistryOptions {
	readonly registerPipelineStage: (descriptor: CustomRenderPassDescriptor) => void;
	readonly unregisterPipelineStage: (id: FramePassStage) => void;
	readonly isRenderTargetRegistered: (id: string) => boolean;
}

export class CustomRenderPassRegistry extends EventEmitter<{
	change: [CustomRenderPassRegistryChange];
}> {
	private readonly _passes = new Map<FramePassStage, CustomRenderPassDescriptor>();
	private readonly _options: CustomRenderPassRegistryOptions;

	public constructor(options: CustomRenderPassRegistryOptions) {
		super();
		this._options = options;
	}

	public register(descriptor: CustomRenderPassDescriptor): void {
		const normalized = normalizeCustomRenderPassDescriptor(descriptor);
		if (this._passes.has(normalized.id)) {
			throw new Error(`Custom render pass "${normalized.id}" is already registered.`);
		}
		if (!this._options.isRenderTargetRegistered(normalized.target)) {
			throw new Error(
				`Custom render pass "${normalized.id}" target "${normalized.target}" is not registered.`
			);
		}
		this._options.registerPipelineStage(normalized);
		this._passes.set(normalized.id, normalized);
		this.emit("change", { id: normalized.id, reason: "register" });
	}

	public unregister(id: FramePassStage): void {
		if (!this._passes.delete(id)) {
			return;
		}
		this._options.unregisterPipelineStage(id);
		this.emit("change", { id, reason: "unregister" });
	}

	public get(id: FramePassStage): CustomRenderPassDescriptor | null {
		const descriptor = this._passes.get(id);
		return descriptor ? cloneCustomRenderPassDescriptor(descriptor) : null;
	}

	public getDescriptors(): readonly CustomRenderPassDescriptor[] {
		return Array.from(this._passes.values()).map(cloneCustomRenderPassDescriptor);
	}

	/** @internal Used by the renderer target registry to enforce dependencies. */
	public getReferencingPassId(target: string): FramePassStage | null {
		for (const pass of this._passes.values()) {
			if (pass.target === target) {
				return pass.id;
			}
		}
		return null;
	}

	public createSnapshot(): CustomRenderPassRegistrySnapshot {
		return new CustomRenderPassRegistrySnapshot(this.getDescriptors());
	}
}

export class CustomRenderPassRegistrySnapshot {
	private readonly _passes = new Map<FramePassStage, CustomRenderPassDescriptor>();

	public constructor(descriptors: readonly CustomRenderPassDescriptor[] = []) {
		for (const descriptor of descriptors) {
			const normalized = normalizeCustomRenderPassDescriptor(descriptor);
			this._passes.set(normalized.id, normalized);
		}
	}

	public get(id: FramePassStage): CustomRenderPassDescriptor | null {
		const descriptor = this._passes.get(id);
		return descriptor ? cloneCustomRenderPassDescriptor(descriptor) : null;
	}

	public has(id: FramePassStage): boolean {
		return this._passes.has(id);
	}

	public getAll(): readonly CustomRenderPassDescriptor[] {
		return Array.from(this._passes.values()).map(cloneCustomRenderPassDescriptor);
	}
}

function normalizeRenderTargetDescriptor(
	descriptor: RenderTargetDescriptor
): RenderTargetDescriptor {
	if (!descriptor?.id) {
		throw new Error("Render target id is required.");
	}
	if (!descriptor.color || descriptor.color.length <= 0) {
		throw new Error(`Render target "${descriptor.id}" requires at least one color attachment.`);
	}
	const sampleCount = descriptor.sampleCount ?? 1;
	if (sampleCount !== 1) {
		throw new Error(`Render target "${descriptor.id}" sampleCount must be 1.`);
	}
	validateSizeDescriptor(descriptor.id, descriptor.size);
	return {
		id: descriptor.id,
		label: descriptor.label,
		size: { ...descriptor.size } as RenderTargetSizeDescriptor,
		color: descriptor.color.map((attachment, index) => {
			if (!attachment?.format) {
				throw new Error(
					`Render target "${descriptor.id}" color attachment ${index} requires a format.`
				);
			}
			const info = getTextureFormatInfo(attachment.format);
			if (info.formatClass !== "color" || !info.isRenderable) {
				throw new Error(
					`Render target "${descriptor.id}" color attachment ${index} requires a renderable color format.`
				);
			}
			return { ...attachment };
		}),
		depth: normalizeDepthAttachment(descriptor),
		sampleCount,
	};
}

function normalizeDepthAttachment(
	descriptor: RenderTargetDescriptor
): RenderTargetDepthAttachmentDescriptor | null {
	if (!descriptor.depth) {
		return null;
	}
	if (!descriptor.depth.format) {
		throw new Error(`Render target "${descriptor.id}" depth attachment requires a format.`);
	}
	const info = getTextureFormatInfo(descriptor.depth.format);
	if (info.formatClass !== "depth" || !info.isRenderable) {
		throw new Error(
			`Render target "${descriptor.id}" depth attachment requires a renderable depth-only format.`
		);
	}
	return { ...descriptor.depth };
}

function validateSizeDescriptor(id: string, size: RenderTargetSizeDescriptor): void {
	if (!size) {
		throw new Error(`Render target "${id}" size is required.`);
	}
	if (size.mode === "fixed") {
		if (
			!Number.isFinite(size.width) ||
			!Number.isFinite(size.height) ||
			size.width <= 0 ||
			size.height <= 0
		) {
			throw new Error(`Render target "${id}" fixed size must be positive.`);
		}
		return;
	}
	if (size.mode === "canvas-scale") {
		const scale = size.scale ?? 1;
		if (!Number.isFinite(scale) || scale <= 0) {
			throw new Error(`Render target "${id}" canvas scale must be positive.`);
		}
		return;
	}
	throw new Error(`Render target "${id}" has unsupported size mode.`);
}

function normalizeCustomRenderPassDescriptor(
	descriptor: CustomRenderPassDescriptor
): CustomRenderPassDescriptor {
	if (!descriptor?.id) {
		throw new Error("Custom render pass id is required.");
	}
	if (!descriptor.target) {
		throw new Error(`Custom render pass "${descriptor.id}" requires a target.`);
	}
	if (typeof descriptor.execute !== "function") {
		throw new Error(`Custom render pass "${descriptor.id}" requires execute().`);
	}
	return cloneCustomRenderPassDescriptor(descriptor);
}

function cloneRenderTargetDescriptor(
	descriptor: RenderTargetDescriptor
): RenderTargetDescriptor {
	return {
		...descriptor,
		size: { ...descriptor.size } as RenderTargetSizeDescriptor,
		color: descriptor.color.map((attachment) => ({ ...attachment })),
		depth: descriptor.depth ? { ...descriptor.depth } : null,
	};
}

function cloneCustomRenderPassDescriptor(
	descriptor: CustomRenderPassDescriptor
): CustomRenderPassDescriptor {
	return {
		...descriptor,
		dependsOn: descriptor.dependsOn?.slice(),
	};
}

function validateReadbackOptions(
	id: string,
	options: RenderTargetReadbackOptions | undefined
): void {
	for (const [name, value] of [
		["width", options?.width],
		["height", options?.height],
	] as const) {
		if (value !== undefined && (!Number.isInteger(value) || value <= 0)) {
			throw new Error(`Render target "${id}" readback ${name} must be a positive integer.`);
		}
	}
}
