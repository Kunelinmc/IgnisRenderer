import type {
	FrameContext,
	FramePass,
} from "../../pipeline/types";
import type {
	CustomRenderPassContext,
	CustomRenderPassResourceFacade,
	CustomRenderTargetExecutionTarget,
	RenderTargetDescriptor,
	RenderTargetReadbackOptions,
	RenderTargetReadbackResult,
	PreparedRenderTargetJob,
} from "../../rendering/CustomRenderTargets";
import {
	getTextureFormatInfo,
	TextureFormat,
} from "../../core/TextureFormat";
import {
	AddressMode,
	BufferUsage,
	FilterMode,
	PrimitiveTopology,
	TextureUsage,
	type BufferDesc,
	type BindingEntry,
	type BindingGroupDesc,
	type ColorTargetState,
	type IShaderModule,
	type IRenderTexture,
	type PipelineDesc,
	type SamplerDesc,
	type ShaderModuleDesc,
	type TextureDesc,
	type VertexAttribute,
} from "../types";
import type {
	ColorAttachment,
	ComputePassDesc,
	DepthStencilAttachment,
	ICommandBuffer,
	ICommandEncoder,
	RenderPassDesc,
	TextureCopySize,
	TextureCopyView,
} from "../ICommandEncoder";
import type {
	IBindingGroup,
	IComputePipeline,
	IRenderBuffer,
	IRenderPipeline,
	IndexFormat,
} from "../types";
import { Logger } from "../../foundation/Logger";
import { float16BitsToFloat32, float32ToFloat16Bits } from "../../foundation/Float16";
import {
	isWebGLFloatColorRenderTargetFormat,
	resolveWebGLColorRenderTargetFormat,
	resolveWebGLDepthRenderTargetFormat,
	type WebGLColorRenderTargetFormat,
} from "./WebGLRenderTargetFormat";
import type { WebGLAuxiliaryUniform } from "./WebGLAuxiliaryRaster";

interface WebGLCustomTexture extends IRenderTexture {
	_gpuResource: WebGLTexture;
	_webglTargetKind: "texture";
	_webglColorFormat?: WebGLColorRenderTargetFormat;
}

interface WebGLCustomBuffer extends IRenderBuffer {
	_webglBuffer: WebGLBuffer;
	_webglUsage: BufferUsage;
}

interface WebGLCustomSampler {
	readonly label?: string;
	_webglSampler: WebGLSampler;
}

interface WebGLCustomShaderModule extends IShaderModule {
	_webglCode: string;
	_webglStage?: ShaderModuleDesc["stage"];
}

interface WebGLCustomPipeline extends IRenderPipeline {
	_webglProgram: WebGLProgram;
	_webglDesc: PipelineDesc;
}

interface WebGLCustomBindingGroup extends IBindingGroup {
	_webglEntries: readonly BindingEntry[];
}

interface WebGLCustomRenderTarget {
	descriptor: RenderTargetDescriptor;
	width: number;
	height: number;
	sampleCount: number;
	sceneDepth: boolean;
	framebuffer: WebGLFramebuffer;
	color: WebGLCustomTexture[];
	depth: WebGLCustomTexture | null;
}

export interface WebGLCustomRenderTargetRuntimeOptions {
	/** @internal Restores the WebGL frame executor baseline after a custom pass. */
	readonly restoreFrameState?: (context: FrameContext) => void;
	/** @internal Records a prepared scene view into an owned target. */
	readonly executeSceneView?: (
		context: FrameContext,
		job: PreparedRenderTargetJob,
		target: {
			framebuffer: WebGLFramebuffer;
			width: number;
			height: number;
		},
	) => void | Promise<void>;
}

export class WebGLCustomRenderTargetRuntime {
	private readonly _gl: WebGL2RenderingContext;
	private readonly _restoreFrameState: (context: FrameContext) => void;
	private readonly _executeSceneView?: WebGLCustomRenderTargetRuntimeOptions["executeSceneView"];
	private readonly _supportsFloatColorBuffer: boolean;
	private readonly _targets = new Map<string, WebGLCustomRenderTarget>();
	private _lastSuccessfulFrame = false;

	public constructor(
		gl: WebGL2RenderingContext,
		options: WebGLCustomRenderTargetRuntimeOptions = {}
	) {
		this._gl = gl;
		this._restoreFrameState =
			options.restoreFrameState ?? ((context) => restoreNeutralFrameState(gl, context));
		this._executeSceneView = options.executeSceneView;
		this._supportsFloatColorBuffer =
			typeof gl.getExtension === "function" &&
			Boolean(gl.getExtension("EXT_color_buffer_float"));
	}

	public sync(context: FrameContext): void {
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
			const sceneDepth = current?.sceneDepth === true ||
				context.renderTargetJobs?.getForTarget(descriptor.id).some(
					(job) => job.descriptor.kind === "scene-view",
				) === true;
			this._validateDescriptor(descriptor, width, height, sampleCount);
			if (
				current &&
				current.width === width &&
				current.height === height &&
				current.sampleCount === sampleCount &&
				current.sceneDepth === sceneDepth &&
				JSON.stringify(current.descriptor) === JSON.stringify(descriptor)
			) {
				continue;
			}
			const replacement = this._createTarget(descriptor, width, height, sceneDepth);
			this._targets.set(descriptor.id, replacement);
			if (current) {
				this._destroyTargetResources(current);
			}
		}
	}

	private _validateDescriptor(
		descriptor: RenderTargetDescriptor,
		width: number,
		height: number,
		sampleCount: number
	): void {
		if (sampleCount !== 1) {
			throw new Error(
				`WebGL custom render target "${descriptor.id}" sampleCount must be 1.`
			);
		}
		const gl = this._gl;
		const maxTextureSize = resolveWebGLLimit(gl, gl.MAX_TEXTURE_SIZE, 1);
		if (width > maxTextureSize || height > maxTextureSize) {
			throw new Error(
				`WebGL custom render target "${descriptor.id}" dimensions ${width}x${height} ` +
					`exceed MAX_TEXTURE_SIZE=${maxTextureSize}.`
			);
		}
		const maxDrawBuffers = resolveWebGLLimit(gl, gl.MAX_DRAW_BUFFERS, 1);
		const maxColorAttachments = resolveWebGLLimit(gl, gl.MAX_COLOR_ATTACHMENTS, 1);
		const colorLimit = Math.min(maxDrawBuffers, maxColorAttachments);
		if (descriptor.color.length > colorLimit) {
			throw new Error(
				`WebGL custom render target "${descriptor.id}" exceeds the runtime ` +
					`color attachment limit ${colorLimit}.`
			);
		}
		for (let index = 0; index < descriptor.color.length; index++) {
			const format = descriptor.color[index].format;
			const resolved = resolveWebGLColorRenderTargetFormat(
				gl,
				format,
				this._supportsFloatColorBuffer
			);
			if (!resolved) {
				if (isWebGLFloatColorRenderTargetFormat(format)) {
					throw new Error(
						`WebGL custom render target "${descriptor.id}" requires ` +
							`EXT_color_buffer_float for color attachment ${index}.`
					);
				}
				throw new Error(
					`WebGL custom render target "${descriptor.id}" color attachment ` +
						`${index} format "${format}" is unsupported.`
				);
			}
		}
		if (
			descriptor.depth &&
			!resolveWebGLDepthRenderTargetFormat(gl, descriptor.depth.format)
		) {
			throw new Error(
				`WebGL custom render target "${descriptor.id}" depth format ` +
					`"${descriptor.depth.format}" is unsupported.`
			);
		}
	}

	public hasPass(pass: FramePass, context: FrameContext): boolean {
		return pass.stage === "render-target-views" &&
			(context.renderTargetJobs?.size ?? 0) > 0;
	}

	public async executePass(pass: FramePass, context: FrameContext): Promise<void> {
		if (pass.stage === "render-target-views") {
			for (const job of context.renderTargetJobs?.getAll() ?? []) {
				const target = this._targets.get(job.targetId);
				if (!target) continue;
				if (job.descriptor.kind === "scene-view") {
					if (!this._executeSceneView) {
						throw new Error("WebGL scene-view target execution is unavailable.");
					}
					await this._executeSceneView(context, job, target);
					this._restoreFrameState(context);
					continue;
				}
				const encoder = new WebGLScopedRasterEncoder(this._gl);
				try {
					await job.descriptor.execute({
						backend: "webgl",
						frameContext: context,
						encoder,
						target: toExecutionTarget(target),
						width: target.width,
						height: target.height,
						resources: createWebGLRasterResourceFacade(this._gl),
					});
					encoder.finish();
				} finally {
					encoder.cleanup();
					this._restoreFrameState(context);
				}
			}
		}
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
		const gl = this._gl;
		const width = resolveReadbackDimension(id, "width", options.width, target.width);
		const height = resolveReadbackDimension(id, "height", options.height, target.height);
		const format = target.descriptor.color[attachmentIndex]?.format ??
			TextureFormat.RGBA8Unorm;
		const resolved = texture._webglColorFormat;
		if (!resolved) {
			throw new Error(
				`Render target "${id}" color attachment ${attachmentIndex} has no readback format.`
			);
		}
		const bytesPerPixel = resolved.bytesPerPixel;
		const bytesPerRow = width * bytesPerPixel;
		let bytes: Uint8Array;
		gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
		try {
			gl.readBuffer(gl.COLOR_ATTACHMENT0 + attachmentIndex);
			if (resolved.readType === gl.FLOAT) {
				const data = new Float32Array(width * height * resolved.channelCount);
				gl.readPixels(0, 0, width, height, resolved.format, gl.FLOAT, data);
				bytes = resolved.repackFloat16 ? packFloat16(data) :
					new Uint8Array(data.buffer.slice(0));
			} else {
				bytes = new Uint8Array(width * height * resolved.channelCount);
				gl.readPixels(
					0,
					0,
					width,
					height,
					resolved.format,
					resolved.readType,
					bytes
				);
			}
		} finally {
			gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		}
		return createTextureReadbackResult({
			bytes: flipReadbackRows(bytes, bytesPerRow, height),
			width,
			height,
			format,
			bytesPerPixel,
			bytesPerRow,
			origin: "top-left",
		});
	}

	public destroy(): void {
		for (const id of Array.from(this._targets.keys())) {
			this._destroyTarget(id);
		}
		this._lastSuccessfulFrame = false;
	}

	private _createTarget(
		descriptor: RenderTargetDescriptor,
		width: number,
		height: number,
		sceneDepth: boolean,
	): WebGLCustomRenderTarget {
		const gl = this._gl;
		const framebuffer = gl.createFramebuffer();
		if (!framebuffer) {
			throw new Error(`Failed to create WebGL custom framebuffer "${descriptor.id}".`);
		}
		const color: WebGLCustomTexture[] = [];
		let depth: WebGLCustomTexture | null = null;
		try {
			for (let index = 0; index < descriptor.color.length; index++) {
				const attachment = descriptor.color[index];
				color.push(
					createWebGLColorTexture(
						gl,
						width,
						height,
						attachment.format,
						this._supportsFloatColorBuffer,
						attachment.label ??
							`WebGLCustomRenderTarget_${descriptor.id}_Color${index}`
					)
				);
			}
			if (descriptor.depth || sceneDepth) {
				depth = createWebGLDepthTexture(
					gl,
					width,
					height,
					descriptor.depth?.format ?? TextureFormat.Depth32Float,
					descriptor.depth?.label ??
						`WebGLCustomRenderTarget_${descriptor.id}_Depth`
				);
			}
			gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
			for (let index = 0; index < color.length; index++) {
				gl.framebufferTexture2D(
					gl.FRAMEBUFFER,
					gl.COLOR_ATTACHMENT0 + index,
					gl.TEXTURE_2D,
					color[index]._gpuResource,
					0
				);
			}
			if (depth) {
				gl.framebufferTexture2D(
					gl.FRAMEBUFFER,
					gl.DEPTH_ATTACHMENT,
					gl.TEXTURE_2D,
					depth._gpuResource,
					0
				);
			}
			gl.drawBuffers(color.map((_texture, index) => gl.COLOR_ATTACHMENT0 + index));
			const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
			if (status !== gl.FRAMEBUFFER_COMPLETE) {
				throw new Error(
					`WebGL custom framebuffer "${descriptor.id}" is incomplete ` +
						`(status=0x${status.toString(16)}).`
				);
			}
			return {
				descriptor,
				width,
				height,
				sampleCount: 1,
				sceneDepth,
				framebuffer,
				color,
				depth,
			};
		} catch (error) {
			for (const texture of color) {
				texture.destroy();
			}
			depth?.destroy();
			gl.deleteFramebuffer(framebuffer);
			throw error;
		} finally {
			gl.bindFramebuffer(gl.FRAMEBUFFER, null);
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

	private _destroyTargetResources(target: WebGLCustomRenderTarget): void {
		for (const texture of target.color) {
			texture.destroy();
		}
		target.depth?.destroy();
		this._gl.deleteFramebuffer(target.framebuffer);
	}
}

function flipReadbackRows(
	bytes: Uint8Array,
	bytesPerRow: number,
	height: number,
): Uint8Array {
	const flipped = new Uint8Array(bytes.length);
	for (let row = 0; row < height; row++) {
		const source = row * bytesPerRow;
		const target = (height - row - 1) * bytesPerRow;
		flipped.set(bytes.subarray(source, source + bytesPerRow), target);
	}
	return flipped;
}

export class WebGLScopedRasterEncoder implements ICommandEncoder {
	private readonly _gl: WebGL2RenderingContext;
	private readonly _scopeState: { active: boolean } | null;
	private _vertexArray: WebGLVertexArrayObject | null;
	private _framebuffer: WebGLFramebuffer | null = null;
	private _pipeline: WebGLCustomPipeline | null = null;
	private readonly _vertexBuffers = new Map<number, IRenderBuffer>();
	private _indexBuffer: IRenderBuffer | null = null;
	private _indexFormat: IndexFormat = "uint16";
	private _discardAttachments: number[] = [];

	public constructor(
		gl: WebGL2RenderingContext,
		scopeState: { active: boolean } | null = null,
	) {
		this._gl = gl;
		this._scopeState = scopeState;
		this._vertexArray = typeof gl.createVertexArray === "function" ?
			gl.createVertexArray() : null;
	}

	public beginRenderPass(desc: RenderPassDesc): void {
		this._assertScopeActive();
		if (this._framebuffer) {
			throw new Error("WebGL custom render pass is already active.");
		}
		for (const attachment of desc.colorAttachments) {
			this._assertScopeResource(attachment.view);
			if (attachment.resolveTarget) {
				throw new Error("WebGL custom render passes do not support resolve targets.");
			}
		}
		const size = resolveRenderPassSize(desc);
		const gl = this._gl;
		if (this._vertexArray) gl.bindVertexArray(this._vertexArray);
		const framebuffer = gl.createFramebuffer();
		if (!framebuffer) {
			throw new Error("Failed to create WebGL custom render pass framebuffer.");
		}
		this._framebuffer = framebuffer;
		gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
		desc.colorAttachments.forEach((attachment, index) => {
			const texture = resolveWebGLTexture(attachment.view);
			gl.framebufferTexture2D(
				gl.FRAMEBUFFER,
				gl.COLOR_ATTACHMENT0 + index,
				gl.TEXTURE_2D,
				texture,
				0
			);
		});
		if (desc.colorAttachments.length > 0) {
			gl.drawBuffers(
				desc.colorAttachments.map((_attachment, index) =>
					gl.COLOR_ATTACHMENT0 + index
				)
			);
		}
		if (desc.depthStencilAttachment) {
			this._assertScopeResource(desc.depthStencilAttachment.view);
			this._attachDepth(desc.depthStencilAttachment);
		}
		const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
		if (status !== gl.FRAMEBUFFER_COMPLETE) {
			this.cleanup();
			throw new Error(
				`WebGL custom render pass framebuffer is incomplete ` +
					`(status=0x${status.toString(16)}).`
			);
		}
		gl.viewport(0, 0, size.width, size.height);
		gl.disable(gl.SCISSOR_TEST);
		desc.colorAttachments.forEach((attachment, index) => {
			if (attachment.loadOp === "clear") {
				gl.clearBufferfv(
					gl.COLOR,
					index,
					colorClearValueToArray(attachment)
				);
			}
		});
		if (desc.depthStencilAttachment) {
			if (desc.depthStencilAttachment.depthLoadOp === "clear") {
				gl.clearBufferfv(
					gl.DEPTH,
					0,
					new Float32Array([
						desc.depthStencilAttachment.depthClearValue ?? 1,
					])
				);
			}
		}
		this._discardAttachments = desc.colorAttachments.flatMap((attachment, index) =>
			attachment.storeOp === "discard" ? [gl.COLOR_ATTACHMENT0 + index] : []
		);
		if (desc.depthStencilAttachment?.depthStoreOp === "discard") {
			this._discardAttachments.push(gl.DEPTH_ATTACHMENT);
		}
		void desc.label;
	}

	public beginComputePass(_desc?: ComputePassDesc): void {
		throw new Error("WebGL custom render passes do not support compute passes.");
	}

	public setPipeline(pipeline: IRenderPipeline): void {
		this._assertScopeResource(pipeline);
		const customPipeline = pipeline as WebGLCustomPipeline;
		if (!customPipeline._webglProgram) {
			throw new Error("WebGL custom render pass received an incompatible pipeline.");
		}
		this._pipeline = customPipeline;
		const gl = this._gl;
		gl.useProgram(customPipeline._webglProgram);
		applyPipelineState(gl, customPipeline._webglDesc);
	}

	public setBindingGroup(_index: number, group: IBindingGroup): void {
		this._assertScopeResource(group);
		const pipeline = this._requirePipeline();
		const customGroup = group as WebGLCustomBindingGroup;
		if (!customGroup._webglEntries) {
			throw new Error("WebGL custom render pass received an incompatible binding group.");
		}
		bindGroupResources(this._gl, pipeline, customGroup._webglEntries);
	}

	public setVertexBuffer(slot: number, buffer: IRenderBuffer): void {
		this._assertScopeResource(buffer);
		this._vertexBuffers.set(slot, buffer);
	}

	public setIndexBuffer(buffer: IRenderBuffer, format: IndexFormat): void {
		this._assertScopeResource(buffer);
		this._indexBuffer = buffer;
		this._indexFormat = format;
	}

	public drawIndexed(
		indexCount: number,
		instanceCount = 1,
		firstIndex = 0,
		baseVertex = 0,
		firstInstance = 0
	): void {
		this._assertScopeActive();
		if (baseVertex !== 0) {
			throw new Error("WebGL custom indexed draws require baseVertex=0.");
		}
		if (firstInstance !== 0) {
			throw new Error("WebGL custom indexed draws require firstInstance=0.");
		}
		const pipeline = this._requirePipeline();
		const indexBuffer = this._indexBuffer as WebGLCustomBuffer | null;
		if (!indexBuffer?._webglBuffer) {
			throw new Error("WebGL custom indexed draw requires an index buffer.");
		}
		this._bindVertexBuffers(pipeline);
		const gl = this._gl;
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer._webglBuffer);
		const indexType = this._indexFormat === "uint32" ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
		const indexSize = this._indexFormat === "uint32" ? 4 : 2;
		const mode = primitiveTopologyToGL(gl, pipeline._webglDesc);
		if (instanceCount > 1) {
			gl.drawElementsInstanced(
				mode,
				indexCount,
				indexType,
				firstIndex * indexSize,
				instanceCount
			);
			return;
		}
		gl.drawElements(mode, indexCount, indexType, firstIndex * indexSize);
	}

	public draw(
		vertexCount: number,
		instanceCount = 1,
		firstVertex = 0,
		firstInstance = 0
	): void {
		this._assertScopeActive();
		if (firstInstance !== 0) {
			throw new Error("WebGL custom draws require firstInstance=0.");
		}
		const pipeline = this._requirePipeline();
		this._bindVertexBuffers(pipeline);
		const gl = this._gl;
		const mode = primitiveTopologyToGL(gl, pipeline._webglDesc);
		if (instanceCount > 1) {
			gl.drawArraysInstanced(mode, firstVertex, vertexCount, instanceCount);
			return;
		}
		gl.drawArrays(mode, firstVertex, vertexCount);
	}
	public setScissorRect(x: number, y: number, width: number, height: number): void {
		this._assertScopeActive();
		this._gl.enable(this._gl.SCISSOR_TEST);
		this._gl.scissor(x, y, width, height);
	}

	public setViewport(
		x: number,
		y: number,
		width: number,
		height: number,
	): void {
		this._assertScopeActive();
		for (const value of [x, y, width, height]) {
			if (!Number.isFinite(value) || value < 0) {
				throw new Error("WebGL auxiliary raster viewport values must be finite and non-negative.");
			}
		}
		this._gl.viewport(x, y, width, height);
	}

	public setUniforms(uniforms: readonly WebGLAuxiliaryUniform[]): void {
		this._assertScopeActive();
		const pipeline = this._requirePipeline();
		for (const uniform of uniforms) {
			applyNamedUniform(
				this._gl,
				pipeline._webglProgram,
				uniform,
			);
		}
	}
	public copyTextureToTexture(
		_source: TextureCopyView,
		_destination: TextureCopyView,
		_copySize: TextureCopySize
	): void {
		throw new Error("WebGL custom render passes do not support texture copies.");
	}
	public endRenderPass(): void {
		this._assertScopeActive();
		if (!this._framebuffer) {
			throw new Error("WebGL custom render pass is not active.");
		}
		if (this._discardAttachments.length > 0) {
			this._gl.invalidateFramebuffer(
				this._gl.FRAMEBUFFER,
				this._discardAttachments
			);
		}
		this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, null);
		this._gl.deleteFramebuffer(this._framebuffer);
		this._framebuffer = null;
		this._discardAttachments = [];
	}
	public setComputePipeline(_pipeline: IComputePipeline): void {
		throw new Error("WebGL custom render passes do not support compute pipelines.");
	}
	public dispatchWorkgroups(): void {
		throw new Error("WebGL custom render passes do not support compute dispatch.");
	}
	public endComputePass(): void {
		throw new Error("WebGL custom render passes do not support compute passes.");
	}
	public finish(): ICommandBuffer {
		if (this._framebuffer) {
			throw new Error("WebGL custom render pass callback left a pass active.");
		}
		return {};
	}

	public cleanup(): void {
		if (this._framebuffer) {
			this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, null);
			this._gl.deleteFramebuffer(this._framebuffer);
			this._framebuffer = null;
		}
		this._discardAttachments = [];
		if (this._vertexArray) {
			this._gl.bindVertexArray(null);
			this._gl.deleteVertexArray(this._vertexArray);
			this._vertexArray = null;
		}
	}

	private _assertScopeActive(): void {
		if (this._scopeState && !this._scopeState.active) {
			throw new Error("WebGL auxiliary raster scope is no longer active.");
		}
	}

	private _assertScopeResource(resource: unknown): void {
		this._assertScopeActive();
		if (
			this._scopeState &&
			(resource as { _webglScopeState?: unknown } | null)?._webglScopeState !==
				this._scopeState
		) {
			throw new Error(
				"WebGL auxiliary raster resources must belong to the active scope.",
			);
		}
	}

	private _requirePipeline(): WebGLCustomPipeline {
		if (!this._pipeline) {
			throw new Error("WebGL custom render pass draw requires a pipeline.");
		}
		return this._pipeline;
	}

	private _bindVertexBuffers(pipeline: WebGLCustomPipeline): void {
		const layouts = pipeline._webglDesc.vertex.buffers ?? [];
		for (let slot = 0; slot < layouts.length; slot++) {
			const buffer = this._vertexBuffers.get(slot) as WebGLCustomBuffer | undefined;
			if (!buffer?._webglBuffer) {
				continue;
			}
			const layout = layouts[slot];
			this._gl.bindBuffer(this._gl.ARRAY_BUFFER, buffer._webglBuffer);
			for (const attribute of layout.attributes) {
				bindVertexAttribute(this._gl, attribute, layout.arrayStride);
				if (layout.stepMode === "instance") {
					this._gl.vertexAttribDivisor(attribute.shaderLocation, 1);
				}
			}
		}
	}

	private _attachDepth(attachment: DepthStencilAttachment): void {
		const resource = (attachment.view as WebGLCustomTexture | undefined)?._gpuResource;
		if (!resource) {
			return;
		}
		this._gl.framebufferTexture2D(
			this._gl.FRAMEBUFFER,
			this._gl.DEPTH_ATTACHMENT,
			this._gl.TEXTURE_2D,
			resource,
			0
		);
	}
}

function createWebGLColorTexture(
	gl: WebGL2RenderingContext,
	width: number,
	height: number,
	format: TextureFormat,
	floatColorSupported: boolean,
	label: string
): WebGLCustomTexture {
	const texture = gl.createTexture();
	if (!texture) {
		throw new Error(`Failed to create ${label}.`);
	}
	const resolved = resolveWebGLColorRenderTargetFormat(
		gl,
		format,
		floatColorSupported
	);
	if (!resolved) {
		gl.deleteTexture(texture);
		throw new Error(`Unsupported WebGL custom render target format "${format}".`);
	}
	gl.bindTexture(gl.TEXTURE_2D, texture);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	gl.texImage2D(
		gl.TEXTURE_2D,
		0,
		resolved.internalFormat,
		width,
		height,
		0,
		resolved.format,
		resolved.allocationType,
		null
	);
	gl.bindTexture(gl.TEXTURE_2D, null);
	return {
		width,
		height,
		format,
		requestedFormat: format,
		_gpuResource: texture,
		_webglTargetKind: "texture",
		_webglColorFormat: resolved,
		destroy: () => gl.deleteTexture(texture),
	};
}

function createWebGLTexture(
	gl: WebGL2RenderingContext,
	desc: TextureDesc
): WebGLCustomTexture {
	if (desc.dimension && desc.dimension !== "2d") {
		throw new Error("WebGL custom render pass textures only support 2d dimension.");
	}
	if ((desc.sampleCount ?? 1) !== 1) {
		throw new Error("WebGL custom render pass textures only support sampleCount=1.");
	}
	if (isDepthFormat(desc.format) && (desc.usage & TextureUsage.RenderAttachment) !== 0) {
		return createWebGLDepthTexture(
			gl,
			desc.width,
			desc.height,
			desc.format,
			desc.label ?? "WebGLCustomTexture_Depth"
		);
	}
	const requiresRenderableFloat =
		(desc.usage & TextureUsage.RenderAttachment) !== 0;
	return createWebGLColorTexture(
		gl,
		desc.width,
		desc.height,
		desc.format,
		!requiresRenderableFloat ||
			(typeof gl.getExtension === "function" &&
				Boolean(gl.getExtension("EXT_color_buffer_float"))),
		desc.label ?? "WebGLCustomTexture_Color"
	);
}

function createWebGLDepthTexture(
	gl: WebGL2RenderingContext,
	width: number,
	height: number,
	format: TextureFormat,
	label: string
): WebGLCustomTexture {
	const texture = gl.createTexture();
	if (!texture) {
		throw new Error(`Failed to create ${label}.`);
	}
	const resolved = resolveWebGLDepthRenderTargetFormat(gl, format);
	if (!resolved) {
		gl.deleteTexture(texture);
		throw new Error(`Unsupported WebGL custom depth format "${format}".`);
	}
	gl.bindTexture(gl.TEXTURE_2D, texture);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	gl.texImage2D(
		gl.TEXTURE_2D,
		0,
		resolved.internalFormat,
		width,
		height,
		0,
		resolved.format,
		resolved.type,
		null
	);
	gl.bindTexture(gl.TEXTURE_2D, null);
	return {
		width,
		height,
		format,
		requestedFormat: format,
		_gpuResource: texture,
		_webglTargetKind: "texture",
		destroy: () => gl.deleteTexture(texture),
	};
}

function isDepthFormat(format: TextureFormat): boolean {
	return format === TextureFormat.Depth16Unorm ||
		format === TextureFormat.Depth24Plus ||
		format === TextureFormat.Depth32Float ||
		format === TextureFormat.Depth24PlusStencil8 ||
		format === TextureFormat.Depth32FloatStencil8;
}

function resolveWebGLTexture(texture: IRenderTexture | undefined): WebGLTexture {
	const resource = (texture as WebGLCustomTexture | undefined)?._gpuResource;
	if (
		!resource ||
		(texture as WebGLCustomTexture | undefined)?._webglTargetKind !== "texture"
	) {
		throw new Error("Render pass color attachment is not a WebGL texture.");
	}
	return resource;
}

function colorClearValueToArray(attachment: ColorAttachment): Float32Array {
	const clear = attachment.clearValue ?? { r: 0, g: 0, b: 0, a: 0 };
	return new Float32Array([clear.r, clear.g, clear.b, clear.a]);
}

function toExecutionTarget(
	target: WebGLCustomRenderTarget
): CustomRenderTargetExecutionTarget {
	return {
		id: target.descriptor.id,
		width: target.width,
		height: target.height,
		sampleCount: target.sampleCount,
		color: target.color.map((texture, index) => ({
			texture,
			format: target.descriptor.color[index].format,
			resolveTexture: null,
		})),
		depth: target.depth && target.descriptor.depth ? {
			texture: target.depth,
			format: target.descriptor.depth.format,
			resolveTexture: null,
		} : null,
	};
}

export function createWebGLRasterResourceFacade(
	gl: WebGL2RenderingContext
): CustomRenderPassResourceFacade {
	return {
		createBuffer: (desc) => createWebGLBuffer(gl, desc),
		createTexture: (desc) => createWebGLTexture(gl, desc),
		createSampler: (desc) => createWebGLSampler(gl, desc),
		createShaderModule: (desc) => createWebGLShaderModule(desc),
		createRenderPipeline: (desc) => createWebGLPipeline(gl, desc),
		createBindingGroup: (desc) => createWebGLBindingGroup(desc),
	};
}

function createWebGLBuffer(
	gl: WebGL2RenderingContext,
	desc: BufferDesc
): WebGLCustomBuffer {
	const buffer = gl.createBuffer();
	if (!buffer) {
		throw new Error(`Failed to create WebGL custom buffer "${desc.label ?? "unnamed"}".`);
	}
	const target = (desc.usage & BufferUsage.Index) !== 0 ?
		gl.ELEMENT_ARRAY_BUFFER
	:	gl.ARRAY_BUFFER;
	const data = desc.initialData ?? desc.size;
	gl.bindBuffer(target, buffer);
	if (typeof data === "number") {
		gl.bufferData(target, data, gl.STATIC_DRAW);
	} else {
		gl.bufferData(target, data, gl.STATIC_DRAW);
	}
	gl.bindBuffer(target, null);
	return {
		size: desc.size,
		_webglBuffer: buffer,
		_webglUsage: desc.usage,
		_cpuData: desc.mappedAtCreation ? new ArrayBuffer(desc.size) : undefined,
		destroy: () => gl.deleteBuffer(buffer),
		unmap: () => {},
	};
}

function createWebGLSampler(
	gl: WebGL2RenderingContext,
	desc: SamplerDesc
): WebGLCustomSampler {
	const sampler = gl.createSampler();
	if (!sampler) {
		throw new Error(`Failed to create WebGL custom sampler "${desc.label ?? "unnamed"}".`);
	}
	gl.samplerParameteri(
		sampler,
		gl.TEXTURE_MIN_FILTER,
		filterModeToGL(gl, desc.minFilter, desc.mipmapFilter)
	);
	gl.samplerParameteri(
		sampler,
		gl.TEXTURE_MAG_FILTER,
		desc.magFilter === FilterMode.Nearest ? gl.NEAREST : gl.LINEAR
	);
	gl.samplerParameteri(
		sampler,
		gl.TEXTURE_WRAP_S,
		addressModeToGL(gl, desc.addressModeU)
	);
	gl.samplerParameteri(
		sampler,
		gl.TEXTURE_WRAP_T,
		addressModeToGL(gl, desc.addressModeV)
	);
	return {
		label: desc.label,
		_webglSampler: sampler,
	};
}

function createWebGLShaderModule(desc: ShaderModuleDesc): WebGLCustomShaderModule {
	return {
		label: desc.label,
		_webglCode: desc.code,
		_webglStage: desc.stage,
	};
}

function createWebGLPipeline(
	gl: WebGL2RenderingContext,
	desc: PipelineDesc
): WebGLCustomPipeline {
	const vertex = compileWebGLShader(
		gl,
		gl.VERTEX_SHADER,
		desc.vertex.module as WebGLCustomShaderModule,
		desc.label
	);
	const fragment = desc.fragment ?
		compileWebGLShader(
			gl,
			gl.FRAGMENT_SHADER,
			desc.fragment.module as WebGLCustomShaderModule,
			desc.label
		)
	:	null;
	const program = gl.createProgram();
	if (!program) {
		gl.deleteShader(vertex);
		if (fragment) {
			gl.deleteShader(fragment);
		}
		throw new Error(`Failed to create WebGL custom pipeline "${desc.label ?? "unnamed"}".`);
	}
	gl.attachShader(program, vertex);
	if (fragment) {
		gl.attachShader(program, fragment);
	}
	gl.linkProgram(program);
	gl.deleteShader(vertex);
	if (fragment) {
		gl.deleteShader(fragment);
	}
	if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
		const log = gl.getProgramInfoLog(program) ?? "unknown link error";
		gl.deleteProgram(program);
		throw new Error(
			`Failed to link WebGL custom pipeline "${desc.label ?? "unnamed"}": ${log}`
		);
	}
	return {
		label: desc.label,
		_webglProgram: program,
		_webglDesc: desc,
		destroy: () => gl.deleteProgram(program),
	} as WebGLCustomPipeline & { destroy(): void };
}

function compileWebGLShader(
	gl: WebGL2RenderingContext,
	type: number,
	module: WebGLCustomShaderModule,
	label?: string
): WebGLShader {
	if (!module._webglCode) {
		throw new Error(`WebGL custom pipeline "${label ?? "unnamed"}" has no GLSL source.`);
	}
	const shader = gl.createShader(type);
	if (!shader) {
		throw new Error(`Failed to create WebGL custom shader "${label ?? "unnamed"}".`);
	}
	gl.shaderSource(shader, module._webglCode);
	gl.compileShader(shader);
	if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
		const log = gl.getShaderInfoLog(shader) ?? "unknown compile error";
		gl.deleteShader(shader);
		throw new Error(
			`Failed to compile WebGL custom shader "${label ?? "unnamed"}": ${log}`
		);
	}
	return shader;
}

function createWebGLBindingGroup(desc: BindingGroupDesc): WebGLCustomBindingGroup {
	return {
		label: desc.label,
		_webglEntries: desc.entries,
		destroy: () => {},
	} as WebGLCustomBindingGroup & { destroy(): void };
}

function bindGroupResources(
	gl: WebGL2RenderingContext,
	pipeline: WebGLCustomPipeline,
	entries: readonly BindingEntry[]
): void {
	for (const entry of entries) {
		const resource = entry.resource as
			| WebGLCustomBuffer
			| WebGLCustomTexture
			| WebGLCustomSampler
			| undefined;
		if ((resource as WebGLCustomTexture | undefined)?._webglTargetKind === "texture") {
			gl.activeTexture(gl.TEXTURE0 + entry.binding);
			gl.bindTexture(
				gl.TEXTURE_2D,
				(resource as WebGLCustomTexture)._gpuResource as WebGLTexture
			);
			const location = findSamplerUniform(gl, pipeline._webglProgram, entry.binding);
			if (location) {
				gl.uniform1i(location, entry.binding);
			}
			continue;
		}
		if ((resource as WebGLCustomSampler | undefined)?._webglSampler) {
			gl.bindSampler(entry.binding, (resource as WebGLCustomSampler)._webglSampler);
			continue;
		}
		if ((resource as WebGLCustomBuffer | undefined)?._webglBuffer) {
			const buffer = resource as WebGLCustomBuffer;
			gl.bindBufferBase(gl.UNIFORM_BUFFER, entry.binding, buffer._webglBuffer);
			const blockIndex = gl.getUniformBlockIndex(
				pipeline._webglProgram,
				`Binding${entry.binding}`
			);
			if (blockIndex !== gl.INVALID_INDEX) {
				gl.uniformBlockBinding(pipeline._webglProgram, blockIndex, entry.binding);
			}
		}
	}
}

function findSamplerUniform(
	gl: WebGL2RenderingContext,
	program: WebGLProgram,
	binding: number
): WebGLUniformLocation | null {
	return gl.getUniformLocation(program, `u_binding${binding}`) ??
		gl.getUniformLocation(program, `u_texture${binding}`) ??
		gl.getUniformLocation(program, `u_sampler${binding}`);
}

function applyNamedUniform(
	gl: WebGL2RenderingContext,
	program: WebGLProgram,
	uniform: WebGLAuxiliaryUniform,
): void {
	if (!uniform.name) {
		throw new Error("WebGL auxiliary raster uniform names must not be empty.");
	}
	const location = gl.getUniformLocation(program, uniform.name);
	if (location === null) return;
	const value = uniform.value;
	if (typeof value === "number") {
		if (!Number.isFinite(value)) {
			throw new Error(`WebGL auxiliary raster uniform "${uniform.name}" is not finite.`);
		}
		switch (uniform.type) {
			case "f32": gl.uniform1f(location, value); return;
			case "i32": gl.uniform1i(location, value); return;
			case "u32": gl.uniform1ui(location, value); return;
			default:
				throw new Error(
					`WebGL auxiliary raster uniform "${uniform.name}" requires flat array data.`,
				);
		}
	}

	const values = Array.from(value);
	const width = resolveUniformElementWidth(uniform.type);
	if (values.length === 0 || values.length % width !== 0) {
		throw new Error(
			`WebGL auxiliary raster uniform "${uniform.name}" data length ` +
				`${values.length} is not a positive multiple of ${width}.`,
		);
	}
	if (values.some((entry) => !Number.isFinite(entry))) {
		throw new Error(`WebGL auxiliary raster uniform "${uniform.name}" is not finite.`);
	}
	const floats = () => Float32Array.from(values);
	const ints = () => Int32Array.from(values);
	const uints = () => Uint32Array.from(values);
	switch (uniform.type) {
		case "f32": gl.uniform1fv(location, floats()); return;
		case "i32": gl.uniform1iv(location, ints()); return;
		case "u32": gl.uniform1uiv(location, uints()); return;
		case "vec2f": gl.uniform2fv(location, floats()); return;
		case "vec3f": gl.uniform3fv(location, floats()); return;
		case "vec4f": gl.uniform4fv(location, floats()); return;
		case "vec2i": gl.uniform2iv(location, ints()); return;
		case "vec3i": gl.uniform3iv(location, ints()); return;
		case "vec4i": gl.uniform4iv(location, ints()); return;
		case "vec2u": gl.uniform2uiv(location, uints()); return;
		case "vec3u": gl.uniform3uiv(location, uints()); return;
		case "vec4u": gl.uniform4uiv(location, uints()); return;
		case "mat3x3f": gl.uniformMatrix3fv(location, false, floats()); return;
		case "mat4x4f": gl.uniformMatrix4fv(location, false, floats()); return;
	}
}

function resolveUniformElementWidth(
	type: WebGLAuxiliaryUniform["type"],
): number {
	switch (type) {
		case "f32":
		case "i32":
		case "u32": return 1;
		case "vec2f":
		case "vec2i":
		case "vec2u": return 2;
		case "vec3f":
		case "vec3i":
		case "vec3u": return 3;
		case "vec4f":
		case "vec4i":
		case "vec4u": return 4;
		case "mat3x3f": return 9;
		case "mat4x4f": return 16;
	}
}

function bindVertexAttribute(
	gl: WebGL2RenderingContext,
	attribute: VertexAttribute,
	stride: number
): void {
	const layout = vertexFormatToGL(gl, attribute.format);
	gl.enableVertexAttribArray(attribute.shaderLocation);
	if (layout.integer) {
		gl.vertexAttribIPointer(
			attribute.shaderLocation,
			layout.size,
			layout.type,
			stride,
			attribute.offset
		);
		return;
	}
	gl.vertexAttribPointer(
		attribute.shaderLocation,
		layout.size,
		layout.type,
		layout.normalized,
		stride,
		attribute.offset
	);
}

function vertexFormatToGL(
	gl: WebGL2RenderingContext,
	format: VertexAttribute["format"]
): { size: number; type: number; normalized: boolean; integer: boolean } {
	switch (format) {
		case "float32":
			return { size: 1, type: gl.FLOAT, normalized: false, integer: false };
		case "float32x2":
			return { size: 2, type: gl.FLOAT, normalized: false, integer: false };
		case "float32x3":
			return { size: 3, type: gl.FLOAT, normalized: false, integer: false };
		case "float32x4":
			return { size: 4, type: gl.FLOAT, normalized: false, integer: false };
		case "float16x2":
			return { size: 2, type: gl.HALF_FLOAT, normalized: false, integer: false };
		case "snorm16x4":
			return { size: 4, type: gl.SHORT, normalized: true, integer: false };
		case "unorm16x4":
			return { size: 4, type: gl.UNSIGNED_SHORT, normalized: true, integer: false };
		case "uint32":
			return { size: 1, type: gl.UNSIGNED_INT, normalized: false, integer: true };
		case "uint32x2":
			return { size: 2, type: gl.UNSIGNED_INT, normalized: false, integer: true };
		case "uint32x3":
			return { size: 3, type: gl.UNSIGNED_INT, normalized: false, integer: true };
		case "uint32x4":
			return { size: 4, type: gl.UNSIGNED_INT, normalized: false, integer: true };
		case "unorm8x4":
			return {
				size: 4,
				type: gl.UNSIGNED_BYTE,
				normalized: true,
				integer: false,
			};
		default:
			throw new Error(`Unsupported WebGL custom vertex format "${format}".`);
	}
}

function primitiveTopologyToGL(
	gl: WebGL2RenderingContext,
	desc: PipelineDesc
): number {
	switch (desc.primitive?.topology ?? PrimitiveTopology.TriangleList) {
		case PrimitiveTopology.PointList:
			return gl.POINTS;
		case PrimitiveTopology.LineList:
			return gl.LINES;
		case PrimitiveTopology.TriangleStrip:
			return gl.TRIANGLE_STRIP;
		case PrimitiveTopology.TriangleList:
		default:
			return gl.TRIANGLES;
	}
}

function applyPipelineState(
	gl: WebGL2RenderingContext,
	desc: PipelineDesc
): void {
	const cullMode = desc.primitive?.cullMode ?? "none";
	if (cullMode === "none") {
		gl.disable(gl.CULL_FACE);
	} else {
		gl.enable(gl.CULL_FACE);
		gl.cullFace(cullMode === "front" ? gl.FRONT : gl.BACK);
	}
	gl.frontFace(desc.primitive?.frontFace === "cw" ? gl.CW : gl.CCW);
	if (desc.depthStencil) {
		gl.enable(gl.DEPTH_TEST);
		gl.depthMask(desc.depthStencil.depthWriteEnabled);
		gl.depthFunc(depthCompareToGL(gl, desc.depthStencil.depthCompare));
	} else {
		gl.disable(gl.DEPTH_TEST);
	}
	if (desc.fragment?.targets.some(hasBlendState)) {
		gl.enable(gl.BLEND);
	} else {
		gl.disable(gl.BLEND);
	}
}

function hasBlendState(target: ColorTargetState): boolean {
	return Boolean(target.blend);
}

function depthCompareToGL(
	gl: WebGL2RenderingContext,
	compare: PipelineDesc["depthStencil"]["depthCompare"]
): number {
	switch (compare) {
		case "always":
			return gl.ALWAYS;
		case "never":
			return gl.NEVER;
		case "equal":
			return gl.EQUAL;
		case "less-equal":
			return gl.LEQUAL;
		case "greater":
			return gl.GREATER;
		case "greater-equal":
			return gl.GEQUAL;
		case "less":
		default:
			return gl.LESS;
	}
}

function filterModeToGL(
	gl: WebGL2RenderingContext,
	filter?: FilterMode,
	mipmapFilter?: FilterMode
): number {
	if (mipmapFilter === FilterMode.Nearest) {
		return filter === FilterMode.Nearest ? gl.NEAREST_MIPMAP_NEAREST : gl.LINEAR_MIPMAP_NEAREST;
	}
	if (mipmapFilter === FilterMode.Linear) {
		return filter === FilterMode.Nearest ? gl.NEAREST_MIPMAP_LINEAR : gl.LINEAR_MIPMAP_LINEAR;
	}
	return filter === FilterMode.Nearest ? gl.NEAREST : gl.LINEAR;
}

function addressModeToGL(
	gl: WebGL2RenderingContext,
	mode?: AddressMode
): number {
	switch (mode) {
		case AddressMode.Repeat:
			return gl.REPEAT;
		case AddressMode.MirrorRepeat:
			return gl.MIRRORED_REPEAT;
		case AddressMode.ClampToEdge:
		default:
			return gl.CLAMP_TO_EDGE;
	}
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

function resolveWebGLLimit(
	gl: WebGL2RenderingContext,
	parameter: number,
	fallback: number
): number {
	if (typeof gl.getParameter !== "function" || !Number.isFinite(parameter)) {
		return fallback;
	}
	const value = Number(gl.getParameter(parameter));
	return Number.isFinite(value) && value > 0 ? Math.floor(value) : fallback;
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

function resolveRenderPassSize(desc: RenderPassDesc): {
	width: number;
	height: number;
} {
	const textures: IRenderTexture[] = [];
	for (const attachment of desc.colorAttachments) {
		if (!attachment.view) {
			throw new Error("WebGL custom render pass color attachments require a texture.");
		}
		textures.push(attachment.view);
	}
	if (desc.depthStencilAttachment) {
		if (!desc.depthStencilAttachment.view) {
			throw new Error("WebGL custom render pass depth attachment requires a texture.");
		}
		textures.push(desc.depthStencilAttachment.view);
	}
	const first = textures[0];
	if (!first) {
		throw new Error("WebGL custom render pass requires at least one attachment.");
	}
	for (const texture of textures) {
		if (texture.width !== first.width || texture.height !== first.height) {
			throw new Error("WebGL custom render pass attachment dimensions must match.");
		}
	}
	return {
		width: first.width,
		height: first.height,
	};
}

function restoreNeutralFrameState(
	gl: WebGL2RenderingContext,
	context: FrameContext
): void {
	gl.bindFramebuffer(gl.FRAMEBUFFER, null);
	gl.viewport(0, 0, context.attachments.width, context.attachments.height);
	gl.drawBuffers([gl.BACK]);
	gl.disable(gl.SCISSOR_TEST);
	gl.disable(gl.BLEND);
	gl.disable(gl.CULL_FACE);
	gl.enable(gl.DEPTH_TEST);
	gl.depthMask(true);
	gl.depthFunc(gl.LESS);
	gl.colorMask(true, true, true, true);
	gl.useProgram(null);
	gl.bindVertexArray(null);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
	gl.activeTexture(gl.TEXTURE0);
}

function createTextureReadbackResult(input: {
	bytes: Uint8Array;
	width: number;
	height: number;
	format: TextureFormat;
	bytesPerPixel: number;
	bytesPerRow: number;
	origin: RenderTargetReadbackResult["origin"];
}): RenderTargetReadbackResult {
	return {
		bytes: input.bytes,
		width: input.width,
		height: input.height,
		format: input.format,
		bytesPerPixel: input.bytesPerPixel,
		bytesPerRow: input.bytesPerRow,
		origin: input.origin,
		toFloat32: () =>
			new Float32Array(
				input.bytes.buffer,
				input.bytes.byteOffset,
				Math.floor(input.bytes.byteLength / 4)
			),
		toRGBAFloat32: () => decodeTextureReadbackToRGBAFloat32(input),
		toNormalizedRGBA8Float32: () => decodeNormalizedRGBA8Readback(input),
	};
}

function decodeTextureReadbackToRGBAFloat32(input: {
	bytes: Uint8Array;
	width: number;
	height: number;
	format: TextureFormat;
	bytesPerPixel: number;
	bytesPerRow: number;
}): Float32Array {
	const info = getTextureFormatInfo(input.format);
	if (
		info.formatClass !== "color" ||
		(info.componentType !== "unorm" && info.componentType !== "float")
	) {
		throw new Error(
			`toRGBAFloat32() does not support texture format "${input.format}".`
		);
	}
	const output = new Float32Array(input.width * input.height * 4);
	const view = new DataView(
		input.bytes.buffer,
		input.bytes.byteOffset,
		input.bytes.byteLength
	);
	const componentBytes = info.bytesPerBlock / info.channelCount;
	for (let y = 0; y < input.height; y++) {
		for (let x = 0; x < input.width; x++) {
			const sourcePixel = y * input.bytesPerRow + x * input.bytesPerPixel;
			const destinationPixel = (y * input.width + x) * 4;
			output[destinationPixel + 3] = 1;
			for (let channel = 0; channel < info.channelCount; channel++) {
				const offset = sourcePixel + channel * componentBytes;
				output[destinationPixel + channel] =
					info.componentType === "unorm" ? view.getUint8(offset) / 255
					: componentBytes === 2 ?
						float16BitsToFloat32(view.getUint16(offset, true))
					:	view.getFloat32(offset, true);
			}
		}
	}
	return output;
}

function decodeNormalizedRGBA8Readback(input: {
	bytes: Uint8Array;
	width: number;
	height: number;
	format: TextureFormat;
	bytesPerPixel: number;
	bytesPerRow: number;
}): Float32Array {
	const info = getTextureFormatInfo(input.format);
	if (
		info.componentType !== "unorm" ||
		info.bytesPerBlock !== info.channelCount
	) {
		throw new Error(
			"toNormalizedRGBA8Float32() is only supported for 8-bit unorm formats."
		);
	}
	return decodeTextureReadbackToRGBAFloat32(input);
}

function packFloat16(values: Float32Array): Uint8Array {
	const bytes = new Uint8Array(values.length * 2);
	const view = new DataView(bytes.buffer);
	for (let index = 0; index < values.length; index++) {
		view.setUint16(index * 2, float32ToFloat16Bits(values[index]), true);
	}
	return bytes;
}
