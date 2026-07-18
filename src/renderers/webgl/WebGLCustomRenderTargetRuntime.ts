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
} from "../CustomRenderTargets";
import type { TextureReadbackResult } from "../IComputeRuntime";
import {
	AddressMode,
	BufferUsage,
	FilterMode,
	PrimitiveTopology,
	TextureFormat,
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

interface WebGLCustomTexture extends IRenderTexture {
	_gpuResource: WebGLTexture | WebGLRenderbuffer;
	_webglTargetKind: "texture" | "renderbuffer";
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
	framebuffer: WebGLFramebuffer;
	color: WebGLCustomTexture[];
	depth: WebGLCustomTexture | null;
}

export class WebGLCustomRenderTargetRuntime {
	private readonly _gl: WebGL2RenderingContext;
	private readonly _targets = new Map<string, WebGLCustomRenderTarget>();
	private _lastSuccessfulFrame = false;

	public constructor(gl: WebGL2RenderingContext) {
		this._gl = gl;
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
			if (
				current &&
				current.width === width &&
				current.height === height &&
				current.sampleCount === sampleCount &&
				JSON.stringify(current.descriptor) === JSON.stringify(descriptor)
			) {
				continue;
			}
			this._destroyTarget(descriptor.id);
			if (sampleCount !== 1) {
				const key = `webgl-custom-render-target-msaa-unsupported-${descriptor.id}`;
				Logger.warn(
					`[${key}] WebGL custom render target "${descriptor.id}" sampleCount=${sampleCount} is not supported in v1; disabling the target.`,
					{ scope: "WebGLCustomRenderTargetRuntime", onceKey: key }
				);
				continue;
			}
			this._targets.set(descriptor.id, this._createTarget(descriptor, width, height));
		}
	}

	public hasPass(pass: FramePass, context: FrameContext): boolean {
		return context.customRenderPasses.has(pass.stage);
	}

	public async executePass(pass: FramePass, context: FrameContext): Promise<void> {
		const descriptor = context.customRenderPasses.get(pass.stage);
		if (!descriptor) {
			return;
		}
		const target = this._targets.get(descriptor.target);
		if (!target) {
			const key = `webgl-custom-render-pass-target-missing-${pass.stage}`;
			Logger.warn(
				`[${key}] WebGL custom render pass "${pass.stage}" target "${descriptor.target}" is unavailable; skipping.`,
				{ scope: "WebGLCustomRenderTargetRuntime", onceKey: key }
			);
			return;
		}
		await descriptor.execute({
			backend: "webgl",
			frameContext: context,
			encoder: new WebGLCustomCommandEncoder(this._gl),
			target: toExecutionTarget(target),
			width: target.width,
			height: target.height,
			resources: createWebGLResourceFacade(this._gl),
		} satisfies CustomRenderPassContext);
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
	): Promise<TextureReadbackResult> {
		if (!this._lastSuccessfulFrame) {
			throw new Error(
				`Render target "${id}" cannot be read before a successful frame completes.`
			);
		}
		const target = this._targets.get(id);
		if (!target) {
			throw new Error(`Render target "${id}" is unavailable.`);
		}
		if (!target.color[attachmentIndex]) {
			throw new Error(
				`Render target "${id}" color attachment ${attachmentIndex} is unavailable.`
			);
		}
		const gl = this._gl;
		const width = Math.max(1, Math.floor(options.width ?? target.width));
		const height = Math.max(1, Math.floor(options.height ?? target.height));
		const format =
			options.format ??
			target.descriptor.color[attachmentIndex]?.format ??
			TextureFormat.RGBA8Unorm;
		const floatRead = format === TextureFormat.RGBA16Float ||
			format === TextureFormat.RGBA32Float;
		const bytesPerPixel = options.bytesPerPixel ?? (floatRead ? 16 : 4);
		const bytesPerRow = width * bytesPerPixel;
		let bytes: Uint8Array;
		gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
		gl.readBuffer(gl.COLOR_ATTACHMENT0 + attachmentIndex);
		if (floatRead) {
			const data = new Float32Array(width * height * 4);
			gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, data);
			bytes = new Uint8Array(data.buffer.slice(0));
		} else {
			bytes = new Uint8Array(width * height * 4);
			gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
		}
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		return createTextureReadbackResult({
			bytes,
			width,
			height,
			format,
			bytesPerPixel,
			bytesPerRow,
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
		height: number
	): WebGLCustomRenderTarget {
		const gl = this._gl;
		const framebuffer = gl.createFramebuffer();
		if (!framebuffer) {
			throw new Error(`Failed to create WebGL custom framebuffer "${descriptor.id}".`);
		}
		const color = descriptor.color.map((attachment, index) =>
			createWebGLColorTexture(
				gl,
				width,
				height,
				attachment.format,
				`WebGLCustomRenderTarget_${descriptor.id}_Color${index}`
			)
		);
		const depth = descriptor.depth ?
			createWebGLDepthRenderbuffer(
				gl,
				width,
				height,
				descriptor.depth.format,
				`WebGLCustomRenderTarget_${descriptor.id}_Depth`
			)
		:	null;

		gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
		for (let index = 0; index < color.length; index++) {
			gl.framebufferTexture2D(
				gl.FRAMEBUFFER,
				gl.COLOR_ATTACHMENT0 + index,
				gl.TEXTURE_2D,
				color[index]._gpuResource as WebGLTexture,
				0
			);
		}
		if (depth) {
			gl.framebufferRenderbuffer(
				gl.FRAMEBUFFER,
				gl.DEPTH_ATTACHMENT,
				gl.RENDERBUFFER,
				depth._gpuResource as WebGLRenderbuffer
			);
		}
		gl.drawBuffers(color.map((_texture, index) => gl.COLOR_ATTACHMENT0 + index));
		const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		if (status !== gl.FRAMEBUFFER_COMPLETE) {
			for (const texture of color) {
				texture.destroy();
			}
			depth?.destroy();
			gl.deleteFramebuffer(framebuffer);
			throw new Error(
				`WebGL custom framebuffer "${descriptor.id}" is incomplete (status=0x${status.toString(16)}).`
			);
		}
		return {
			descriptor,
			width,
			height,
			sampleCount: 1,
			framebuffer,
			color,
			depth,
		};
	}

	private _destroyTarget(id: string): void {
		const target = this._targets.get(id);
		if (!target) {
			return;
		}
		for (const texture of target.color) {
			texture.destroy();
		}
		target.depth?.destroy();
		this._gl.deleteFramebuffer(target.framebuffer);
		this._targets.delete(id);
	}
}

class WebGLCustomCommandEncoder implements ICommandEncoder {
	private readonly _gl: WebGL2RenderingContext;
	private _framebuffer: WebGLFramebuffer | null = null;
	private _pipeline: WebGLCustomPipeline | null = null;
	private readonly _vertexBuffers = new Map<number, IRenderBuffer>();
	private _indexBuffer: IRenderBuffer | null = null;
	private _indexFormat: IndexFormat = "uint16";

	public constructor(gl: WebGL2RenderingContext) {
		this._gl = gl;
	}

	public beginRenderPass(desc: RenderPassDesc): void {
		const gl = this._gl;
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
			if (attachment.loadOp === "clear") {
				gl.clearBufferfv(
					gl.COLOR,
					index,
					colorClearValueToArray(attachment)
				);
			}
		});
		if (desc.colorAttachments.length > 0) {
			gl.drawBuffers(
				desc.colorAttachments.map((_attachment, index) =>
					gl.COLOR_ATTACHMENT0 + index
				)
			);
		}
		if (desc.depthStencilAttachment) {
			this._attachDepth(desc.depthStencilAttachment);
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
		void desc.label;
	}

	public beginComputePass(_desc?: ComputePassDesc): void {
		throw new Error("WebGL custom render passes do not support compute passes.");
	}

	public setPipeline(pipeline: IRenderPipeline): void {
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
		const pipeline = this._requirePipeline();
		const customGroup = group as WebGLCustomBindingGroup;
		if (!customGroup._webglEntries) {
			throw new Error("WebGL custom render pass received an incompatible binding group.");
		}
		bindGroupResources(this._gl, pipeline, customGroup._webglEntries);
	}

	public setVertexBuffer(slot: number, buffer: IRenderBuffer): void {
		this._vertexBuffers.set(slot, buffer);
	}

	public setIndexBuffer(buffer: IRenderBuffer, format: IndexFormat): void {
		this._indexBuffer = buffer;
		this._indexFormat = format;
	}

	public drawIndexed(
		indexCount: number,
		instanceCount = 1,
		firstIndex = 0,
		baseVertex = 0,
		_firstInstance = 0
	): void {
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
		void baseVertex;
		gl.drawElements(mode, indexCount, indexType, firstIndex * indexSize);
	}

	public draw(
		vertexCount: number,
		instanceCount = 1,
		firstVertex = 0,
		_firstInstance = 0
	): void {
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
		this._gl.scissor(x, y, width, height);
	}
	public copyTextureToTexture(
		_source: TextureCopyView,
		_destination: TextureCopyView,
		_copySize: TextureCopySize
	): void {
		throw new Error("WebGL custom render passes do not support texture copies.");
	}
	public endRenderPass(): void {
		this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, null);
		if (this._framebuffer) {
			this._gl.deleteFramebuffer(this._framebuffer);
			this._framebuffer = null;
		}
	}
	public setComputePipeline(_pipeline: IComputePipeline): void {}
	public dispatchWorkgroups(): void {}
	public endComputePass(): void {}
	public finish(): ICommandBuffer {
		if (this._framebuffer) {
			throw new Error("Pass still active");
		}
		return {};
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
		if (
			(attachment.view as WebGLCustomTexture | undefined)?._webglTargetKind ===
			"renderbuffer"
		) {
			this._gl.framebufferRenderbuffer(
				this._gl.FRAMEBUFFER,
				this._gl.DEPTH_ATTACHMENT,
				this._gl.RENDERBUFFER,
				resource
			);
			return;
		}
		this._gl.framebufferTexture2D(
			this._gl.FRAMEBUFFER,
			this._gl.DEPTH_ATTACHMENT,
			this._gl.TEXTURE_2D,
			resource as WebGLTexture,
			0
		);
	}
}

function createWebGLColorTexture(
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
	const resolved = resolveColorFormat(gl, format);
	gl.bindTexture(gl.TEXTURE_2D, texture);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
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
		return createWebGLDepthRenderbuffer(
			gl,
			desc.width,
			desc.height,
			desc.format,
			desc.label ?? "WebGLCustomTexture_Depth"
		);
	}
	return createWebGLColorTexture(
		gl,
		desc.width,
		desc.height,
		desc.format,
		desc.label ?? "WebGLCustomTexture_Color"
	);
}

function createWebGLDepthRenderbuffer(
	gl: WebGL2RenderingContext,
	width: number,
	height: number,
	format: TextureFormat,
	label: string
): WebGLCustomTexture {
	const renderbuffer = gl.createRenderbuffer();
	if (!renderbuffer) {
		throw new Error(`Failed to create ${label}.`);
	}
	gl.bindRenderbuffer(gl.RENDERBUFFER, renderbuffer);
	gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, width, height);
	gl.bindRenderbuffer(gl.RENDERBUFFER, null);
	return {
		width,
		height,
		format,
		requestedFormat: format,
		_gpuResource: renderbuffer,
		_webglTargetKind: "renderbuffer",
		destroy: () => gl.deleteRenderbuffer(renderbuffer),
	};
}

function resolveColorFormat(
	gl: WebGL2RenderingContext,
	format: TextureFormat
): { internalFormat: number; format: number; type: number } {
	switch (format) {
		case TextureFormat.RGBA16Float:
		case TextureFormat.RGBA32Float:
			return {
				internalFormat: gl.RGBA16F,
				format: gl.RGBA,
				type: gl.HALF_FLOAT,
			};
		default:
			return {
				internalFormat: gl.RGBA8,
				format: gl.RGBA,
				type: gl.UNSIGNED_BYTE,
			};
	}
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
		})),
		depth: target.depth && target.descriptor.depth ? {
			texture: target.depth,
			format: target.descriptor.depth.format,
		} : null,
	};
}

function createWebGLResourceFacade(
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

function createTextureReadbackResult(input: {
	bytes: Uint8Array;
	width: number;
	height: number;
	format: TextureFormat;
	bytesPerPixel: number;
	bytesPerRow: number;
}): TextureReadbackResult {
	return {
		bytes: input.bytes,
		width: input.width,
		height: input.height,
		format: input.format,
		bytesPerPixel: input.bytesPerPixel,
		bytesPerRow: input.bytesPerRow,
		toFloat32: () =>
			new Float32Array(
				input.bytes.buffer,
				input.bytes.byteOffset,
				Math.floor(input.bytes.byteLength / 4)
			),
		toRGBAFloat32: () => {
			if (input.bytesPerPixel === 16) {
				return new Float32Array(
					input.bytes.buffer.slice(
						input.bytes.byteOffset,
						input.bytes.byteOffset + input.bytes.byteLength
					)
				);
			}
			return normalizeRGBA8(input.bytes, input.width, input.height);
		},
		toNormalizedRGBA8Float32: () =>
			normalizeRGBA8(input.bytes, input.width, input.height),
	};
}

function normalizeRGBA8(
	bytes: Uint8Array,
	width: number,
	height: number
): Float32Array {
	const out = new Float32Array(width * height * 4);
	const count = Math.min(out.length, bytes.length);
	for (let i = 0; i < count; i++) {
		out[i] = bytes[i] / 255;
	}
	return out;
}
