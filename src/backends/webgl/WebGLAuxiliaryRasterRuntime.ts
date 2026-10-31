import type {
	CustomRenderPassResourceFacade,
} from "../../rendering/CustomRenderTargets";
import { float32ToFloat16Bits } from "../../foundation/Float16";
import type {
	ShaderBackendCompileStage,
	ShaderRuntime,
} from "../../shaders/runtime";
import type {
	ReadTextureOptions,
	TextureReadbackResult,
	WriteTextureSize,
} from "../IComputeRuntime";
import type {
	BindingGroupDesc,
	IRenderPipeline,
	IRenderTexture,
	IShaderModule,
	PipelineDesc,
	ShaderModuleDesc,
	TextureDataLayout,
} from "../types";
import { TextureFormat } from "../../core/TextureFormat";
import {
	createWebGLRasterResourceFacade,
	WebGLScopedRasterEncoder,
} from "./WebGLCustomRenderTargetRuntime";
import type {
	IWebGLAuxiliaryRasterEncoder,
	WebGLAuxiliaryRasterContext,
	WebGLAuxiliaryRasterResourceFacade,
} from "./WebGLAuxiliaryRaster";
import { WebGLResourceStore } from "./WebGLResourceStore";

interface WebGLScopedResource {
	destroy(): void;
}

/** Owns cached raster programs and creates isolated auxiliary task scopes. */
export class WebGLAuxiliaryRasterRuntime {
	private readonly _gl: WebGL2RenderingContext;
	private readonly _shaderRuntime: ShaderRuntime;
	private readonly _shaderCompileStage: ShaderBackendCompileStage;
	private readonly _baseResources: CustomRenderPassResourceFacade;
	private readonly _resources = new WebGLResourceStore();
	private readonly _pipelineCache = new Map<string, IRenderPipeline>();
	private _cacheRevision = "";
	private _destroyed = false;

	public constructor(
		gl: WebGL2RenderingContext,
		shaderRuntime: ShaderRuntime,
		shaderCompileStage: ShaderBackendCompileStage,
	) {
		this._gl = gl;
		this._shaderRuntime = shaderRuntime;
		this._shaderCompileStage = shaderCompileStage;
		this._baseResources = createWebGLRasterResourceFacade(gl, this._resources);
	}

	public hasExtension(name: string): boolean {
		this._assertAlive();
		try {
			return !!this._gl.getExtension(name);
		} catch {
			return false;
		}
	}

	public async execute<T>(
		generation: number,
		signal: AbortSignal,
		task: (context: WebGLAuxiliaryRasterContext) => T | Promise<T>,
	): Promise<T> {
		this._assertAlive();
		if (signal.aborted) throw createAbortError(signal.reason);
		const scopeState = { active: true };
		const ownedResources = new Set<WebGLScopedResource>();
		const encoder = new WebGLScopedRasterEncoder(
			this._gl,
			scopeState,
			this._resources,
		);
		const resources = this._createScopeResources(
			scopeState,
			ownedResources,
		);
		try {
			const result = await task({
				generation,
				signal,
				encoder: encoder as IWebGLAuxiliaryRasterEncoder,
				resources,
			});
			if (signal.aborted) throw createAbortError(signal.reason);
			encoder.finish();
			return result;
		} finally {
			encoder.cleanup();
			scopeState.active = false;
			for (const resource of Array.from(ownedResources).reverse()) {
				resource.destroy();
			}
		}
	}

	public destroy(): void {
		if (this._destroyed) return;
		this._destroyed = true;
		this._clearPipelineCache();
		this._resources.destroy();
	}

	private _createScopeResources(
		scopeState: { active: boolean },
		ownedResources: Set<WebGLScopedResource>,
	): WebGLAuxiliaryRasterResourceFacade {
		const assertActive = (): void => {
			this._assertAlive();
			if (!scopeState.active) {
				throw new Error("WebGL auxiliary raster scope is no longer active.");
			}
		};
		const markOwned = <T extends object & WebGLScopedResource>(resource: T): T => {
			this._resources.markScope(resource, scopeState);
			ownedResources.add(resource);
			return resource;
		};
		const assertResource = (resource: unknown): void => {
			assertActive();
			this._resources.assertScope(resource, scopeState);
		};

		return {
			createBuffer: (desc) => {
				assertActive();
				return markOwned(this._baseResources.createBuffer(desc));
			},
			createTexture: (desc) => {
				assertActive();
				return markOwned(this._baseResources.createTexture(desc));
			},
			createSampler: (desc) => {
				assertActive();
				return markOwned(this._baseResources.createSampler(desc));
			},
			createShaderModule: (desc) => {
				assertActive();
				return markOwned(this._createShaderModule(desc));
			},
			createRenderPipeline: (desc) => {
				assertActive();
				assertResource(desc.vertex.module);
				if (desc.fragment) assertResource(desc.fragment.module);
				const pipeline = this._getOrCreatePipeline(desc);
				return markOwned(this._resources.aliasPipeline(pipeline));
			},
			createBindingGroup: (desc) => {
				assertActive();
				this._assertBindingGroupResources(desc, assertResource);
				return markOwned(this._baseResources.createBindingGroup(desc));
			},
			writeTexture: (texture, data, layout, size) => {
				assertResource(texture);
				this._writeTexture(texture, data, layout, size);
			},
			readTexture: async (options) => {
				assertResource(options.texture);
				return this._readTexture(options);
			},
		};
	}

	private _createShaderModule(desc: ShaderModuleDesc): IShaderModule {
		const compiled = this._shaderCompileStage.compile({
			code: desc.code,
			language: desc.language ?? "glsl",
			stage: desc.stage ?? "unknown",
			entryPoint: desc.entryPoint,
			label: desc.label,
			sourceKind: desc.sourceKind ?? "unknown",
			sourceMap: desc.sourceMap,
		});
		const module = this._baseResources.createShaderModule({
			...desc,
			code: compiled.code,
			sourceMap: compiled.sourceMap,
			directiveFingerprint: compiled.directiveFingerprint,
		} as ShaderModuleDesc);
		if (module instanceof Promise) {
			throw new Error("WebGL auxiliary raster shader modules must compile synchronously.");
		}
		return module;
	}

	private _getOrCreatePipeline(desc: PipelineDesc): IRenderPipeline {
		const revision = `${this._shaderRuntime.revision}:` +
			`${this._shaderCompileStage.revision}`;
		if (revision !== this._cacheRevision) {
			this._clearPipelineCache();
			this._cacheRevision = revision;
		}
		const key = createPipelineFingerprint(desc, revision, this._resources);
		const cached = this._pipelineCache.get(key);
		if (cached) return cached;
		const pipeline = this._baseResources.createRenderPipeline(desc);
		if (pipeline instanceof Promise) {
			throw new Error("WebGL auxiliary raster pipelines must compile synchronously.");
		}
		this._pipelineCache.set(key, pipeline);
		return pipeline;
	}

	private _assertBindingGroupResources(
		desc: BindingGroupDesc,
		assertResource: (resource: unknown) => void,
	): void {
		for (const entry of desc.entries) {
			assertResource("buffer" in entry.resource ? entry.resource.buffer : entry.resource);
		}
	}

	private _writeTexture(
		texture: IRenderTexture,
		data: BufferSource,
		layout: TextureDataLayout,
		size: WriteTextureSize,
	): void {
		const state = this._resources.getTexture(texture);
		const resource = state.texture;
		const level = Math.max(0, Math.floor(layout.mipLevel ?? 0));
		const format = texture.requestedFormat ?? texture.format;
		const upload = resolveTextureUpload(this._gl, format, data);
		this._gl.bindTexture(this._gl.TEXTURE_2D, resource);
		this._gl.pixelStorei(this._gl.UNPACK_ALIGNMENT, 1);
		this._gl.texImage2D(
			this._gl.TEXTURE_2D,
			level,
			upload.internalFormat,
			size.width,
			size.height,
			0,
			upload.format,
			upload.type,
			upload.data,
		);
		state.maxMipLevel = Math.max(state.maxMipLevel, level);
		this._gl.texParameteri(
			this._gl.TEXTURE_2D,
			this._gl.TEXTURE_MAX_LEVEL,
			state.maxMipLevel,
		);
		this._gl.bindTexture(this._gl.TEXTURE_2D, null);
	}

	private _readTexture(options: ReadTextureOptions): TextureReadbackResult {
		const texture = options.texture;
		const resource = this._resources.getTexture(texture).texture;
		const width = Math.max(1, Math.floor(options.width ?? texture.width));
		const height = Math.max(1, Math.floor(options.height ?? texture.height));
		const level = Math.max(0, Math.floor(options.mipLevel ?? 0));
		const format = options.format ?? texture.requestedFormat ??
			texture.format ?? TextureFormat.RGBA8Unorm;
		const framebuffer = this._gl.createFramebuffer();
		if (!framebuffer) throw new Error("Failed to create WebGL auxiliary readback framebuffer.");
		try {
			this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, framebuffer);
			this._gl.framebufferTexture2D(
				this._gl.FRAMEBUFFER,
				this._gl.COLOR_ATTACHMENT0,
				this._gl.TEXTURE_2D,
				resource,
				level,
			);
			if (this._gl.checkFramebufferStatus(this._gl.FRAMEBUFFER) !==
				this._gl.FRAMEBUFFER_COMPLETE) {
				throw new Error("WebGL auxiliary readback framebuffer is incomplete.");
			}
			if (
				format === TextureFormat.RGBA16Float ||
				format === TextureFormat.RGBA32Float
			) {
				const values = new Float32Array(width * height * 4);
				this._gl.readPixels(
					0, 0, width, height,
					this._gl.RGBA, this._gl.FLOAT, values,
				);
				return createFloatReadback(values, width, height, format);
			}
			const bytes = new Uint8Array(width * height * 4);
			this._gl.readPixels(
				0, 0, width, height,
				this._gl.RGBA, this._gl.UNSIGNED_BYTE, bytes,
			);
			return createByteReadback(bytes, width, height, format);
		} finally {
			this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, null);
			this._gl.deleteFramebuffer(framebuffer);
		}
	}

	private _clearPipelineCache(): void {
		for (const pipeline of this._pipelineCache.values()) {
			pipeline.destroy();
		}
		this._pipelineCache.clear();
	}

	private _assertAlive(): void {
		if (this._destroyed) {
			throw new Error("WebGL auxiliary raster runtime has been destroyed.");
		}
	}
}

function createPipelineFingerprint(
	desc: PipelineDesc,
	revision: string,
	resources: WebGLResourceStore,
): string {
	return JSON.stringify({
		revision,
		vertex: { ...desc.vertex, module: resources.getShader(desc.vertex.module) },
		fragment: desc.fragment ? {
			...desc.fragment, module: resources.getShader(desc.fragment.module),
		} : undefined,
		primitive: desc.primitive,
		depthStencil: desc.depthStencil,
		sampleCount: desc.sampleCount,
	}, (_key, value) => {
		if (_key === "label") return undefined;
		return value;
	});
}

function resolveTextureUpload(
	gl: WebGL2RenderingContext,
	format: TextureFormat | undefined,
	data: BufferSource,
): {
	internalFormat: number;
	format: number;
	type: number;
	data: ArrayBufferView;
} {
	if (format === TextureFormat.RGBA8UnormSrgb) {
		if (typeof gl.SRGB8_ALPHA8 !== "number") {
			throw new Error("WebGL auxiliary sRGB uploads require SRGB8_ALPHA8.");
		}
		return {
			internalFormat: gl.SRGB8_ALPHA8,
			format: gl.RGBA,
			type: gl.UNSIGNED_BYTE,
			data: asArrayBufferView(data),
		};
	}
	if (format === TextureFormat.RGBA16Float) {
		const source = asArrayBufferView(data);
		const half = source instanceof Uint16Array ? source :
			source instanceof Float32Array ?
				Uint16Array.from(source, float32ToFloat16Bits) :
				Uint16Array.from(
					new Uint8Array(
						source.buffer,
						source.byteOffset,
						source.byteLength,
					),
					(value) => float32ToFloat16Bits(value / 255),
				);
		return {
			internalFormat: gl.RGBA16F,
			format: gl.RGBA,
			type: gl.HALF_FLOAT,
			data: half,
		};
	}
	if (format === TextureFormat.RGBA32Float) {
		return {
			internalFormat: gl.RGBA32F,
			format: gl.RGBA,
			type: gl.FLOAT,
			data: asArrayBufferView(data),
		};
	}
	return {
		internalFormat: gl.RGBA8,
		format: gl.RGBA,
		type: gl.UNSIGNED_BYTE,
		data: asArrayBufferView(data),
	};
}

function asArrayBufferView(data: BufferSource): ArrayBufferView {
	return data instanceof ArrayBuffer ? new Uint8Array(data) : data;
}

function createFloatReadback(
	values: Float32Array,
	width: number,
	height: number,
	format: TextureFormat,
): TextureReadbackResult {
	const bytes = new Uint8Array(
		values.buffer,
		values.byteOffset,
		values.byteLength,
	);
	return {
		bytes,
		width,
		height,
		format,
		bytesPerPixel: 16,
		bytesPerRow: width * 16,
		toFloat32: () => values.slice(),
		toRGBAFloat32: () => values.slice(),
		toNormalizedRGBA8Float32: () => Float32Array.from(
			values,
			(value) => Math.max(0, Math.min(1, value)),
		),
	};
}

function createByteReadback(
	bytes: Uint8Array,
	width: number,
	height: number,
	format: TextureFormat,
): TextureReadbackResult {
	const normalized = () => Float32Array.from(bytes, (value) => value / 255);
	return {
		bytes,
		width,
		height,
		format,
		bytesPerPixel: 4,
		bytesPerRow: width * 4,
		toFloat32: normalized,
		toRGBAFloat32: normalized,
		toNormalizedRGBA8Float32: normalized,
	};
}

function createAbortError(reason?: unknown): Error {
	const error = new Error("WebGL auxiliary raster work was aborted", {
		cause: reason,
	});
	error.name = "AbortError";
	return error;
}
