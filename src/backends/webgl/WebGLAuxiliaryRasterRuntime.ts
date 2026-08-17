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
	ISampler,
	IRenderPipeline,
	IRenderTexture,
	PipelineDesc,
	ShaderModuleDesc,
	TextureDataLayout,
} from "../types";
import { TextureFormat } from "../types";
import {
	createWebGLRasterResourceFacade,
	WebGLScopedRasterEncoder,
} from "./WebGLCustomRenderTargetRuntime";
import type {
	IWebGLAuxiliaryRasterEncoder,
	WebGLAuxiliaryRasterContext,
	WebGLAuxiliaryRasterResourceFacade,
} from "./WebGLAuxiliaryRaster";

interface WebGLScopedResource {
	_webglScopeState?: { active: boolean };
	destroy?: () => void;
}

type WebGLScopedTexture = IRenderTexture & WebGLScopedResource & {
	_gpuResource?: WebGLTexture;
	_webglMaxMipLevel?: number;
};

/** Owns cached raster programs and creates isolated auxiliary task scopes. */
export class WebGLAuxiliaryRasterRuntime {
	private readonly _gl: WebGL2RenderingContext;
	private readonly _shaderRuntime: ShaderRuntime;
	private readonly _shaderCompileStage: ShaderBackendCompileStage;
	private readonly _baseResources: CustomRenderPassResourceFacade;
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
		this._baseResources = createWebGLRasterResourceFacade(gl);
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
				resource.destroy?.();
			}
		}
	}

	public destroy(): void {
		if (this._destroyed) return;
		this._destroyed = true;
		this._clearPipelineCache();
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
		const markOwned = <T extends object>(resource: T): T => {
			const scoped = resource as T & WebGLScopedResource;
			scoped._webglScopeState = scopeState;
			const destroy = scoped.destroy?.bind(scoped);
			let destroyed = false;
			scoped.destroy = () => {
				if (destroyed) return;
				destroyed = true;
				ownedResources.delete(scoped);
				destroy?.();
			};
			ownedResources.add(scoped);
			return resource;
		};
		const markScoped = <T extends object>(resource: T): T => {
			(resource as T & WebGLScopedResource)._webglScopeState = scopeState;
			return resource;
		};
		const assertResource = (resource: unknown): void => {
			assertActive();
			if (
				(resource as WebGLScopedResource | null)?._webglScopeState !==
				scopeState
			) {
				throw new Error(
					"WebGL auxiliary raster resources must belong to the active scope.",
				);
			}
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
				const sampler = this._baseResources.createSampler(desc) as
					ISampler & WebGLScopedResource & {
						_webglSampler?: WebGLSampler;
					};
				if (!sampler.destroy) {
					sampler.destroy = () => {
						if (sampler._webglSampler) {
							this._gl.deleteSampler(sampler._webglSampler);
						}
					};
				}
				return markOwned(sampler);
			},
			createShaderModule: (desc) => {
				assertActive();
				return markScoped(this._createShaderModule(desc));
			},
			createRenderPipeline: (desc) => {
				assertActive();
				const pipeline = this._getOrCreatePipeline(desc);
				return markScoped({ ...pipeline });
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

	private _createShaderModule(desc: ShaderModuleDesc) {
		const compiled = this._shaderCompileStage.compile({
			code: desc.code,
			language: desc.language ?? "glsl",
			stage: desc.stage ?? "unknown",
			entryPoint: desc.entryPoint,
			label: desc.label,
			sourceKind: desc.sourceKind ?? "unknown",
			sourceMap: desc.sourceMap,
		});
		return this._baseResources.createShaderModule({
			...desc,
			code: compiled.code,
			sourceMap: compiled.sourceMap,
			directiveFingerprint: compiled.directiveFingerprint,
		} as ShaderModuleDesc);
	}

	private _getOrCreatePipeline(desc: PipelineDesc): IRenderPipeline {
		const revision = `${this._shaderRuntime.revision}:` +
			`${this._shaderCompileStage.revision}`;
		if (revision !== this._cacheRevision) {
			this._clearPipelineCache();
			this._cacheRevision = revision;
		}
		const key = createPipelineFingerprint(desc, revision);
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
		for (const entry of desc.entries) assertResource(entry.resource);
	}

	private _writeTexture(
		texture: IRenderTexture,
		data: BufferSource,
		layout: TextureDataLayout,
		size: WriteTextureSize,
	): void {
		const scoped = texture as WebGLScopedTexture;
		const resource = scoped._gpuResource;
		if (!resource) throw new Error("WebGL auxiliary texture is unavailable.");
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
		scoped._webglMaxMipLevel = Math.max(
			scoped._webglMaxMipLevel ?? 0,
			level,
		);
		this._gl.texParameteri(
			this._gl.TEXTURE_2D,
			this._gl.TEXTURE_MAX_LEVEL,
			scoped._webglMaxMipLevel,
		);
		this._gl.bindTexture(this._gl.TEXTURE_2D, null);
	}

	private _readTexture(options: ReadTextureOptions): TextureReadbackResult {
		const texture = options.texture as WebGLScopedTexture;
		const resource = texture._gpuResource;
		if (!resource) throw new Error("WebGL auxiliary texture is unavailable.");
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
			(pipeline as { destroy?: () => void }).destroy?.();
		}
		this._pipelineCache.clear();
	}

	private _assertAlive(): void {
		if (this._destroyed) {
			throw new Error("WebGL auxiliary raster runtime has been destroyed.");
		}
	}
}

function createPipelineFingerprint(desc: PipelineDesc, revision: string): string {
	return JSON.stringify({
		revision,
		vertex: desc.vertex,
		fragment: desc.fragment,
		primitive: desc.primitive,
		depthStencil: desc.depthStencil,
		sampleCount: desc.sampleCount,
	}, (_key, value) => {
		if (
			_key === "label" ||
			_key === "_gpuResource" ||
			_key === "_webglScopeState"
		) return undefined;
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
