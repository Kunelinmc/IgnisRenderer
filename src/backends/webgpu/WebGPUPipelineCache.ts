/// <reference types="@webgpu/types" />
import {
	formatShaderCompilerMessages,
	mapShaderCompilerMessages,
	normalizeWebGPUCompilationMessages,
	ShaderCompileError,
	type ShaderCompilerMessage,
} from "../../shaders/runtime";
import {
	WebGPUPipelineCreationInvalidatedError,
	WebGPUShaderModuleCreationInvalidatedError,
} from "../../foundation/Error";
import { Logger } from "../../foundation/Logger";
import {
	WEBGPU_PIPELINE_CACHE_LIMIT,
	WEBGPU_PIPELINE_LAYOUT_CACHE_LIMIT,
} from "./constants";
import { getWebGPUShaderModule } from "./WebGPUResourceAccess";
import type { WebGPUShaderModuleCompiler } from "./WebGPUShaderModuleCompiler";
import type {
	ComputePipelineDesc,
	IComputePipeline,
	IRenderPipeline,
	ISampler,
	IShaderModule,
	PipelineDesc,
	SamplerDesc,
	ShaderModuleDesc,
} from "../types";
import {
	hashString64,
	type WebGPUObjectIdentity,
} from "./WebGPUObjectIdentity";

interface InternalSampler extends ISampler {
	destroy(): void;
	_gpuResource: GPUSampler;
}

interface InternalShaderModule extends IShaderModule {
	destroy(): void;
	_gpuResource: GPUShaderModule;
}

interface InternalRenderPipeline extends IRenderPipeline {
	destroy(): void;
	_gpuResource: GPURenderPipeline;
}

interface InternalComputePipeline extends IComputePipeline {
	destroy(): void;
	_gpuResource: GPUComputePipeline;
}

interface CachedSamplerEntry {
	key: string;
	label?: string;
	gpuResource: GPUSampler;
	refCount: number;
}

interface CachedShaderModuleEntry {
	key: string;
	label?: string;
	refCount: number;
	gpuResource: GPUShaderModule;
}

interface CachedRenderPipelineEntry {
	key: string;
	label?: string;
	refCount: number;
	gpuResource: GPURenderPipeline;
}

interface CachedComputePipelineEntry {
	key: string;
	label?: string;
	refCount: number;
	gpuResource: GPUComputePipeline;
}

type PipelineDescLayout = PipelineDesc["layout"] | undefined;
type ComputePipelineDescLayout = ComputePipelineDesc["layout"] | undefined;

const SHADER_CODE_HASH_CACHE_LIMIT = 128;

export interface WebGPUPipelineCacheHost {
	readonly device: GPUDevice | null;
	readonly shaderModuleCompiler: WebGPUShaderModuleCompiler;
	readonly warmupLogCompilationInfo: boolean;
	readonly objectIdentity: WebGPUObjectIdentity;
	assertDeviceOperational(operation: string): void;
	resolveSupportedSampleCount(
		requested: number,
		probeFormats?: readonly GPUTextureFormat[]
	): number;
	createManagedDestroy(
		target: object,
		options: {
			label: string;
			dispose: () => void;
		}
	): () => void;
	runValidationScopeAsync<T>(
		label: string,
		operation: () => Promise<T>
	): Promise<T>;
}

export interface WebGPUPipelineCacheDebugStats {
	samplerEntries: number;
	shaderModuleEntries: number;
	shaderCodeHashEntries: number;
	shaderModuleInFlight: number;
	renderPipelineEntries: number;
	computePipelineEntries: number;
	renderPipelineInFlight: number;
	computePipelineInFlight: number;
	samplerRefCounts: number[];
	shaderModuleRefCounts: number[];
	renderPipelineRefCounts: number[];
	computePipelineRefCounts: number[];
}

export class WebGPUPipelineCache {
	private _samplerCache = new Map<string, CachedSamplerEntry>();
	private _shaderModuleCache = new Map<string, CachedShaderModuleEntry>();
	private _shaderCodeHashCache = new Map<string, string>();
	private _shaderModuleInFlight = new Map<string, Promise<CachedShaderModuleEntry>>();
	private _shaderModuleGeneration = 0;
	private _renderPipelineCache = new Map<string, CachedRenderPipelineEntry>();
	private _computePipelineCache = new Map<string, CachedComputePipelineEntry>();
	private _renderPipelineInFlight = new Map<string, Promise<CachedRenderPipelineEntry>>();
	private _computePipelineInFlight = new Map<string, Promise<CachedComputePipelineEntry>>();

	public constructor(private readonly _host: WebGPUPipelineCacheHost) {}

	public createSampler(desc: SamplerDesc): ISampler {
		this._host.assertDeviceOperational("create samplers");
		const device = this._requireDevice("create samplers");
		const cacheKey = this._getSamplerCacheKey(desc);
		const cached = this._getLruCacheEntry(this._samplerCache, cacheKey);
		if (cached) {
			return this._acquireSamplerHandle(cached);
		}

		const gpuSampler = device.createSampler({
			addressModeU: desc.addressModeU as GPUAddressMode | undefined,
			addressModeV: desc.addressModeV as GPUAddressMode | undefined,
			magFilter: desc.magFilter as GPUFilterMode | undefined,
			minFilter: desc.minFilter as GPUFilterMode | undefined,
			mipmapFilter: desc.mipmapFilter as GPUFilterMode | undefined,
			compare: desc.compare as GPUCompareFunction | undefined,
			label: desc.label,
		});

		const entry: CachedSamplerEntry = {
			key: cacheKey,
			refCount: 0,
			label: desc.label,
			gpuResource: gpuSampler,
		};
		this._samplerCache.set(cacheKey, entry);
		const sampler = this._acquireSamplerHandle(entry);
		this._trimRefCountedCache(
			this._samplerCache,
			WEBGPU_PIPELINE_LAYOUT_CACHE_LIMIT
		);
		return sampler;
	}

	public async createShaderModule(desc: ShaderModuleDesc): Promise<IShaderModule> {
		this._host.assertDeviceOperational("create shader modules");
		const creationDevice = this._requireDevice("create shader modules");
		const creationGeneration = this._shaderModuleGeneration;
		const processed =
			await this._host.shaderModuleCompiler.processShaderSource(desc);
		this._assertShaderModuleCreationCurrent(
			creationDevice,
			creationGeneration,
			desc,
		);
		if (processed.hasErrors) {
			this._host.shaderModuleCompiler.reportShaderRuntimeDiagnostics(
				desc,
				processed
			);
		}
		const effectiveSourceMap = processed.sourceMap ?? desc.sourceMap ?? null;
		const effectiveCodeHash = processed.code === desc.code ? desc.codeHash : undefined;
		const effectiveDesc: ShaderModuleDesc = {
			...desc,
			code: processed.code,
			sourceMap: effectiveSourceMap,
			codeHash: effectiveCodeHash,
			directiveFingerprint: processed.directiveFingerprint,
			logCompilationInfo:
				desc.logCompilationInfo ?? this._host.warmupLogCompilationInfo,
		};
		const cacheKey = this._getShaderModuleCacheKey(effectiveDesc);
		const cached = this._shaderModuleCache.get(cacheKey);
		if (cached) {
			this._touchCacheEntry(this._shaderModuleCache, cacheKey, cached);
			return this._acquireShaderModuleHandleAndTrim(cached);
		}

		const inFlight = this._shaderModuleInFlight.get(cacheKey);
		if (inFlight) {
			const entry = await inFlight;
			this._assertShaderModuleCreationCurrent(
				creationDevice,
				creationGeneration,
				effectiveDesc,
			);
			return this._acquireShaderModuleHandleAndTrim(entry);
		}

		const creationPromise = (async () => {
			let lastError: unknown = null;
			for (let attempt = 0; attempt < 2; attempt++) {
				try {
					this._assertShaderModuleCreationCurrent(
						creationDevice,
						creationGeneration,
						effectiveDesc,
					);
					const gpuModule = creationDevice.createShaderModule({
						code: effectiveDesc.code,
						label: effectiveDesc.label,
					});
					let compileMessages: ShaderCompilerMessage[] = [];
					if (typeof gpuModule.getCompilationInfo === "function") {
						try {
							const info = await gpuModule.getCompilationInfo();
							this._assertShaderModuleCreationCurrent(
								creationDevice,
								creationGeneration,
								effectiveDesc,
							);
							compileMessages = normalizeWebGPUCompilationMessages(info.messages);
						} catch (error) {
							this._assertShaderModuleCreationCurrent(
								creationDevice,
								creationGeneration,
								effectiveDesc,
							);
							Logger.warn(
								`WebGPU shader compilation info unavailable [${effectiveDesc.label ?? "unnamed"}]: ${String(error)}`,
								{ scope: "WebGPUBackend" },
							);
						}
					}
					this._assertShaderModuleCreationCurrent(
						creationDevice,
						creationGeneration,
						effectiveDesc,
					);
					if (compileMessages.length > 0) {
						const mappedMessages = mapShaderCompilerMessages(
							compileMessages,
							effectiveDesc.code,
							effectiveDesc.sourceMap,
						);
						if (effectiveDesc.logCompilationInfo === true) {
							const label = effectiveDesc.label ?? "unnamed";
							console.group(`WebGPU Shader Compilation Info [${label}]`);
							console.log(formatShaderCompilerMessages(mappedMessages));
							console.groupEnd();
						}
						const hasErrors = mappedMessages.some(
							(message) => message.type === "error",
						);
						if (hasErrors) {
							throw new ShaderCompileError({
								backend: "webgpu",
								language: effectiveDesc.language ?? "wgsl",
								stage: effectiveDesc.stage ?? "unknown",
								label: effectiveDesc.label,
								sourceKind: effectiveDesc.sourceKind ?? "unknown",
								variantKey: effectiveDesc.variantKey,
								materialId: effectiveDesc.materialId,
								code: effectiveDesc.code,
								sourceMap: effectiveDesc.sourceMap,
								messages: compileMessages,
							});
						}
					}

					this._assertShaderModuleCreationCurrent(
						creationDevice,
						creationGeneration,
						effectiveDesc,
					);
					const entry: CachedShaderModuleEntry = {
						key: cacheKey,
						refCount: 0,
						label: effectiveDesc.label,
						gpuResource: gpuModule,
					};
					this._shaderModuleCache.set(cacheKey, entry);
					return entry;
				} catch (error) {
					if (error instanceof WebGPUShaderModuleCreationInvalidatedError) {
						throw error;
					}
					lastError = this._host.shaderModuleCompiler.createShaderModuleError(
						error,
						effectiveDesc
					);
					if (attempt === 0) {
						continue;
					}
				}
			}
			throw lastError;
		})();
		this._shaderModuleInFlight.set(cacheKey, creationPromise);
		try {
			const entry = await creationPromise;
			this._assertShaderModuleCreationCurrent(
				creationDevice,
				creationGeneration,
				effectiveDesc,
			);
			return this._acquireShaderModuleHandleAndTrim(entry);
		} finally {
			if (this._shaderModuleInFlight.get(cacheKey) === creationPromise) {
				this._shaderModuleInFlight.delete(cacheKey);
			}
		}
	}

	public async createPipeline(desc: PipelineDesc): Promise<IRenderPipeline> {
		this._host.assertDeviceOperational("create render pipelines");
		const creationDevice = this._requireDevice("create render pipelines");
		const creationGeneration = this._shaderModuleGeneration;
		const layout = this._resolveRenderPipelineLayout(desc);
		const cacheKey = this._getRenderPipelineCacheKey(desc, layout);
		const cached = this._getLruCacheEntry(this._renderPipelineCache, cacheKey);
		if (cached) {
			return this._acquireRenderPipelineHandle(cached);
		}

		const inFlight = this._renderPipelineInFlight.get(cacheKey);
		if (inFlight) {
			const entry = await inFlight;
			this._assertPipelineCreationCurrent(
				creationDevice,
				creationGeneration,
				desc.label,
			);
			return this._acquireRenderPipelineHandle(entry);
		}

		const creationPromise = (async () => {
			this._assertPipelineCreationCurrent(
				creationDevice,
				creationGeneration,
				desc.label,
			);
			const descriptor = this._createRenderPipelineDescriptor(desc, layout);
			const gpuPipeline = await this._host.runValidationScopeAsync(
				`createRenderPipeline:${desc.label ?? "unnamed"}`,
				() => creationDevice.createRenderPipelineAsync(descriptor),
			);
			this._assertPipelineCreationCurrent(
				creationDevice,
				creationGeneration,
				desc.label,
			);
			const existing = this._getLruCacheEntry(this._renderPipelineCache, cacheKey);
			if (existing) {
				return existing;
			}
			const entry: CachedRenderPipelineEntry = {
				key: cacheKey,
				refCount: 0,
				label: desc.label,
				gpuResource: gpuPipeline,
			};
			this._renderPipelineCache.set(cacheKey, entry);
			this._trimRefCountedCache(
				this._renderPipelineCache,
				WEBGPU_PIPELINE_CACHE_LIMIT,
			);
			return entry;
		})();
		this._renderPipelineInFlight.set(cacheKey, creationPromise);
		try {
			const entry = await creationPromise;
			this._assertPipelineCreationCurrent(
				creationDevice,
				creationGeneration,
				desc.label,
			);
			return this._acquireRenderPipelineHandle(entry);
		} finally {
			if (this._renderPipelineInFlight.get(cacheKey) === creationPromise) {
				this._renderPipelineInFlight.delete(cacheKey);
			}
		}
	}

	public async createComputePipeline(
		desc: ComputePipelineDesc
	): Promise<IComputePipeline> {
		this._host.assertDeviceOperational("create compute pipelines");
		const creationDevice = this._requireDevice("create compute pipelines");
		const creationGeneration = this._shaderModuleGeneration;
		const layout = this._resolveComputePipelineLayout(desc);
		const cacheKey = this._getComputePipelineCacheKey(desc, layout);
		const cached = this._getLruCacheEntry(this._computePipelineCache, cacheKey);
		if (cached) {
			return this._acquireComputePipelineHandle(cached);
		}

		const inFlight = this._computePipelineInFlight.get(cacheKey);
		if (inFlight) {
			const entry = await inFlight;
			this._assertPipelineCreationCurrent(
				creationDevice,
				creationGeneration,
				desc.label,
			);
			return this._acquireComputePipelineHandle(entry);
		}

		const creationPromise = (async () => {
			this._assertPipelineCreationCurrent(
				creationDevice,
				creationGeneration,
				desc.label,
			);
			const descriptor = this._createComputePipelineDescriptor(desc, layout);
			const gpuPipeline = await this._host.runValidationScopeAsync(
				`createComputePipeline:${desc.label ?? "unnamed"}`,
				() => creationDevice.createComputePipelineAsync(descriptor),
			);
			this._assertPipelineCreationCurrent(
				creationDevice,
				creationGeneration,
				desc.label,
			);
			const existing = this._getLruCacheEntry(this._computePipelineCache, cacheKey);
			if (existing) {
				return existing;
			}
			const entry: CachedComputePipelineEntry = {
				key: cacheKey,
				refCount: 0,
				label: desc.label,
				gpuResource: gpuPipeline,
			};
			this._computePipelineCache.set(cacheKey, entry);
			this._trimRefCountedCache(
				this._computePipelineCache,
				WEBGPU_PIPELINE_CACHE_LIMIT,
			);
			return entry;
		})();
		this._computePipelineInFlight.set(cacheKey, creationPromise);
		try {
			const entry = await creationPromise;
			this._assertPipelineCreationCurrent(
				creationDevice,
				creationGeneration,
				desc.label,
			);
			return this._acquireComputePipelineHandle(entry);
		} finally {
			if (this._computePipelineInFlight.get(cacheKey) === creationPromise) {
				this._computePipelineInFlight.delete(cacheKey);
			}
		}
	}

	public clearPipelineCaches(): void {
		this._renderPipelineCache.clear();
		this._computePipelineCache.clear();
		this._renderPipelineInFlight.clear();
		this._computePipelineInFlight.clear();
	}

	public invalidateShaderDependentCaches(): void {
		this._shaderModuleGeneration++;
		this._shaderModuleCache.clear();
		this._shaderCodeHashCache.clear();
		this._shaderModuleInFlight.clear();
		this.clearPipelineCaches();
	}

	public reset(): void {
		this.invalidateShaderDependentCaches();
		this._samplerCache.clear();
	}

	public getDebugStats(): WebGPUPipelineCacheDebugStats {
		return {
			samplerEntries: this._samplerCache.size,
			shaderModuleEntries: this._shaderModuleCache.size,
			shaderCodeHashEntries: this._shaderCodeHashCache.size,
			shaderModuleInFlight: this._shaderModuleInFlight.size,
			renderPipelineEntries: this._renderPipelineCache.size,
			computePipelineEntries: this._computePipelineCache.size,
			renderPipelineInFlight: this._renderPipelineInFlight.size,
			computePipelineInFlight: this._computePipelineInFlight.size,
			samplerRefCounts: Array.from(this._samplerCache.values()).map(
				(entry) => entry.refCount
			),
			shaderModuleRefCounts: Array.from(this._shaderModuleCache.values()).map(
				(entry) => entry.refCount
			),
			renderPipelineRefCounts: Array.from(this._renderPipelineCache.values()).map(
				(entry) => entry.refCount
			),
			computePipelineRefCounts: Array.from(this._computePipelineCache.values()).map(
				(entry) => entry.refCount
			),
		};
	}

	private _requireDevice(operation: string): GPUDevice {
		const device = this._host.device;
		if (!device) {
			throw new Error(`WebGPU backend is not initialized; cannot ${operation}.`);
		}
		return device;
	}

	private _assertShaderModuleCreationCurrent(
		device: GPUDevice,
		generation: number,
		desc: ShaderModuleDesc,
	): void {
		if (this._shaderModuleGeneration !== generation || this._host.device !== device) {
			throw new WebGPUShaderModuleCreationInvalidatedError(desc.label);
		}
	}

	private _assertPipelineCreationCurrent(
		device: GPUDevice,
		generation: number,
		label?: string,
	): void {
		if (this._shaderModuleGeneration !== generation || this._host.device !== device) {
			throw new WebGPUPipelineCreationInvalidatedError(label);
		}
	}

	private _getSamplerCacheKey(desc: SamplerDesc): string {
		return [
			desc.addressModeU ?? "",
			desc.addressModeV ?? "",
			desc.magFilter ?? "",
			desc.minFilter ?? "",
			desc.mipmapFilter ?? "",
			desc.compare ?? "",
		].join("|");
	}

	private _getShaderModuleCacheKey(desc: ShaderModuleDesc): string {
		const directiveFingerprint = desc.directiveFingerprint ?? "none";
		const explicitHash = desc.codeHash;
		if (explicitHash) {
			return `directive:${directiveFingerprint}|codeHash:${explicitHash}`;
		}
		const hash = this._getCachedShaderCodeHash(desc.code);
		return `directive:${directiveFingerprint}|len:${desc.code.length}|hash:${hash}`;
	}

	private _acquireSamplerHandle(entry: CachedSamplerEntry): InternalSampler {
		this._bumpRefCount(entry);
		const sampler = {
			label: entry.label,
			destroy: () => {},
			_gpuResource: entry.gpuResource,
		} as InternalSampler;
		sampler.destroy = this._host.createManagedDestroy(sampler, {
			label: entry.label ?? "WebGPUSampler",
			dispose: () => {
				this._releaseSamplerCacheEntry(entry.key, entry.gpuResource);
			},
		});
		return sampler;
	}

	private _acquireShaderModuleHandle(entry: CachedShaderModuleEntry): InternalShaderModule {
		this._bumpRefCount(entry);
		const module = {
			label: entry.label,
			destroy: () => {},
			_gpuResource: entry.gpuResource,
		} as InternalShaderModule;
		module.destroy = this._host.createManagedDestroy(module, {
			label: entry.label ?? "WebGPUShaderModule",
			dispose: () => {
				this._releaseShaderModuleCacheEntry(entry.key, entry.gpuResource);
			},
		});
		return module;
	}

	private _acquireShaderModuleHandleAndTrim(
		entry: CachedShaderModuleEntry,
	): InternalShaderModule {
		const module = this._acquireShaderModuleHandle(entry);
		this._trimRefCountedCache(this._shaderModuleCache, WEBGPU_PIPELINE_CACHE_LIMIT);
		return module;
	}

	private _acquireRenderPipelineHandle(entry: CachedRenderPipelineEntry): InternalRenderPipeline {
		this._bumpRefCount(entry);
		const pipeline = {
			label: entry.label,
			destroy: () => {},
			_gpuResource: entry.gpuResource,
		} as InternalRenderPipeline;
		pipeline.destroy = this._host.createManagedDestroy(pipeline, {
			label: entry.label ?? "WebGPURenderPipeline",
			dispose: () => {
				this._releasePipelineCacheEntry(
					this._renderPipelineCache,
					entry.key,
					entry.gpuResource,
				);
			},
		});
		return pipeline;
	}

	private _acquireComputePipelineHandle(
		entry: CachedComputePipelineEntry,
	): InternalComputePipeline {
		this._bumpRefCount(entry);
		const pipeline = {
			label: entry.label,
			destroy: () => {},
			_gpuResource: entry.gpuResource,
		} as InternalComputePipeline;
		pipeline.destroy = this._host.createManagedDestroy(pipeline, {
			label: entry.label ?? "WebGPUComputePipeline",
			dispose: () => {
				this._releasePipelineCacheEntry(
					this._computePipelineCache,
					entry.key,
					entry.gpuResource,
				);
			},
		});
		return pipeline;
	}

	private _releaseSamplerCacheEntry(key: string, sampler: GPUSampler): void {
		const cached = this._samplerCache.get(key);
		if (!cached || cached.gpuResource !== sampler) {
			return;
		}
		cached.refCount = Math.max(0, cached.refCount - 1);
		if (cached.refCount <= 0) {
			this._samplerCache.delete(key);
		}
	}

	private _releaseShaderModuleCacheEntry(key: string, module: GPUShaderModule): void {
		const cached = this._shaderModuleCache.get(key);
		if (!cached || cached.gpuResource !== module) {
			return;
		}
		cached.refCount = Math.max(0, cached.refCount - 1);
		if (cached.refCount <= 0) {
			this._shaderModuleCache.delete(key);
		}
	}

	private _bumpRefCount(entry: { refCount: number }): void {
		entry.refCount = Math.min(65535, entry.refCount + 1);
	}

	private _releasePipelineCacheEntry<TPipeline extends object>(
		cache: Map<string, { refCount: number; gpuResource: TPipeline }>,
		key: string,
		pipeline: TPipeline,
	): void {
		const cached = cache.get(key);
		if (!cached || cached.gpuResource !== pipeline) {
			return;
		}
		cached.refCount = Math.max(0, cached.refCount - 1);
		if (cached.refCount <= 0) {
			cache.delete(key);
		}
	}

	private _resolveRenderPipelineLayout(
		desc: PipelineDesc,
	): GPUPipelineLayout | GPUAutoLayoutMode {
		const explicitLayout = this._resolveExplicitPipelineLayout(desc.layout);
		if (explicitLayout) {
			return explicitLayout;
		}
		return "auto";
	}

	private _resolveComputePipelineLayout(
		desc: ComputePipelineDesc,
	): GPUPipelineLayout | GPUAutoLayoutMode {
		const explicitLayout = this._resolveExplicitPipelineLayout(desc.layout);
		if (explicitLayout) {
			return explicitLayout;
		}
		return "auto";
	}

	private _resolveExplicitPipelineLayout(
		layout: PipelineDescLayout | ComputePipelineDescLayout,
	): GPUPipelineLayout | null {
		if (layout === null || layout === undefined || layout === "auto") {
			return null;
		}
		return layout as GPUPipelineLayout;
	}

	private _createRenderPipelineDescriptor(
		desc: PipelineDesc,
		layout: GPUPipelineLayout | GPUAutoLayoutMode,
	): GPURenderPipelineDescriptor {
		const probeFormats = this._getRenderPipelineProbeFormats(desc);
		const sampleCount = this._host.resolveSupportedSampleCount(
			Math.max(1, Math.floor(desc.sampleCount ?? 1)),
			probeFormats,
		);
		return {
			layout,
			vertex: {
				module: getWebGPUShaderModule(desc.vertex.module),
				entryPoint: desc.vertex.entryPoint,
				buffers:
					desc.vertex.buffers?.map((buffer) => ({
						arrayStride: buffer.arrayStride,
						stepMode: buffer.stepMode ?? "vertex",
						attributes: buffer.attributes.map((attribute) => ({
							format: attribute.format as GPUVertexFormat,
							offset: attribute.offset,
							shaderLocation: attribute.shaderLocation,
						})),
					})) ?? [],
			},
			fragment: desc.fragment
				? {
						module: getWebGPUShaderModule(desc.fragment.module),
						entryPoint: desc.fragment.entryPoint,
						targets: desc.fragment.targets.map((target) => ({
							format: target.format as GPUTextureFormat,
							blend: target.blend,
							writeMask: target.writeMask,
						})),
					}
				: undefined,
			primitive: {
				topology: desc.primitive?.topology ?? "triangle-list",
				cullMode: desc.primitive?.cullMode ?? "none",
				frontFace: desc.primitive?.frontFace ?? "ccw",
			},
			depthStencil: desc.depthStencil
				? {
						format: desc.depthStencil.format as GPUTextureFormat,
						depthWriteEnabled: desc.depthStencil.depthWriteEnabled,
						depthCompare: desc.depthStencil.depthCompare as GPUCompareFunction,
					}
				: undefined,
			multisample: {
				count: sampleCount,
			},
			label: desc.label,
		};
	}

	private _createComputePipelineDescriptor(
		desc: ComputePipelineDesc,
		layout: GPUPipelineLayout | GPUAutoLayoutMode,
	): GPUComputePipelineDescriptor {
		return {
			layout,
			compute: {
				module: getWebGPUShaderModule(desc.compute.module),
				entryPoint: desc.compute.entryPoint,
			},
			label: desc.label,
		};
	}

	private _getRenderPipelineProbeFormats(desc: PipelineDesc): GPUTextureFormat[] {
		const formats: GPUTextureFormat[] = [];
		if (desc.fragment?.targets) {
			for (const target of desc.fragment.targets) {
				formats.push(target.format as GPUTextureFormat);
			}
		}
		if (desc.depthStencil?.format) {
			formats.push(desc.depthStencil.format as GPUTextureFormat);
		}
		return formats;
	}

	private _getRenderPipelineCacheKey(
		desc: PipelineDesc,
		layout: GPUPipelineLayout | GPUAutoLayoutMode,
	): string {
		const parts: string[] = [];
		parts.push(`layout:${this._host.objectIdentity.getCacheToken(layout)}`);
		parts.push(`vs.module:${this._host.objectIdentity.getCacheToken(desc.vertex.module)}`);
		parts.push(`vs.entry:${desc.vertex.entryPoint}`);

		const vertexBuffers = desc.vertex.buffers ?? [];
		parts.push(`vs.buffers:${vertexBuffers.length}`);
		for (let i = 0; i < vertexBuffers.length; i++) {
			const buffer = vertexBuffers[i];
			parts.push(
				`vsb${i}.stride:${buffer.arrayStride}`,
				`vsb${i}.step:${buffer.stepMode ?? "vertex"}`,
				`vsb${i}.attrs:${buffer.attributes.length}`,
			);
			for (let j = 0; j < buffer.attributes.length; j++) {
				const attribute = buffer.attributes[j];
				parts.push(
					`vsa${i}.${j}.fmt:${attribute.format}`,
					`vsa${i}.${j}.off:${attribute.offset}`,
					`vsa${i}.${j}.loc:${attribute.shaderLocation}`,
				);
			}
		}

		if (desc.fragment) {
			parts.push(`fs.module:${this._host.objectIdentity.getCacheToken(desc.fragment.module)}`);
			parts.push(`fs.entry:${desc.fragment.entryPoint}`);
			parts.push(`fs.targets:${desc.fragment.targets.length}`);
			for (let i = 0; i < desc.fragment.targets.length; i++) {
				const target = desc.fragment.targets[i];
				parts.push(`fst${i}.fmt:${target.format}`);
				parts.push(`fst${i}.blend:${this._serializeBlendState(target.blend)}`);
				parts.push(`fst${i}.write:${this._serializeWriteMask(target.writeMask)}`);
			}
		} else {
			parts.push("fs:none");
		}

		parts.push(`primitive.topology:${desc.primitive?.topology ?? "triangle-list"}`);
		parts.push(`primitive.cull:${desc.primitive?.cullMode ?? "none"}`);
		parts.push(`primitive.front:${desc.primitive?.frontFace ?? "ccw"}`);
		const probeFormats = this._getRenderPipelineProbeFormats(desc);
		const sampleCount = this._host.resolveSupportedSampleCount(
			Math.max(1, Math.floor(desc.sampleCount ?? 1)),
			probeFormats,
		);
		parts.push(`multisample.count:${sampleCount}`);
		if (desc.depthStencil) {
			parts.push(`depth.format:${desc.depthStencil.format}`);
			parts.push(`depth.write:${desc.depthStencil.depthWriteEnabled ? 1 : 0}`);
			parts.push(`depth.compare:${desc.depthStencil.depthCompare}`);
		} else {
			parts.push("depth:none");
		}
		return parts.join("|");
	}

	private _getComputePipelineCacheKey(
		desc: ComputePipelineDesc,
		layout: GPUPipelineLayout | GPUAutoLayoutMode,
	): string {
		return [
			`layout:${this._host.objectIdentity.getCacheToken(layout)}`,
			`cs.module:${this._host.objectIdentity.getCacheToken(desc.compute.module)}`,
			`cs.entry:${desc.compute.entryPoint}`,
		].join("|");
	}

	private _serializeBlendState(blend: unknown): string {
		if (!blend || typeof blend !== "object") {
			return "none";
		}
		const asBlend = blend as {
			color?: {
				srcFactor?: string;
				dstFactor?: string;
				operation?: string;
			};
			alpha?: {
				srcFactor?: string;
				dstFactor?: string;
				operation?: string;
			};
		};
		return [
			`c:${asBlend.color?.srcFactor ?? "none"},${asBlend.color?.dstFactor ?? "none"},${asBlend.color?.operation ?? "add"}`,
			`a:${asBlend.alpha?.srcFactor ?? "none"},${asBlend.alpha?.dstFactor ?? "none"},${asBlend.alpha?.operation ?? "add"}`,
		].join("/");
	}

	private _serializeWriteMask(writeMask: unknown): string {
		if (typeof writeMask !== "number" || !Number.isFinite(writeMask)) {
			return "all";
		}
		return String(Math.max(0, Math.floor(writeMask)));
	}

	private _getCachedShaderCodeHash(code: string): string {
		const cached = this._getLruCacheEntry(this._shaderCodeHashCache, code);
		if (cached !== undefined) {
			return cached;
		}
		const hash = hashString64(code).toString(16);
		this._shaderCodeHashCache.set(code, hash);
		this._trimCache(this._shaderCodeHashCache, SHADER_CODE_HASH_CACHE_LIMIT);
		return hash;
	}

	private _touchCacheEntry<K, T>(cache: Map<K, T>, key: K, value: T): void {
		cache.delete(key);
		cache.set(key, value);
	}

	private _getLruCacheEntry<K, T>(cache: Map<K, T>, key: K): T | undefined {
		const cached = cache.get(key);
		if (cached === undefined && !cache.has(key)) {
			return undefined;
		}
		cache.delete(key);
		cache.set(key, cached as T);
		return cached as T;
	}

	private _trimCache<K, T>(cache: Map<K, T>, maxSize: number): void {
		if (cache.size <= maxSize) {
			return;
		}

		const toEvict = cache.size - maxSize;
		let evicted = 0;
		for (const key of cache.keys()) {
			cache.delete(key);
			evicted++;
			if (evicted >= toEvict) {
				break;
			}
		}
	}

	private _trimRefCountedCache<
		K,
		T extends {
			refCount: number;
		},
	>(cache: Map<K, T>, maxSize: number): void {
		if (cache.size <= maxSize) {
			return;
		}
		const toEvict = cache.size - maxSize;
		let evicted = 0;
		for (const [key, entry] of cache.entries()) {
			if (entry.refCount > 0) {
				continue;
			}
			cache.delete(key);
			evicted++;
			if (evicted >= toEvict) {
				break;
			}
		}
	}
}
