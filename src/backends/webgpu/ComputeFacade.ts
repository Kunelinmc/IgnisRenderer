import type { Texture } from "../../core/Texture";
import type { IRenderBackend } from "../IRenderBackend";
import type { ICommandBuffer, ICommandEncoder } from "../ICommandEncoder";
import { WEBGPU_COMPUTE_EXTENSION } from "../BackendExtensions";
import type {
	BindingGroupDesc,
	BufferDesc,
	ComputePipelineDesc,
	IBindingGroup,
	IComputePipeline,
	IRenderBuffer,
	IRenderTexture,
	ISampler,
	IShaderModule,
	SamplerDesc,
	ShaderModuleDesc,
	TextureDataLayout,
	TextureDesc,
} from "../types";
import { Logger } from "../../foundation/Logger";

export const WEBGPU_COMPUTE_FACADE_BRAND = Symbol(
	"IgnisRenderer.WebGPUComputeFacade"
);

type CreateTextureViewMethod = (
	texture: IRenderTexture,
	desc?: GPUTextureViewDescriptor
) => GPUTextureView;

type ResolveTextureForSlotMethod = (
	texture: Texture | null,
	slotIndex: number
) => IRenderTexture;

/**
 * Backend-private operations used to implement the public WebGPU compute
 * extension without exposing device services on `WebGPUBackend`.
 *
 * @internal Owned by `WebGPUBackend`.
 */
export interface WebGPUComputeFacadeHost {
	device: GPUDevice | null;
	queue?: GPUQueue | null;
	createSampler(desc: SamplerDesc): ISampler;
	createShaderModule(desc: ShaderModuleDesc): Promise<IShaderModule>;
	createComputePipeline(
		desc: ComputePipelineDesc
	): Promise<IComputePipeline>;
	createBuffer(desc: BufferDesc): IRenderBuffer;
	createTexture(desc: TextureDesc): IRenderTexture;
	createBindingGroup(desc: BindingGroupDesc): IBindingGroup;
	createBindGroupLayout?: (
		desc: GPUBindGroupLayoutDescriptor
	) => GPUBindGroupLayout;
	createPipelineLayout?: (
		desc: GPUPipelineLayoutDescriptor
	) => GPUPipelineLayout;
	createTextureView: CreateTextureViewMethod;
	createCommandEncoder(): ICommandEncoder;
	submit(commands: ICommandBuffer[]): void;
	writeBuffer(
		buffer: IRenderBuffer,
		data: BufferSource,
		offset?: number
	): void;
	writeTexture(
		texture: IRenderTexture,
		data: BufferSource,
		desc: TextureDataLayout,
		size: { width: number; height: number; depthOrArrayLayers?: number }
	): void;
	resolveTextureForSlot: ResolveTextureForSlotMethod;
	registerExternalTexture(
		texture: Texture,
		resource: IRenderTexture,
		uploadedVersion?: number,
		mipLevelCount?: number
	): void;
	unregisterExternalTexture(texture: Texture): void;
}

export type WebGPUComputeFacadeSource =
	| IRenderBackend
	| IWebGPUComputeFacade;

export interface IWebGPUComputeFacade {
	readonly [WEBGPU_COMPUTE_FACADE_BRAND]: true;
	readonly device: GPUDevice | null;
	readonly queue: GPUQueue | null;
	createSampler(desc: SamplerDesc): ISampler;
	createShaderModule(desc: ShaderModuleDesc): Promise<IShaderModule>;
	createComputePipeline(desc: ComputePipelineDesc): Promise<IComputePipeline>;
	createBuffer(desc: BufferDesc): IRenderBuffer;
	createTexture(desc: TextureDesc): IRenderTexture;
	createBindingGroup(desc: BindingGroupDesc): IBindingGroup;
	createBindGroupLayout(desc: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout;
	createPipelineLayout(desc: GPUPipelineLayoutDescriptor): GPUPipelineLayout;
	createTextureView(
		texture: IRenderTexture,
		desc?: GPUTextureViewDescriptor
	): GPUTextureView;
	createCommandEncoder(): ICommandEncoder;
	submit(commands: ICommandBuffer[]): void;
	writeBuffer(buffer: IRenderBuffer, data: BufferSource, offset?: number): void;
	writeTexture(
		texture: IRenderTexture,
		data: BufferSource,
		desc: TextureDataLayout,
		size: { width: number; height: number; depthOrArrayLayers?: number }
	): void;
	resolveTextureForSlot(
		texture: Texture | null,
		slotIndex: number
	): IRenderTexture;
	registerExternalTexture(
		texture: Texture,
		resource: IRenderTexture,
		uploadedVersion?: number,
		mipLevelCount?: number
	): void;
	unregisterExternalTexture(texture: Texture): void;
	destroy(): void;
}

let WEBGPU_COMPUTE_FACADE_CACHE = new WeakMap<object, IWebGPUComputeFacade>();
let WEBGPU_COMPUTE_FACADE_CACHE_ENTRY_COUNT = 0;

interface AdaptedFacadeOps {
	deviceSource: { device?: GPUDevice | null; queue?: GPUQueue | null };
	createSampler: (desc: SamplerDesc) => ISampler;
	createShaderModule: (desc: ShaderModuleDesc) => Promise<IShaderModule>;
	createComputePipeline: (desc: ComputePipelineDesc) => Promise<IComputePipeline>;
	createBuffer: (desc: BufferDesc) => IRenderBuffer;
	createTexture: (desc: TextureDesc) => IRenderTexture;
	createBindingGroup: (desc: BindingGroupDesc) => IBindingGroup;
	createBindGroupLayout?:
		| ((desc: GPUBindGroupLayoutDescriptor) => GPUBindGroupLayout)
		| null;
	createPipelineLayout?:
		| ((desc: GPUPipelineLayoutDescriptor) => GPUPipelineLayout)
		| null;
	createTextureView: CreateTextureViewMethod;
	createCommandEncoder: () => ICommandEncoder;
	submit: (commands: ICommandBuffer[]) => void;
	writeBuffer: (
		buffer: IRenderBuffer,
		data: BufferSource,
		offset?: number
	) => void;
	writeTexture: (
		texture: IRenderTexture,
		data: BufferSource,
		desc: TextureDataLayout,
		size: { width: number; height: number; depthOrArrayLayers?: number }
	) => void;
	resolveTextureForSlot: ResolveTextureForSlotMethod;
	registerExternalTexture: (
		texture: Texture,
		resource: IRenderTexture,
		uploadedVersion?: number,
		mipLevelCount?: number
	) => void;
	unregisterExternalTexture: (texture: Texture) => void;
	layoutDeviceSource: { device?: GPUDevice | null };
	onDestroy?: () => void;
}

function createAdaptedFacade(ops: AdaptedFacadeOps): IWebGPUComputeFacade {
	let destroyed = false;
	const trackedExternalTextures = new Set<Texture>();

	const assertAlive = (operation: string): void => {
		if (destroyed) {
			throw new Error(
				`WebGPU compute facade is destroyed; cannot ${operation}.`
			);
		}
	};

	return {
		[WEBGPU_COMPUTE_FACADE_BRAND]: true,
		get device() {
			return ops.deviceSource.device ?? null;
		},
		get queue() {
			return ops.deviceSource.queue ?? ops.deviceSource.device?.queue ?? null;
		},
		createSampler: (desc) => {
			assertAlive("create samplers");
			return ops.createSampler(desc);
		},
		createShaderModule: (desc) => {
			assertAlive("create shader modules");
			return ops.createShaderModule(desc);
		},
		createComputePipeline: (desc) => {
			assertAlive("create compute pipelines");
			return ops.createComputePipeline(desc);
		},
		createBuffer: (desc) => {
			assertAlive("create buffers");
			return ops.createBuffer(desc);
		},
		createTexture: (desc) => {
			assertAlive("create textures");
			return ops.createTexture(desc);
		},
		createBindingGroup: (desc) => {
			assertAlive("create binding groups");
			return ops.createBindingGroup(desc);
		},
		createBindGroupLayout: (desc) => {
			assertAlive("create bind group layouts");
			if (ops.createBindGroupLayout) {
				return ops.createBindGroupLayout(desc);
			}
			const device = resolveLayoutDevice(
				ops.layoutDeviceSource,
				"createBindGroupLayout"
			);
			return device.createBindGroupLayout(desc);
		},
		createPipelineLayout: (desc) => {
			assertAlive("create pipeline layouts");
			if (ops.createPipelineLayout) {
				return ops.createPipelineLayout(desc);
			}
			const device = resolveLayoutDevice(
				ops.layoutDeviceSource,
				"createPipelineLayout"
			);
			return device.createPipelineLayout(desc);
		},
		createTextureView: (texture, desc) => {
			assertAlive("create texture views");
			return ops.createTextureView(texture, desc);
		},
		createCommandEncoder: () => {
			assertAlive("create command encoders");
			return ops.createCommandEncoder();
		},
		submit: (commands) => {
			assertAlive("submit command buffers");
			ops.submit(commands);
		},
		writeBuffer: (buffer, data, offset = 0) => {
			assertAlive("write buffers");
			ops.writeBuffer(buffer, data, offset);
		},
		writeTexture: (texture, data, desc, size) => {
			assertAlive("write textures");
			ops.writeTexture(texture, data, desc, size);
		},
		resolveTextureForSlot: (texture, slotIndex) => {
			assertAlive("resolve texture slots");
			return ops.resolveTextureForSlot(texture, slotIndex);
		},
		registerExternalTexture: (
			texture,
			resource,
			uploadedVersion = texture.version,
			mipLevelCount = 1
		) => {
			assertAlive("register external textures");
			ops.registerExternalTexture(
				texture,
				resource,
				uploadedVersion,
				mipLevelCount
			);
			trackedExternalTextures.add(texture);
		},
		unregisterExternalTexture: (texture) => {
			assertAlive("unregister external textures");
			ops.unregisterExternalTexture(texture);
			trackedExternalTextures.delete(texture);
		},
		destroy: () => {
			if (destroyed) {
				return;
			}
			destroyed = true;
			const trackedTextures = Array.from(trackedExternalTextures);
			trackedExternalTextures.clear();
			for (const texture of trackedTextures) {
				try {
					ops.unregisterExternalTexture(texture);
				} catch (error) {
					Logger.warn(
						`WebGPU adapted compute facade failed to unregister external texture during destroy(): ${String(error)}`,
						{ scope: "WebGPUComputeFacade" }
					);
				}
			}
			ops.onDestroy?.();
		},
	};
}

function hasComputeFacadeMethodSurface(
	candidate: Record<string, unknown>
): boolean {
	return (
		typeof candidate.createSampler === "function" &&
		typeof candidate.createShaderModule === "function" &&
		typeof candidate.createComputePipeline === "function" &&
		typeof candidate.createBuffer === "function" &&
		typeof candidate.createTexture === "function" &&
		typeof candidate.createBindingGroup === "function" &&
		typeof candidate.createBindGroupLayout === "function" &&
		typeof candidate.createPipelineLayout === "function" &&
		typeof candidate.createTextureView === "function" &&
		typeof candidate.createCommandEncoder === "function" &&
		typeof candidate.submit === "function" &&
		typeof candidate.writeBuffer === "function" &&
		typeof candidate.writeTexture === "function" &&
		typeof candidate.resolveTextureForSlot === "function" &&
		typeof candidate.registerExternalTexture === "function" &&
		typeof candidate.unregisterExternalTexture === "function" &&
		typeof candidate.destroy === "function"
	);
}

function isWebGPUComputeFacade(value: unknown): value is IWebGPUComputeFacade {
	if (!value || typeof value !== "object") {
		return false;
	}
	const candidate = value as Record<string, unknown>;
	if (
		(candidate as { [WEBGPU_COMPUTE_FACADE_BRAND]?: unknown })[
			WEBGPU_COMPUTE_FACADE_BRAND
		] !== true
	) {
		return false;
	}
	return hasComputeFacadeMethodSurface(candidate);
}

function resolveLayoutDevice(
	source: { device?: GPUDevice | null },
	operation: "createBindGroupLayout" | "createPipelineLayout"
): GPUDevice {
	const device = source.device;
	if (!device) {
		throw new Error(
			"WebGPU compute facade requires an initialized GPU device."
		);
	}
	const hasBindGroupLayout = typeof device.createBindGroupLayout === "function";
	const hasPipelineLayout = typeof device.createPipelineLayout === "function";
	if (!hasBindGroupLayout || !hasPipelineLayout) {
		throw new Error(
			`WebGPU compute facade device does not support ${operation}().`
		);
	}
	return device;
}

function getCachedFacade(host: object): IWebGPUComputeFacade | null {
	return WEBGPU_COMPUTE_FACADE_CACHE.get(host) ?? null;
}

function setCachedFacade(
	host: object,
	facade: IWebGPUComputeFacade
): void {
	const hadEntry = WEBGPU_COMPUTE_FACADE_CACHE.has(host);
	WEBGPU_COMPUTE_FACADE_CACHE.set(host, facade);
	if (!hadEntry) {
		WEBGPU_COMPUTE_FACADE_CACHE_ENTRY_COUNT++;
	}
}

function removeCachedFacade(
	host: object
): IWebGPUComputeFacade | null {
	const cached = WEBGPU_COMPUTE_FACADE_CACHE.get(host) ?? null;
	if (cached && WEBGPU_COMPUTE_FACADE_CACHE.delete(host)) {
		WEBGPU_COMPUTE_FACADE_CACHE_ENTRY_COUNT = Math.max(
			0,
			WEBGPU_COMPUTE_FACADE_CACHE_ENTRY_COUNT - 1
		);
	}
	return cached;
}

export function getWebGPUComputeFacadeCacheStats(): {
	entryCount: number;
} {
	return {
		entryCount: WEBGPU_COMPUTE_FACADE_CACHE_ENTRY_COUNT,
	};
}

export function resetWebGPUComputeFacadeCacheForTesting(): void {
	WEBGPU_COMPUTE_FACADE_CACHE = new WeakMap<object, IWebGPUComputeFacade>();
	WEBGPU_COMPUTE_FACADE_CACHE_ENTRY_COUNT = 0;
}

export function invalidateWebGPUComputeFacade(host: object): void {
	const cached = removeCachedFacade(host);
	if (!cached) {
		return;
	}
	try {
		cached.destroy();
	} catch (error) {
		Logger.warn(
			`Failed to destroy cached WebGPU compute facade during invalidation: ${String(error)}`,
			{ scope: "WebGPUComputeFacade" }
		);
	}
}

export function createWebGPUComputeFacade(
	host: WebGPUComputeFacadeHost
): IWebGPUComputeFacade {
	const cached = getCachedFacade(host);
	if (cached) {
		return cached;
	}
	const facade = createAdaptedFacade({
		deviceSource: host,
		createSampler: (desc) => host.createSampler(desc),
		createShaderModule: (desc) => host.createShaderModule(desc),
		createComputePipeline: (desc) => host.createComputePipeline(desc),
		createBuffer: (desc) => host.createBuffer(desc),
		createTexture: (desc) => host.createTexture(desc),
		createBindingGroup: (desc) => host.createBindingGroup(desc),
		createBindGroupLayout:
			host.createBindGroupLayout ?
				(desc) => host.createBindGroupLayout!(desc)
			:	null,
		createPipelineLayout:
			host.createPipelineLayout ?
				(desc) => host.createPipelineLayout!(desc)
			:	null,
		createTextureView: (texture, desc) =>
			host.createTextureView(texture, desc),
		createCommandEncoder: () => host.createCommandEncoder(),
		submit: (commands) => host.submit(commands),
		writeBuffer: (buffer, data, offset) =>
			host.writeBuffer(buffer, data, offset),
		writeTexture: (texture, data, desc, size) =>
			host.writeTexture(texture, data, desc, size),
		resolveTextureForSlot: (texture, slotIndex) =>
			host.resolveTextureForSlot(texture, slotIndex),
		registerExternalTexture: (texture, resource, uploadedVersion, mipLevelCount) =>
			host.registerExternalTexture(
				texture,
				resource,
				uploadedVersion,
				mipLevelCount,
			),
		unregisterExternalTexture: (texture) =>
			host.unregisterExternalTexture(texture),
		layoutDeviceSource: host,
		onDestroy: () => {
			removeCachedFacade(host);
		},
	});
	setCachedFacade(host, facade);
	return facade;
}

export function resolveWebGPUComputeFacade(
	source: WebGPUComputeFacadeSource
): IWebGPUComputeFacade {
	if (!source || typeof source !== "object") {
		throw new Error(
			"WebGPU compute facade requires a WebGPU backend or compute facade source."
		);
	}

	if (isWebGPUComputeFacade(source)) {
		return source;
	}

	const extensions = (source as Partial<IRenderBackend>).extensions;
	if (!extensions || typeof extensions.getBackendExtension !== "function") {
		throw new Error(
			'The provided source is neither a WebGPU compute facade nor an IRenderBackend exposing "webgpu.compute".'
		);
	}
	const facade = extensions.getBackendExtension(WEBGPU_COMPUTE_EXTENSION);
	if (facade && isWebGPUComputeFacade(facade)) {
		return facade;
	}

	throw new Error(
		'The provided render backend does not expose the "webgpu.compute" extension.'
	);
}
