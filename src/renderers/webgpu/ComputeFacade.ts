import type { Texture } from "../../core/Texture";
import type { Renderer } from "../Renderer";
import type { ICommandBuffer, ICommandEncoder } from "../ICommandEncoder";
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
	TextureDesc,
} from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";

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

interface IWebGPUBackendLike {
	type?: unknown;
	device?: GPUDevice;
	createSampler?: (desc: SamplerDesc) => ISampler;
	createShaderModule?: (desc: ShaderModuleDesc) => Promise<IShaderModule>;
	createComputePipeline?: (desc: ComputePipelineDesc) => IComputePipeline;
	createBuffer?: (desc: BufferDesc) => IRenderBuffer;
	createTexture?: (desc: TextureDesc) => IRenderTexture;
	createBindingGroup?: (desc: BindingGroupDesc) => IBindingGroup;
	createBindGroupLayout?: (
		desc: GPUBindGroupLayoutDescriptor
	) => GPUBindGroupLayout;
	createPipelineLayout?: (
		desc: GPUPipelineLayoutDescriptor
	) => GPUPipelineLayout;
	createTextureView?: CreateTextureViewMethod;
	createCommandEncoder?: () => ICommandEncoder;
	submit?: (commands: ICommandBuffer[]) => void;
	writeBuffer?: (
		buffer: IRenderBuffer,
		data: BufferSource,
		offset?: number
	) => void;
	resolveTextureForSlot?: ResolveTextureForSlotMethod;
	getTextureForSlot?: ResolveTextureForSlotMethod;
	registerExternalTexture?: (
		texture: Texture,
		resource: IRenderTexture,
		uploadedVersion?: number,
		mipLevelCount?: number
	) => void;
	unregisterExternalTexture?: (texture: Texture) => void;
}

export interface IWebGPUComputeFacadeResolverSource {
	backend?: WebGPUComputeFacadeSource | null;
	type?: unknown;
	getComputeFacade?: () => WebGPUComputeFacadeSource | null | undefined;
}

export type WebGPUComputeFacadeSource =
	| Renderer
	| WebGPUBackend
	| IWebGPUComputeFacade
	| IWebGPUComputeFacadeResolverSource
	| IWebGPUBackendLike;

export interface IWebGPUComputeFacade {
	readonly [WEBGPU_COMPUTE_FACADE_BRAND]: true;
	createSampler(desc: SamplerDesc): ISampler;
	createShaderModule(desc: ShaderModuleDesc): Promise<IShaderModule>;
	createComputePipeline(desc: ComputePipelineDesc): IComputePipeline;
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

let WEBGPU_COMPUTE_FACADE_CACHE = new WeakMap<
	WebGPUBackend,
	IWebGPUComputeFacade
>();
let WEBGPU_COMPUTE_FACADE_CACHE_ENTRY_COUNT = 0;

class WebGPUBackendComputeFacade implements IWebGPUComputeFacade {
	public readonly [WEBGPU_COMPUTE_FACADE_BRAND] = true;

	private _destroyed = false;
	private _trackedExternalTextures = new Set<Texture>();

	constructor(
		private _backend: WebGPUBackend,
		private _onDestroy: () => void
	) {}

	private _assertAlive(operation: string): void {
		if (this._destroyed) {
			throw new Error(
				`WebGPU compute facade is destroyed; cannot ${operation}.`
			);
		}
	}

	public createSampler(desc: SamplerDesc): ISampler {
		this._assertAlive("create samplers");
		return this._backend.createSampler(desc);
	}

	public createShaderModule(desc: ShaderModuleDesc): Promise<IShaderModule> {
		this._assertAlive("create shader modules");
		return this._backend.createShaderModule(desc);
	}

	public createComputePipeline(desc: ComputePipelineDesc): IComputePipeline {
		this._assertAlive("create compute pipelines");
		return this._backend.createComputePipeline(desc);
	}

	public createBuffer(desc: BufferDesc): IRenderBuffer {
		this._assertAlive("create buffers");
		return this._backend.createBuffer(desc);
	}

	public createTexture(desc: TextureDesc): IRenderTexture {
		this._assertAlive("create textures");
		return this._backend.createTexture(desc);
	}

	public createBindingGroup(desc: BindingGroupDesc): IBindingGroup {
		this._assertAlive("create binding groups");
		return this._backend.createBindingGroup(desc);
	}

	public createBindGroupLayout(
		desc: GPUBindGroupLayoutDescriptor
	): GPUBindGroupLayout {
		this._assertAlive("create bind group layouts");
		const device = resolveLayoutDevice(this._backend, "createBindGroupLayout");
		return device.createBindGroupLayout(desc);
	}

	public createPipelineLayout(
		desc: GPUPipelineLayoutDescriptor
	): GPUPipelineLayout {
		this._assertAlive("create pipeline layouts");
		const device = resolveLayoutDevice(this._backend, "createPipelineLayout");
		return device.createPipelineLayout(desc);
	}

	public createTextureView(
		texture: IRenderTexture,
		desc?: GPUTextureViewDescriptor
	): GPUTextureView {
		this._assertAlive("create texture views");
		const createTextureView = (this._backend as IWebGPUBackendLike)
			.createTextureView;
		if (typeof createTextureView !== "function") {
			throw new Error("WebGPU backend does not expose createTextureView().");
		}
		return createTextureView.call(this._backend, texture, desc);
	}

	public createCommandEncoder(): ICommandEncoder {
		this._assertAlive("create command encoders");
		return this._backend.createCommandEncoder();
	}

	public submit(commands: ICommandBuffer[]): void {
		this._assertAlive("submit command buffers");
		this._backend.submit(commands);
	}

	public writeBuffer(
		buffer: IRenderBuffer,
		data: BufferSource,
		offset: number = 0
	): void {
		this._assertAlive("write buffers");
		this._backend.writeBuffer(buffer, data, offset);
	}

	public resolveTextureForSlot(
		texture: Texture | null,
		slotIndex: number
	): IRenderTexture {
		this._assertAlive("resolve texture slots");
		return this._backend.getTextureForSlot(texture, slotIndex);
	}

	public registerExternalTexture(
		texture: Texture,
		resource: IRenderTexture,
		uploadedVersion: number = texture.version,
		mipLevelCount: number = 1
	): void {
		this._assertAlive("register external textures");
		this._backend.registerExternalTexture(
			texture,
			resource,
			uploadedVersion,
			mipLevelCount
		);
		this._trackedExternalTextures.add(texture);
	}

	public unregisterExternalTexture(texture: Texture): void {
		this._assertAlive("unregister external textures");
		this._backend.unregisterExternalTexture(texture);
		this._trackedExternalTextures.delete(texture);
	}

	public destroy(): void {
		if (this._destroyed) {
			return;
		}
		this._destroyed = true;
		const trackedTextures = Array.from(this._trackedExternalTextures);
		this._trackedExternalTextures.clear();
		for (const texture of trackedTextures) {
			try {
				this._backend.unregisterExternalTexture(texture);
			} catch (error) {
				console.warn(
					`WebGPU compute facade failed to unregister external texture during destroy(): ${String(error)}`
				);
			}
		}
		this._onDestroy();
	}
}

interface AdaptedFacadeOps {
	createSampler: (desc: SamplerDesc) => ISampler;
	createShaderModule: (desc: ShaderModuleDesc) => Promise<IShaderModule>;
	createComputePipeline: (desc: ComputePipelineDesc) => IComputePipeline;
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
	resolveTextureForSlot: ResolveTextureForSlotMethod;
	registerExternalTexture: (
		texture: Texture,
		resource: IRenderTexture,
		uploadedVersion?: number,
		mipLevelCount?: number
	) => void;
	unregisterExternalTexture: (texture: Texture) => void;
	layoutDeviceSource: { device?: GPUDevice };
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
					console.warn(
						`WebGPU adapted compute facade failed to unregister external texture during destroy(): ${String(error)}`
					);
				}
			}
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

function bindMethod<
	TMethod extends (...args: any[]) => any = (...args: any[]) => any,
>(target: object, methodName: string): TMethod | null {
	const method = (target as Record<string, unknown>)[methodName];
	if (typeof method !== "function") {
		return null;
	}
	return method.bind(target) as TMethod;
}

function tryCreateFacadeFromBackendLike(
	value: unknown
): IWebGPUComputeFacade | null {
	if (!value || typeof value !== "object") {
		return null;
	}

	const sourceObject = value as object;
	const source = value as IWebGPUBackendLike;
	const createSampler = bindMethod<(desc: SamplerDesc) => ISampler>(
		sourceObject,
		"createSampler"
	);
	const createShaderModule = bindMethod<
		(desc: ShaderModuleDesc) => Promise<IShaderModule>
	>(sourceObject, "createShaderModule");
	const createComputePipeline = bindMethod<
		(desc: ComputePipelineDesc) => IComputePipeline
	>(sourceObject, "createComputePipeline");
	const createBuffer = bindMethod<(desc: BufferDesc) => IRenderBuffer>(
		sourceObject,
		"createBuffer"
	);
	const createTexture = bindMethod<(desc: TextureDesc) => IRenderTexture>(
		sourceObject,
		"createTexture"
	);
	const createBindingGroup = bindMethod<
		(desc: BindingGroupDesc) => IBindingGroup
	>(sourceObject, "createBindingGroup");
	const createTextureView = bindMethod<CreateTextureViewMethod>(
		sourceObject,
		"createTextureView"
	);
	const createCommandEncoder = bindMethod<() => ICommandEncoder>(
		sourceObject,
		"createCommandEncoder"
	);
	const submit = bindMethod<(commands: ICommandBuffer[]) => void>(
		sourceObject,
		"submit"
	);
	const writeBuffer = bindMethod<
		(buffer: IRenderBuffer, data: BufferSource, offset?: number) => void
	>(sourceObject, "writeBuffer");
	const registerExternalTexture = bindMethod<
		(
			texture: Texture,
			resource: IRenderTexture,
			uploadedVersion?: number,
			mipLevelCount?: number
		) => void
	>(sourceObject, "registerExternalTexture");
	const unregisterExternalTexture = bindMethod<(texture: Texture) => void>(
		sourceObject,
		"unregisterExternalTexture"
	);

	if (
		!createSampler ||
		!createShaderModule ||
		!createComputePipeline ||
		!createBuffer ||
		!createTexture ||
		!createBindingGroup ||
		!createTextureView ||
		!createCommandEncoder ||
		!submit ||
		!writeBuffer ||
		!registerExternalTexture ||
		!unregisterExternalTexture
	) {
		return null;
	}

	const resolveTextureForSlot =
		bindMethod<ResolveTextureForSlotMethod>(
			sourceObject,
			"resolveTextureForSlot"
		) ??
		bindMethod<ResolveTextureForSlotMethod>(sourceObject, "getTextureForSlot");
	if (!resolveTextureForSlot) {
		return null;
	}

	const createBindGroupLayout = bindMethod<
		(desc: GPUBindGroupLayoutDescriptor) => GPUBindGroupLayout
	>(sourceObject, "createBindGroupLayout");
	const createPipelineLayout = bindMethod<
		(desc: GPUPipelineLayoutDescriptor) => GPUPipelineLayout
	>(sourceObject, "createPipelineLayout");

	return createAdaptedFacade({
		createSampler,
		createShaderModule,
		createComputePipeline,
		createBuffer,
		createTexture,
		createBindingGroup,
		createBindGroupLayout,
		createPipelineLayout,
		createTextureView,
		createCommandEncoder,
		submit,
		writeBuffer,
		resolveTextureForSlot,
		registerExternalTexture,
		unregisterExternalTexture,
		layoutDeviceSource: source,
	});
}

function resolveLayoutDevice(
	source: { device?: GPUDevice },
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

function getCachedFacade(backend: WebGPUBackend): IWebGPUComputeFacade | null {
	return WEBGPU_COMPUTE_FACADE_CACHE.get(backend) ?? null;
}

function setCachedFacade(
	backend: WebGPUBackend,
	facade: IWebGPUComputeFacade
): void {
	const hadEntry = WEBGPU_COMPUTE_FACADE_CACHE.has(backend);
	WEBGPU_COMPUTE_FACADE_CACHE.set(backend, facade);
	if (!hadEntry) {
		WEBGPU_COMPUTE_FACADE_CACHE_ENTRY_COUNT++;
	}
}

function removeCachedFacade(
	backend: WebGPUBackend
): IWebGPUComputeFacade | null {
	const cached = WEBGPU_COMPUTE_FACADE_CACHE.get(backend) ?? null;
	if (cached && WEBGPU_COMPUTE_FACADE_CACHE.delete(backend)) {
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
	WEBGPU_COMPUTE_FACADE_CACHE = new WeakMap<
		WebGPUBackend,
		IWebGPUComputeFacade
	>();
	WEBGPU_COMPUTE_FACADE_CACHE_ENTRY_COUNT = 0;
}

export function invalidateWebGPUComputeFacade(backend: WebGPUBackend): void {
	const cached = removeCachedFacade(backend);
	if (!cached) {
		return;
	}
	try {
		cached.destroy();
	} catch (error) {
		console.warn(
			`Failed to destroy cached WebGPU compute facade during invalidation: ${String(error)}`
		);
	}
}

export function createWebGPUComputeFacade(
	backend: WebGPUBackend
): IWebGPUComputeFacade {
	const cached = getCachedFacade(backend);
	if (cached) {
		return cached;
	}
	const facade = new WebGPUBackendComputeFacade(backend, () => {
		removeCachedFacade(backend);
	});
	setCachedFacade(backend, facade);
	return facade;
}

export function resolveWebGPUComputeFacade(
	source: WebGPUComputeFacadeSource
): IWebGPUComputeFacade {
	if (!source || typeof source !== "object") {
		throw new Error(
			"WebGPU compute facade requires a Renderer, WebGPU backend, or compute facade source."
		);
	}

	const visited = new WeakSet<object>();
	let current: unknown = source;
	let depth = 0;
	const maxDepth = 32;

	while (current && typeof current === "object") {
		if (depth++ > maxDepth) {
			throw new Error(
				"Failed to resolve WebGPU compute facade: resolution depth exceeded safe limit."
			);
		}

		const currentObject = current as object;
		if (visited.has(currentObject)) {
			throw new Error(
				"Failed to resolve WebGPU compute facade: cyclic source references detected."
			);
		}
		visited.add(currentObject);

		if (isWebGPUComputeFacade(current)) {
			return current;
		}

		const resolverLike = current as IWebGPUComputeFacadeResolverSource;
		if (typeof resolverLike.getComputeFacade === "function") {
			const resolved = resolverLike.getComputeFacade();
			if (resolved && resolved !== currentObject) {
				current = resolved;
				continue;
			}
		}

		if (resolverLike.backend && resolverLike.backend !== currentObject) {
			current = resolverLike.backend;
			continue;
		}

		if (
			typeof resolverLike.type === "string" &&
			resolverLike.type.length > 0 &&
			resolverLike.type !== "webgpu"
		) {
			throw new Error(
				`WebGPU compute facade requires WebGPU backend, received "${resolverLike.type}".`
			);
		}

		const adapted = tryCreateFacadeFromBackendLike(current);
		if (adapted) {
			return adapted;
		}

		break;
	}

	throw new Error(
		"Failed to resolve WebGPU compute facade from provided source."
	);
}
