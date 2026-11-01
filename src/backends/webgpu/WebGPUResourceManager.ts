import {
	BufferUsage,
	TextureUsage,
	type BindingResource,
	type BufferDesc,
	type IBindingGroup,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderPipeline,
	type IRenderTexture,
	type ISampler,
	type IShaderModule,
	type TextureDataLayout,
	type TextureDesc,
} from "../types";
import {
	getTextureFormatFallback,
	getTextureFormatInfo,
	textureFormatRequiresFeature,
	type TextureFormat,
} from "../../core/TextureFormat";
import type { Texture } from "../../core/Texture";
import { Logger } from "../../foundation/Logger";
import { createTextureMipUploadLevels } from "./texture";
import { supportsWebGPUTextureSourceUploadFormat } from "./WebGPUTextureFormatInfo";

const WEBGPU_TEXTURE_COPY_DST = globalThis.GPUTextureUsage?.COPY_DST ?? 0x02;
const WEBGPU_TEXTURE_RENDER_ATTACHMENT = globalThis.GPUTextureUsage?.RENDER_ATTACHMENT ?? 0x10;

/** @internal Device services for the WebGPU resource owner; applications use `Renderer`. */
export interface WebGPUResourceManagerHost {
	readonly device: GPUDevice | null;
	readonly queue: GPUQueue | null;
	assertDeviceOperational(operation: string): void;
	resolveSupportedSampleCount(
		requested: number,
		probeFormats?: readonly GPUTextureFormat[],
	): number;
	createManagedDestroy(
		target: object,
		options: {
			label: string;
			dispose: () => void;
		},
	): () => void;
	runValidationScope<T>(label: string, operation: () => T): T;
}

/** @internal WebGPU texture allocation and its default sampling view. */
export interface WebGPUTexture {
	readonly texture: GPUTexture;
	readonly view: GPUTextureView;
}

type ResourceKind = IRenderBuffer["kind"] | IRenderTexture["kind"] | ISampler["kind"] |
	IShaderModule["kind"] | IRenderPipeline["kind"] | IComputePipeline["kind"] |
	IBindingGroup["kind"];

interface ResourceRecord<T extends object> {
	native: T | null;
	available: boolean;
	dispose: () => void;
}

/**
 * Owns native allocations, opaque registrations, uploads, and resource lifetimes
 * for one WebGPU backend and its device generations.
 *
 * @internal Owned by `WebGPUBackend`; applications use `Renderer` or its compute facade.
 */
export class WebGPUResourceManager {
	private readonly _buffers = new WeakMap<object, ResourceRecord<GPUBuffer>>();
	private readonly _textures = new WeakMap<object, ResourceRecord<WebGPUTexture>>();
	private readonly _samplers = new WeakMap<object, ResourceRecord<GPUSampler>>();
	private readonly _shaders = new WeakMap<object, ResourceRecord<GPUShaderModule>>();
	private readonly _renderPipelines = new WeakMap<object, ResourceRecord<GPURenderPipeline>>();
	private readonly _computePipelines = new WeakMap<object, ResourceRecord<GPUComputePipeline>>();
	private readonly _groups = new WeakMap<object, ResourceRecord<GPUBindGroup>>();
	private readonly _records = new WeakMap<object, ResourceRecord<object>>();
	// Records retain native allocations for explicit teardown, without retaining
	// facade keys and preventing the existing forgotten-destroy diagnostics.
	private readonly _live = new Set<ResourceRecord<object>>();

	constructor(private readonly _host: WebGPUResourceManagerHost) {}

	public createBuffer(desc: BufferDesc): IRenderBuffer {
		this._host.assertDeviceOperational("create buffers");
		const device = this._requireDevice("create buffers");
		const hasInitialData = !!desc.initialData;
		const mappedAtCreation = hasInitialData || !!desc.mappedAtCreation;
		const gpuBuffer = device.createBuffer({
			size: desc.size,
			usage: mapBufferUsage(desc.usage),
			mappedAtCreation,
			label: desc.label,
		});
		if (hasInitialData) {
			const source = desc.initialData as BufferSource;
			const mappedRange = gpuBuffer.getMappedRange();
			const target = new Uint8Array(mappedRange);
			const srcView = toUint8View(source);
			const copyLength = Math.min(target.byteLength, srcView.byteLength);
			target.set(srcView.subarray(0, copyLength), 0);
			gpuBuffer.unmap();
		}

		const disposal = { current: () => {} };
		const buffer = this.registerBuffer(gpuBuffer, {
			size: desc.size,
		}, createDeferredBufferDispose(disposal));
		// Manager-retained callbacks must not share the scope holding the facade.
		buffer.unmap = createBufferUnmap(this, buffer);
		disposal.current = this._host.createManagedDestroy(buffer, {
			label: desc.label ?? "WebGPUBuffer",
			dispose: createNativeBufferDispose(gpuBuffer),
		});
		return buffer;
	}

	public createTexture(desc: TextureDesc): IRenderTexture {
		this._host.assertDeviceOperational("create textures");
		const device = this._requireDevice("create textures");
		const dimension = (desc.dimension ?? "2d") as GPUTextureDimension;
		const resolvedWidth = resolvePositiveInteger(desc.width, 1);
		const resolvedHeight = resolvePositiveInteger(desc.height, 1);
		const depthOrArrayLayers = resolvePositiveInteger(desc.depthOrArrayLayers ?? 1, 1);
		const formatResolution = resolveWebGPUTextureFormat(desc.format, device.features);
		const requestedSampleCount = Math.max(1, Math.floor(desc.sampleCount ?? 1));
		const sampleCount =
			dimension === "2d"
				? this._host.resolveSupportedSampleCount(requestedSampleCount, [
						formatResolution.format as GPUTextureFormat,
					])
				: 1;
		const size: GPUExtent3DStrict =
			dimension === "1d"
				? {
						width: resolvedWidth,
					}
				: {
						width: resolvedWidth,
						height: resolvedHeight,
						depthOrArrayLayers,
					};
		const baseDescriptor: GPUTextureDescriptor = {
			size,
			dimension,
			sampleCount,
			format: formatResolution.format as GPUTextureFormat,
			usage: mapTextureUsage(desc.usage),
			mipLevelCount: Math.max(1, desc.mipLevelCount ?? 1),
			viewFormats:
				formatResolution.format === desc.format
					? (desc.viewFormats as GPUTextureFormat[] | undefined)
					: undefined,
			label: desc.label,
		};
		if (formatResolution.reason) {
			Logger.warn(`[webgpu-texture-format-fallback] ${formatResolution.reason}`, {
				scope: "WebGPUResourceManager",
				onceKey: `webgpu-texture-format-fallback-${desc.format}-${formatResolution.format}`,
			});
		}
		const gpuTexture = device.createTexture(baseDescriptor);
		const webgpuTexture = { texture: gpuTexture, view: gpuTexture.createView() };
		let dispose = () => {};
		const texture = this.registerTexture(webgpuTexture, {
			width: resolvedWidth,
			height: dimension === "1d" ? 1 : resolvedHeight,
			requestedFormat: desc.format,
			format: formatResolution.format,
			formatFallbackReason: formatResolution.reason,
		}, () => dispose());
		dispose = this._host.createManagedDestroy(texture, {
			label: desc.label ?? "WebGPUTexture",
			dispose: () => gpuTexture.destroy(),
		});
		return texture;
	}

	public writeBuffer(buffer: IRenderBuffer, data: BufferSource, offset: number = 0): void {
		this._host.assertDeviceOperational("write buffers");
		this._requireQueue("write buffers").writeBuffer(
			this.getBuffer(buffer), offset, data,
		);
	}

	/**
	 * Writes authored source mips at the target origin using its actual format.
	 *
	 * @internal Owned by WebGPU resource uploads. Applications should use
	 * `ComputeRuntime.createTexture({ source })` for runtime-owned snapshots.
	 * @param target - Live texture owned by this manager with sufficient copy usage.
	 * @param source - Engine texture containing raw mips or an external image source.
	 * @throws When ownership, usage, layout, range, or format conversion is invalid.
	 * The existing target is never destroyed by this operation.
	 */
	public writeTexture(target: IRenderTexture, source: Texture): void;
	/**
	 * Writes bytes using an explicit layout and copy extent without conversion.
	 *
	 * @internal Owned by WebGPU resource uploads; applications use
	 * `ComputeRuntime.writeTexture()` for explicit byte uploads.
	 */
	public writeTexture(
		target: IRenderTexture,
		data: BufferSource,
		layout: TextureDataLayout,
		size: { width: number; height: number; depthOrArrayLayers?: number },
	): void;
	public writeTexture(
		target: IRenderTexture,
		input: BufferSource | Texture,
		layout?: TextureDataLayout,
		size?: { width: number; height: number; depthOrArrayLayers?: number },
	): void {
		this._host.assertDeviceOperational("write textures");
		const gpuTexture = this.getTexture(target).texture;
		const source = input as Texture;
		if (arguments.length !== 2) {
			if (arguments.length !== 4 || !layout || !size ||
				typeof source?.getUploadSource === "function") {
				throw new TypeError("writeTexture requires bytes, layout and size, or a Texture alone.");
			}
			this._requireQueue("write textures").writeTexture({
				texture: gpuTexture,
				mipLevel: Math.max(0, layout.mipLevel ?? 0),
			}, input as BufferSource, {
				offset: layout.offset ?? 0,
				bytesPerRow: layout.bytesPerRow,
				rowsPerImage: layout.rowsPerImage,
			}, size);
			return;
		}
		if (!source || typeof source !== "object" ||
			!Array.isArray(source.levels) || !Array.isArray(source.mipmaps) ||
			typeof source.getUploadSource !== "function" ||
			typeof source.getMipLevelDescriptor !== "function" ||
			typeof source.readPixelData !== "function") {
			throw new TypeError("writeTexture requires a Texture source or bytes with layout and size.");
		}
		if (gpuTexture.sampleCount !== 1) {
			throw new Error("Texture source writes require a single-sampled target.");
		}
		if (!(gpuTexture.usage & WEBGPU_TEXTURE_COPY_DST)) {
			throw new Error("Texture source writes require target CopyDst usage.");
		}
		if (!Number.isInteger(source.width) || source.width <= 0 ||
			!Number.isInteger(source.height) || source.height <= 0) {
			throw new RangeError("Texture source dimensions must be positive integers.");
		}
		if (Math.max(1, source.levels.length, source.mipmaps.length) > gpuTexture.mipLevelCount) {
			throw new RangeError("Texture source mip count exceeds the target mip range.");
		}
		for (const level of source.levels) {
			if ((level?.depthOrArrayLayers ?? 1) !== 1) {
				throw new Error("Texture source writes require single-layer 2D mips.");
			}
		}
		const format = (gpuTexture.format ?? target.format) as TextureFormat;
		if (!supportsWebGPUTextureSourceUploadFormat(format)) {
			throw new Error(`Unsupported WebGPU texture source upload format "${format}".`);
		}
		if (getTextureFormatInfo(source.format).isCompressed &&
			source.format.replace(/-srgb$/, "") !== format.replace(/-srgb$/, "")) {
			throw new Error("WebGPU compressed source requires a storage-compatible texture format.");
		}
		if (this._tryUploadExternalSource(source, gpuTexture)) return;

		// Validate every mip before any write can modify the existing target.
		const uploads = createTextureMipUploadLevels(source, format);
		for (const upload of uploads) {
			assertTextureWriteRange(gpuTexture, format, upload.mipLevel,
				upload.copyWidth, upload.copyHeight);
		}
		for (const upload of uploads) {
			const data = upload.data.buffer instanceof ArrayBuffer ?
				new Uint8Array(upload.data.buffer, upload.data.byteOffset, upload.data.byteLength) :
				new Uint8Array(upload.data);
			this.writeTexture(target, data, {
				bytesPerRow: upload.bytesPerRow,
				rowsPerImage: upload.rowsPerImage,
				mipLevel: upload.mipLevel,
			}, {
				width: upload.copyWidth,
				height: upload.copyHeight,
				depthOrArrayLayers: 1,
			});
		}
	}

	public createTextureView(
		texture: IRenderTexture,
		desc?: GPUTextureViewDescriptor,
	): GPUTextureView {
		this._host.assertDeviceOperational("create texture views");
		const gpuTexture = this.getTexture(texture);
		return this._host.runValidationScope("device.createTextureView", () => {
			if (desc) {
				return gpuTexture.texture.createView(desc);
			}
			return gpuTexture.view;
		});
	}

	private _requireDevice(operation: string): GPUDevice {
		const device = this._host.device;
		if (!device) {
			throw new Error(`WebGPU backend is not initialized; cannot ${operation}.`);
		}
		return device;
	}

	private _requireQueue(operation: string): GPUQueue {
		const queue = this._host.queue;
		if (!queue) {
			throw new Error(`WebGPU backend is not initialized; cannot ${operation}.`);
		}
		return queue;
	}

	/** @internal Tests owner identity; getters additionally require a live record. */
	public owns(resource: unknown): boolean {
		return resource !== null && typeof resource === "object" && this._records.has(resource);
	}

	/** @internal Registers a buffer allocated by the owning WebGPU resource factory. */
	public registerBuffer(
		native: GPUBuffer,
		metadata: Omit<IRenderBuffer, "kind" | "destroy">,
		dispose: () => void = () => {},
	): IRenderBuffer {
		return this._register("buffer", this._buffers, native, metadata, dispose);
	}

	/** @internal Registers an owned or borrowed texture and its default view. */
	public registerTexture(
		native: WebGPUTexture,
		metadata: Omit<IRenderTexture, "kind" | "destroy">,
		dispose: () => void = () => {},
	): IRenderTexture {
		return this._register("texture", this._textures, native, metadata, dispose);
	}

	/** @internal Registers one managed reference to a cached WebGPU sampler. */
	public registerSampler(native: GPUSampler, label?: string, dispose = () => {}): ISampler {
		return this._register("sampler", this._samplers, native, { label }, dispose);
	}

	/** @internal Registers one managed reference to a cached WebGPU shader module. */
	public registerShaderModule(
		native: GPUShaderModule, label?: string, dispose = () => {},
	): IShaderModule {
		return this._register("shader-module", this._shaders, native, { label }, dispose);
	}

	/** @internal Registers one managed reference to a cached WebGPU render pipeline. */
	public registerRenderPipeline(
		native: GPURenderPipeline, label?: string, dispose = () => {},
	): IRenderPipeline {
		return this._register("render-pipeline", this._renderPipelines, native, { label }, dispose);
	}

	/** @internal Registers one managed reference to a cached WebGPU compute pipeline. */
	public registerComputePipeline(
		native: GPUComputePipeline, label?: string, dispose = () => {},
	): IComputePipeline {
		return this._register("compute-pipeline", this._computePipelines, native, { label }, dispose);
	}

	/** @internal Registers a WebGPU binding and its existing cache release operation. */
	public registerBindingGroup(
		native: GPUBindGroup, label?: string, dispose = () => {},
	): IBindingGroup {
		return this._register("binding-group", this._groups, native, { label }, dispose);
	}

	/** @internal Resolves live buffer state owned by this WebGPU backend. */
	public getBuffer(resource: unknown): GPUBuffer {
		return this._require(resource, this._buffers, "buffer");
	}
	/** @internal Resolves live texture state owned by this WebGPU backend. */
	public getTexture(resource: unknown): WebGPUTexture {
		return this._require(resource, this._textures, "texture");
	}
	/** @internal Resolves live sampler state owned by this WebGPU backend. */
	public getSampler(resource: unknown): GPUSampler {
		return this._require(resource, this._samplers, "sampler");
	}
	/** @internal Resolves live shader state owned by this WebGPU backend. */
	public getShaderModule(resource: unknown): GPUShaderModule {
		return this._require(resource, this._shaders, "shader module");
	}
	/** @internal Resolves live render pipeline state owned by this WebGPU backend. */
	public getRenderPipeline(resource: unknown): GPURenderPipeline {
		return this._require(resource, this._renderPipelines, "render pipeline");
	}
	/** @internal Resolves live compute pipeline state owned by this WebGPU backend. */
	public getComputePipeline(resource: unknown): GPUComputePipeline {
		return this._require(resource, this._computePipelines, "compute pipeline");
	}
	/** @internal Resolves live bind-group state owned by this WebGPU backend. */
	public getBindingGroup(resource: unknown): GPUBindGroup {
		return this._require(resource, this._groups, "binding group");
	}

	/** @internal Validates the renderer-only input boundary of the compute extension. */
	public assertBindingResource(resource: BindingResource): void {
		if (!resource || typeof resource !== "object") {
			throw new Error("Binding resource must be an owned renderer resource.");
		}
		if ("buffer" in resource) {
			this.getBuffer(resource.buffer);
			return;
		}
		switch (resource.kind) {
			case "buffer": this.getBuffer(resource); return;
			case "texture": this.getTexture(resource); return;
			case "sampler": this.getSampler(resource); return;
			default: throw new Error("Binding resource must be an owned renderer resource.");
		}
	}

	/** @internal Invalidates a facade while its native state awaits deferred disposal. */
	public invalidate(resource: object): void {
		const record = this._records.get(resource);
		if (!record) throw new Error("Resource is not owned by this WebGPU backend.");
		record.available = false;
	}

	/** @internal Retires this device generation and explicitly releases live records. */
	public reset(): void {
		let failure: unknown;
		for (const record of this._live) {
			try {
				this._dispose(record);
			} catch (error) {
				failure ??= error;
			}
		}
		if (failure) throw failure;
	}

	private _tryUploadExternalSource(source: Texture, target: GPUTexture): boolean {
		if (Math.max(source.levels.length, source.mipmaps.length) > 1) return false;
		if (target.dimension !== "2d" || !(target.usage & WEBGPU_TEXTURE_RENDER_ATTACHMENT)) {
			return false;
		}
		const info = getTextureFormatInfo(target.format as TextureFormat);
		// Unsupported native copies fail asynchronously, so catch cannot trigger fallback.
		if (info.formatClass !== "color" || info.componentType === "uint" ||
			info.componentType === "sint" || info.componentType === "snorm") {
			return false;
		}
		const queue = this._host.queue;
		if (!queue || typeof queue.copyExternalImageToTexture !== "function") return false;
		const image = source.getUploadSource(0);
		if (!image || ArrayBuffer.isView(image) || source.width <= 0 || source.height <= 0) {
			return false;
		}
		assertTextureWriteRange(target, target.format as TextureFormat, 0, source.width, source.height);
		try {
			queue.copyExternalImageToTexture({ source: image }, { texture: target }, {
				width: source.width,
				height: source.height,
				depthOrArrayLayers: 1,
			});
			return true;
		} catch {
			return false;
		}
	}

	private _register<K extends ResourceKind, T extends object, M extends object>(
		kind: K, records: WeakMap<object, ResourceRecord<T>>, native: T,
		metadata: M, dispose: () => void,
	): M & { readonly kind: K; destroy(): void } {
		const record: ResourceRecord<T> = { native, available: true, dispose };
		// Only this manager's private registration records establish provenance.
		const facade = Object.assign({ kind }, metadata, {
			destroy: () => this._dispose(record),
		});
		records.set(facade, record);
		this._records.set(facade, record);
		this._live.add(record);
		return facade;
	}

	private _dispose<T extends object>(record: ResourceRecord<T>): void {
		if (!record.native) return;
		record.available = false;
		record.native = null;
		this._live.delete(record);
		const dispose = record.dispose;
		record.dispose = () => {};
		dispose();
	}

	private _require<T extends object>(
		resource: unknown, records: WeakMap<object, ResourceRecord<T>>, kind: string,
	): T {
		// A record in this manager's typed table already proves resource provenance.
		const record = resource !== null && typeof resource === "object" ?
			records.get(resource) : undefined;
		if (record?.available && record.native) return record.native;
		if (!this.owns(resource)) {
			throw new Error(`WebGPU ${kind} is not owned by this backend.`);
		}
		throw new Error(`WebGPU ${kind} is not live: wrong kind, destroyed or retired resource.`);
	}
}

function assertTextureWriteRange(
	target: GPUTexture,
	format: TextureFormat,
	mipLevel: number,
	width: number,
	height: number,
): void {
	if (!Number.isInteger(mipLevel) || mipLevel < 0 ||
		!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
		throw new RangeError("Texture source copy extents must be positive integers.");
	}
	const info = getTextureFormatInfo(format);
	const mipWidth = Math.max(1, Math.floor(target.width / 2 ** mipLevel));
	const mipHeight = Math.max(1, Math.floor(target.height / 2 ** mipLevel));
	const physicalWidth = Math.ceil(mipWidth / info.blockWidth) * info.blockWidth;
	const physicalHeight = Math.ceil(mipHeight / info.blockHeight) * info.blockHeight;
	if (mipLevel >= target.mipLevelCount || width > physicalWidth || height > physicalHeight) {
		throw new RangeError("Texture source copy extent exceeds the target mip range.");
	}
}

function resolvePositiveInteger(value: number, fallback: number): number {
	if (!Number.isFinite(value)) {
		return fallback;
	}
	return Math.max(1, Math.floor(value));
}

function toUint8View(data: BufferSource): Uint8Array {
	if (data instanceof ArrayBuffer) {
		return new Uint8Array(data);
	}
	return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

function tryUnmapBuffer(buffer: GPUBuffer | null): void {
	if (!buffer || (buffer.mapState ?? "unmapped") === "unmapped") {
		return;
	}
	try {
		buffer.unmap();
	} catch (error) {
		Logger.warn(`WebGPU buffer unmap failed: ${String(error)}`, {
			scope: "WebGPUResourceManager",
		});
	}
}

function createBufferUnmap(resources: WebGPUResourceManager, buffer: IRenderBuffer): () => void {
	return () => tryUnmapBuffer(resources.getBuffer(buffer));
}

function createDeferredBufferDispose(disposal: { current: () => void }): () => void {
	return () => disposal.current();
}

function createNativeBufferDispose(buffer: GPUBuffer): () => void {
	return () => {
		tryUnmapBuffer(buffer);
		buffer.destroy();
	};
}

function mapBufferUsage(usage: number): GPUBufferUsageFlags {
	let flags = 0;
	if (usage & BufferUsage.Vertex) flags |= GPUBufferUsage.VERTEX;
	if (usage & BufferUsage.Index) flags |= GPUBufferUsage.INDEX;
	if (usage & BufferUsage.Uniform) flags |= GPUBufferUsage.UNIFORM;
	if (usage & BufferUsage.Storage) flags |= GPUBufferUsage.STORAGE;
	if (usage & BufferUsage.CopySrc) flags |= GPUBufferUsage.COPY_SRC;
	if (usage & BufferUsage.CopyDst) flags |= GPUBufferUsage.COPY_DST;
	if (usage & BufferUsage.MapRead) flags |= GPUBufferUsage.MAP_READ;
	if (usage & BufferUsage.MapWrite) flags |= GPUBufferUsage.MAP_WRITE;
	if (usage & BufferUsage.Indirect) flags |= GPUBufferUsage.INDIRECT;
	return flags;
}

function mapTextureUsage(usage: number): GPUTextureUsageFlags {
	let flags = 0;
	if (usage & TextureUsage.CopySrc) flags |= GPUTextureUsage.COPY_SRC;
	if (usage & TextureUsage.CopyDst) flags |= GPUTextureUsage.COPY_DST;
	if (usage & TextureUsage.TextureBinding) {
		flags |= GPUTextureUsage.TEXTURE_BINDING;
	}
	if (usage & TextureUsage.StorageBinding) {
		flags |= GPUTextureUsage.STORAGE_BINDING;
	}
	if (usage & TextureUsage.RenderAttachment) {
		flags |= GPUTextureUsage.RENDER_ATTACHMENT;
	}
	if (usage & TextureUsage.ComputeStorage) {
		flags |= GPUTextureUsage.STORAGE_BINDING;
	}
	return flags;
}

function resolveWebGPUTextureFormat(
	format: TextureDesc["format"],
	features: GPUSupportedFeatures | undefined,
): { format: TextureDesc["format"]; reason?: string } {
	if (!textureFormatRequiresFeature(format, features)) {
		return { format };
	}
	const fallbackFormat = getTextureFormatFallback(format);
	return {
		format: fallbackFormat,
		reason: `WebGPU texture format "${format}" requires an unavailable device feature; using "${fallbackFormat}" instead.`,
	};
}
