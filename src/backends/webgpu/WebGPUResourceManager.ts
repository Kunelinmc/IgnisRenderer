import {
	BufferUsage,
	type BufferDesc,
	type IRenderBuffer,
	type IRenderTexture,
	type TextureDataLayout,
	type TextureDesc,
	TextureUsage,
} from "../types";
import { getTextureFormatFallback, textureFormatRequiresFeature } from "../../core/TextureFormat";
import { Logger } from "../../foundation/Logger";
import {
	attachWebGPUTexture,
	createWebGPUTexture,
	getWebGPUBuffer,
	getWebGPUTexture,
} from "./WebGPUResourceAccess";

interface InternalRenderBuffer extends IRenderBuffer {
	_gpuResource: GPUBuffer;
}

interface InternalTexture extends IRenderTexture {
	requestedFormat: TextureDesc["format"];
	format: TextureDesc["format"];
	formatFallbackReason?: string;
	_gpuResource: GPUTexture;
	_gpuTexture: GPUTexture;
	_gpuView: GPUTextureView;
}

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

/**
 * Owns WebGPU-native resource creation and queue-backed resource uploads.
 *
 * @internal Owned by `WebGPUBackend`; applications must use `Renderer`.
 */
export class WebGPUResourceManager {
	constructor(private _host: WebGPUResourceManagerHost) {}

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

		const buffer = {
			size: desc.size,
			destroy: () => {},
			unmap: () => {},
			_gpuResource: gpuBuffer,
		} as InternalRenderBuffer;
		buffer.unmap = () => {
			tryUnmapBuffer(gpuBuffer);
		};
		buffer.destroy = this._host.createManagedDestroy(buffer, {
			label: desc.label ?? "WebGPUBuffer",
			dispose: () => {
				tryUnmapBuffer(gpuBuffer);
				gpuBuffer.destroy();
			},
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
		const webgpuTexture = createWebGPUTexture(gpuTexture);
		const texture: InternalTexture = {
			width: resolvedWidth,
			height: dimension === "1d" ? 1 : resolvedHeight,
			requestedFormat: desc.format,
			format: formatResolution.format,
			formatFallbackReason: formatResolution.reason,
			destroy: () => {},
			_gpuResource: gpuTexture,
			_gpuTexture: gpuTexture,
			_gpuView: webgpuTexture.view,
		};
		texture.destroy = this._host.createManagedDestroy(texture, {
			label: desc.label ?? "WebGPUTexture",
			dispose: () => gpuTexture.destroy(),
		});
		attachWebGPUTexture(texture, webgpuTexture);
		return texture;
	}

	public writeBuffer(buffer: IRenderBuffer, data: BufferSource, offset: number = 0): void {
		this._host.assertDeviceOperational("write buffers");
		this._requireQueue("write buffers").writeBuffer(getWebGPUBuffer(buffer), offset, data);
	}

	public writeTexture(
		texture: IRenderTexture,
		data: BufferSource,
		desc: TextureDataLayout,
		size: { width: number; height: number; depthOrArrayLayers?: number },
	): void {
		this._host.assertDeviceOperational("write textures");
		const gpuTexture = getWebGPUTexture(texture).texture;
		this._requireQueue("write textures").writeTexture(
			{
				texture: gpuTexture,
				mipLevel: Math.max(0, desc.mipLevel ?? 0),
			},
			data,
			{
				offset: desc.offset ?? 0,
				bytesPerRow: desc.bytesPerRow,
				rowsPerImage: desc.rowsPerImage,
			},
			size,
		);
	}

	public createTextureView(
		texture: IRenderTexture,
		desc?: GPUTextureViewDescriptor,
	): GPUTextureView {
		this._host.assertDeviceOperational("create texture views");
		const gpuTexture = getWebGPUTexture(texture);
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
