import type {
	BufferDesc,
	IRenderBuffer,
	IRenderTexture,
	TextureDataLayout,
	TextureDesc,
} from "../types";
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
	_gpuResource: GPUTexture;
	_gpuTexture: GPUTexture;
	_gpuView: GPUTextureView;
}

export interface WebGPUResourceManagerHost {
	readonly device: GPUDevice | null;
	readonly queue: GPUQueue | null;
	assertDeviceOperational(operation: string): void;
	mapBufferUsage(usage: number): GPUBufferUsageFlags;
	mapTextureUsage(usage: number): GPUTextureUsageFlags;
	resolveSupportedMSAASampleCount(
		requested: number,
		probeFormats?: readonly GPUTextureFormat[]
	): number;
	resolvePositiveInteger(value: number, fallback: number): number;
	toUint8View(data: BufferSource): Uint8Array;
	tryUnmapBuffer(buffer: GPUBuffer | null): void;
	createManagedDestroy(
		target: object,
		options: {
			label: string;
			dispose: () => void;
		}
	): () => void;
	runValidationScope<T>(label: string, operation: () => T): T;
}

export class WebGPUResourceManager {
	constructor(private _host: WebGPUResourceManagerHost) {}

	public createBuffer(desc: BufferDesc): IRenderBuffer {
		this._host.assertDeviceOperational("create buffers");
		const device = this._requireDevice("create buffers");
		const hasInitialData = !!desc.initialData;
		const mappedAtCreation = hasInitialData || !!desc.mappedAtCreation;
		const gpuBuffer = device.createBuffer({
			size: desc.size,
			usage: this._host.mapBufferUsage(desc.usage),
			mappedAtCreation,
			label: desc.label,
		});
		if (hasInitialData) {
			const source = desc.initialData as BufferSource;
			const mappedRange = gpuBuffer.getMappedRange();
			const target = new Uint8Array(mappedRange);
			const srcView = this._host.toUint8View(source);
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
			this._host.tryUnmapBuffer(gpuBuffer);
		};
		buffer.destroy = this._host.createManagedDestroy(buffer, {
			label: desc.label ?? "WebGPUBuffer",
			dispose: () => {
				this._host.tryUnmapBuffer(gpuBuffer);
				gpuBuffer.destroy();
			},
		});
		return buffer;
	}

	public createTexture(desc: TextureDesc): IRenderTexture {
		this._host.assertDeviceOperational("create textures");
		const device = this._requireDevice("create textures");
		const dimension = (desc.dimension ?? "2d") as GPUTextureDimension;
		const resolvedWidth = this._host.resolvePositiveInteger(desc.width, 1);
		const resolvedHeight = this._host.resolvePositiveInteger(desc.height, 1);
		const depthOrArrayLayers = this._host.resolvePositiveInteger(
			desc.depthOrArrayLayers ?? 1,
			1
		);
		const requestedSampleCount = Math.max(1, Math.floor(desc.sampleCount ?? 1));
		const sampleCount =
			dimension === "2d"
				? this._host.resolveSupportedMSAASampleCount(requestedSampleCount, [
						desc.format as GPUTextureFormat,
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
			format: desc.format as GPUTextureFormat,
			usage: this._host.mapTextureUsage(desc.usage),
			mipLevelCount: Math.max(1, desc.mipLevelCount ?? 1),
			viewFormats: desc.viewFormats as GPUTextureFormat[] | undefined,
			label: desc.label,
		};
		const gpuTexture = device.createTexture(baseDescriptor);
		const webgpuTexture = createWebGPUTexture(gpuTexture);
		const texture: InternalTexture = {
			width: resolvedWidth,
			height: dimension === "1d" ? 1 : resolvedHeight,
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

	public writeBuffer(
		buffer: IRenderBuffer,
		data: BufferSource,
		offset: number = 0
	): void {
		this._host.assertDeviceOperational("write buffers");
		this._requireQueue("write buffers").writeBuffer(getWebGPUBuffer(buffer), offset, data);
	}

	public writeTexture(
		texture: IRenderTexture,
		data: BufferSource,
		desc: TextureDataLayout,
		size: { width: number; height: number; depthOrArrayLayers?: number }
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
			size
		);
	}

	public createTextureView(
		texture: IRenderTexture,
		desc?: GPUTextureViewDescriptor
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
