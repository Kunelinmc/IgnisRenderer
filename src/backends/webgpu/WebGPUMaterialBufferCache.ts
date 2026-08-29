import {
	BufferUsage,
	type IRenderBuffer,
} from "../types";

import {
	WEBGPU_FLAT_MATERIAL_UNIFORM_BYTE_SIZE,
	WEBGPU_MATERIAL_COMMON_UNIFORM_BYTE_SIZE,
	WEBGPU_PBR_MATERIAL_UNIFORM_BYTE_SIZE,
	WEBGPU_PHONG_MATERIAL_UNIFORM_BYTE_SIZE,
} from "./constants";
import {
	packFlatMaterialUniformData,
	packMaterialCommonUniformData,
	packPBRMaterialUniformData,
	packPhongMaterialUniformData,
} from "./packing";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type {
	WebGPUMaterialUniformData,
	WebGPUShadingFamily,
} from "./types";

/** @internal Shared immutable GPU buffers for one material snapshot. */
export interface WebGPUMaterialBufferResources {
	readonly shadingFamily: WebGPUShadingFamily;
	readonly commonBuffer: IRenderBuffer;
	readonly lightingBuffer: IRenderBuffer | null;
}

/** @internal Retained material-buffer reference owned by one binding cache entry. */
export interface WebGPUMaterialBufferLease {
	readonly resources: WebGPUMaterialBufferResources;
	release(): void;
}

interface MaterialBufferEntry extends WebGPUMaterialBufferResources {
	refCount: number;
}

/** @internal Shares immutable material uniform buffers across draw paths. */
export class WebGPUMaterialBufferCache {
	private _entries = new Map<WebGPUMaterialUniformData, MaterialBufferEntry>();
	private _destroyed = false;

	public constructor(private readonly _backend: WebGPUDeviceResourceHost) {}

	public acquire(data: WebGPUMaterialUniformData): WebGPUMaterialBufferLease {
		if (this._destroyed) {
			throw new Error("WebGPU material buffer cache is destroyed.");
		}
		let entry = this._entries.get(data);
		if (!entry) {
			entry = this._createEntry(data);
			this._entries.set(data, entry);
		}
		entry.refCount++;
		let released = false;
		return {
			resources: entry,
			release: () => {
				if (released) return;
				released = true;
				this._release(data, entry as MaterialBufferEntry);
			},
		};
	}

	public destroy(): void {
		if (this._destroyed) return;
		this._destroyed = true;
		for (const entry of this._entries.values()) {
			entry.commonBuffer.destroy();
			entry.lightingBuffer?.destroy();
		}
		this._entries.clear();
	}

	public getDebugStats(): { readonly entries: number; readonly references: number } {
		let references = 0;
		for (const entry of this._entries.values()) references += entry.refCount;
		return { entries: this._entries.size, references };
	}

	private _createEntry(data: WebGPUMaterialUniformData): MaterialBufferEntry {
		const commonBuffer = this._createBuffer(
			WEBGPU_MATERIAL_COMMON_UNIFORM_BYTE_SIZE,
			`WebGPUMaterialCommon_${data.shadingFamily}`,
			packMaterialCommonUniformData(data.common),
		);
		let lightingBuffer: IRenderBuffer | null = null;
		try {
			switch (data.shadingFamily) {
				case "pbr":
					lightingBuffer = this._createBuffer(
						WEBGPU_PBR_MATERIAL_UNIFORM_BYTE_SIZE,
						"WebGPUPBRMaterial",
						packPBRMaterialUniformData(data.lighting),
					);
					break;
				case "phong":
					lightingBuffer = this._createBuffer(
						WEBGPU_PHONG_MATERIAL_UNIFORM_BYTE_SIZE,
						"WebGPUPhongMaterial",
						packPhongMaterialUniformData(data.lighting),
					);
					break;
				case "flat":
					lightingBuffer = this._createBuffer(
						WEBGPU_FLAT_MATERIAL_UNIFORM_BYTE_SIZE,
						"WebGPUFlatMaterial",
						packFlatMaterialUniformData(data.lighting),
					);
					break;
				case "unlit":
					break;
			}
		} catch (error) {
			commonBuffer.destroy();
			lightingBuffer?.destroy();
			throw error;
		}
		return {
			shadingFamily: data.shadingFamily,
			commonBuffer,
			lightingBuffer,
			refCount: 0,
		};
	}

	private _createBuffer(
		size: number,
		label: string,
		data: Float32Array<ArrayBuffer>,
	): IRenderBuffer {
		const buffer = this._backend.createBuffer({
			size,
			usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			label,
		});
		this._backend.writeBuffer(buffer, data);
		return buffer;
	}

	private _release(
		data: WebGPUMaterialUniformData,
		entry: MaterialBufferEntry,
	): void {
		if (this._destroyed) return;
		entry.refCount--;
		if (entry.refCount > 0) return;
		if (this._entries.get(data) !== entry) return;
		this._entries.delete(data);
		entry.commonBuffer.destroy();
		entry.lightingBuffer?.destroy();
	}
}
