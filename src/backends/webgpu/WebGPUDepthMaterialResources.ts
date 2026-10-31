import { AlphaMode, type Material } from "../../materials/Material";
import { clamp } from "../../maths/Common";
import type { Texture } from "../../core/Texture";
import {
	BufferUsage, type IBindingGroup, type IRenderBuffer,
	type IRenderTexture, type ISampler,
} from "../types";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";
import type { WebGPUTextureRegistry } from "./WebGPUTextureRegistry";
import type { WebGPUResourceManager } from "./WebGPUResourceManager";

/** @internal Immutable alpha-only bindings owned by WebGPU depth preparation. */
export interface WebGPUResolvedDepthMaterialSnapshot {
	readonly binding: IBindingGroup;
	readonly revision: number;
}

interface DepthMaterialEntry {
	snapshot: WebGPUResolvedDepthMaterialSnapshot;
	buffer: IRenderBuffer;
	projection: Float32Array<ArrayBuffer>;
	map: Texture | null;
	texture: IRenderTexture;
	sampler: ISampler;
	textureHandle: GPUTexture;
	samplerHandle: GPUSampler;
	lastUsedFrame: number;
}

const MAX_DEPTH_MATERIAL_ENTRIES = 4096;
const DEPTH_MATERIAL_UNIFORM_BYTE_SIZE = 48;
const NEUTRAL_PROJECTION = new Float32Array([1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]);

/** @internal Owns alpha coverage resources; scene materials remain application-owned. */
export class WebGPUDepthMaterialResources {
	private readonly _resources: WebGPUResourceManager;
	private _entries = new Map<Material, DepthMaterialEntry>();
	private _neutral: DepthMaterialEntry | null = null;
	private _projections = new WeakMap<Material, Float32Array<ArrayBuffer>>();
	private _projectionScratch = new Float32Array(12);
	private _retired: DepthMaterialEntry[] = [];
	private _frame = 0;
	private _revision = 0;
	private _destroyed = false;

	constructor(
		private readonly _backend: WebGPUDeviceResourceHost,
		private readonly _layouts: WebGPUPipelineLayouts,
		private readonly _textures: WebGPUTextureRegistry,
	) {
		this._resources = _backend.resourceManager;
	}

	public beginFrame(): void {
		if (this._destroyed) return;
		this._frame++;
		for (const entry of this._retired) this._destroyEntry(entry);
		this._retired.length = 0;
		this._trim();
	}

	/**
	 * @internal Resolves only alpha coverage and the base map for built-in depth draws.
	 * Replaced resources stay alive until the next frame begins.
	 */
	public async resolve(material: Material): Promise<WebGPUResolvedDepthMaterialSnapshot> {
		if (this._destroyed) throw new Error("WebGPU depth material resources are destroyed.");
		const masked = material.alphaMode === AlphaMode.Mask;
		const map = masked ? material.map ?? null : null;
		const data = masked ? this._resolveProjection(material, map) : NEUTRAL_PROJECTION;
		// Always revisit the registry: upload/native replacements need no material revision.
		const texture = await this._textures.getTextureForSlotAsync(map, 0);
		if (this._destroyed) throw new Error("WebGPU depth material resources are destroyed.");
		const sampler = this._textures.getSamplerForTexture(map);
		const textureHandle = this._resources.getTexture(texture).texture;
		const samplerHandle = this._resources.getSampler(sampler);
		const cached = masked ? this._entries.get(material) : this._neutral;
		if (cached && equalProjection(cached.projection, data) && cached.texture === texture &&
			cached.map === map &&
			cached.sampler === sampler && cached.textureHandle === textureHandle &&
			cached.samplerHandle === samplerHandle) {
			cached.lastUsedFrame = this._frame;
			return cached.snapshot;
		}
		const buffer = this._backend.createBuffer({
			size: DEPTH_MATERIAL_UNIFORM_BYTE_SIZE,
			usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			label: "WebGPUDepthAlphaUniform",
		});
		let binding: IBindingGroup;
		try {
			this._backend.writeBuffer(buffer, data);
			binding = this._backend.createBindingGroup({
				label: "WebGPUDepthAlphaBinding",
				layout: this._layouts.depthMaterialBindGroupLayout,
				entries: [
					{ binding: 0, resource: buffer },
					{ binding: 1, resource: texture },
					{ binding: 2, resource: sampler },
				],
				cache: false,
			});
		} catch (error) {
			buffer.destroy();
			throw error;
		}
		const entry: DepthMaterialEntry = {
			snapshot: Object.freeze({ binding, revision: ++this._revision }),
			buffer, projection: data, map, texture, sampler, textureHandle, samplerHandle,
			lastUsedFrame: this._frame,
		};
		if (cached) this._retired.push(cached);
		if (masked) {
			this._entries.delete(material);
			this._entries.set(material, entry);
		} else {
			this._neutral = entry;
		}
		this._trim();
		return entry.snapshot;
	}

	/**
	 * @internal Depth draw preparation rejects stale alpha inputs after asynchronous uploads.
	 * Checks existing resources only; it does not upload textures or create GPU handles.
	 */
	public isCurrent(material: Material, snapshot: WebGPUResolvedDepthMaterialSnapshot): boolean {
		if (this._destroyed) return false;
		const masked = material.alphaMode === AlphaMode.Mask;
		const entry = masked ? this._entries.get(material) : this._neutral;
		if (!entry || entry.snapshot !== snapshot) return false;
		const map = masked ? material.map ?? null : null;
		if (entry.map !== map) return false;
		if (masked) {
			this._fillProjectionScratch(material, map);
			if (!equalProjection(entry.projection, this._projectionScratch)) return false;
		}
		return entry.textureHandle === this._resources.tryGetTexture(entry.texture)?.texture &&
			entry.samplerHandle === this._resources.tryGetSampler(entry.sampler) &&
			this._textures.isTextureBindingCurrent(map, entry.texture, entry.sampler);
	}

	public destroy(): void {
		if (this._destroyed) return;
		this._destroyed = true;
		for (const entry of this._entries.values()) this._destroyEntry(entry);
		for (const entry of this._retired) this._destroyEntry(entry);
		if (this._neutral) this._destroyEntry(this._neutral);
		this._entries.clear();
		this._retired.length = 0;
		this._neutral = null;
		this._projections = new WeakMap();
	}

	private _resolveProjection(material: Material, map: Texture | null): Float32Array<ArrayBuffer> {
		this._fillProjectionScratch(material, map);
		const scratch = this._projectionScratch;
		const cached = this._projections.get(material);
		if (cached && equalProjection(cached, scratch)) return cached;
		// Capture changed inputs before awaiting upload; future requests reuse the scratch.
		const projection = new Float32Array(scratch);
		this._projections.set(material, projection);
		return projection;
	}

	private _fillProjectionScratch(material: Material, map: Texture | null): void {
		const scratch = this._projectionScratch;
		const uvSet = (material as Material & { albedoMapUV?: number }).albedoMapUV ?? 0;
		scratch[0] = clamp(material.opacity ?? 1, 0, 1);
		scratch[1] = clamp(material.alphaCutoff ?? 0.5, 0, 1);
		scratch[4] = map?.offset.x ?? 0;
		scratch[5] = map?.offset.y ?? 0;
		scratch[6] = map?.repeat.x ?? 1;
		scratch[7] = map?.repeat.y ?? 1;
		scratch[8] = map?.rotation ?? 0;
		scratch[9] = Number.isFinite(uvSet) ? clamp(Math.floor(uvSet), 0, 3) : 0;
	}

	private _trim(): void {
		if (this._entries.size <= MAX_DEPTH_MATERIAL_ENTRIES) return;
		for (const [material, entry] of this._entries) {
			if (entry.lastUsedFrame === this._frame) continue;
			this._entries.delete(material);
			this._destroyEntry(entry);
			if (this._entries.size <= MAX_DEPTH_MATERIAL_ENTRIES) break;
		}
	}

	private _destroyEntry(entry: DepthMaterialEntry): void {
		entry.snapshot.binding.destroy();
		entry.buffer.destroy();
	}
}

function equalProjection(left: Float32Array, right: Float32Array): boolean {
	if (left === right) return true;
	for (let i = 0; i < 12; i++) {
		if (!Object.is(left[i], right[i])) return false;
	}
	return true;
}
