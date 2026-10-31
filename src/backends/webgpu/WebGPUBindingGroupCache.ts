/// <reference types="@webgpu/types" />
import {
	WEBGPU_BINDING_GROUP_CACHE_LIMIT,
	WEBGPU_BINDING_GROUP_CACHE_TTL_FRAMES,
	WEBGPU_PIPELINE_LAYOUT_CACHE_LIMIT,
} from "./constants";
import type { WebGPUResourceManager } from "./WebGPUResourceManager";
import type {
	BindingGroupDesc,
	BindingResource,
	BufferBindingRange,
	IBindingGroup,
	IComputePipeline,
	IRenderBuffer,
	IRenderPipeline,
	IRenderTexture,
	ISampler,
} from "../types";
import {
	createHash64,
	hash64Combine,
	type WebGPUObjectIdentity,
} from "./WebGPUObjectIdentity";

/** @internal WebGPU binding inputs; applications use engine resource facades. */
export type WebGPUBindingResource =
	| BindingResource
	| GPUBindingResource
	| GPUTexture
	| GPUBuffer;

/** @internal WebGPU cache descriptor; applications use `BindingGroupDesc`. */
export interface WebGPUBindingGroupDesc extends Omit<BindingGroupDesc, "entries" | "pipeline"> {
	pipeline?: IRenderPipeline | IComputePipeline;
	entries: { binding: number; resource: WebGPUBindingResource }[];
}

interface BindingResourceSignature {
	binding: number;
	kind: number;
	primaryId: number;
	secondaryId: number;
	offset: number;
	size: number;
}

interface CachedBindingGroupEntry {
	hashKey: bigint;
	layoutId: number;
	signatures: BindingResourceSignature[];
	native: GPUBindGroup;
	label?: string;
	lastUsedFrame: number;
	lastTouchedTick: number;
	refCount: number;
}

type BindingResourceInput = WebGPUBindingResource;

export interface WebGPUBindingGroupCacheHost {
	readonly device: GPUDevice | null;
	readonly resourceManager: WebGPUResourceManager;
	readonly frameSerial: number;
	readonly objectIdentity: WebGPUObjectIdentity;
	assertDeviceOperational(operation: string): void;
	createManagedDestroy(
		target: object,
		options: {
			label: string;
			dispose: () => void;
		}
	): () => void;
	runValidationScope<T>(label: string, operation: () => T): T;
}

export interface WebGPUBindingGroupCacheDebugStats {
	entryCount: number;
	bucketCount: number;
	bucketSizes: number[];
	layoutEntries: number;
	refCounts: number[];
}

export class WebGPUBindingGroupCache {
	private readonly _resources: WebGPUResourceManager;
	private _bindingGroupCache = new Map<bigint, CachedBindingGroupEntry[]>();
	private _bindingGroupLru = new Map<CachedBindingGroupEntry, true>();
	private _bindingGroupCacheEntryCount = 0;
	private _pipelineBindGroupLayoutCache = new Map<string, GPUBindGroupLayout>();
	private _bindingGroupTouchTick = 0;
	private _hashOverrideForTesting:
		| ((
				layoutId: number,
				signatures: readonly BindingResourceSignature[]
		  ) => bigint)
		| null = null;

	public constructor(private readonly _host: WebGPUBindingGroupCacheHost) {
		this._resources = _host.resourceManager;
	}

	public createBindingGroup(desc: WebGPUBindingGroupDesc): IBindingGroup {
		this._host.assertDeviceOperational("create binding groups");
		const device = this._requireDevice("create binding groups");
		const pipeline = desc.pipeline
			? desc.pipeline.kind === "compute-pipeline"
				? this._resources.getComputePipeline(desc.pipeline)
				: this._resources.getRenderPipeline(desc.pipeline)
			: null;
		const layout =
			(desc.layout as GPUBindGroupLayout | undefined) ??
			this._getPipelineBindGroupLayout(
				pipeline as GPURenderPipeline | GPUComputePipeline | null,
				desc.layoutIndex ?? 0,
			);

		if (!layout) {
			throw new Error(
				`WebGPU binding group ${desc.label ?? "(unnamed)"} requires an explicit layout or pipeline`,
			);
		}
		if (desc.cache === false) {
			const gpuBindGroup = this._host.runValidationScope(
				`createBindGroup:${desc.label ?? "unnamed"}`,
				() => device.createBindGroup({
					layout,
					entries: desc.entries.map((entry) => ({
						binding: entry.binding,
						resource: this.mapBindingResource(entry.resource),
					})),
					label: desc.label,
				}),
			);
			return this._resources.registerBindingGroup(gpuBindGroup, desc.label);
		}

		const layoutId = this._host.objectIdentity.getObjectId(layout);
		const signatures = desc.entries.map((entry) =>
			this._getBindingResourceSignature(entry.binding, entry.resource),
		);
		const cacheKey = this._getBindingGroupCacheKey(layoutId, signatures);
		const cached = this._findBindingGroupCacheEntry(cacheKey, layoutId, signatures);
		if (cached) {
			cached.lastUsedFrame = this._host.frameSerial;
			cached.lastTouchedTick = ++this._bindingGroupTouchTick;
			this._touchBindingGroupEntry(cached);
			return this._acquireBindingGroupHandle(cached);
		}

		const gpuBindGroup = this._host.runValidationScope(
			`createBindGroup:${desc.label ?? "unnamed"}`,
			() =>
				device.createBindGroup({
					layout,
					entries: desc.entries.map((entry) => ({
						binding: entry.binding,
						resource: this.mapBindingResource(entry.resource),
					})),
					label: desc.label,
				}),
		);

		const entry: CachedBindingGroupEntry = {
			hashKey: cacheKey,
			layoutId,
			signatures: signatures.map((signature) => ({ ...signature })),
			native: gpuBindGroup,
			label: desc.label,
			lastUsedFrame: this._host.frameSerial,
			lastTouchedTick: ++this._bindingGroupTouchTick,
			refCount: 0,
		};
		const bucket = this._bindingGroupCache.get(cacheKey);
		if (bucket) {
			bucket.push(entry);
		} else {
			this._bindingGroupCache.set(cacheKey, [entry]);
		}
		this._bindingGroupCacheEntryCount++;
		this._bindingGroupLru.set(entry, true);
		this.trim();
		return this._acquireBindingGroupHandle(entry);
	}

	public mapBindingResource(resource: BindingResourceInput): GPUBindingResource {
		this._assertBindingObject(resource);
		const resources = this._resources;
		const kind = (resource as { kind?: string }).kind;
		if (kind === "texture") return resources.getTexture(resource as IRenderTexture).view;
		if (kind === "buffer") return { buffer: resources.getBuffer(resource as IRenderBuffer) };
		if (kind === "sampler") return resources.getSampler(resource as ISampler);
		if (kind) throw new Error(`Unsupported WebGPU binding resource kind: ${kind}.`);
		const range = resource as BufferBindingRange | GPUBufferBinding;
		if (range.buffer) {
			if (!(range.buffer as { kind?: string }).kind) {
				return resource as GPUBufferBinding;
			}
			const buffer = resources.getBuffer(range.buffer as IRenderBuffer);
			return { buffer, offset: range.offset ?? 0, size: range.size };
		}
		if (typeof (resource as GPUTexture).createView === "function") {
			return (resource as GPUTexture).createView();
		}
		if (typeof (resource as GPUBuffer).destroy === "function") {
			return { buffer: resource as GPUBuffer };
		}
		return resource as GPUBindingResource;
	}

	private _assertBindingObject(resource: unknown): asserts resource is object {
		if (!resource || typeof resource !== "object" || "_gpuResource" in resource) {
			throw new Error("Unsupported WebGPU binding resource; expected an owned facade or native input.");
		}
	}

	public evictStale(): void {
		let inspected = 0;
		for (const entry of this._bindingGroupLru.keys()) {
			if (inspected++ >= 128) break;
			const frameAge = Math.max(0, this._host.frameSerial - entry.lastUsedFrame);
			if (frameAge <= WEBGPU_BINDING_GROUP_CACHE_TTL_FRAMES) break;
			this._removeBindingGroupCacheEntry(entry.hashKey, entry);
		}
	}

	public clear(): void {
		this._bindingGroupCache.clear();
		this._bindingGroupLru.clear();
		this._bindingGroupCacheEntryCount = 0;
		this._pipelineBindGroupLayoutCache.clear();
		this._bindingGroupTouchTick = 0;
	}

	public clearBindingGroups(): void {
		this._bindingGroupCache.clear();
		this._bindingGroupLru.clear();
		this._bindingGroupCacheEntryCount = 0;
		this._bindingGroupTouchTick = 0;
	}

	public setHashOverrideForTesting(
		override:
			| ((
					layoutId: number,
					signatures: readonly BindingResourceSignature[]
			  ) => bigint)
			| null
	): void {
		this._hashOverrideForTesting = override;
	}

	public getDebugStats(): WebGPUBindingGroupCacheDebugStats {
		const bucketSizes = Array.from(this._bindingGroupCache.values()).map(
			(bucket) => bucket.length
		);
		const refCounts: number[] = [];
		for (const bucket of this._bindingGroupCache.values()) {
			for (const entry of bucket) {
				refCounts.push(entry.refCount);
			}
		}
		return {
			entryCount: this._bindingGroupCacheEntryCount,
			bucketCount: this._bindingGroupCache.size,
			bucketSizes,
			layoutEntries: this._pipelineBindGroupLayoutCache.size,
			refCounts,
		};
	}

	private _requireDevice(operation: string): GPUDevice {
		const device = this._host.device;
		if (!device) {
			throw new Error(`WebGPU backend is not initialized; cannot ${operation}.`);
		}
		return device;
	}

	private _getBindingGroupCacheKey(
		layoutId: number,
		signatures: BindingResourceSignature[],
	): bigint {
		if (this._hashOverrideForTesting) {
			return this._hashOverrideForTesting(layoutId, signatures);
		}
		let hash = createHash64();
		hash = hash64Combine(hash, layoutId);
		hash = hash64Combine(hash, signatures.length);
		for (const signature of signatures) {
			hash = hash64Combine(hash, signature.binding);
			hash = hash64Combine(hash, signature.kind);
			hash = hash64Combine(hash, signature.primaryId);
			hash = hash64Combine(hash, signature.secondaryId);
			hash = hash64Combine(hash, signature.offset);
			hash = hash64Combine(hash, signature.size);
		}
		return hash;
	}

	private _findBindingGroupCacheEntry(
		hashKey: bigint,
		layoutId: number,
		signatures: BindingResourceSignature[],
	): CachedBindingGroupEntry | null {
		const bucket = this._bindingGroupCache.get(hashKey);
		if (!bucket) {
			return null;
		}
		for (const candidate of bucket) {
			if (
				candidate.layoutId === layoutId &&
				this._isBindingSignaturesMatch(candidate.signatures, signatures)
			) {
				return candidate;
			}
		}
		return null;
	}

	private _acquireBindingGroupHandle(entry: CachedBindingGroupEntry): IBindingGroup {
		entry.refCount++;
		const native = entry.native;
		const hashKey = entry.hashKey;
		// Each acquisition owns a reference; native cache reuse must not share destruction.
		let managedDispose = () => {};
		const group = this._resources.registerBindingGroup(
			native,
			entry.label,
			() => managedDispose(),
		);
		// Manager-driven device reset must also unregister managed lifetime diagnostics.
		managedDispose = this._host.createManagedDestroy(group, {
			label: entry.label ?? "WebGPUBindGroup",
			dispose: () => this._releaseBindingGroupCacheEntry(hashKey, native),
		});
		return group;
	}

	private _releaseBindingGroupCacheEntry(hashKey: bigint, native: GPUBindGroup): void {
		const bucket = this._bindingGroupCache.get(hashKey);
		if (!bucket) {
			return;
		}
		for (let i = 0; i < bucket.length; i++) {
			const candidate = bucket[i];
			if (candidate.native === native) {
				candidate.refCount = Math.max(0, candidate.refCount - 1);
				if (candidate.refCount <= 0) {
					this._removeBindingGroupCacheEntry(hashKey, candidate);
				}
				return;
			}
		}
	}

	private _isBindingSignaturesMatch(
		left: BindingResourceSignature[],
		right: BindingResourceSignature[],
	): boolean {
		if (left.length !== right.length) {
			return false;
		}
		for (let i = 0; i < left.length; i++) {
			const a = left[i];
			const b = right[i];
			if (
				a.binding !== b.binding ||
				a.kind !== b.kind ||
				a.primaryId !== b.primaryId ||
				a.secondaryId !== b.secondaryId ||
				a.offset !== b.offset ||
				a.size !== b.size
			) {
				return false;
			}
		}
		return true;
	}

	private _getBindingResourceSignature(
		binding: number,
		resource: BindingResourceInput,
	): BindingResourceSignature {
		this._assertBindingObject(resource);
		const resources = this._resources;
		const kind = (resource as { kind?: string }).kind;
		if (kind === "texture") {
			const texture = resources.getTexture(resource as IRenderTexture);
			return {
				binding,
				kind: 1,
				primaryId: this._host.objectIdentity.getObjectId(texture.texture),
				secondaryId: this._host.objectIdentity.getObjectId(texture.view),
				offset: 0,
				size: -1,
			};
		}

		if (kind === "buffer") {
			const buffer = resources.getBuffer(resource as IRenderBuffer);
			return {
				binding,
				kind: 2,
				primaryId: this._host.objectIdentity.getObjectId(buffer),
				secondaryId: 0,
				offset: 0,
				size: -1,
			};
		}

		if (kind === "sampler") {
			return {
				binding,
				kind: 5,
				primaryId: this._host.objectIdentity.getObjectId(resources.getSampler(resource as ISampler)),
				secondaryId: 0,
				offset: 0,
				size: -1,
			};
		}
		if (kind) throw new Error(`Unsupported WebGPU binding resource kind: ${kind}.`);
		const range = resource as BufferBindingRange | GPUBufferBinding;
		if (range.buffer) {
			const buffer = (range.buffer as { kind?: string }).kind
				? resources.getBuffer(range.buffer as IRenderBuffer)
				: range.buffer as GPUBuffer;
			return {
				binding,
				kind: 3,
				primaryId: this._host.objectIdentity.getObjectId(buffer),
				secondaryId: 0,
				offset: Math.max(0, range.offset ?? 0),
				size: Math.max(-1, range.size ?? -1),
			};
		}

		if (typeof (resource as GPUTexture).createView === "function") {
			return {
				binding,
				kind: 4,
				primaryId: this._host.objectIdentity.getObjectId(resource),
				secondaryId: 0,
				offset: 0,
				size: -1,
			};
		}
		return {
			binding,
			kind: typeof (resource as GPUBuffer).destroy === "function" ? 2 : 6,
			primaryId: this._host.objectIdentity.getObjectId(resource),
			secondaryId: 0,
			offset: 0,
			size: -1,
		};
	}

	private _getPipelineBindGroupLayout(
		pipeline: GPURenderPipeline | GPUComputePipeline | null,
		layoutIndex: number,
	): GPUBindGroupLayout | undefined {
		if (!pipeline) {
			return undefined;
		}
		const cacheKey = `${this._host.objectIdentity.getCacheToken(pipeline)}:${layoutIndex}`;
		const cached = this._getLruCacheEntry(this._pipelineBindGroupLayoutCache, cacheKey);
		if (cached) {
			return cached;
		}
		const layout = pipeline.getBindGroupLayout(layoutIndex);
		this._pipelineBindGroupLayoutCache.set(cacheKey, layout);
		this._trimCache(
			this._pipelineBindGroupLayoutCache,
			WEBGPU_PIPELINE_LAYOUT_CACHE_LIMIT
		);
		return layout;
	}

	private trim(): void {
		this.evictStale();
		while (this._bindingGroupCacheEntryCount > WEBGPU_BINDING_GROUP_CACHE_LIMIT) {
			const oldest = this._bindingGroupLru.keys().next().value as
				| CachedBindingGroupEntry
				| undefined;
			if (!oldest) break;
			this._removeBindingGroupCacheEntry(oldest.hashKey, oldest);
		}
	}

	private _touchBindingGroupEntry(entry: CachedBindingGroupEntry): void {
		this._bindingGroupLru.delete(entry);
		this._bindingGroupLru.set(entry, true);
	}

	private _removeBindingGroupCacheEntry(
		hashKey: bigint,
		target: CachedBindingGroupEntry,
	): boolean {
		const bucket = this._bindingGroupCache.get(hashKey);
		if (!bucket) {
			return false;
		}
		const index = bucket.indexOf(target);
		if (index < 0) {
			return false;
		}
		bucket.splice(index, 1);
		this._bindingGroupLru.delete(target);
		this._bindingGroupCacheEntryCount = Math.max(
			0,
			this._bindingGroupCacheEntryCount - 1
		);
		if (bucket.length <= 0) {
			this._bindingGroupCache.delete(hashKey);
		}
		return true;
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
}
