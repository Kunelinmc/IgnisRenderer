/// <reference types="@webgpu/types" />
import {
	WEBGPU_BINDING_GROUP_CACHE_LIMIT,
	WEBGPU_BINDING_GROUP_CACHE_TTL_FRAMES,
	WEBGPU_PIPELINE_LAYOUT_CACHE_LIMIT,
} from "./constants";
import {
	getWebGPUPipeline,
	tryGetWebGPUBuffer,
	tryGetWebGPUTexture,
} from "./WebGPUResourceAccess";
import type {
	BindingGroupDesc,
	IBindingGroup,
	IComputePipeline,
	IRenderPipeline,
} from "../types";
import {
	createHash64,
	hash64Combine,
	type WebGPUObjectIdentity,
} from "./WebGPUObjectIdentity";

interface InternalBindingGroup extends IBindingGroup {
	destroy(): void;
	_gpuResource: GPUBindGroup;
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
	group: InternalBindingGroup;
	lastUsedFrame: number;
	lastTouchedTick: number;
	refCount: number;
}

type BindingResourceInput = BindingGroupDesc["entries"][number]["resource"];

export interface WebGPUBindingGroupCacheHost {
	readonly device: GPUDevice | null;
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
	private _bindingGroupCache = new Map<bigint, CachedBindingGroupEntry[]>();
	private _bindingGroupCacheEntryCount = 0;
	private _pipelineBindGroupLayoutCache = new Map<string, GPUBindGroupLayout>();
	private _bindingGroupTouchTick = 0;
	private _hashOverrideForTesting:
		| ((
				layoutId: number,
				signatures: readonly BindingResourceSignature[]
		  ) => bigint)
		| null = null;

	public constructor(private readonly _host: WebGPUBindingGroupCacheHost) {}

	public createBindingGroup(desc: BindingGroupDesc): IBindingGroup {
		this._host.assertDeviceOperational("create binding groups");
		const device = this._requireDevice("create binding groups");
		const pipeline = desc.pipeline
			? getWebGPUPipeline(desc.pipeline as IRenderPipeline | IComputePipeline)
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

		const layoutId = this._host.objectIdentity.getObjectId(layout);
		const signatures = desc.entries.map((entry) =>
			this._getBindingResourceSignature(entry.binding, entry.resource),
		);
		const cacheKey = this._getBindingGroupCacheKey(layoutId, signatures);
		const cached = this._findBindingGroupCacheEntry(cacheKey, layoutId, signatures);
		if (cached) {
			cached.lastUsedFrame = this._host.frameSerial;
			cached.lastTouchedTick = ++this._bindingGroupTouchTick;
			return cached.group;
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

		const group = {
			label: desc.label,
			destroy: () => {},
			_gpuResource: gpuBindGroup,
		} as InternalBindingGroup;
		group.destroy = this._host.createManagedDestroy(group, {
			label: desc.label ?? "WebGPUBindGroup",
			dispose: () => {
				this._releaseBindingGroupCacheEntry(cacheKey, group as InternalBindingGroup);
			},
		});
		const entry: CachedBindingGroupEntry = {
			hashKey: cacheKey,
			layoutId,
			signatures: signatures.map((signature) => ({ ...signature })),
			group,
			lastUsedFrame: this._host.frameSerial,
			lastTouchedTick: ++this._bindingGroupTouchTick,
			refCount: 1,
		};
		const bucket = this._bindingGroupCache.get(cacheKey);
		if (bucket) {
			bucket.push(entry);
		} else {
			this._bindingGroupCache.set(cacheKey, [entry]);
		}
		this._bindingGroupCacheEntryCount++;
		this.trim();
		return group;
	}

	public mapBindingResource(resource: BindingResourceInput): GPUBindingResource {
		const texture = tryGetWebGPUTexture(resource);
		if (texture) {
			return texture.view;
		}

		if (resource && typeof (resource as GPUTexture).createView === "function") {
			return (resource as GPUTexture).createView();
		}

		const buffer = tryGetWebGPUBuffer(resource);
		if (buffer) {
			return { buffer };
		}

		if (resource && typeof resource === "object") {
			const bufferBinding = resource as GPUBufferBinding;
			if (
				bufferBinding.buffer &&
				typeof (bufferBinding.buffer as GPUBuffer).destroy === "function"
			) {
				return bufferBinding;
			}
		}

		if (resource && typeof resource === "object") {
			const resourceWithHandle = resource as { _gpuResource?: unknown };
			const handle = resourceWithHandle._gpuResource;
			if (handle) {
				if (typeof (handle as GPUTexture).createView === "function") {
					return (handle as GPUTexture).createView();
				}
				if (typeof (handle as GPUBuffer).destroy === "function") {
					return { buffer: handle as GPUBuffer };
				}
				return handle as GPUBindingResource;
			}
			return resource as GPUBindingResource;
		}

		throw new Error(
			"Unsupported WebGPU binding resource: expected texture, buffer, sampler, or GPU-backed resource object.",
		);
	}

	public evictStale(): void {
		if (this._bindingGroupCache.size <= 0) {
			return;
		}
		for (const [hashKey, bucket] of this._bindingGroupCache.entries()) {
			for (let i = bucket.length - 1; i >= 0; i--) {
				const entry = bucket[i];
				const frameAge = Math.max(0, this._host.frameSerial - entry.lastUsedFrame);
				const ttlBudget = WEBGPU_BINDING_GROUP_CACHE_TTL_FRAMES;
				if (frameAge > ttlBudget) {
					this._removeBindingGroupCacheEntry(hashKey, entry);
				}
			}
		}
	}

	public clear(): void {
		this._bindingGroupCache.clear();
		this._bindingGroupCacheEntryCount = 0;
		this._pipelineBindGroupLayoutCache.clear();
		this._bindingGroupTouchTick = 0;
	}

	public clearBindingGroups(): void {
		this._bindingGroupCache.clear();
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

	private _releaseBindingGroupCacheEntry(hashKey: bigint, group: InternalBindingGroup): void {
		const bucket = this._bindingGroupCache.get(hashKey);
		if (!bucket) {
			return;
		}
		for (let i = 0; i < bucket.length; i++) {
			const candidate = bucket[i];
			if (candidate.group === group) {
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
		const texture = tryGetWebGPUTexture(resource);
		if (texture) {
			return {
				binding,
				kind: 1,
				primaryId: this._host.objectIdentity.getObjectId(texture.texture),
				secondaryId: this._host.objectIdentity.getObjectId(texture.view),
				offset: 0,
				size: -1,
			};
		}

		const buffer = tryGetWebGPUBuffer(resource);
		if (buffer) {
			return {
				binding,
				kind: 2,
				primaryId: this._host.objectIdentity.getObjectId(buffer),
				secondaryId: 0,
				offset: 0,
				size: -1,
			};
		}

		if (resource && typeof resource === "object") {
			const asBufferBinding = resource as GPUBufferBinding;
			if (
				asBufferBinding.buffer &&
				typeof (asBufferBinding.buffer as GPUBuffer).destroy === "function"
			) {
				return {
					binding,
					kind: 3,
					primaryId: this._host.objectIdentity.getObjectId(asBufferBinding.buffer),
					secondaryId: 0,
					offset: Math.max(0, asBufferBinding.offset ?? 0),
					size: Math.max(-1, asBufferBinding.size ?? -1),
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

			const resourceWithHandle = resource as {
				_gpuResource?: unknown;
			};
			if (
				resourceWithHandle._gpuResource &&
				typeof resourceWithHandle._gpuResource === "object"
			) {
				return {
					binding,
					kind: 5,
					primaryId: this._host.objectIdentity.getObjectId(
						resourceWithHandle._gpuResource as object
					),
					secondaryId: 0,
					offset: 0,
					size: -1,
				};
			}

			return {
				binding,
				kind: 6,
				primaryId: this._host.objectIdentity.getObjectId(resource),
				secondaryId: 0,
				offset: 0,
				size: -1,
			};
		}

		throw new Error(
			`Unsupported binding resource for binding ${binding}: expected object-backed WebGPU resource.`,
		);
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
		const entryCount = this._bindingGroupCacheEntryCount;
		if (entryCount <= WEBGPU_BINDING_GROUP_CACHE_LIMIT) {
			return;
		}

		const candidates: Array<{
			hashKey: bigint;
			entry: CachedBindingGroupEntry;
		}> = [];
		for (const [hashKey, bucket] of this._bindingGroupCache.entries()) {
			for (const entry of bucket) {
				candidates.push({
					hashKey,
					entry,
				});
			}
		}
		candidates.sort((a, b) => {
			const frameDelta = a.entry.lastUsedFrame - b.entry.lastUsedFrame;
			if (frameDelta !== 0) {
				return frameDelta;
			}
			const touchDelta = a.entry.lastTouchedTick - b.entry.lastTouchedTick;
			if (touchDelta !== 0) {
				return touchDelta;
			}
			return a.entry.refCount - b.entry.refCount;
		});

		let remainingToEvict = entryCount - WEBGPU_BINDING_GROUP_CACHE_LIMIT;
		for (const candidate of candidates) {
			if (remainingToEvict <= 0) {
				break;
			}
			if (this._removeBindingGroupCacheEntry(candidate.hashKey, candidate.entry)) {
				remainingToEvict--;
			}
		}
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
