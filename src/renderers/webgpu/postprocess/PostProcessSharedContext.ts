import type { IBindingGroup, IComputePipeline, ISampler } from "../../types";
import { AddressMode, FilterMode } from "../../types";
import type { IWebGPUComputeFacade } from "../ComputeFacade";
import { destroyResource } from "../computeUtils";

interface CachedBindGroup {
	group: IBindingGroup;
	resources: readonly unknown[];
}

/**
 * Shared post-process context for common resources and bind-group caching.
 */
export class PostProcessSharedContext {
	private _compute: IWebGPUComputeFacade;
	private _warn: (key: string, message: string) => void;
	private _sampler: ISampler | null = null;
	private _bindGroupCache = new Map<string, CachedBindGroup>();
	private _frameBindGroupLayout: GPUBindGroupLayout | null;

	constructor(
		compute: IWebGPUComputeFacade,
		warn: (key: string, message: string) => void,
		frameBindGroupLayout?: GPUBindGroupLayout
	) {
		this._compute = compute;
		this._warn = warn;
		this._frameBindGroupLayout = frameBindGroupLayout || null;
	}

	public get compute(): IWebGPUComputeFacade {
		return this._compute;
	}

	public get frameBindGroupLayout(): GPUBindGroupLayout | null {
		return this._frameBindGroupLayout;
	}

	public get sampler(): ISampler | null {
		return this._sampler;
	}

	public warn(key: string, message: string): void {
		this._warn(key, message);
	}

	public async ensureCommonResources(): Promise<void> {
		if (!this._sampler) {
			this._sampler = this._compute.createSampler({
				label: "WebGPUPostSampler",
				magFilter: FilterMode.Linear,
				minFilter: FilterMode.Linear,
				mipmapFilter: FilterMode.Linear,
				addressModeU: AddressMode.ClampToEdge,
				addressModeV: AddressMode.ClampToEdge,
			});
		}
	}

	public getCachedBindGroup(
		key: string,
		pipeline: IComputePipeline,
		entries: Array<{ binding: number; resource: unknown }>,
		label: string
	): IBindingGroup {
		const resources = entries.map((entry) => entry.resource);
		const cached = this._bindGroupCache.get(key);
		if (cached && cached.resources.length === resources.length) {
			let match = true;
			for (let i = 0; i < resources.length; i++) {
				if (cached.resources[i] !== resources[i]) {
					match = false;
					break;
				}
			}
			if (match) {
				return cached.group;
			}
		}
		if (cached) {
			this.destroyBindingGroup(cached.group);
		}
		const group = this._compute.createBindingGroup({
			pipeline,
			layoutIndex: 0,
			entries: entries as Array<{ binding: number; resource: any }>,
			label,
		});
		this._bindGroupCache.set(key, { group, resources });
		return group;
	}

	public invalidateBindings(): void {
		this._destroyCachedBindGroups();
	}

	public onShaderRuntimeChanged(): void {
		this._destroyCachedBindGroups();
	}

	public invalidateBindingsByPrefix(prefix: string): void {
		for (const key of Array.from(this._bindGroupCache.keys())) {
			if (!key.startsWith(prefix)) {
				continue;
			}
			const cached = this._bindGroupCache.get(key);
			if (cached) {
				this.destroyBindingGroup(cached.group);
			}
			this._bindGroupCache.delete(key);
		}
	}

	public destroyBindingGroup(group: IBindingGroup | null): void {
		destroyResource(group);
	}

	private _destroyCachedBindGroups(): void {
		for (const cached of this._bindGroupCache.values()) {
			this.destroyBindingGroup(cached.group);
		}
		this._bindGroupCache.clear();
	}
}
