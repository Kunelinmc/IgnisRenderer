import type { IBindingGroup, IComputePipeline, ISampler } from "../types";
import { AddressMode, FilterMode } from "../types";
import type { IWebGPUComputeFacade } from "./ComputeFacade";
import type { WebGPUPostProcessServices } from "./WebGPUPostProcessContracts";
import { WebGPUHiZBuilder } from "./WebGPUHiZBuilder";
import { PostProcessCopyHelper } from "./postprocess/PostProcessCopyHelper";

interface CachedBindGroup {
	group: IBindingGroup;
	resources: readonly unknown[];
}

/** @internal Owns shared WebGPU post-process services, not pass dispatch. */
export class WebGPUPostProcessRuntime implements WebGPUPostProcessServices {
	private readonly _compute: IWebGPUComputeFacade;
	private readonly _warn: (key: string, message: string) => void;
	private _sampler: ISampler | null = null;
	private _hiZBuilder: WebGPUHiZBuilder | null;
	private readonly _ownsHiZBuilder: boolean;
	private _copyHelper: PostProcessCopyHelper | null = null;
	private readonly _bindGroupCache = new Map<string, CachedBindGroup>();
	private readonly _frameBindGroupLayout: GPUBindGroupLayout | null;

	constructor(
		computeFacade: IWebGPUComputeFacade,
		warn: (key: string, message: string) => void,
		frameBindGroupLayout?: GPUBindGroupLayout,
		hiZBuilder?: WebGPUHiZBuilder
	) {
		this._compute = computeFacade;
		this._warn = warn;
		this._frameBindGroupLayout = frameBindGroupLayout ?? null;
		this._hiZBuilder = hiZBuilder ?? null;
		this._ownsHiZBuilder = !hiZBuilder;
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

	/** @internal Returns the frame-graph-owned shared Hi-Z builder. */
	public getHiZBuilder(): WebGPUHiZBuilder {
		if (!this._hiZBuilder) {
			this._hiZBuilder = new WebGPUHiZBuilder(this._compute);
		}
		return this._hiZBuilder;
	}

	/**
	 * Returns the shared copy helper used by WebGPU post-process passes.
	 *
	 * @returns Lazily allocated helper owned by this runtime.
	 * @sideEffects Allocates the helper object on first use.
	 */
	public getCopyHelper(): PostProcessCopyHelper {
		if (!this._copyHelper) {
			this._copyHelper = new PostProcessCopyHelper(this);
		}
		return this._copyHelper;
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

	/**
	 * Invalidate all cached bind groups. Call when frame targets are
	 * destroyed/rebuilt (e.g. on resize) so stale texture references are
	 * not reused.
	 */
	public invalidateBindings(): void {
		this._destroyCachedBindGroups();
	}

	public onShaderRuntimeChanged(): void {
		this._destroyCachedBindGroups();
		if (this._ownsHiZBuilder) this._hiZBuilder?.invalidateShaderResources();
		this._copyHelper?.destroy();
		this._copyHelper = null;
	}

	public destroy(): void {
		if (this._ownsHiZBuilder) {
			this._hiZBuilder?.destroy();
			this._hiZBuilder = null;
		}
		this._copyHelper?.destroy();
		this._copyHelper = null;
		this._destroyCachedBindGroups();
		this.destroyManagedResource(this._sampler, "post-process sampler");
		this._sampler = null;
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

	public destroyManagedResource(
		resource: unknown,
		description = "post-process resource"
	): void {
		const destroyFn = (resource as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn !== "function") {
			return;
		}
		try {
			destroyFn.call(resource);
		} catch (error) {
			const detail = error instanceof Error ? error.message : String(error);
			throw new Error(`Failed to destroy ${description}: ${detail}`);
		}
	}

	public destroyBindingGroup(group: IBindingGroup | null): void {
		const destroyFn = (group as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn !== "function") {
			return;
		}
		try {
			destroyFn.call(group);
		} catch (error) {
			const detail = error instanceof Error ? error.message : String(error);
			throw new Error(
				`Failed to destroy post-process binding group: ${detail}`
			);
		}
	}

	private _destroyCachedBindGroups(): void {
		for (const cached of this._bindGroupCache.values()) {
			this.destroyBindingGroup(cached.group);
		}
		this._bindGroupCache.clear();
	}
}
