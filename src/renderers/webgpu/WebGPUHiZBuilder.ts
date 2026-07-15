import type { ICommandEncoder } from "../ICommandEncoder";
import type { IBindingGroup, IComputePipeline, IRenderTexture, IShaderModule } from "../types";
import { ceilDiv } from "../../maths/Misc";
import { ShaderSource } from "../../shaders/ShaderSource";
import type { IWebGPUComputeFacade } from "./ComputeFacade";
import { WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WORKGROUP_SIZE } from "./constants";

export interface WebGPUHiZBuildRequest {
	readonly encoder: ICommandEncoder;
	readonly depth: IRenderTexture;
	readonly hiZ: IRenderTexture;
	readonly force?: boolean;
}

export interface WebGPUHiZBuildResult {
	readonly texture: IRenderTexture;
	readonly mipViews: readonly GPUTextureView[];
	readonly mipLevelCount: number;
	readonly maxMip: number;
}

interface CachedBinding {
	group: IBindingGroup;
	resources: readonly unknown[];
}

/** @internal Shared WebGPU Hi-Z pyramid builder. */
export class WebGPUHiZBuilder {
	private _module: IShaderModule | null = null;
	private _initPipeline: IComputePipeline | null = null;
	private _reducePipeline: IComputePipeline | null = null;
	private _viewCache = new WeakMap<object, GPUTextureView[]>();
	private _bindings = new Map<string, CachedBinding>();
	private _lastBuild:
		| (WebGPUHiZBuildResult & {
				encoder: ICommandEncoder;
				depth: IRenderTexture;
		  })
		| null = null;

	constructor(private readonly _compute: IWebGPUComputeFacade) {}

	public async ensureResources(): Promise<void> {
		if (!this._module) {
			const shader = await ShaderSource.load("webgpu.postprocess.hiz.composite");
			this._module = await this._compute.createShaderModule({
				label: "WebGPUHiZShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._initPipeline) {
			this._initPipeline = await this._compute.createComputePipeline({
				label: "WebGPUHiZInitPipeline",
				compute: { module: this._module, entryPoint: "csInit" },
			});
		}
		if (!this._reducePipeline) {
			this._reducePipeline = await this._compute.createComputePipeline({
				label: "WebGPUHiZReducePipeline",
				compute: { module: this._module, entryPoint: "csReduce" },
			});
		}
	}

	public async build(request: WebGPUHiZBuildRequest): Promise<WebGPUHiZBuildResult> {
		await this.ensureResources();
		const mipViews = this.getMipViews(request.hiZ);
		if (!this._initPipeline || !this._reducePipeline || mipViews.length === 0) {
			throw new Error("WebGPU Hi-Z resources are incomplete.");
		}
		if (
			!request.force &&
			this._lastBuild?.encoder === request.encoder &&
			this._lastBuild.depth === request.depth &&
			this._lastBuild.texture === request.hiZ
		) {
			return this._lastBuild;
		}
		let binding = this._getBinding(
			"hiz-init",
			this._initPipeline,
			[
				{ binding: 0, resource: request.depth },
				{ binding: 1, resource: mipViews[0] },
			],
			"WebGPUHiZInitBinding",
		);
		request.encoder.beginComputePass({ label: "WebGPUHiZInit" });
		request.encoder.setComputePipeline(this._initPipeline);
		request.encoder.setBindingGroup(0, binding);
		request.encoder.dispatchWorkgroups(
			ceilDiv(request.hiZ.width, WORKGROUP_SIZE),
			ceilDiv(request.hiZ.height, WORKGROUP_SIZE),
			1,
		);
		request.encoder.endComputePass();
		let width = request.hiZ.width;
		let height = request.hiZ.height;
		for (let mip = 1; mip < mipViews.length; mip++) {
			const nextWidth = Math.max(1, width >> 1);
			const nextHeight = Math.max(1, height >> 1);
			binding = this._getBinding(
				`hiz-reduce-${mip}`,
				this._reducePipeline,
				[
					{ binding: 0, resource: mipViews[mip - 1] },
					{ binding: 1, resource: mipViews[mip] },
				],
				`WebGPUHiZReduceBinding_${mip}`,
			);
			request.encoder.beginComputePass({ label: `WebGPUHiZReduce_${mip}` });
			request.encoder.setComputePipeline(this._reducePipeline);
			request.encoder.setBindingGroup(0, binding);
			request.encoder.dispatchWorkgroups(
				ceilDiv(nextWidth, WORKGROUP_SIZE),
				ceilDiv(nextHeight, WORKGROUP_SIZE),
				1,
			);
			request.encoder.endComputePass();
			width = nextWidth;
			height = nextHeight;
		}
		const result = {
			texture: request.hiZ,
			mipViews,
			mipLevelCount: mipViews.length,
			maxMip: mipViews.length - 1,
			encoder: request.encoder,
			depth: request.depth,
		};
		this._lastBuild = result;
		return result;
	}

	public getMipViews(texture: IRenderTexture): GPUTextureView[] {
		const cached = this._viewCache.get(texture as object);
		if (cached) return cached;
		const count = Math.floor(Math.log2(Math.max(texture.width, texture.height))) + 1;
		const views: GPUTextureView[] = [];
		for (let mip = 0; mip < count; mip++) {
			views.push(
				this._compute.createTextureView(texture, { baseMipLevel: mip, mipLevelCount: 1 }),
			);
		}
		this._viewCache.set(texture as object, views);
		return views;
	}

	public invalidateBindings(): void {
		for (const { group } of this._bindings.values()) this._destroy(group);
		this._bindings.clear();
		this._lastBuild = null;
	}

	public invalidateShaderResources(): void {
		this.invalidateBindings();
		this._destroy(this._initPipeline);
		this._destroy(this._reducePipeline);
		this._destroy(this._module);
		this._initPipeline = null;
		this._reducePipeline = null;
		this._module = null;
	}

	public destroy(): void {
		this.invalidateShaderResources();
		this._viewCache = new WeakMap<object, GPUTextureView[]>();
	}

	private _getBinding(
		key: string,
		pipeline: IComputePipeline,
		entries: Array<{ binding: number; resource: unknown }>,
		label: string,
	): IBindingGroup {
		const resources = entries.map((entry) => entry.resource);
		const cached = this._bindings.get(key);
		if (
			cached &&
			cached.resources.length === resources.length &&
			cached.resources.every((resource, index) => resource === resources[index])
		)
			return cached.group;
		if (cached) this._destroy(cached.group);
		const group = this._compute.createBindingGroup({
			pipeline,
			layoutIndex: 0,
			entries: entries as Array<{ binding: number; resource: any }>,
			label,
		});
		this._bindings.set(key, { group, resources });
		return group;
	}

	private _destroy(resource: unknown): void {
		try {
			(resource as { destroy?: () => void } | null)?.destroy?.();
		} catch {
			/* Device loss may already release resources. */
		}
	}
}
