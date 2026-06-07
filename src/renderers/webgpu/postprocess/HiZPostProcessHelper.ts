import type { ICommandEncoder } from "../../ICommandEncoder";
import type {
	IComputePipeline,
	IRenderTexture,
	IShaderModule,
} from "../../types";
import { ceilDiv } from "../../../maths/Misc";
import { ShaderSource } from "../../../shaders/ShaderSource";
import {
	WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WORKGROUP_SIZE,
} from "../constants";
import type { PostProcessSharedContext } from "./PostProcessSharedContext";

/**
 * Explicit textures used to build a WebGPU Hi-Z pyramid.
 */
export interface WebGPUHiZBuildRequest {
	/**
	 * Active command encoder receiving the Hi-Z compute passes.
	 */
	readonly encoder: ICommandEncoder;
	/**
	 * Source texture containing current frame depth data.
	 */
	readonly depth: IRenderTexture;
	/**
	 * Destination texture that stores the generated Hi-Z mip chain.
	 */
	readonly hiZ: IRenderTexture;
	/**
	 * Forces command encoding even when the same encoder already built this
	 * depth/Hi-Z pair.
	 */
	readonly force?: boolean;
}

/**
 * Builds a Hi-Z pyramid used by depth-aware WebGPU post-process passes.
 */
export class WebGPUHiZPostProcessHelper {
	private _shared: PostProcessSharedContext;
	private _module: IShaderModule | null = null;
	private _initPipeline: IComputePipeline | null = null;
	private _reducePipeline: IComputePipeline | null = null;
	private _viewCache = new WeakMap<object, GPUTextureView[]>();
	private _lastBuild:
		| {
			encoder: ICommandEncoder;
			depth: IRenderTexture;
			hiZ: IRenderTexture;
			mips: GPUTextureView[];
		}
		| null = null;

	constructor(shared: PostProcessSharedContext) {
		this._shared = shared;
	}

	/**
	 * Ensures the Hi-Z shader and compute pipelines are available.
	 *
	 * @returns Nothing.
	 * @sideEffects Allocates WebGPU shader and pipeline resources lazily.
	 */
	public async ensureResources(): Promise<void> {
		await this._shared.ensureCommonResources();
		if (!this._module) {
			const shader = await ShaderSource.load("webgpu.postprocess.hiz.composite");
			this._module = await this._shared.compute.createShaderModule({
				label: "WebGPUHiZShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._initPipeline) {
			this._initPipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUHiZInitPipeline",
				compute: { module: this._module, entryPoint: "csInit" },
			});
		}
		if (!this._reducePipeline) {
			this._reducePipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUHiZReducePipeline",
				compute: { module: this._module, entryPoint: "csReduce" },
			});
		}
	}

	/**
	 * Builds the Hi-Z pyramid from an explicit depth source.
	 *
	 * @param request Textures and encoder used for the pyramid build.
	 * @returns Mip views for the generated pyramid, or an empty list on failure.
	 * @sideEffects Encodes Hi-Z init and reduction compute passes.
	 */
	public async build(request: WebGPUHiZBuildRequest): Promise<GPUTextureView[]> {
		await this.ensureResources();
		const hiZMips = this.getMipViews(request.hiZ);
		if (!this._initPipeline || !this._reducePipeline || hiZMips.length === 0) {
			return [];
		}
		if (
			!request.force &&
			this._lastBuild?.encoder === request.encoder &&
			this._lastBuild.depth === request.depth &&
			this._lastBuild.hiZ === request.hiZ
		) {
			return this._lastBuild.mips;
		}

		let binding = this._shared.getCachedBindGroup(
			"hiz-init",
			this._initPipeline,
			[
				{ binding: 0, resource: request.depth },
				{ binding: 1, resource: hiZMips[0] },
			],
			"WebGPUHiZInitBinding"
		);
		request.encoder.beginComputePass({ label: "WebGPUHiZInit" });
		request.encoder.setComputePipeline(this._initPipeline);
		request.encoder.setBindingGroup(0, binding);
		request.encoder.dispatchWorkgroups(
			ceilDiv(request.hiZ.width, WORKGROUP_SIZE),
			ceilDiv(request.hiZ.height, WORKGROUP_SIZE),
			1
		);
		request.encoder.endComputePass();

		let srcW = request.hiZ.width;
		let srcH = request.hiZ.height;
		for (let mip = 1; mip < hiZMips.length; mip++) {
			const dstW = Math.max(1, srcW >> 1);
			const dstH = Math.max(1, srcH >> 1);
			binding = this._shared.getCachedBindGroup(
				`hiz-reduce-${mip}`,
				this._reducePipeline,
				[
					{ binding: 0, resource: hiZMips[mip - 1] },
					{ binding: 1, resource: hiZMips[mip] },
				],
				`WebGPUHiZReduceBinding_${mip}`
			);
			request.encoder.beginComputePass({ label: `WebGPUHiZReduce_${mip}` });
			request.encoder.setComputePipeline(this._reducePipeline);
			request.encoder.setBindingGroup(0, binding);
			request.encoder.dispatchWorkgroups(
				ceilDiv(dstW, WORKGROUP_SIZE),
				ceilDiv(dstH, WORKGROUP_SIZE),
				1
			);
			request.encoder.endComputePass();
			srcW = dstW;
			srcH = dstH;
		}
		this._lastBuild = {
			encoder: request.encoder,
			depth: request.depth,
			hiZ: request.hiZ,
			mips: hiZMips,
		};
		return hiZMips;
	}

	/**
	 * Returns cached texture views for every Hi-Z mip level.
	 *
	 * @param texture Hi-Z texture.
	 * @returns Mip-level texture views.
	 * @sideEffects Allocates texture views on first use per texture.
	 */
	public getMipViews(texture: IRenderTexture): GPUTextureView[] {
		const cached = this._viewCache.get(texture as object);
		if (cached) {
			return cached;
		}
		const mipCount =
			Math.floor(Math.log2(Math.max(texture.width, texture.height))) + 1;
		const views: GPUTextureView[] = [];
		for (let i = 0; i < mipCount; i++) {
			views.push(
				this._shared.compute.createTextureView(texture, {
					baseMipLevel: i,
					mipLevelCount: 1,
				})
			);
		}
		this._viewCache.set(texture as object, views);
		return views;
	}

	/**
	 * Destroys helper-owned shader and pipeline resources.
	 *
	 * @returns Nothing.
	 * @sideEffects Releases managed WebGPU resources.
	 */
	public destroy(): void {
		this._shared.invalidateBindingsByPrefix("hiz-");
		this._shared.destroyManagedResource(this._initPipeline, "Hi-Z init pipeline");
		this._shared.destroyManagedResource(
			this._reducePipeline,
			"Hi-Z reduce pipeline"
		);
		this._shared.destroyManagedResource(this._module, "Hi-Z shader module");
		this._module = null;
		this._initPipeline = null;
		this._reducePipeline = null;
		this._viewCache = new WeakMap<object, GPUTextureView[]>();
		this._lastBuild = null;
	}
}
