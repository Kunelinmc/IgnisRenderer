import type { ICommandEncoder } from "../../ICommandEncoder";
import type {
	IComputePipeline,
	IRenderTexture,
	IShaderModule,
} from "../../types";
import { ceilDiv } from "../../../maths/Misc";
import { loadPostProcessShaderPartComposite } from "../../../shaders/webgpu/shaderSource";
import type { WebGPUFrameTargets } from "../WebGPUPostProcessContracts";
import type { PostProcessSharedContext } from "./PostProcessSharedContext";

const WORKGROUP_SIZE = 8;

/**
 * Builds a Hi-Z pyramid used by depth-aware WebGPU post-process passes.
 */
export class WebGPUHiZPostProcessHelper {
	private _shared: PostProcessSharedContext;
	private _module: IShaderModule | null = null;
	private _initPipeline: IComputePipeline | null = null;
	private _reducePipeline: IComputePipeline | null = null;
	private _viewCache = new WeakMap<object, GPUTextureView[]>();

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
			const shader = await loadPostProcessShaderPartComposite("hiz");
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
	 * Builds the Hi-Z pyramid from the frame motion-depth target.
	 *
	 * @param encoder Active command encoder.
	 * @param targets WebGPU frame targets.
	 * @returns Mip views for the generated pyramid, or an empty list on failure.
	 * @sideEffects Encodes Hi-Z init and reduction compute passes.
	 */
	public async build(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets
	): Promise<GPUTextureView[]> {
		await this.ensureResources();
		const hiZMips = this.getMipViews(targets.hiZ);
		if (!this._initPipeline || !this._reducePipeline || hiZMips.length === 0) {
			return [];
		}

		let binding = this._shared.getCachedBindGroup(
			"hiz-init",
			this._initPipeline,
			[
				{ binding: 0, resource: targets.gMotionDepth },
				{ binding: 1, resource: hiZMips[0] },
			],
			"WebGPUSSR_HiZInitBinding"
		);
		encoder.beginComputePass({ label: "WebGPUSSR_HiZInit" });
		encoder.setComputePipeline(this._initPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(targets.hiZ.width, WORKGROUP_SIZE),
			ceilDiv(targets.hiZ.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();

		let srcW = targets.hiZ.width;
		let srcH = targets.hiZ.height;
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
				`WebGPUSSR_HiZReduceBinding_${mip}`
			);
			encoder.beginComputePass({ label: `WebGPUSSR_HiZReduce_${mip}` });
			encoder.setComputePipeline(this._reducePipeline);
			encoder.setBindingGroup(0, binding);
			encoder.dispatchWorkgroups(
				ceilDiv(dstW, WORKGROUP_SIZE),
				ceilDiv(dstH, WORKGROUP_SIZE),
				1
			);
			encoder.endComputePass();
			srcW = dstW;
			srcH = dstH;
		}
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
	}
}
