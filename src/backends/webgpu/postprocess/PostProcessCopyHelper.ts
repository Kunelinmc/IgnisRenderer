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
import type { WebGPUPostProcessServices } from "../WebGPUPostProcessContracts";

/**
 * Request for an ordered WebGPU post-process texture copy.
 */
export interface PostProcessCopyTextureRequest {
	/**
	 * Active frame command encoder that receives the copy compute pass.
	 */
	readonly encoder: ICommandEncoder;
	/**
	 * Source texture to sample from.
	 */
	readonly source: IRenderTexture;
	/**
	 * Destination storage texture to write.
	 */
	readonly destination: IRenderTexture;
	/**
	 * Optional cache key for the helper-owned bind group.
	 */
	readonly cacheKey?: string;
	/**
	 * Optional compute pass label.
	 */
	readonly label?: string;
}

/**
 * Shared compute-copy helper for WebGPU post-process passes.
 *
 * @internal WebGPU post-process helper. It records work on the caller-provided
 * command encoder so pass ordering is preserved.
 */
export class PostProcessCopyHelper {
	private _services: WebGPUPostProcessServices;
	private _module: IShaderModule | null = null;
	private _pipeline: IComputePipeline | null = null;

	/**
	 * Creates a helper bound to one shared WebGPU post-process context.
	 *
	 * @param services Runtime services that own helper resources.
	 */
	public constructor(services: WebGPUPostProcessServices) {
		this._services = services;
	}

	/**
	 * Ensures the post-process copy shader and pipeline are available.
	 *
	 * @returns Nothing.
	 * @sideEffects Allocates WebGPU shader and pipeline resources lazily.
	 */
	public async ensureResources(): Promise<void> {
		if (!this._module) {
			const shader = await ShaderSource.load("webgpu.postprocess.copy.composite");
			this._module = await this._services.compute.createShaderModule({
				label: "PostProcessCopyShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._pipeline) {
			this._pipeline = await this._services.compute.createComputePipeline({
				label: "PostProcessCopyPipeline",
				compute: { module: this._module, entryPoint: "csMain" },
			});
		}
	}

	/**
	 * Copies one post-process texture into another using the active encoder.
	 *
	 * @param request Source, destination, and encoder for the copy.
	 * @returns Nothing.
	 * @sideEffects Encodes one compute pass unless source and destination match.
	 */
	public async copyTexture(
		request: PostProcessCopyTextureRequest
	): Promise<void> {
		if (request.source === request.destination) {
			return;
		}
		await this.ensureResources();
		if (!this._pipeline) {
			return;
		}
		const label = request.label ?? "WebGPUPost_Copy";
		const cacheKey = this._resolveCacheKey(request.cacheKey);
		const binding = this._services.getCachedBindGroup(
			cacheKey,
			this._pipeline,
			[
				{ binding: 0, resource: request.source },
				{ binding: 1, resource: request.destination },
			],
			`${label}Binding`
		);
		request.encoder.beginComputePass({ label });
		request.encoder.setComputePipeline(this._pipeline);
		request.encoder.setBindingGroup(0, binding);
		request.encoder.dispatchWorkgroups(
			ceilDiv(request.destination.width, WORKGROUP_SIZE),
			ceilDiv(request.destination.height, WORKGROUP_SIZE),
			1
		);
		request.encoder.endComputePass();
	}

	private _resolveCacheKey(cacheKey: string | undefined): string {
		if (!cacheKey) {
			return "copy-texture";
		}
		if (cacheKey.startsWith("copy-")) {
			return cacheKey;
		}
		return `copy-${cacheKey}`;
	}

	/**
	 * Destroys helper-owned shader and pipeline resources.
	 *
	 * @returns Nothing.
	 * @sideEffects Releases managed WebGPU resources and cached copy bindings.
	 */
	public destroy(): void {
		this._services.invalidateBindingsByPrefix("copy-");
		this._services.destroyManagedResource(
			this._pipeline,
			"post-process copy pipeline"
		);
		this._services.destroyManagedResource(
			this._module,
			"post-process copy shader module"
		);
		this._pipeline = null;
		this._module = null;
	}
}
