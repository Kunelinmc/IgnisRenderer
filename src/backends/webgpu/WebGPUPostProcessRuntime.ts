import type { IWebGPUComputeFacade } from "./ComputeFacade";
import { WebGPUHiZBuilder } from "./WebGPUHiZBuilder";
import { PostProcessSharedContext } from "./postprocess/PostProcessSharedContext";

/** @internal Owns shared WebGPU post-process services, not pass dispatch. */
export class WebGPUPostProcessRuntime {
	private _shared: PostProcessSharedContext;

	constructor(
		computeFacade: IWebGPUComputeFacade,
		warn: (key: string, message: string) => void,
		frameBindGroupLayout?: GPUBindGroupLayout,
		hiZBuilder?: WebGPUHiZBuilder
	) {
		this._shared = new PostProcessSharedContext(
			computeFacade,
			warn,
			frameBindGroupLayout,
			hiZBuilder
		);
	}

	public get sharedContext(): PostProcessSharedContext {
		return this._shared;
	}

	/**
	 * Invalidate all cached bind groups. Call when frame targets are
	 * destroyed/rebuilt (e.g. on resize) so stale texture references are
	 * not reused.
	 */
	public invalidateBindings(): void {
		this._shared.invalidateBindings();
	}

	public onShaderRuntimeChanged(): void {
		this._shared.onShaderRuntimeChanged();
	}

	public destroy(): void {
		this._shared.destroy();
	}
}
