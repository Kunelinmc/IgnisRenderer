import type { ShaderCompileError } from "../../shaders/runtime";
import { toShaderCompileError } from "../../pipeline/WarmupPlanner";
import type { IWebGPUComputeFacade } from "./computeFacade";
import { PostProcessSharedContext } from "./postprocess/PostProcessSharedContext";
import { ScreenPostProcessDelegate } from "./postprocess/ScreenPostProcessDelegate";
import { SpatialPostProcessDelegate } from "./postprocess/SpatialPostProcessDelegate";
import { TemporalPostProcessDelegate } from "./postprocess/TemporalPostProcessDelegate";
import type {
	WebGPUPostProcessExecuteRequest,
	WebGPUPostProcessExecuteResult,
	WebGPUPostProcessPassDelegate,
	WebGPUPostProcessPassId,
} from "./postprocess/postprocessTypes";

export type {
	WebGPUPostProcessExecuteRequest,
	WebGPUPostProcessExecuteResult,
	WebGPUPostProcessPassId,
} from "./postprocess/postprocessTypes";

export class WebGPUPostProcessRuntime {
	private _compute: IWebGPUComputeFacade;
	private _shared: PostProcessSharedContext;
	private _delegates: readonly WebGPUPostProcessPassDelegate[];
	private _passDelegateById = new Map<
		WebGPUPostProcessPassId,
		WebGPUPostProcessPassDelegate
	>();

	constructor(
		computeFacade: IWebGPUComputeFacade,
		warn: (key: string, message: string) => void,
		frameBindGroupLayout?: GPUBindGroupLayout
	) {
		this._compute = computeFacade;
		this._shared = new PostProcessSharedContext(
			computeFacade,
			warn,
			frameBindGroupLayout
		);
		this._delegates = [
			new SpatialPostProcessDelegate(this._shared),
			new TemporalPostProcessDelegate(this._shared),
			new ScreenPostProcessDelegate(this._shared),
		];
		for (const delegate of this._delegates) {
			for (const passId of delegate.passIds) {
				this._passDelegateById.set(passId, delegate);
			}
		}
	}

	/**
	 * Invalidate all cached bind groups. Call when frame targets are
	 * destroyed/rebuilt (e.g. on resize) so stale texture references are
	 * not reused.
	 */
	public invalidateBindings(): void {
		this._shared.invalidateBindings();
		for (const delegate of this._delegates) {
			delegate.invalidateBindings();
		}
	}

	public onShaderRuntimeChanged(): void {
		this._shared.onShaderRuntimeChanged();
		for (const delegate of this._delegates) {
			delegate.onShaderRuntimeChanged();
		}
	}

	public async warmupHints(hints: readonly string[]): Promise<{
		compiled: number;
		failed: number;
		errors: ShaderCompileError[];
	}> {
		let compiled = 0;
		let failed = 0;
		const errors: ShaderCompileError[] = [];
		const seen = new Set<string>();
		for (const hint of hints) {
			if (seen.has(hint)) {
				continue;
			}
			seen.add(hint);
			try {
				const warmed = await this._warmupHint(hint);
				if (warmed) {
					compiled++;
				}
			} catch (error) {
				failed++;
				errors.push(
					toShaderCompileError(error, "webgpu", `WebGPUPostWarmup:${hint}`)
				);
			}
		}
		return {
			compiled,
			failed,
			errors,
		};
	}

	public async executePass(
		request: WebGPUPostProcessExecuteRequest
	): Promise<WebGPUPostProcessExecuteResult> {
		const delegate = this._passDelegateById.get(request.passId);
		if (!delegate) {
			return { ran: false };
		}
		const result = await delegate.execute(request);
		return result ?? { ran: false };
	}

	private async _warmupHint(hint: string): Promise<boolean> {
		for (const delegate of this._delegates) {
			const warmed = await delegate.warmupHint(hint);
			if (warmed) {
				return true;
			}
		}
		return false;
	}
}
