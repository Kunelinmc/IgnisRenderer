import type {
	WebGPUFrameGraphNodeKind,
} from "./types";
import type { WebGPUFrameNodeExecutor } from "./WebGPUFrameNodeExecutorRegistry";
import type { FrameContext } from "../../../pipeline/types";
import type { WarmupPlan } from "../../../pipeline/WarmupPlanner";
import type { WarmupOptions } from "../../IRenderBackend";

export interface WebGPUFrameNodeRuntime {
	readonly id: string;
	readonly executors: Readonly<Partial<Record<WebGPUFrameGraphNodeKind, WebGPUFrameNodeExecutor>>>;
	warmup?(context: FrameContext, plan: WarmupPlan, options: WarmupOptions): Promise<void>;
	beginFrame?(context: FrameContext): void;
	invalidateFrameResources?(): void;
	onShaderRuntimeChanged?(): void;
	destroy(): void;
}

class CallbackNodeRuntime implements WebGPUFrameNodeRuntime {
	public constructor(
		public readonly id: string,
		public readonly executors: Readonly<
			Partial<Record<WebGPUFrameGraphNodeKind, WebGPUFrameNodeExecutor>>
		>,
		private readonly _lifecycle: {
			warmup?: (context: FrameContext, plan: WarmupPlan, options: WarmupOptions) => Promise<void>;
		beginFrame?: (context: FrameContext) => void;
			invalidateFrameResources?: () => void;
			onShaderRuntimeChanged?: () => void;
			destroy?: () => void;
		} = {},
	) {}

	public async warmup(
		context: FrameContext,
		plan: WarmupPlan,
		options: WarmupOptions,
	): Promise<void> {
		await this._lifecycle.warmup?.(context, plan, options);
	}

	public beginFrame(context: FrameContext): void {
		this._lifecycle.beginFrame?.(context);
	}

	public invalidateFrameResources(): void {
		this._lifecycle.invalidateFrameResources?.();
	}

	public onShaderRuntimeChanged(): void {
		this._lifecycle.onShaderRuntimeChanged?.();
	}

	public destroy(): void {
		this._lifecycle.destroy?.();
	}
}

export class WebGPUSceneNodeRuntime extends CallbackNodeRuntime {}
export class WebGPUShadowNodeRuntime extends CallbackNodeRuntime {}
export class WebGPUDeferredNodeRuntime extends CallbackNodeRuntime {}
export class WebGPUTransparencyNodeRuntime extends CallbackNodeRuntime {}
export class WebGPUReflectionNodeRuntime extends CallbackNodeRuntime {}
export class WebGPUVisibilityNodeRuntime extends CallbackNodeRuntime {}
export class WebGPUPostProcessNodeRuntime extends CallbackNodeRuntime {}
export class WebGPUPresentationNodeRuntime extends CallbackNodeRuntime {}
