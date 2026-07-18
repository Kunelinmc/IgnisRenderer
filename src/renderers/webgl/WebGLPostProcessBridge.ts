import type { FrameContext } from "../../pipeline/types";
import type {
	PostProcessPassExecutionContextRequest,
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassResult,
} from "../../postprocess";
import type { WebGLProgramCompiler } from "./WebGLProgramCompiler";
import {
	isWebGLPostProcessContextMetadata,
	type WebGLPostProcessContextMetadata,
} from "./WebGLPostProcessContracts";

export interface WebGLPostProcessBridgeCallbacks {
	getGL(): WebGL2RenderingContext;
	getProgramCompiler(): WebGLProgramCompiler;
	getFullscreenVao(): WebGLVertexArrayObject | null;
	getPostFramebuffer(): WebGLFramebuffer | null;
	getSceneColorTexture(): WebGLTexture | null;
	getSceneMotionTexture(): WebGLTexture | null;
	getSceneNormalTexture(): WebGLTexture | null;
	getSSAORawTexture(): WebGLTexture | null;
	getSSAOBlurTexture(): WebGLTexture | null;
	getWidth(): number;
	getHeight(): number;
	getSSAODownsample(): number;
	getActiveContext(): FrameContext | null;
	getSourceTexture(): WebGLTexture | null;
	resolveTargetTexture(sourceTexture: WebGLTexture): WebGLTexture | null;
	bindColorTarget(texture: WebGLTexture): void;
	drawFullscreen(
		width: number,
		height: number,
		context: FrameContext | null
	): void;
	publishColorTexture(texture: WebGLTexture): void;
	markTAAHistoryValid(): void;
	nextFrameJitter(): number;
	applyPipelineHistories(request: PostProcessPassRequest): void;
	warn(key: string, message: string): void;
}

/** @internal Packs implementation-declared WebGL post-process execution contexts. */
export class WebGLPostProcessBridge {
	private readonly _callbacks: WebGLPostProcessBridgeCallbacks;
	private _pendingColorTexture: WebGLTexture | null = null;
	private _pendingMarksTAAHistoryValid = false;
	private _transactionActive = false;

	/** @internal Starts the runtime-owned controlled-publication transaction. */
	public beginFrameTransaction(): void {
		this._transactionActive = true;
		this.clearPendingPassState();
	}

	/** @internal Ends the controlled-publication transaction. */
	public endFrameTransaction(): void {
		this._transactionActive = false;
		this.clearPendingPassState();
	}

	/**
	 * Creates the WebGL post-process bridge.
	 *
	 * @internal WebGL frame executor helper. Prefer registering pass
	 * implementations with `renderer.postProcess` over using this directly.
	 *
	 * @param callbacks Executor-owned accessors and side-effect hooks.
	 * @sideEffects None. Callback side effects occur only when methods create
	 * execution contexts.
	 */
	public constructor(callbacks: WebGLPostProcessBridgeCallbacks) {
		this._callbacks = callbacks;
	}

	/**
	 * Provides a WebGL context for a pass-owned implementation.
	 *
	 * @internal Called by `WebGLFrameExecutor.getPassExecutionContext`.
	 *
	 * @param request Current pass-owned implementation context request.
	 * @returns Context declared by implementation metadata, or `undefined`.
	 * @sideEffects May synchronize executor temporal-history aliases when the
	 * implementation declares `syncPipelineHistories`.
	 */
	public getPassExecutionContext(
		request: PostProcessPassExecutionContextRequest
	): unknown {
		if (this._transactionActive) this.clearPendingPassState();
		const metadata = request.implementation.metadata?.context;
		if (!isWebGLPostProcessContextMetadata(metadata)) {
			return undefined;
		}
		if (metadata.syncPipelineHistories) {
			this._callbacks.applyPipelineHistories(request);
		}
		return this._createContext(metadata, request, "execute");
	}

	/** @internal Commits a controlled color publication after a successful pass. */
	public completePass(
		request: PostProcessPassRequest,
		result: PostProcessPassResult
	): void {
		if (!this._transactionActive) return;
		const texture = this._pendingColorTexture;
		const markTAAHistoryValid = this._pendingMarksTAAHistoryValid;
		this.clearPendingPassState();
		if (!texture) return;
		if (result.ran === false) {
			throw new Error(
				`Post-process pass "${request.passId}" published a color texture and then reported ran: false.`
			);
		}
		this._callbacks.publishColorTexture(texture);
		if (markTAAHistoryValid) this._callbacks.markTAAHistoryValid();
	}

	/** @internal Clears an uncommitted pass publication after an aborted frame. */
	public clearPendingFrameState(): void {
		this.clearPendingPassState();
		this._transactionActive = false;
	}

	/**
	 * Provides a WebGL context for implementation warmup.
	 *
	 * @internal Called by WebGL warmup planning.
	 *
	 * @param implementation Backend implementation being warmed.
	 * @returns Warmup-safe context declared by implementation metadata, or
	 * `undefined`.
	 * @sideEffects None.
	 */
	public getPassWarmupExecutionContext(
		implementation: PostProcessPassImplementation
	): unknown {
		const metadata = implementation.metadata?.context;
		if (!isWebGLPostProcessContextMetadata(metadata)) {
			return undefined;
		}
		return this._createContext(metadata, null, "warmup");
	}

	private _createContext(
		metadata: WebGLPostProcessContextMetadata,
		request: PostProcessPassRequest | null,
		mode: "execute" | "warmup"
	): Record<string, unknown> | undefined {
		const context: Record<string, unknown> = {
			gl: this._callbacks.getGL(),
			programCompiler: this._callbacks.getProgramCompiler(),
			fullscreenVao: this._callbacks.getFullscreenVao(),
			postFramebuffer: this._callbacks.getPostFramebuffer(),
			sceneColorTexture: this._callbacks.getSceneColorTexture(),
			width: this._callbacks.getWidth(),
			height: this._callbacks.getHeight(),
			getSourceTexture: () => this._callbacks.getSourceTexture(),
			resolveTargetTexture: (sourceTexture: WebGLTexture) =>
				this._callbacks.resolveTargetTexture(sourceTexture),
			bindColorTarget: (texture: WebGLTexture) =>
				this._callbacks.bindColorTarget(texture),
			drawFullscreen: (
				width = this._callbacks.getWidth(),
				height = this._callbacks.getHeight(),
				frameContext = this._callbacks.getActiveContext()
			) => this._callbacks.drawFullscreen(width, height, frameContext),
			publishColorTexture: (texture: WebGLTexture) => {
				if (mode !== "execute") return;
				if (!this._transactionActive) {
					this._callbacks.publishColorTexture(texture);
					if (metadata.markTAAHistoryValidOnPublish) {
						this._callbacks.markTAAHistoryValid();
					}
					return;
				}
				this._pendingColorTexture = texture;
				this._pendingMarksTAAHistoryValid =
					metadata.markTAAHistoryValidOnPublish === true;
			},
		};

		if (metadata.sceneMotionTexture) {
			context.sceneMotionTexture = this._callbacks.getSceneMotionTexture();
		}
		if (metadata.sceneNormalTexture) {
			context.sceneNormalTexture = this._callbacks.getSceneNormalTexture();
		}
		if (metadata.ssaoTargets) {
			context.ssaoRawTexture = this._callbacks.getSSAORawTexture();
			context.ssaoBlurTexture = this._callbacks.getSSAOBlurTexture();
			context.ssaoDownsample = this._callbacks.getSSAODownsample();
		}
		if (metadata.frameJitter) {
			context.nextFrameJitter = () => this._callbacks.nextFrameJitter();
		}
		if (metadata.warn) {
			context.warn = (key: string, message: string) =>
				this._callbacks.warn(key, message);
		}
		if (request && mode === "execute") {
			for (const binding of metadata.histories ?? []) {
				context[binding.property] =
					(request.histories[binding.historyId]?.[binding.side]
						.resource as WebGLTexture | null) ?? null;
			}
		}
		return context;
	}

	private clearPendingPassState(): void {
		this._pendingColorTexture = null;
		this._pendingMarksTAAHistoryValid = false;
	}
}
