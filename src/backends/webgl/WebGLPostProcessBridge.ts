import type { FrameContext } from "../../pipeline/types";
import type {
	LogicalGBufferSemantic,
	PostProcessExecutionDeclaration,
	PostProcessPassExecutionContextRequest,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessPassCompletion,
} from "../../postprocess";
import { createPostProcessResourceAccessor } from "../../postprocess/PostProcessResourceAccessor";
import type { WebGLProgramCompiler } from "./WebGLProgramCompiler";

export interface WebGLPostProcessBridgeCallbacks {
	getGL(): WebGL2RenderingContext;
	getProgramCompiler(): WebGLProgramCompiler;
	getFullscreenVao(): WebGLVertexArrayObject | null;
	getPostFramebuffer(): WebGLFramebuffer | null;
	getSceneColorTexture(): WebGLTexture | null;
	getSceneMotionTexture(): WebGLTexture | null;
	getSceneNormalTexture(): WebGLTexture | null;
	getWidth(): number;
	getHeight(): number;
	getActiveContext(): FrameContext | null;
	getSourceTexture(): WebGLTexture | null;
	resolveTargetTexture(sourceTexture: WebGLTexture): WebGLTexture | null;
	bindColorTarget(texture: WebGLTexture): void;
	drawFullscreen(
		width: number,
		height: number,
		context: FrameContext | null
	): void;
	commitColorTexture(texture: WebGLTexture): void;
	markTAAHistoryValid(): void;
	applyPipelineHistories(request: PostProcessPassRequest): void;
	warn(key: string, message: string): void;
}

/** @internal Packs implementation-declared WebGL post-process execution contexts. */
export class WebGLPostProcessBridge {
	private readonly _callbacks: WebGLPostProcessBridgeCallbacks;
	private _expectedColorTexture: WebGLTexture | null = null;
	private readonly _physicalIds = new WeakMap<WebGLTexture, string>();
	private _nextPhysicalId = 1;
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
	constructor(callbacks: WebGLPostProcessBridgeCallbacks) {
		this._callbacks = callbacks;
	}

	/**
	 * Provides a WebGL context for a pass-owned implementation.
	 *
	 * @internal Called by `WebGLFrameServiceOwner.createPassExecutionContext`.
	 *
	 * @param request Current pass-owned implementation context request.
	 * @returns Context declared by implementation metadata, or `undefined`.
	 * @sideEffects May synchronize executor temporal-history aliases when the
	 * implementation declares `syncPipelineHistories`.
	 */
	public createPassExecutionContext(request: PostProcessPassExecutionContextRequest): unknown {
		if (this._transactionActive) this.clearPendingPassState();
		if ((request.declaration.histories?.length ?? 0) > 0) {
			this._callbacks.applyPipelineHistories(request);
		}
		const source = this._callbacks.getSourceTexture();
		this._expectedColorTexture =
			request.declaration.color.output === "new-version" && source ?
				this._resolveTargetTexture(source) : null;
		if (
			request.declaration.color.output === "new-version" &&
			!this._expectedColorTexture
		) {
			throw new Error(
				`Post-process pass "${request.passId}" cannot create its required ` +
					"WebGL color output binding.",
			);
		}
		return this._createContext(request);
	}

	/** @internal Commits a controlled color publication after a successful pass. */
	public completePass(
		request: PostProcessPassRequest,
		result: PostProcessPassResult,
	): PostProcessPassCompletion {
		if (!this._transactionActive) return {};
		const expectedTexture = this._expectedColorTexture;
		this.clearPendingPassState();
		if (result.ran === false || request.declaration.color.output === "preserve") {
			return { committed: false };
		}
		if (!expectedTexture) {
			throw new Error(
				`Post-process pass "${request.passId}" has no required physical color binding.`,
			);
		}
		this._callbacks.commitColorTexture(expectedTexture);
		if (result.updatedHistoryIds?.includes("taa")) {
			this._callbacks.markTAAHistoryValid();
		}
		return { committed: true, physicalId: this._getPhysicalId(expectedTexture) };
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
		passId: string,
		declaration: PostProcessExecutionDeclaration,
	): unknown {
		return this._createWarmupContext(passId, declaration);
	}

	private _createContext(
		request: PostProcessPassExecutionContextRequest,
	): Record<string, unknown> {
		return Object.freeze({
			gl: this._callbacks.getGL(),
			programCompiler: this._callbacks.getProgramCompiler(),
			fullscreenVao: this._callbacks.getFullscreenVao(),
			postFramebuffer: this._callbacks.getPostFramebuffer(),
			sceneColorTexture: this._callbacks.getSceneColorTexture(),
			width: this._callbacks.getWidth(),
			height: this._callbacks.getHeight(),
			getSourceTexture: () => this._callbacks.getSourceTexture(),
			bindColorTarget: (texture: WebGLTexture) => this._callbacks.bindColorTarget(texture),
			drawFullscreen: (
				width = this._callbacks.getWidth(),
				height = this._callbacks.getHeight(),
				frameContext = this._callbacks.getActiveContext(),
			) => this._callbacks.drawFullscreen(width, height, frameContext),
			warn: (key: string, message: string) => this._callbacks.warn(key, message),
			resources: createPostProcessResourceAccessor<WebGLTexture>({
				passId: request.passId,
				declaration: request.declaration,
				colorInput: this._callbacks.getSourceTexture(),
				colorOutput: this._expectedColorTexture,
				getGBuffer: (semantic) => this._getGBufferTexture(request, semantic),
				getHistory: (id) => {
					const slot = request.histories[id];
					return slot ? {
						read: (slot.read.resource as WebGLTexture | null) ?? null,
						write: (slot.write.resource as WebGLTexture | null) ?? null,
						valid: slot.valid,
					} : null;
				},
				getTransient: (id) => this._getTransientTexture(request, id),
				getShared: () => null,
			}),
		});
	}

	private _createWarmupContext(
		passId: string,
		declaration: PostProcessExecutionDeclaration,
	): Record<string, unknown> {
		return Object.freeze({
			gl: this._callbacks.getGL(),
			programCompiler: this._callbacks.getProgramCompiler(),
			fullscreenVao: this._callbacks.getFullscreenVao(),
			postFramebuffer: this._callbacks.getPostFramebuffer(),
			sceneColorTexture: this._callbacks.getSceneColorTexture(),
			width: this._callbacks.getWidth(),
			height: this._callbacks.getHeight(),
			getSourceTexture: (): null => null,
			bindColorTarget: (texture: WebGLTexture) => this._callbacks.bindColorTarget(texture),
			drawFullscreen: () => {},
			warn: (key: string, message: string) => this._callbacks.warn(key, message),
			resources: createPostProcessResourceAccessor<WebGLTexture>({
				passId,
				declaration,
				colorInput: null,
				colorOutput: null,
				getGBuffer: () => null,
				getHistory: () => null,
				getTransient: () => null,
				getShared: () => null,
			}),
		});
	}

	private clearPendingPassState(): void {
		this._expectedColorTexture = null;
	}

	private _getGBufferTexture(
		request: PostProcessPassRequest,
		semantic: LogicalGBufferSemantic,
	): WebGLTexture | null {
		const handle = request.gBuffer.channels[semantic]?.handle;
		return handle?.backend === "webgl" && "texture" in handle ? handle.texture : null;
	}

	private _getTransientTexture(
		request: PostProcessPassRequest,
		id: string,
	): WebGLTexture | null {
		const slot = request.transients[id];
		return (slot?.handle.resource as WebGLTexture | null) ?? null;
	}

	private _resolveTargetTexture(sourceTexture: WebGLTexture): WebGLTexture | null {
		const target = this._callbacks.resolveTargetTexture(sourceTexture);
		if (target === sourceTexture) {
			throw new Error(
				"WebGL post-process graph selected one texture for sampled input and color output.",
			);
		}
		this._expectedColorTexture = target;
		return target;
	}

	private _getPhysicalId(texture: WebGLTexture): string {
		let id = this._physicalIds.get(texture);
		if (!id) {
			id = `webgl:${this._nextPhysicalId++}`;
			this._physicalIds.set(texture, id);
		}
		return id;
	}
}
