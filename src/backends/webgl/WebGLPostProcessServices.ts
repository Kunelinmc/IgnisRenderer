import type { FrameContext } from "../../pipeline/types";
import type { PostProcessColorDomain } from "../../postprocess/PostProcessPass";
import type { DisplayOutputState } from "../../rendering/DisplayOutput";
import type { BackendPostProcessRuntime } from "../../postprocess/BackendPostProcessRuntime";
import type {
	LogicalGBufferBridge,
	LogicalGBufferSemantic,
	PostProcessExecutionDeclaration,
	PostProcessPassCompletion,
	PostProcessPassExecutionContextRequest,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "../../postprocess";
import { createPostProcessResourceAccessor } from "../../postprocess/PostProcessResourceAccessor";
import { Logger } from "../../foundation/Logger";
import type { WebGLProgramCompiler } from "./WebGLProgramCompiler";
import type { WebGLFrameTargetManager } from "./WebGLFrameTargetManager";
import type {
	WebGLProgramWarmupContributor,
	WebGLProgramWarmupRequest,
	WebGLProgramWarmupTask,
} from "./WebGLWarmupCoordinator";

export interface WebGLPostProcessServicesHost {
	readonly gl: WebGL2RenderingContext;
	readonly targets: WebGLFrameTargetManager;
	getProgramCompiler(): WebGLProgramCompiler;
	getFullscreenVao(): WebGLVertexArrayObject | null;
	getWidth(): number;
	getHeight(): number;
	getActiveContext(): FrameContext | null;
	getDisplayOutputState(): DisplayOutputState | undefined;
	drawFullscreen(width: number, height: number, context: FrameContext | null): void;
}

/** Context-scoped WebGL post-process resources and controlled publication. */
export class WebGLPostProcessServices {
	private readonly _host: WebGLPostProcessServicesHost;
	private _expectedColorTexture: WebGLTexture | null = null;
	private readonly _physicalIds = new WeakMap<WebGLTexture, string>();
	private _nextPhysicalId = 1;
	private _transactionActive = false;
	private _outputColorDomain: PostProcessColorDomain = "scene-linear-hdr";

	public constructor(host: WebGLPostProcessServicesHost) {
		this._host = host;
	}

	public get outputColorDomain(): PostProcessColorDomain {
		return this._outputColorDomain;
	}

	public setInitialColorDomain(domain: PostProcessColorDomain): void {
		this._outputColorDomain = domain;
	}

	/** @internal Starts the runtime-owned controlled-publication transaction. */
	public beginFrame(): void {
		this._transactionActive = true;
		this._clearPendingPassState();
	}

	/** @internal Ends the controlled-publication transaction. */
	public endFrame(): void {
		this._transactionActive = false;
		this._clearPendingPassState();
	}

	public createResource(
		desc: PostProcessResourceDescriptor,
	): PostProcessResourceHandle {
		const gl = this._host.gl;
		const texture = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		const actualFormat = "rgba16float";
		const internalFormat = gl.RGBA16F;
		const type = gl.HALF_FLOAT;
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			internalFormat,
			desc.width,
			desc.height,
			0,
			gl.RGBA,
			type,
			null,
		);
		gl.bindTexture(gl.TEXTURE_2D, null);
		return {
			id: desc.id,
			backend: "webgl",
			width: desc.width,
			height: desc.height,
			format: actualFormat,
			resource: texture,
		};
	}

	public destroyResource(handle: PostProcessResourceHandle): void {
		this._host.gl.deleteTexture(handle.resource as WebGLTexture | null);
	}

	public createGBufferBridge(context: FrameContext): LogicalGBufferBridge {
		return this._host.targets.createGBufferBridge(context);
	}

	/**
	 * Provides a WebGL context for a pass-owned implementation.
	 *
	 * @internal Called by `WebGLFrameServices.createPassExecutionContext`.
	 *
	 * @param request Current pass-owned implementation context request.
	 * @returns Context declared by implementation metadata.
	 * @sideEffects Resolves the pass output binding for controlled publication.
	 */
	public createPassExecutionContext(
		request: PostProcessPassExecutionContextRequest,
	): unknown {
		if (this._transactionActive) this._clearPendingPassState();
		const source = this._getSourceTexture();
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
		this._clearPendingPassState();
		if (result.ran) {
			const contract = request.pass.colorContract;
			if (contract?.input === this._outputColorDomain) {
				this._outputColorDomain = contract.output;
			}
		}
		if (result.ran === false || request.declaration.color.output === "preserve") {
			return { committed: false };
		}
		if (!expectedTexture) {
			throw new Error(
				`Post-process pass "${request.passId}" has no required physical color binding.`,
			);
		}
		this._host.targets._presentSourceTexture = expectedTexture;
		return { committed: true, physicalId: this._getPhysicalId(expectedTexture) };
	}

	/** @internal Clears an uncommitted pass publication after an aborted frame. */
	public abortFrame(): void {
		this._clearPendingPassState();
		this._transactionActive = false;
	}

	/**
	 * Provides a WebGL context for implementation warmup.
	 *
	 * @internal Called by WebGL warmup planning.
	 *
	 * @param passId Pass whose implementation is being warmed.
	 * @param declaration Pass resource declaration.
	 * @returns Warmup-safe context declared by implementation metadata.
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
			gl: this._host.gl,
			displayOutput: this._host.getDisplayOutputState(),
			programCompiler: this._host.getProgramCompiler(),
			fullscreenVao: this._host.getFullscreenVao(),
			postFramebuffer: this._host.targets._postFramebuffer,
			sceneColorTexture: this._host.targets._sceneColorTexture,
			width: this._host.getWidth(),
			height: this._host.getHeight(),
			getSourceTexture: () => this._getSourceTexture(),
			bindColorTarget: (texture: WebGLTexture) =>
				this._host.targets.bindPostSingleColorTarget(texture),
			drawFullscreen: (
				width = this._host.getWidth(),
				height = this._host.getHeight(),
				frameContext = this._host.getActiveContext(),
			) => this._host.drawFullscreen(width, height, frameContext),
			warn: (key: string, message: string) => this._warn(key, message),
			resources: createPostProcessResourceAccessor<WebGLTexture>({
				passId: request.passId,
				declaration: request.declaration,
				colorInput: this._getSourceTexture(),
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
			gl: this._host.gl,
			displayOutput: this._host.getDisplayOutputState(),
			programCompiler: this._host.getProgramCompiler(),
			fullscreenVao: this._host.getFullscreenVao(),
			postFramebuffer: this._host.targets._postFramebuffer,
			sceneColorTexture: this._host.targets._sceneColorTexture,
			width: this._host.getWidth(),
			height: this._host.getHeight(),
			getSourceTexture: (): null => null,
			bindColorTarget: (texture: WebGLTexture) =>
				this._host.targets.bindPostSingleColorTarget(texture),
			drawFullscreen: () => {},
			warn: (key: string, message: string) => this._warn(key, message),
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

	private _clearPendingPassState(): void {
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
		const target = this._host.targets.resolvePostProcessTargetTexture(sourceTexture);
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

	private _getSourceTexture(): WebGLTexture | null {
		return this._host.targets._presentSourceTexture ??
			this._host.targets._sceneColorTexture;
	}

	private _warn(key: string, message: string): void {
		Logger.warn(`[${key}] ${message}`, {
			scope: "WebGLPostProcessServices",
			onceKey: key,
		});
	}
}

/** @internal Produces warmup tasks from one validated post-process plan. */
export class WebGLPostProcessWarmupContributor
	implements WebGLProgramWarmupContributor
{
	private readonly _runtime: BackendPostProcessRuntime;
	private readonly _services: WebGLPostProcessServices;

	constructor(
		runtime: BackendPostProcessRuntime,
		services: WebGLPostProcessServices,
	) {
		this._runtime = runtime;
		this._services = services;
	}

	public collectWarmupTasks(
		request: WebGLProgramWarmupRequest,
	): readonly WebGLProgramWarmupTask[] {
		const graph =
			request.postProcessPlan ?? this._runtime.planWarmup(request.context);
		const warmed = new Set<string>();
		const tasks: WebGLProgramWarmupTask[] = [];
		for (const passId of request.plan.postProcessPasses) {
			if (warmed.has(passId)) continue;
			const compiled = graph?.passes.find((pass) => pass.id === passId);
			const implementation = compiled?.implementation;
			if (!compiled || typeof implementation?.warmup !== "function") continue;
			warmed.add(passId);
			tasks.push({
				label: `WebGLPostWarmup:${passId}`,
				priority: "postprocess",
				run: async () => {
					const context = this._services.getPassWarmupExecutionContext(
						compiled.id,
						compiled.declaration,
					);
					await implementation.warmup?.(context, {
						frameContext: request.context,
						postProcess: request.context.postProcess,
						backend: "webgl",
						context,
						options: compiled.options,
					});
				},
			});
		}
		return tasks;
	}
}
