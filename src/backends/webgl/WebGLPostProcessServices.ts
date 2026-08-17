import type { FrameContext } from "../../pipeline/types";
import type { BackendPostProcessRuntime } from "../../postprocess/BackendPostProcessRuntime";
import type {
	LogicalGBufferBridge,
	PostProcessExecutionDeclaration,
	PostProcessPassCompletion,
	PostProcessPassExecutionContextRequest,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "../../postprocess";
import { Logger } from "../../foundation/Logger";

import type { WebGLProgramCompiler } from "./WebGLProgramCompiler";
import type { WebGLFrameTargetManager } from "./WebGLFrameTargetManager";
import { WebGLPostProcessBridge } from "./WebGLPostProcessBridge";
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
	drawFullscreen(width: number, height: number, context: FrameContext | null): void;
}

/** Context-scoped WebGL post-process resources and publication bridge. */
export class WebGLPostProcessServices {
	private readonly _host: WebGLPostProcessServicesHost;
	private readonly _bridge: WebGLPostProcessBridge;

	public constructor(host: WebGLPostProcessServicesHost) {
		this._host = host;
		this._bridge = new WebGLPostProcessBridge({
			getGL: () => host.gl,
			getProgramCompiler: () => host.getProgramCompiler(),
			getFullscreenVao: () => host.getFullscreenVao(),
			getPostFramebuffer: () => host.targets._postFramebuffer,
			getSceneColorTexture: () => host.targets._sceneColorTexture,
			getSceneMotionTexture: () => host.targets._sceneMotionTexture,
			getSceneNormalTexture: () => host.targets._sceneNormalTexture,
			getWidth: () => host.getWidth(),
			getHeight: () => host.getHeight(),
			getActiveContext: () => host.getActiveContext(),
			getSourceTexture: () =>
				host.targets._presentSourceTexture ?? host.targets._sceneColorTexture,
			resolveTargetTexture: (sourceTexture) =>
				host.targets.resolvePostProcessTargetTexture(sourceTexture),
			bindColorTarget: (texture) => host.targets.bindPostSingleColorTarget(texture),
			drawFullscreen: (width, height, context) =>
				host.drawFullscreen(width, height, context),
			commitColorTexture: (texture) => {
				host.targets._presentSourceTexture = texture;
			},
			warn: (key, message) =>
				Logger.warn(`[${key}] ${message}`, {
					scope: "WebGLPostProcessServices",
					onceKey: key,
				}),
		});
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

	public createPassExecutionContext(
		request: PostProcessPassExecutionContextRequest,
	): unknown {
		return this._bridge.createPassExecutionContext(request);
	}

	public getPassWarmupExecutionContext(
		passId: string,
		declaration: PostProcessExecutionDeclaration,
	): unknown {
		return this._bridge.getPassWarmupExecutionContext(passId, declaration);
	}

	public beginFrame(): void {
		this._bridge.beginFrameTransaction();
	}

	public endFrame(): void {
		this._bridge.endFrameTransaction();
	}

	public abortFrame(): void {
		this._bridge.clearPendingFrameState();
	}

	public completePass(
		request: PostProcessPassRequest,
		result: PostProcessPassResult,
	): PostProcessPassCompletion {
		return this._bridge.completePass(request, result);
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
			request.postProcessPlan ??
			(request.plan.includePostProcess ?
				this._runtime.planWarmup(request.context)
			: null);
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
