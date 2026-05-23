import {
	DEFAULT_FOG_OPTIONS,
	type FogOptions,
} from "../../pipeline/types";
import type { ICommandEncoder } from "../../renderers/ICommandEncoder";
import {
	BufferUsage,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type IShaderModule,
} from "../../renderers/types";
import type { WebGPUFrameTargets } from "../../renderers/webgpu/WebGPUPostProcessContracts";
import type { PostProcessSharedContext } from "../../renderers/webgpu/postprocess/PostProcessSharedContext";
import type { WebGLProgramLibrary } from "../../renderers/webgl/WebGLProgramLibrary";
import { ceilDiv } from "../../maths/Misc";
import { loadPostProcessShaderPartComposite } from "../../shaders/webgpu/shaderSource";
import {
	PostProcessPass,
	type PostProcessPassConfig,
} from "../PostProcessPass";
import type {
	IPostProcessExecutor,
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassRequirements,
	PostProcessPassResult,
} from "../types";
import { resolveFogParamData, resolveFogUniformParams } from "./fogUtils";

const WORKGROUP_SIZE = 8;

export interface WebGPUFogContext {
	readonly encoder?: ICommandEncoder;
	readonly targets?: WebGPUFrameTargets;
	readonly shared: PostProcessSharedContext;
	publishColorTarget?(texture: IRenderTexture): void;
}

export interface WebGLFogContext {
	readonly gl: WebGL2RenderingContext;
	readonly programs: WebGLProgramLibrary;
	readonly fullscreenVao: WebGLVertexArrayObject | null;
	readonly postFramebuffer: WebGLFramebuffer | null;
	readonly sceneColorTexture: WebGLTexture | null;
	readonly sceneMotionTexture: WebGLTexture | null;
	readonly width: number;
	readonly height: number;
	getSourceTexture(): WebGLTexture | null;
	resolveTargetTexture(sourceTexture: WebGLTexture): WebGLTexture | null;
	bindColorTarget(texture: WebGLTexture): void;
	drawFullscreen(): void;
	publishColorTexture(texture: WebGLTexture): void;
}

interface WebGPUFogResources {
	shared: PostProcessSharedContext;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
	paramData: Float32Array<ArrayBuffer>;
}

/**
 * WebGPU implementation of the cross-backend fog pass.
 */
export class WebGPUFogImplementation
	implements PostProcessPassImplementation<WebGPUFogContext, FogOptions>
{
	public readonly id = "fog:webgpu";
	private _resources = new WeakMap<PostProcessSharedContext, WebGPUFogResources>();
	private _resourceSet = new Set<WebGPUFogResources>();

	public async warmup(context: WebGPUFogContext | undefined): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		request: PostProcessPassRequest<FogOptions>,
		context: WebGPUFogContext | undefined
	): Promise<PostProcessPassResult> {
		if (!context?.encoder || !context.targets) {
			return { ran: false };
		}
		const ran = await this._runFogKernel(request, context);
		return ran ? { ran: true } : { ran: false };
	}

	public invalidate(): void {
		for (const resources of this._resourceSet) {
			resources.shared.invalidateBindingsByPrefix("fog-");
		}
	}

	public destroy(): void {
		for (const resources of this._resourceSet) {
			resources.shared.destroyManagedResource(resources.pipeline, "fog pipeline");
			resources.shared.destroyManagedResource(resources.module, "fog shader module");
			resources.shared.destroyManagedResource(resources.params, "fog params buffer");
			resources.shared.invalidateBindingsByPrefix("fog-");
			resources.module = null;
			resources.pipeline = null;
			resources.params = null;
		}
		this._resourceSet.clear();
		this._resources = new WeakMap<PostProcessSharedContext, WebGPUFogResources>();
	}

	private async _runFogKernel(
		request: PostProcessPassRequest<FogOptions>,
		context: WebGPUFogContext
	): Promise<boolean> {
		const resources = await this._ensureResources(context.shared);
		if (
			!context.encoder ||
			!context.targets ||
			!context.shared.sampler ||
			!resources.pipeline ||
			!resources.params
		) {
			return false;
		}
		const targets = context.targets;
		const target =
			targets.sceneColor === targets.postPong ? targets.postPing : targets.postPong;
		resolveFogParamData(request.options, true, resources.paramData);
		context.shared.compute.writeBuffer(resources.params, resources.paramData);
		const binding = context.shared.getCachedBindGroup(
			`fog-${target === targets.postPing ? "ping" : "pong"}`,
			resources.pipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: context.shared.sampler },
				{ binding: 3, resource: resources.params },
				{ binding: 4, resource: target },
			],
			"WebGPUFog_Binding"
		);
		context.encoder.beginComputePass({ label: "WebGPUFog" });
		context.encoder.setComputePipeline(resources.pipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WORKGROUP_SIZE),
			ceilDiv(target.height, WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();
		context.publishColorTarget?.(target);
		return true;
	}

	private async _ensureResources(
		shared: PostProcessSharedContext
	): Promise<WebGPUFogResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = {
				shared,
				module: null,
				pipeline: null,
				params: null,
				paramData: new Float32Array(8),
			};
			this._resources.set(shared, resources);
			this._resourceSet.add(resources);
		}
		await shared.ensureCommonResources();
		if (!resources.module) {
			const shader = await loadPostProcessShaderPartComposite("fog");
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUFogShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!resources.pipeline) {
			resources.pipeline = shared.compute.createComputePipeline({
				label: "WebGPUFogPipeline",
				compute: { module: resources.module, entryPoint: "csMain" },
			});
		}
		if (!resources.params) {
			resources.params = shared.compute.createBuffer({
				label: "WebGPUFogParams",
				size: 8 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		return resources;
	}
}

/**
 * WebGL implementation of the cross-backend fog pass.
 */
export class WebGLFogImplementation
	implements PostProcessPassImplementation<WebGLFogContext, FogOptions>
{
	public readonly id = "fog:webgl";
	private _fogParams0 = new Float32Array(4);
	private _fogParams1 = new Float32Array(4);

	public warmup(context: WebGLFogContext | undefined): void {
		context?.programs.getFogProgram();
	}

	public execute(
		request: PostProcessPassRequest<FogOptions>,
		context: WebGLFogContext | undefined
	): PostProcessPassResult {
		if (!context) {
			return { ran: false };
		}
		return this._runFogPass(request, context);
	}

	private _runFogPass(
		request: PostProcessPassRequest<FogOptions>,
		context: WebGLFogContext
	): PostProcessPassResult {
		if (
			!context.sceneMotionTexture ||
			!context.postFramebuffer ||
			!context.sceneColorTexture ||
			!context.fullscreenVao
		) {
			return { ran: false };
		}
		const sourceTexture = context.getSourceTexture();
		if (!sourceTexture) {
			return { ran: false };
		}
		const targetTexture = context.resolveTargetTexture(sourceTexture);
		if (!targetTexture) {
			return { ran: false };
		}

		resolveFogUniformParams(
			request.options,
			true,
			this._fogParams0,
			this._fogParams1
		);
		const gl = context.gl;
		const fogProgram = context.programs.getFogProgram();
		gl.bindFramebuffer(gl.FRAMEBUFFER, context.postFramebuffer);
		context.bindColorTarget(targetTexture);
		gl.viewport(0, 0, context.width, context.height);
		gl.useProgram(fogProgram.program);
		gl.bindVertexArray(context.fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, context.sceneMotionTexture);

		const uniforms = fogProgram.uniforms;
		if (uniforms.sceneColor) {
			gl.uniform1i(uniforms.sceneColor, 0);
		}
		if (uniforms.motionDepthMap) {
			gl.uniform1i(uniforms.motionDepthMap, 1);
		}
		if (uniforms.fogParams0) {
			gl.uniform4fv(uniforms.fogParams0, this._fogParams0);
		}
		if (uniforms.fogParams1) {
			gl.uniform4fv(uniforms.fogParams1, this._fogParams1);
		}
		context.drawFullscreen();
		gl.bindVertexArray(null);
		context.publishColorTexture(targetTexture);
		return { ran: true };
	}
}

export interface FogPassConfig
	extends Omit<
		PostProcessPassConfig<FogOptions>,
		"id" | "placement" | "implementations"
	> {}

/**
 * Stateful logical fog pass shared by WebGPU and WebGL backends.
 */
export class FogPass extends PostProcessPass<FogOptions, FogOptions> {
	public constructor(config: FogPassConfig = {}) {
		super({
			...config,
			id: "fog",
			placement: "atmosphere",
			implementations: {
				webgpu: new WebGPUFogImplementation(),
				webgl: new WebGLFogImplementation(),
			},
		});
	}

	public override normalizeOptions(): FogOptions {
		return {
			...DEFAULT_FOG_OPTIONS,
			...this.getRawOptions(),
		};
	}

	public override getRequirements(): PostProcessPassRequirements {
		return { gBuffer: ["depth"] };
	}

	public override execute(
		request: PostProcessPassRequest<FogOptions>,
		context: unknown,
		executor: IPostProcessExecutor
	): PostProcessPassResult | Promise<PostProcessPassResult> {
		if ((request.options.application ?? "postprocess") === "scene") {
			return { ran: false };
		}
		return super.execute(request, context, executor);
	}
}
