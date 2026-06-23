import { clamp } from "../../maths/Common";
import type { ICommandEncoder } from "../../renderers/ICommandEncoder";
import {
	BufferUsage,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type IShaderModule,
} from "../../renderers/types";
import {
	WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WORKGROUP_SIZE,
} from "../../renderers/webgpu/constants";
import {
	WEBGPU_SCREEN_POST_PROCESS_CONTEXT_METADATA,
	type WebGPUPostProcessFrameTargets,
} from "../../renderers/webgpu/WebGPUPostProcessContracts";
import {
	WEBGL_SCREEN_POST_PROCESS_CONTEXT_METADATA,
} from "../../renderers/webgl/WebGLPostProcessContracts";
import type { PostProcessSharedContext } from "../../renderers/webgpu/postprocess/PostProcessSharedContext";
import type {
	WebGLProgramCompiler,
	WebGLProgramSlot,
} from "../../renderers/webgl/WebGLProgramCompiler";
import { ceilDiv, finiteOr } from "../../maths/Misc";
import { ShaderSource } from "../../shaders/ShaderSource";
import {
	PostProcessPass,
	type PostProcessPassConfig,
	type PostProcessPassResolveRequest,
} from "../PostProcessPass";
import type { PostProcessPassMetadata } from "../ordering";
import type {
	IPostProcessExecutor,
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassRequirements,
	PostProcessPassResult,
} from "../types";

export const FOG_PASS_ID = "fog";
export const FOG_PASS_ORDER = {
	id: FOG_PASS_ID,
	placement: "atmosphere",
	order: 310,
	incremental: {
		firstPass: "fog",
		grade: "cinematic",
		inflationRadius: 20,
		isEnabled: (postProcess) =>
			postProcess.isEnabled(FOG_PASS_ID) &&
			(postProcess.getOptions<FogOptions>(FOG_PASS_ID)?.application ??
				"postprocess") !== "scene",
	},
} as const satisfies PostProcessPassMetadata;

export interface FogOptions {
	/** Distance falloff model used to convert depth into fog opacity. */
	mode?: "linear" | "exp" | "exp2";
	/** Executes fog in the post-process stack or during scene shading. */
	application?: "postprocess" | "scene";
	/** Linear RGB fog color mixed over the scene. */
	color?: [number, number, number];
	/** World/view depth where linear fog starts contributing. */
	start?: number;
	/** World/view depth where linear fog reaches full configured strength. */
	end?: number;
	/** Exponential fog density for `exp` and `exp2` modes. */
	density?: number;
	/** Final fog opacity multiplier. `0` disables visible fog. */
	strength?: number;
	/** Allows backend-specific experimental fog options. */
	[key: string]: unknown;
}

export const DEFAULT_FOG_OPTIONS: Required<
	Pick<
		FogOptions,
		| "mode"
		| "application"
		| "color"
		| "start"
		| "end"
		| "density"
		| "strength"
	>
> = {
	mode: "linear",
	application: "postprocess",
	color: [0.58, 0.64, 0.72],
	start: 20,
	end: 200,
	density: 0.015,
	strength: 1,
};

/** @internal WebGPU context supplied to the built-in fog implementation. */
export interface WebGPUFogContext {
	readonly encoder?: ICommandEncoder;
	readonly targets?: WebGPUPostProcessFrameTargets;
	readonly shared: PostProcessSharedContext;
	publishColorTarget?(texture: IRenderTexture): void;
}

/** @internal WebGL context supplied to the built-in fog implementation. */
export interface WebGLFogContext {
	readonly gl: WebGL2RenderingContext;
	readonly programCompiler: WebGLProgramCompiler;
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

interface WebGLFogProgram {
	readonly program: WebGLProgram;
	readonly uniforms: {
		readonly sceneColor: WebGLUniformLocation | null;
		readonly motionDepthMap: WebGLUniformLocation | null;
		readonly fogParams0: WebGLUniformLocation | null;
		readonly fogParams1: WebGLUniformLocation | null;
	};
}

interface WebGPUFogResources {
	shared: PostProcessSharedContext;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
	paramData: Float32Array<ArrayBuffer>;
}

/**
 * Resolves the numeric fog mode code shared by WebGPU and WebGL shaders.
 *
 * @param mode User-provided fog mode.
 * @returns `0` for linear, `1` for exponential, and `2` for exponential-squared.
 * @sideEffects None.
 */
export function resolveFogModeCode(mode: FogOptions["mode"] | undefined): number {
	switch (mode) {
		case "exp":
			return 1;
		case "exp2":
			return 2;
		default:
			return 0;
	}
}

/**
 * Resolves fog uniforms shared by scene and post-process fog paths.
 *
 * @param options User-provided fog options.
 * @param enabled Whether fog strength should be applied.
 * @param params0 Destination for mode/start/end/density.
 * @param params1 Destination for color/strength.
 * @returns The destination tuple passed in by the caller.
 * @sideEffects Mutates `params0` and `params1`.
 */
export function resolveFogUniformParams(
	options: FogOptions | undefined,
	enabled: boolean,
	params0: Float32Array,
	params1: Float32Array
): readonly [Float32Array, Float32Array] {
	const source = options ?? DEFAULT_FOG_OPTIONS;
	const color = source.color ?? DEFAULT_FOG_OPTIONS.color;
	const start = Math.max(
		0,
		finiteOr(source.start, DEFAULT_FOG_OPTIONS.start)
	);
	const end = Math.max(
		start + 1e-4,
		finiteOr(source.end, DEFAULT_FOG_OPTIONS.end)
	);
	const density = Math.max(
		0,
		finiteOr(source.density, DEFAULT_FOG_OPTIONS.density)
	);
	const strength = enabled ?
		Math.max(0, finiteOr(source.strength, DEFAULT_FOG_OPTIONS.strength))
	:	0;

	params0[0] = resolveFogModeCode(source.mode);
	params0[1] = start;
	params0[2] = end;
	params0[3] = density;

	params1[0] = clamp(finiteOr(color[0], DEFAULT_FOG_OPTIONS.color[0]), 0, 1);
	params1[1] = clamp(finiteOr(color[1], DEFAULT_FOG_OPTIONS.color[1]), 0, 1);
	params1[2] = clamp(finiteOr(color[2], DEFAULT_FOG_OPTIONS.color[2]), 0, 1);
	params1[3] = strength;
	return [params0, params1];
}

/**
 * Resolves fog uniforms into one contiguous WebGPU parameter buffer.
 *
 * @param options User-provided fog options.
 * @param enabled Whether fog strength should be applied.
 * @param data Destination with at least eight floats.
 * @returns The populated destination.
 * @sideEffects Mutates `data`.
 */
export function resolveFogParamData(
	options: FogOptions | undefined,
	enabled: boolean,
	data: Float32Array
): Float32Array {
	const params0 = data.subarray(0, 4);
	const params1 = data.subarray(4, 8);
	resolveFogUniformParams(options, enabled, params0, params1);
	return data;
}

/**
 * WebGPU implementation of the cross-backend fog pass.
 */
/** @internal WebGPU implementation for the built-in fog pass. */
export class WebGPUFogImplementation
	implements PostProcessPassImplementation<WebGPUFogContext, FogOptions>
{
	public readonly id = "fog:webgpu";
	public readonly metadata = {
		context: WEBGPU_SCREEN_POST_PROCESS_CONTEXT_METADATA,
	};
	private _resources = new Map<PostProcessSharedContext, WebGPUFogResources>();

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
		for (const resources of this._resources.values()) {
			resources.shared.invalidateBindingsByPrefix("fog-");
		}
	}

	public destroy(): void {
		for (const resources of this._resources.values()) {
			resources.shared.destroyManagedResource(resources.pipeline, "fog pipeline");
			resources.shared.destroyManagedResource(resources.module, "fog shader module");
			resources.shared.destroyManagedResource(resources.params, "fog params buffer");
			resources.shared.invalidateBindingsByPrefix("fog-");
			resources.module = null;
			resources.pipeline = null;
			resources.params = null;
		}
		this._resources.clear();
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
		}
		await shared.ensureCommonResources();
		if (!resources.module) {
			const shader = await ShaderSource.load("webgpu.postprocess.fog.composite");
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
			resources.pipeline = await shared.compute.createComputePipeline({
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
/** @internal WebGL implementation for the built-in fog pass. */
export class WebGLFogImplementation
	implements PostProcessPassImplementation<WebGLFogContext, FogOptions>
{
	public readonly id = "fog:webgl";
	public readonly metadata = {
		context: {
			...WEBGL_SCREEN_POST_PROCESS_CONTEXT_METADATA,
			sceneMotionTexture: true,
		},
	};
	private _fogParams0 = new Float32Array(4);
	private _fogParams1 = new Float32Array(4);
	private _programCompiler: WebGLProgramCompiler | null = null;
	private _programSlot: WebGLProgramSlot<WebGLFogProgram> | null = null;

	public warmup(context: WebGLFogContext | undefined): void {
		if (context) {
			this._getProgramSlot(context.programCompiler).warmup();
		}
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
		const fogProgram = this._getProgramSlot(
			context.programCompiler
		).tryGet();
		if (!fogProgram) {
			return { ran: false };
		}
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

	public destroy(): void {
		this._programSlot?.destroy();
		this._programSlot = null;
		this._programCompiler = null;
	}

	private _getProgramSlot(
		compiler: WebGLProgramCompiler
	): WebGLProgramSlot<WebGLFogProgram> {
		if (this._programCompiler !== compiler) {
			this._programSlot?.destroy();
			this._programCompiler = compiler;
			this._programSlot = compiler.createSlot({
				label: "WebGLFogProgram",
				vertex: () => ShaderSource.get("webgl.part.presentVertex.raw"),
				fragment: () => ShaderSource.get("webgl.part.fogFragment.raw"),
				reflect: (gl, program) => ({
					program,
					uniforms: {
						sceneColor: gl.getUniformLocation(program, "uSceneColor"),
						motionDepthMap: gl.getUniformLocation(
							program,
							"uMotionDepthMap"
						),
						fogParams0: gl.getUniformLocation(program, "uFogParams0"),
						fogParams1: gl.getUniformLocation(program, "uFogParams1"),
					},
				}),
			});
		}
		return this._programSlot!;
	}
}

export interface FogPassConfig
	extends Omit<
		PostProcessPassConfig<FogOptions>,
		| "id"
		| "builtIn"
		| "warningLabel"
		| "placement"
		| "order"
		| "implementations"
	> {}

/**
 * Stateful logical fog pass shared by WebGPU and WebGL backends.
 */
export class FogPass extends PostProcessPass<FogOptions, FogOptions> {
	public constructor(config: FogPassConfig = {}) {
		super({
			...config,
			...FOG_PASS_ORDER,
			incremental: config.incremental ?? FOG_PASS_ORDER.incremental,
			warningLabel: "fog",
			implementations: {
				webgpu: () => new WebGPUFogImplementation(),
				webgl: () => new WebGLFogImplementation(),
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

	public override shouldExecute(
		request: PostProcessPassResolveRequest<FogOptions>
	): boolean {
		return (request.options?.application ?? "postprocess") !== "scene";
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
