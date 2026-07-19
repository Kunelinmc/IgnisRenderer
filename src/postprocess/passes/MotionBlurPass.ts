import { clamp } from "../../maths/Common";
import { ceilDiv, finiteOr } from "../../maths/Misc";
import {
	BufferUsage,
	type IComputePipeline,
	type IRenderBuffer,
	type IShaderModule,
} from "../../backends/types";
import {
	WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WEBGPU_WORKGROUP_SIZE,
} from "../../backends/webgpu/constants";
import {
	WEBGPU_SCREEN_POST_PROCESS_CONTEXT_METADATA,
} from "../../backends/webgpu/WebGPUPostProcessContracts";
import {
	WEBGL_SCREEN_POST_PROCESS_CONTEXT_METADATA,
} from "../../backends/webgl/WebGLPostProcessContracts";
import type { PostProcessSharedContext } from "../../backends/webgpu/postprocess/PostProcessSharedContext";
import {
	MOTION_BLUR_CENTER_WEIGHT_RANGE,
	MOTION_BLUR_DEPTH_REJECT_RANGE,
	MOTION_BLUR_MAX_SAMPLES_RANGE,
	MOTION_BLUR_SHUTTER_SCALE_RANGE,
	MOTION_BLUR_VELOCITY_CLAMP_RANGE,
} from "../../backends/webgl/constants";
import { sanitizeFiniteClamped } from "../../backends/webgl/WebGLFrameMath";
import type {
	WebGLProgramCompiler,
	WebGLProgramSlot,
} from "../../backends/webgl/WebGLProgramCompiler";
import { ShaderSource } from "../../shaders/ShaderSource";
import { PostProcessPass, type PostProcessPassConfig } from "../PostProcessPass";
import type { PostProcessPassMetadata } from "../ordering";
import type {
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassRequirements,
	PostProcessPassResult,
} from "../types";
import {
	bindWebGLPostTarget,
	publishWebGPUColorTarget,
	resolveWebGLTarget,
	resolveWebGPUTarget,
	type WebGLScreenPostProcessContext,
	type WebGPURuntimePostProcessContext,
} from "./ScreenPassShared";

export const MOTION_BLUR_PASS_ID = "motion-blur";

interface WebGLMotionBlurProgram {
	readonly program: WebGLProgram;
	readonly uniforms: {
		readonly sourceMap: WebGLUniformLocation | null;
		readonly motionDepthMap: WebGLUniformLocation | null;
		readonly texelSize: WebGLUniformLocation | null;
		readonly motionParams: WebGLUniformLocation | null;
		readonly centerWeight: WebGLUniformLocation | null;
	};
}
export const MOTION_BLUR_PASS_ORDER = {
	id: MOTION_BLUR_PASS_ID,
	placement: "camera",
	order: 400,
	incremental: {
		firstPass: "motion-blur",
		grade: "cinematic",
		inflationRadius: 24,
	},
} as const satisfies PostProcessPassMetadata;
export interface MotionBlurOptions {
	/** Virtual shutter duration multiplier. Higher values lengthen blur trails. */
	shutterScale?: number;
	/** Maximum samples per pixel along the motion vector. */
	maxSamples?: number;
	/** Maximum normalized screen velocity used for blur length. */
	velocityClamp?: number;
	/** Depth-difference threshold for rejecting samples across silhouettes. */
	depthReject?: number;
	/** Weight of the current pixel in the blur accumulation. */
	centerWeight?: number;
	/** Allows backend-specific experimental motion blur options. */
	[key: string]: unknown;
}
export const DEFAULT_MOTION_BLUR_OPTIONS: Required<
	Pick<
		MotionBlurOptions,
		| "shutterScale"
		| "maxSamples"
		| "velocityClamp"
		| "depthReject"
		| "centerWeight"
	>
> = {
	shutterScale: 1,
	maxSamples: 16,
	velocityClamp: 0.06,
	depthReject: 0.025,
	centerWeight: 1,
};
export type WebGPUMotionBlurContext = WebGPURuntimePostProcessContext;
export type WebGLMotionBlurContext = WebGLScreenPostProcessContext;

interface WebGPUMotionBlurResources {
	shared: PostProcessSharedContext;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
	paramData: Float32Array<ArrayBuffer>;
	paramUploaded: boolean;
}
/** @internal WebGPU implementation for the built-in motion blur pass. */
export class WebGPUMotionBlurImplementation
	implements PostProcessPassImplementation<WebGPUMotionBlurContext, MotionBlurOptions>
{
	public readonly id = "motion-blur:webgpu";
	public readonly metadata = {
		context: WEBGPU_SCREEN_POST_PROCESS_CONTEXT_METADATA,
	};
	private _resources =
		new Map<PostProcessSharedContext, WebGPUMotionBlurResources>();

	public async warmup(
		context: WebGPUMotionBlurContext | undefined
	): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		request: PostProcessPassRequest<MotionBlurOptions>,
		context: WebGPUMotionBlurContext | undefined
	): Promise<PostProcessPassResult> {
		if (!context?.encoder || !context.targets) {
			return { ran: false };
		}
		const ran = await this._runMotionBlurKernel(request, context);
		return ran ? { ran: true } : { ran: false };
	}

	public invalidate(): void {
		for (const resources of this._resources.values()) {
			resources.shared.invalidateBindingsByPrefix("motion-blur-");
		}
	}

	public destroy(): void {
		for (const resources of this._resources.values()) {
			resources.shared.destroyManagedResource(
				resources.pipeline,
				"motion blur pipeline"
			);
			resources.shared.destroyManagedResource(
				resources.module,
				"motion blur shader module"
			);
			resources.shared.destroyManagedResource(
				resources.params,
				"motion blur params buffer"
			);
			resources.shared.invalidateBindingsByPrefix("motion-blur-");
			resources.module = null;
			resources.pipeline = null;
			resources.params = null;
			resources.paramUploaded = false;
		}
		this._resources.clear();
	}

	private async _runMotionBlurKernel(
		request: PostProcessPassRequest<MotionBlurOptions>,
		context: WebGPUMotionBlurContext
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
		const target = resolveWebGPUTarget(targets);
		const options = request.options ?? DEFAULT_MOTION_BLUR_OPTIONS;
		const shutterScale = clamp(
			finiteOr(options.shutterScale, DEFAULT_MOTION_BLUR_OPTIONS.shutterScale),
			0,
			2
		);
		const maxSamples = clamp(
			Math.round(
				finiteOr(options.maxSamples, DEFAULT_MOTION_BLUR_OPTIONS.maxSamples)
			),
			4,
			64
		);
		const velocityClamp = clamp(
			finiteOr(
				options.velocityClamp,
				DEFAULT_MOTION_BLUR_OPTIONS.velocityClamp
			),
			0.005,
			0.25
		);
		const depthReject = clamp(
			finiteOr(options.depthReject, DEFAULT_MOTION_BLUR_OPTIONS.depthReject),
			0.0001,
			0.25
		);
		const centerWeight = clamp(
			finiteOr(options.centerWeight, DEFAULT_MOTION_BLUR_OPTIONS.centerWeight),
			0,
			4
		);
		uploadWebGPUMotionBlurParams(
			context.shared,
			resources,
			target.width,
			target.height,
			shutterScale,
			maxSamples,
			velocityClamp,
			depthReject,
			centerWeight
		);
		const binding = context.shared.getCachedBindGroup(
			`motion-blur-${target === targets.postPing ? "ping" : "pong"}`,
			resources.pipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: context.shared.sampler },
				{ binding: 3, resource: resources.params },
				{ binding: 4, resource: target },
			],
			"WebGPUMotionBlur_Binding"
		);
		context.encoder.beginComputePass({ label: "WebGPUMotionBlur" });
		context.encoder.setComputePipeline(resources.pipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WEBGPU_WORKGROUP_SIZE),
			ceilDiv(target.height, WEBGPU_WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();
		publishWebGPUColorTarget(context, target);
		return true;
	}

	private async _ensureResources(
		shared: PostProcessSharedContext
	): Promise<WebGPUMotionBlurResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = {
				shared,
				module: null,
				pipeline: null,
				params: null,
				paramData: new Float32Array(8),
				paramUploaded: false,
			};
			this._resources.set(shared, resources);
		}
		await shared.ensureCommonResources();
		if (!resources.module) {
			const shader = await ShaderSource.load(
				"webgpu.postprocess.motionBlur.composite"
			);
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUMotionBlurShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!resources.pipeline) {
			resources.pipeline = await shared.compute.createComputePipeline({
				label: "WebGPUMotionBlurPipeline",
				compute: { module: resources.module, entryPoint: "csMain" },
			});
		}
		if (!resources.params) {
			resources.params = shared.compute.createBuffer({
				label: "WebGPUMotionBlurParams",
				size: 8 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		return resources;
	}
}
/** @internal WebGL implementation for the built-in motion blur pass. */
export class WebGLMotionBlurImplementation
	implements PostProcessPassImplementation<WebGLMotionBlurContext, MotionBlurOptions>
{
	public readonly id = "motion-blur:webgl";
	public readonly metadata = {
		context: {
			...WEBGL_SCREEN_POST_PROCESS_CONTEXT_METADATA,
			sceneMotionTexture: true,
		},
	};
	private _programCompiler: WebGLProgramCompiler | null = null;
	private _programSlot: WebGLProgramSlot<WebGLMotionBlurProgram> | null = null;

	public warmup(context: WebGLMotionBlurContext | undefined): void {
		if (context) {
			this._getProgramSlot(context.programCompiler).warmup();
		}
	}

	public execute(
		request: PostProcessPassRequest<MotionBlurOptions>,
		context: WebGLMotionBlurContext | undefined
	): PostProcessPassResult {
		if (!context?.sceneMotionTexture) {
			return { ran: false };
		}
		const target = resolveWebGLTarget(context);
		if (!target) {
			return { ran: false };
		}
		const options = request.options;
		const shutterScale = sanitizeFiniteClamped(
			options?.shutterScale,
			DEFAULT_MOTION_BLUR_OPTIONS.shutterScale,
			MOTION_BLUR_SHUTTER_SCALE_RANGE[0],
			MOTION_BLUR_SHUTTER_SCALE_RANGE[1]
		);
		const maxSamples = clamp(
			Math.round(
				finiteOr(options?.maxSamples, DEFAULT_MOTION_BLUR_OPTIONS.maxSamples)
			),
			MOTION_BLUR_MAX_SAMPLES_RANGE[0],
			MOTION_BLUR_MAX_SAMPLES_RANGE[1]
		);
		const velocityClamp = sanitizeFiniteClamped(
			options?.velocityClamp,
			DEFAULT_MOTION_BLUR_OPTIONS.velocityClamp,
			MOTION_BLUR_VELOCITY_CLAMP_RANGE[0],
			MOTION_BLUR_VELOCITY_CLAMP_RANGE[1]
		);
		const depthReject = sanitizeFiniteClamped(
			options?.depthReject,
			DEFAULT_MOTION_BLUR_OPTIONS.depthReject,
			MOTION_BLUR_DEPTH_REJECT_RANGE[0],
			MOTION_BLUR_DEPTH_REJECT_RANGE[1]
		);
		const centerWeight = sanitizeFiniteClamped(
			options?.centerWeight,
			DEFAULT_MOTION_BLUR_OPTIONS.centerWeight,
			MOTION_BLUR_CENTER_WEIGHT_RANGE[0],
			MOTION_BLUR_CENTER_WEIGHT_RANGE[1]
		);

		const gl = context.gl;
		const program = this._getProgramSlot(context.programCompiler).tryGet();
		if (!program) {
			return { ran: false };
		}
		bindWebGLPostTarget(context, program.program, target.texture);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, target.source);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, context.sceneMotionTexture);
		const uniforms = program.uniforms;
		if (uniforms.sourceMap) gl.uniform1i(uniforms.sourceMap, 0);
		if (uniforms.motionDepthMap) gl.uniform1i(uniforms.motionDepthMap, 1);
		if (uniforms.texelSize) {
			gl.uniform2f(
				uniforms.texelSize,
				1 / Math.max(1, context.width),
				1 / Math.max(1, context.height)
			);
		}
		if (uniforms.motionParams) {
			gl.uniform4f(
				uniforms.motionParams,
				shutterScale,
				maxSamples,
				velocityClamp,
				depthReject
			);
		}
		if (uniforms.centerWeight) {
			gl.uniform1f(uniforms.centerWeight, centerWeight);
		}
		context.drawFullscreen();
		gl.bindVertexArray(null);
		context.publishColorTexture(target.texture);
		return { ran: true };
	}

	public destroy(): void {
		this._programSlot?.destroy();
		this._programSlot = null;
		this._programCompiler = null;
	}

	private _getProgramSlot(
		compiler: WebGLProgramCompiler
	): WebGLProgramSlot<WebGLMotionBlurProgram> {
		if (this._programCompiler !== compiler) {
			this._programSlot?.destroy();
			this._programCompiler = compiler;
			this._programSlot = compiler.createSlot({
				label: "WebGLMotionBlurProgram",
				vertex: () => ShaderSource.get("webgl.part.presentVertex.raw"),
				fragment: () =>
					ShaderSource.get("webgl.part.motionBlurFragment.raw"),
				reflect: (gl, program) => ({
					program,
					uniforms: {
						sourceMap: gl.getUniformLocation(program, "uSourceMap"),
						motionDepthMap: gl.getUniformLocation(
							program,
							"uMotionDepthMap"
						),
						texelSize: gl.getUniformLocation(program, "uTexelSize"),
						motionParams: gl.getUniformLocation(program, "uMotionParams"),
						centerWeight: gl.getUniformLocation(program, "uCenterWeight"),
					},
				}),
			});
		}
		return this._programSlot!;
	}
}
export interface MotionBlurPassConfig
	extends Omit<
		PostProcessPassConfig<MotionBlurOptions>,
		| "id"
		| "builtIn"
		| "warningLabel"
		| "placement"
		| "order"
		| "implementations"
	> {}

/**
 * Stateful logical motion blur pass.
 */
export class MotionBlurPass extends PostProcessPass<
	MotionBlurOptions,
	MotionBlurOptions
> {
	public constructor(config: MotionBlurPassConfig = {}) {
		super({
			...config,
			...MOTION_BLUR_PASS_ORDER,
			incremental: config.incremental ?? MOTION_BLUR_PASS_ORDER.incremental,
			warningLabel: "motion blur",
			implementations: {
				webgpu: () => new WebGPUMotionBlurImplementation(),
				webgl: () => new WebGLMotionBlurImplementation(),
			},
		});
	}

	public override normalizeOptions(): MotionBlurOptions {
		return {
			...DEFAULT_MOTION_BLUR_OPTIONS,
			...this.getRawOptions(),
		};
	}

	public override getRequirements(): PostProcessPassRequirements {
		return { gBuffer: ["depth", "motion"] };
	}
}
function uploadWebGPUMotionBlurParams(
	shared: PostProcessSharedContext,
	resources: WebGPUMotionBlurResources,
	width: number,
	height: number,
	shutterScale: number,
	maxSamples: number,
	velocityClamp: number,
	depthReject: number,
	centerWeight: number
): void {
	if (!resources.params) {
		return;
	}
	const data = resources.paramData;
	let changed = !resources.paramUploaded;
	changed =
		setWebGPUParamIfChanged(data, 0, 1 / Math.max(width, 1)) || changed;
	changed =
		setWebGPUParamIfChanged(data, 1, 1 / Math.max(height, 1)) || changed;
	changed = setWebGPUParamIfChanged(data, 2, shutterScale) || changed;
	changed = setWebGPUParamIfChanged(data, 3, maxSamples) || changed;
	changed = setWebGPUParamIfChanged(data, 4, velocityClamp) || changed;
	changed = setWebGPUParamIfChanged(data, 5, depthReject) || changed;
	changed = setWebGPUParamIfChanged(data, 6, centerWeight) || changed;
	changed = setWebGPUParamIfChanged(data, 7, 0) || changed;
	if (!changed) {
		return;
	}
	shared.compute.writeBuffer(resources.params, data);
	resources.paramUploaded = true;
}

function setWebGPUParamIfChanged(
	data: Float32Array<ArrayBuffer>,
	index: number,
	value: number
): boolean {
	const nextValue = Math.fround(value);
	if (data[index] === nextValue) {
		return false;
	}
	data[index] = nextValue;
	return true;
}
