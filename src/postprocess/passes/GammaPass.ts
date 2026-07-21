import { linearToSRGB } from "../../maths/Common";
import { ceilDiv } from "../../maths/Misc";
import { DEFAULT_GAMMA } from "../../backends/constants";
import { WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WEBGPU_WORKGROUP_SIZE } from "../../backends/webgpu/constants";
import {
	BufferUsage,
	type IComputePipeline,
	type IRenderBuffer,
	type IShaderModule,
} from "../../backends/types";
import type { PostProcessSharedContext } from "../../backends/webgpu/postprocess/PostProcessSharedContext";
import type {
	WebGLProgramCompiler,
	WebGLProgramSlot,
} from "../../backends/webgl/WebGLProgramCompiler";
import { ShaderSource } from "../../shaders/ShaderSource";
import type { PostProcessIncrementalMetadata } from "../../pipeline/incremental";
import { PostProcessPass, type PostProcessPassConfig } from "../PostProcessPass";
import type { PostProcessScheduleEntry } from "../ordering";
import { createPostProcessExecutionDeclaration } from "../executionDeclarations";
import type {
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassResult,
} from "../types";
import {
	bindWebGLPostTarget,
	forEachSoftwareDirtyRect,
	resolveSoftwareDirtyRects,
	resolveWebGLTarget,
	resolveWebGPUTarget,
	type EmptyOptions,
	type SoftwareBuiltinPostProcessContext,
	type WebGLScreenPostProcessContext,
	type WebGPUGammaContext,
} from "./ScreenPassShared";
export type { WebGPUGammaContext } from "./ScreenPassShared";

export const GAMMA_PASS_ID = "gamma";
export const GAMMA_PASS_INCREMENTAL = {
	firstPass: GAMMA_PASS_ID,
	grade: "light",
	inflationRadius: 0,
} as const satisfies PostProcessIncrementalMetadata;
export const GAMMA_PASS_ORDER = {
	id: GAMMA_PASS_ID,
	placement: "present",
	order: 900,
	incremental: GAMMA_PASS_INCREMENTAL,
} as const satisfies PostProcessScheduleEntry;

/** @internal Software implementation for the built-in gamma pass. */
export class SoftwareGammaImplementation implements PostProcessPassImplementation<SoftwareBuiltinPostProcessContext> {
	public readonly id = "gamma:software";
	public describeExecution() {
		return createPostProcessExecutionDeclaration("software");
	}
	private readonly _sRGBLUT = new Uint8Array(256);
	private _lutBuilt = false;
	private _lastGamma = -1;

	public execute(
		request: PostProcessPassRequest,
		context: SoftwareBuiltinPostProcessContext | undefined,
	): PostProcessPassResult {
		const width = request.frameContext.attachments.width;
		const height = request.frameContext.attachments.height;
		const canvasContext = context?.canvasContext ?? null;
		let pixels = request.frameContext.attachments.pixels;
		let imageData: ImageData | null = null;
		if (!pixels) {
			if (!canvasContext) {
				return { ran: false };
			}
			imageData = canvasContext.getImageData(0, 0, width, height);
			pixels = imageData.data;
		}
		if (pixels.length === 0) {
			return { ran: false };
		}
		const gamma = request.frameContext.postProcess.isEnabled(GAMMA_PASS_ID) ? DEFAULT_GAMMA : 1;
		const dirtyRects = resolveSoftwareDirtyRects(request.frameContext);
		this._buildSRGBLUT(gamma);
		const lut = this._sRGBLUT;
		forEachSoftwareDirtyRect(dirtyRects, (rect) => {
			for (let y = rect.minY; y <= rect.maxY; y++) {
				const row = y * width;
				for (let x = rect.minX; x <= rect.maxX; x++) {
					const i = (row + x) << 2;
					pixels[i] = lut[pixels[i]];
					pixels[i + 1] = lut[pixels[i + 1]];
					pixels[i + 2] = lut[pixels[i + 2]];
				}
			}
		});
		if (imageData && canvasContext) {
			canvasContext.putImageData(imageData, 0, 0);
		}
		return { ran: true };
	}

	private _buildSRGBLUT(gamma: number): void {
		if (this._lutBuilt && this._lastGamma === gamma) {
			return;
		}
		const isStandardSRGB = Math.abs(gamma - DEFAULT_GAMMA) < 0.001;
		const invGamma = 1 / gamma;
		for (let i = 0; i < 256; i++) {
			const value = i / 255;
			this._sRGBLUT[i] = isStandardSRGB
				? Math.round(linearToSRGB(value) * 255)
				: Math.round(Math.pow(value, invGamma) * 255);
		}
		this._lutBuilt = true;
		this._lastGamma = gamma;
	}
}

interface WebGPUGammaResources {
	shared: PostProcessSharedContext;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
	paramData: Float32Array<ArrayBuffer>;
}

interface WebGLGammaProgram {
	readonly program: WebGLProgram;
	readonly uniforms: {
		readonly sourceMap: WebGLUniformLocation | null;
	};
}

/** @internal WebGPU implementation for the built-in gamma pass. */
export class WebGPUGammaImplementation implements PostProcessPassImplementation<
	WebGPUGammaContext,
	EmptyOptions
> {
	public readonly id = "gamma:webgpu";
	public describeExecution() {
		return createPostProcessExecutionDeclaration("webgpu");
	}
	private _resources = new Map<PostProcessSharedContext, WebGPUGammaResources>();

	public async warmup(context: WebGPUGammaContext | undefined): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		_request: PostProcessPassRequest<EmptyOptions>,
		context: WebGPUGammaContext | undefined,
	): Promise<PostProcessPassResult> {
		if (!context?.encoder || !context.targets) {
			return { ran: false };
		}
		const ran = await this._runGammaKernel(context);
		return ran ? { ran: true } : { ran: false };
	}

	public invalidate(): void {
		for (const resources of this._resources.values()) {
			resources.shared.invalidateBindingsByPrefix("gamma-");
		}
	}

	public destroy(): void {
		for (const resources of this._resources.values()) {
			resources.shared.destroyManagedResource(resources.pipeline, "gamma pipeline");
			resources.shared.destroyManagedResource(resources.module, "gamma shader module");
			resources.shared.destroyManagedResource(resources.params, "gamma params buffer");
			resources.shared.invalidateBindingsByPrefix("gamma-");
			resources.pipeline = null;
			resources.module = null;
			resources.params = null;
		}
		this._resources.clear();
	}

	private async _runGammaKernel(context: WebGPUGammaContext): Promise<boolean> {
		const resources = await this._ensureResources(context.shared);
		if (!context.encoder || !context.targets || !resources.pipeline || !resources.params) {
			return false;
		}
		const targets = context.targets;
		const target = resolveWebGPUTarget(context);
		const input = context.resources.color.input;
		if (!input) return false;
		const binding = context.shared.getCachedBindGroup(
			`gamma-${target === targets.postPing ? "ping" : "pong"}`,
			resources.pipeline,
			[
				{ binding: 0, resource: input },
				{ binding: 1, resource: resources.params },
				{ binding: 2, resource: target },
			],
			"WebGPUGamma_Binding",
		);
		context.encoder.beginComputePass({ label: "WebGPUGamma" });
		context.encoder.setComputePipeline(resources.pipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WEBGPU_WORKGROUP_SIZE),
			ceilDiv(target.height, WEBGPU_WORKGROUP_SIZE),
			1,
		);
		context.encoder.endComputePass();
		return true;
	}

	private async _ensureResources(
		shared: PostProcessSharedContext,
	): Promise<WebGPUGammaResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = {
				shared,
				module: null,
				pipeline: null,
				params: null,
				paramData: new Float32Array(4),
			};
			this._resources.set(shared, resources);
		}
		if (!resources.module) {
			const shader = await ShaderSource.load("webgpu.postprocess.gamma.composite");
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUGammaShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!resources.pipeline) {
			resources.pipeline = await shared.compute.createComputePipeline({
				label: "WebGPUGammaPipeline",
				compute: { module: resources.module, entryPoint: "csMain" },
			});
		}
		if (!resources.params) {
			resources.params = shared.compute.createBuffer({
				label: "WebGPUGammaParams",
				size: 16,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
			resources.paramData[0] = DEFAULT_GAMMA;
			shared.compute.writeBuffer(resources.params, resources.paramData);
		}
		return resources;
	}
}

/** @internal WebGL implementation for the built-in gamma pass. */
export class WebGLGammaImplementation implements PostProcessPassImplementation<
	WebGLScreenPostProcessContext,
	EmptyOptions
> {
	public readonly id = "gamma:webgl";
	public describeExecution() {
		return createPostProcessExecutionDeclaration("webgl");
	}
	private _programCompiler: WebGLProgramCompiler | null = null;
	private _programSlot: WebGLProgramSlot<WebGLGammaProgram> | null = null;

	public warmup(context: WebGLScreenPostProcessContext | undefined): void {
		if (context) {
			this._getProgramSlot(context.programCompiler).warmup();
		}
	}

	public execute(
		_request: PostProcessPassRequest<EmptyOptions>,
		context: WebGLScreenPostProcessContext | undefined,
	): PostProcessPassResult {
		const target = resolveWebGLTarget(context);
		if (!target) {
			return { ran: false };
		}
		const gl = context.gl;
		const program = this._getProgramSlot(context.programCompiler).tryGet();
		if (!program) {
			return { ran: false };
		}
		bindWebGLPostTarget(context, program.program, target.texture);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, target.source);
		if (program.uniforms.sourceMap) {
			gl.uniform1i(program.uniforms.sourceMap, 0);
		}
		context.drawFullscreen();
		gl.bindVertexArray(null);
		return { ran: true };
	}

	public destroy(): void {
		this._programSlot?.destroy();
		this._programSlot = null;
		this._programCompiler = null;
	}

	private _getProgramSlot(compiler: WebGLProgramCompiler): WebGLProgramSlot<WebGLGammaProgram> {
		if (this._programCompiler !== compiler) {
			this._programSlot?.destroy();
			this._programCompiler = compiler;
			this._programSlot = compiler.createSlot({
				label: "WebGLGammaProgram",
				vertex: () => ShaderSource.get("webgl.part.presentVertex.raw"),
				fragment: () => ShaderSource.get("webgl.part.gammaFragment.raw"),
				reflect: (gl, program) => ({
					program,
					uniforms: {
						sourceMap: gl.getUniformLocation(program, "uSourceMap"),
					},
				}),
			});
		}
		return this._programSlot!;
	}
}

/**
 * Stateful logical gamma correction pass.
 */
export class GammaPass extends PostProcessPass<EmptyOptions, EmptyOptions> {
	public constructor(
		config: Omit<
			PostProcessPassConfig<EmptyOptions>,
			"id" | "builtIn" | "warningLabel" | "implementations"
		> = {},
	) {
		super({
			...config,
			id: GAMMA_PASS_ORDER.id,
			schedule: {
				placement: config.schedule?.placement ?? GAMMA_PASS_ORDER.placement,
				order: config.schedule?.order ?? GAMMA_PASS_ORDER.order,
				incremental: config.schedule?.incremental ?? GAMMA_PASS_ORDER.incremental,
			},
			builtIn: true,
			warningLabel: "gamma correction",
			implementations: {
				software: () => new SoftwareGammaImplementation(),
				webgpu: () => new WebGPUGammaImplementation(),
				webgl: () => new WebGLGammaImplementation(),
			},
		});
	}
}
