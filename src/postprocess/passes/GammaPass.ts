import { ceilDiv } from "../../maths/Misc";
import { WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WEBGPU_WORKGROUP_SIZE } from "../../backends/webgpu/constants";
import {
	BufferUsage,
	type IComputePipeline,
	type IRenderBuffer,
	type IShaderModule,
} from "../../backends/types";
import type { WebGPUPostProcessServices } from "../../backends/webgpu/WebGPUPostProcessContracts";
import type {
	WebGLProgramCompiler,
	WebGLProgramSlot,
} from "../../backends/webgl/WebGLProgramCompiler";
import { ShaderSource } from "../../shaders/ShaderSource";
import type { PostProcessIncrementalMetadata } from "../../pipeline/incremental";
import { PostProcessPass, type PostProcessPassConfig } from "../PostProcessPass";
import type { PostProcessScheduleEntry } from "../ordering";
import {
	SOFTWARE_IN_PLACE_EXECUTION,
	WEBGL_VERSIONED_EXECUTION,
	WEBGPU_VERSIONED_EXECUTION,
} from "../executionDeclarations";
import type {
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassResult,
} from "../types";
import {
	bindWebGLPostTarget,
	forEachSoftwareDirtyRect,
	resolveWebGLTarget,
	resolveWebGPUTarget,
	type EmptyOptions,
	type SoftwareBuiltinPostProcessContext,
	type WebGLScreenPostProcessContext,
	type WebGPUScreenPostProcessContext,
} from "./ScreenPassShared";

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

/**
 * Converts D65 linear-sRGB values to D65 linear Display-P3.
 *
 * @internal Shared by CPU presentation and numerical contract tests.
 */
export function linearSrgbToDisplayP3(
	color: readonly [number, number, number],
	out: [number, number, number] = [0, 0, 0],
): [number, number, number] {
	const [red, green, blue] = color;
	out[0] = 0.82259287 * red + 0.17753395 * green;
	out[1] = 0.03319951 * red + 0.9667835 * green;
	out[2] = 0.01708535 * red + 0.07239572 * green + 0.91030148 * blue;
	return out;
}

/** @internal Piecewise sRGB transfer function with an optional extended range. */
export function encodeLinearSRGB(value: number, extended = false): number {
	const linear = Math.max(0, value);
	const encoded = linear <= 0.0031308 ?
		12.92 * linear
	:	1.055 * Math.pow(linear, 1 / 2.4) - 0.055;
	return extended ? encoded : Math.min(1, encoded);
}

/** @internal Software implementation for the built-in gamma pass. */
export class SoftwareGammaImplementation implements PostProcessPassImplementation<SoftwareBuiltinPostProcessContext> {
	public readonly id = "gamma:software";
	private readonly _inputColor: [number, number, number] = [0, 0, 0];
	private readonly _p3Color: [number, number, number] = [0, 0, 0];

	public describeExecution() {
		return SOFTWARE_IN_PLACE_EXECUTION;
	}

	public execute(
		request: PostProcessPassRequest,
		context: SoftwareBuiltinPostProcessContext | undefined,
	): PostProcessPassResult {
		if (!context) {
			return { ran: false };
		}
		const width = request.frameContext.attachments.width;
		const pixels = context.resources.color.input;
		if (!(pixels instanceof Float32Array) || pixels.length === 0) {
			return { ran: false };
		}
		const dirtyRects = context.dirtyRects;
		const hdr = context.displayOutput?.activeDynamicRange === "hdr";
		const upperBound = hdr ? (context.displayOutput?.requested.hdrHeadroom ?? 4) : 1;
		forEachSoftwareDirtyRect(dirtyRects, (rect) => {
			for (let y = rect.minY; y <= rect.maxY; y++) {
				const row = y * width;
				for (let x = rect.minX; x <= rect.maxX; x++) {
					const i = (row + x) << 2;
					this._inputColor[0] = pixels[i];
					this._inputColor[1] = pixels[i + 1];
					this._inputColor[2] = pixels[i + 2];
					const linear = hdr
						? linearSrgbToDisplayP3(this._inputColor, this._p3Color)
						: this._inputColor;
					pixels[i] = encodeLinearSRGB(Math.min(upperBound, Math.max(0, linear[0])), hdr);
					pixels[i + 1] = encodeLinearSRGB(
						Math.min(upperBound, Math.max(0, linear[1])),
						hdr,
					);
					pixels[i + 2] = encodeLinearSRGB(
						Math.min(upperBound, Math.max(0, linear[2])),
						hdr,
					);
				}
			}
		});
		return { ran: true };
	}
}

interface WebGPUGammaResources {
	shared: WebGPUPostProcessServices;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
	paramData: Float32Array<ArrayBuffer>;
}

interface WebGLGammaProgram {
	readonly program: WebGLProgram;
	readonly uniforms: {
		readonly sourceMap: WebGLUniformLocation | null;
		readonly hdrEnabled: WebGLUniformLocation | null;
		readonly hdrHeadroom: WebGLUniformLocation | null;
	};
}

/** @internal WebGPU implementation for the built-in gamma pass. */
export class WebGPUGammaImplementation implements PostProcessPassImplementation<
	WebGPUScreenPostProcessContext,
	EmptyOptions
> {
	public readonly id = "gamma:webgpu";

	public describeExecution() {
		return WEBGPU_VERSIONED_EXECUTION;
	}
	private _resources = new Map<WebGPUPostProcessServices, WebGPUGammaResources>();

	public async warmup(context: WebGPUScreenPostProcessContext | undefined): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		_request: PostProcessPassRequest<EmptyOptions>,
		context: WebGPUScreenPostProcessContext | undefined,
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

	private async _runGammaKernel(context: WebGPUScreenPostProcessContext): Promise<boolean> {
		const resources = await this._ensureResources(context.shared);
		if (!context.encoder || !context.targets || !resources.pipeline || !resources.params) {
			return false;
		}
		const targets = context.targets;
		const target = resolveWebGPUTarget(context);
		const input = context.resources.color.input;
		if (!input) return false;
		const display = context.shared.getDisplayOutputState?.();
		resources.paramData[0] = display?.activeDynamicRange === "hdr" ? 1 : 0;
		resources.paramData[1] = display?.requested.hdrHeadroom ?? 4;
		resources.paramData[2] = 0;
		resources.paramData[3] = 0;
		context.shared.compute.writeBuffer(resources.params, resources.paramData);
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
		shared: WebGPUPostProcessServices,
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
		return WEBGL_VERSIONED_EXECUTION;
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
		const display = context.displayOutput;
		if (program.uniforms.hdrEnabled) {
			gl.uniform1f(
				program.uniforms.hdrEnabled,
				display?.activeDynamicRange === "hdr" ? 1 : 0,
			);
		}
		if (program.uniforms.hdrHeadroom) {
			gl.uniform1f(
				program.uniforms.hdrHeadroom,
				display?.requested.hdrHeadroom ?? 4,
			);
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
						hdrEnabled: gl.getUniformLocation(program, "uHdrEnabled"),
						hdrHeadroom: gl.getUniformLocation(program, "uHdrHeadroom"),
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
			"id" | "builtIn" | "label" | "implementations"
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
			label: "gamma correction",
			colorContract: config.colorContract ?? {
				input: "display-linear",
				output: "display-encoded",
			},
			implementations: {
				software: () => new SoftwareGammaImplementation(),
				webgpu: () => new WebGPUGammaImplementation(),
				webgl: () => new WebGLGammaImplementation(),
			},
		});
	}
}
