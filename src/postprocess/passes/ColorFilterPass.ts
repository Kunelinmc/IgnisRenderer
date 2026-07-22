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
import type { WebGPUPostProcessServices } from "../../backends/webgpu/WebGPUPostProcessContracts";
import { sanitizeFiniteClamped } from "../../backends/webgl/WebGLFrameMath";
import type {
	WebGLProgramCompiler,
	WebGLProgramSlot,
} from "../../backends/webgl/WebGLProgramCompiler";
import { ShaderSource } from "../../shaders/ShaderSource";
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
	resolveSoftwareDirtyRects,
	resolveWebGLTarget,
	resolveWebGPUTarget,
	type SoftwareBuiltinPostProcessContext,
	type WebGLScreenPostProcessContext,
	type WebGPURuntimePostProcessContext,
} from "./ScreenPassShared";

export const COLOR_FILTER_PASS_ID = "color-filter";

interface WebGLColorFilterProgram {
	readonly program: WebGLProgram;
	readonly uniforms: {
		readonly sourceMap: WebGLUniformLocation | null;
		readonly filterParams0: WebGLUniformLocation | null;
		readonly filterParams1: WebGLUniformLocation | null;
	};
}
export const COLOR_FILTER_PASS_ORDER = {
	id: COLOR_FILTER_PASS_ID,
	placement: "ldr",
	order: 700,
	incremental: {
		firstPass: "color-filter",
		grade: "light",
		inflationRadius: 2,
	},
} as const satisfies PostProcessScheduleEntry;
export interface ColorFilterOptions {
	/** Additive brightness shift in normalized color space. */
	brightness?: number;
	/** Saturation multiplier. `0` is grayscale, `1` preserves source color. */
	saturation?: number;
	/** Contrast multiplier around mid-gray. */
	contrast?: number;
	/** Warm/cool color balance shift; positive values warm the image. */
	temperature?: number;
	/** Green/magenta tint shift; positive values bias toward magenta. */
	tint?: number;
	/** Allows backend-specific experimental color-filter options. */
	[key: string]: unknown;
}
export const DEFAULT_COLOR_FILTER_OPTIONS: Required<
	Pick<
		ColorFilterOptions,
		"brightness" | "saturation" | "contrast" | "temperature" | "tint"
	>
> = {
	brightness: 0,
	saturation: 1,
	contrast: 1,
	temperature: 0,
	tint: 0,
};
export type WebGPUColorFilterContext = WebGPURuntimePostProcessContext;
export type WebGLColorFilterContext = WebGLScreenPostProcessContext;

/** @internal Software implementation for the built-in color filter pass. */
export class SoftwareColorFilterImplementation
	implements PostProcessPassImplementation<SoftwareBuiltinPostProcessContext>
{
	public readonly id = "color-filter:software";
	public describeExecution() {
		return SOFTWARE_IN_PLACE_EXECUTION;
	}

	public execute(
		request: PostProcessPassRequest,
		_context: SoftwareBuiltinPostProcessContext | undefined
	): PostProcessPassResult {
		const pixels = request.frameContext.attachments.pixels;
		if (!pixels || pixels.length === 0) {
			return { ran: false };
		}
		const options = {
			...DEFAULT_COLOR_FILTER_OPTIONS,
			...((request.options as ColorFilterOptions | undefined) ?? {}),
		};
		const dirtyRects = resolveSoftwareDirtyRects(request.frameContext);
		const brightness = clamp(finiteOr(options.brightness, 0), -1, 1);
		const saturation = clamp(finiteOr(options.saturation, 1), 0, 2);
		const contrast = clamp(finiteOr(options.contrast, 1), 0, 2);
		const temperature = clamp(finiteOr(options.temperature, 0), -1, 1);
		const tint = clamp(finiteOr(options.tint, 0), -1, 1);
		const tempShiftR = temperature * 0.1 + tint * 0.05;
		const tempShiftG = -tint * 0.1;
		const tempShiftB = -temperature * 0.1 + tint * 0.05;
		const width = request.frameContext.attachments.width;
		forEachSoftwareDirtyRect(dirtyRects, (rect) => {
			for (let y = rect.minY; y <= rect.maxY; y++) {
				const row = y * width;
				for (let x = rect.minX; x <= rect.maxX; x++) {
					const index = (row + x) << 2;
					let red = pixels[index] / 255;
					let green = pixels[index + 1] / 255;
					let blue = pixels[index + 2] / 255;

					red += brightness;
					green += brightness;
					blue += brightness;

					const luma = red * 0.2126 + green * 0.7152 + blue * 0.0722;
					red = luma + (red - luma) * saturation;
					green = luma + (green - luma) * saturation;
					blue = luma + (blue - luma) * saturation;

					red = (red - 0.5) * contrast + 0.5;
					green = (green - 0.5) * contrast + 0.5;
					blue = (blue - 0.5) * contrast + 0.5;

					red += tempShiftR;
					green += tempShiftG;
					blue += tempShiftB;

					pixels[index] = Math.round(clamp(red, 0, 1) * 255);
					pixels[index + 1] = Math.round(clamp(green, 0, 1) * 255);
					pixels[index + 2] = Math.round(clamp(blue, 0, 1) * 255);
				}
			}
		});
		return { ran: true };
	}
}
interface WebGPUColorFilterResources {
	shared: WebGPUPostProcessServices;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
	paramData: Float32Array<ArrayBuffer>;
}
/** @internal WebGPU implementation for the built-in color filter pass. */
export class WebGPUColorFilterImplementation
	implements PostProcessPassImplementation<WebGPUColorFilterContext, ColorFilterOptions>
{
	public readonly id = "color-filter:webgpu";
	public describeExecution() {
		return WEBGPU_VERSIONED_EXECUTION;
	}
	private _resources =
		new Map<WebGPUPostProcessServices, WebGPUColorFilterResources>();

	public async warmup(
		context: WebGPUColorFilterContext | undefined
	): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		request: PostProcessPassRequest<ColorFilterOptions>,
		context: WebGPUColorFilterContext | undefined
	): Promise<PostProcessPassResult> {
		if (!context?.encoder || !context.targets) {
			return { ran: false };
		}
		const ran = await this._runColorFilterKernel(request, context);
		return ran ? { ran: true } : { ran: false };
	}

	public invalidate(): void {
		for (const resources of this._resources.values()) {
			resources.shared.invalidateBindingsByPrefix("color-filter-");
		}
	}

	public destroy(): void {
		for (const resources of this._resources.values()) {
			resources.shared.destroyManagedResource(
				resources.pipeline,
				"color filter pipeline"
			);
			resources.shared.destroyManagedResource(
				resources.module,
				"color filter shader module"
			);
			resources.shared.destroyManagedResource(
				resources.params,
				"color filter params buffer"
			);
			resources.shared.invalidateBindingsByPrefix("color-filter-");
			resources.module = null;
			resources.pipeline = null;
			resources.params = null;
		}
		this._resources.clear();
	}

	private async _runColorFilterKernel(
		request: PostProcessPassRequest<ColorFilterOptions>,
		context: WebGPUColorFilterContext
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
		const target = resolveWebGPUTarget(context);
		const input = context.resources.color.input;
		if (!input) return false;
		const options = request.options ?? DEFAULT_COLOR_FILTER_OPTIONS;
		const data = resources.paramData;
		data[0] = clamp(
			finiteOr(options.brightness, DEFAULT_COLOR_FILTER_OPTIONS.brightness),
			-1,
			1
		);
		data[1] = clamp(
			finiteOr(options.saturation, DEFAULT_COLOR_FILTER_OPTIONS.saturation),
			0,
			2
		);
		data[2] = clamp(
			finiteOr(options.contrast, DEFAULT_COLOR_FILTER_OPTIONS.contrast),
			0,
			2
		);
		data[3] = clamp(
			finiteOr(options.temperature, DEFAULT_COLOR_FILTER_OPTIONS.temperature),
			-1,
			1
		);
		data[4] = clamp(
			finiteOr(options.tint, DEFAULT_COLOR_FILTER_OPTIONS.tint),
			-1,
			1
		);
		data[5] = 0;
		data[6] = 0;
		data[7] = 0;
		context.shared.compute.writeBuffer(resources.params, data);
		const binding = context.shared.getCachedBindGroup(
			`color-filter-${target === targets.postPing ? "ping" : "pong"}`,
			resources.pipeline,
			[
				{ binding: 0, resource: input },
				{ binding: 1, resource: context.shared.sampler },
				{ binding: 2, resource: resources.params },
				{ binding: 3, resource: target },
			],
			"WebGPUColorFilter_Binding"
		);
		context.encoder.beginComputePass({ label: "WebGPUColorFilter" });
		context.encoder.setComputePipeline(resources.pipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WEBGPU_WORKGROUP_SIZE),
			ceilDiv(target.height, WEBGPU_WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();
		return true;
	}

	private async _ensureResources(
		shared: WebGPUPostProcessServices
	): Promise<WebGPUColorFilterResources> {
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
			const shader = await ShaderSource.load(
				"webgpu.postprocess.colorFilter.composite"
			);
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUColorFilterShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!resources.pipeline) {
			resources.pipeline = await shared.compute.createComputePipeline({
				label: "WebGPUColorFilterPipeline",
				compute: { module: resources.module, entryPoint: "csMain" },
			});
		}
		if (!resources.params) {
			resources.params = shared.compute.createBuffer({
				label: "WebGPUColorFilterParams",
				size: 8 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		return resources;
	}
}
/** @internal WebGL implementation for the built-in color filter pass. */
export class WebGLColorFilterImplementation
	implements PostProcessPassImplementation<WebGLColorFilterContext, ColorFilterOptions>
{
	public readonly id = "color-filter:webgl";
	public describeExecution() {
		return WEBGL_VERSIONED_EXECUTION;
	}
	private _programCompiler: WebGLProgramCompiler | null = null;
	private _programSlot: WebGLProgramSlot<WebGLColorFilterProgram> | null = null;

	public warmup(context: WebGLColorFilterContext | undefined): void {
		if (context) {
			this._getProgramSlot(context.programCompiler).warmup();
		}
	}

	public execute(
		request: PostProcessPassRequest<ColorFilterOptions>,
		context: WebGLColorFilterContext | undefined
	): PostProcessPassResult {
		const target = resolveWebGLTarget(context);
		if (!target) {
			return { ran: false };
		}
		const options = request.options;
		const brightness = sanitizeFiniteClamped(
			options?.brightness,
			DEFAULT_COLOR_FILTER_OPTIONS.brightness,
			-1,
			1
		);
		const saturation = sanitizeFiniteClamped(
			options?.saturation,
			DEFAULT_COLOR_FILTER_OPTIONS.saturation,
			0,
			2
		);
		const contrast = sanitizeFiniteClamped(
			options?.contrast,
			DEFAULT_COLOR_FILTER_OPTIONS.contrast,
			0,
			2
		);
		const temperature = sanitizeFiniteClamped(
			options?.temperature,
			DEFAULT_COLOR_FILTER_OPTIONS.temperature,
			-1,
			1
		);
		const tint = sanitizeFiniteClamped(
			options?.tint,
			DEFAULT_COLOR_FILTER_OPTIONS.tint,
			-1,
			1
		);

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
		if (program.uniforms.filterParams0) {
			gl.uniform4f(
				program.uniforms.filterParams0,
				brightness,
				saturation,
				contrast,
				temperature
			);
		}
		if (program.uniforms.filterParams1) {
			gl.uniform4f(program.uniforms.filterParams1, tint, 0, 0, 0);
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

	private _getProgramSlot(
		compiler: WebGLProgramCompiler
	): WebGLProgramSlot<WebGLColorFilterProgram> {
		if (this._programCompiler !== compiler) {
			this._programSlot?.destroy();
			this._programCompiler = compiler;
			this._programSlot = compiler.createSlot({
				label: "WebGLColorFilterProgram",
				vertex: () => ShaderSource.get("webgl.part.presentVertex.raw"),
				fragment: () =>
					ShaderSource.get("webgl.part.colorFilterFragment.raw"),
				reflect: (gl, webglProgram) => ({
					program: webglProgram,
					uniforms: {
						sourceMap: gl.getUniformLocation(webglProgram, "uSourceMap"),
						filterParams0: gl.getUniformLocation(
							webglProgram,
							"uFilterParams0"
						),
						filterParams1: gl.getUniformLocation(
							webglProgram,
							"uFilterParams1"
						),
					},
				}),
			});
		}
		return this._programSlot!;
	}
}
export interface ColorFilterPassConfig
	extends Omit<
		PostProcessPassConfig<ColorFilterOptions>,
		| "id"
		| "builtIn"
		| "label"
		| "placement"
		| "order"
		| "implementations"
	> {}

/**
 * Stateful logical color filter pass.
 */
export class ColorFilterPass extends PostProcessPass<
	ColorFilterOptions,
	ColorFilterOptions
> {
	public constructor(config: ColorFilterPassConfig = {}) {
		super({
			...config,
			id: COLOR_FILTER_PASS_ORDER.id,
			schedule: {
				placement: config.schedule?.placement ?? COLOR_FILTER_PASS_ORDER.placement,
				order: config.schedule?.order ?? COLOR_FILTER_PASS_ORDER.order,
				incremental: config.schedule?.incremental ?? COLOR_FILTER_PASS_ORDER.incremental,
			},
			label: "color filter",
			implementations: {
				software: () => new SoftwareColorFilterImplementation(),
				webgpu: () => new WebGPUColorFilterImplementation(),
				webgl: () => new WebGLColorFilterImplementation(),
			},
		});
	}

	public override normalizeOptions(): ColorFilterOptions {
		return {
			...DEFAULT_COLOR_FILTER_OPTIONS,
			...this.getRawOptions(),
		};
	}
}
