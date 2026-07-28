import {
	PostProcessPass,
	type PostProcessPassConfig,
	type PostProcessPassRequest,
	type PostProcessPassResult,
	type PostProcessPassImplementation,
} from "../../src/index";

import {
	BufferUsage,
	type IComputePipeline,
	type IRenderBuffer,
	type IShaderModule,
} from "../../src/backends/types";

import type {
	WebGPUPostProcessServices,
} from "../../src/backends/webgpu/WebGPUPostProcessContracts";

import type {
	WebGLProgramCompiler,
	WebGLProgramSlot,
} from "../../src/backends/webgl/WebGLProgramCompiler";

import { ShaderSource } from "../../src/shaders/ShaderSource";

import {
	bindWebGLPostTarget,
	forEachSoftwareDirtyRect,
	resolveSoftwareDirtyRects,
	resolveWebGLTarget,
	resolveWebGPUTarget,
	type SoftwareBuiltinPostProcessContext,
	type WebGLScreenPostProcessContext,
	type WebGPURuntimePostProcessContext,
} from "../../src/postprocess/passes/ScreenPassShared";

import { ceilDiv } from "../../src/maths/Misc";
import { WEBGPU_2D_COMPUTE_WORKGROUP_SIZE } from "../../src/backends/webgpu/constants";
import {
	SOFTWARE_IN_PLACE_EXECUTION,
	WEBGL_VERSIONED_EXECUTION,
	WEBGPU_VERSIONED_EXECUTION,
} from "../../src/postprocess/executionDeclarations";

// Import raw shaders via Vite ?raw queries
import blurShaderWGSL from "./shaders/blur.wgsl?raw";
import blurShaderGLSL from "./shaders/blur.glsl?raw";

export const CUSTOM_GAUSSIAN_BLUR_PASS_ID = "custom-gaussian-blur";

export interface CustomGaussianBlurOptions {
	/** Blur radius, typically 1 to 5. Higher values are more expensive. */
	radius?: number;
	/** Standard deviation (sigma) for the Gaussian function. */
	sigma?: number;
	[key: string]: unknown;
}

export const DEFAULT_CUSTOM_GAUSSIAN_BLUR_OPTIONS: Required<
	Pick<CustomGaussianBlurOptions, "radius" | "sigma">
> = {
	radius: 3,
	sigma: 2.0,
};

// -----------------------------------------------------------------------------
// Software (CPU) Implementation
// -----------------------------------------------------------------------------
export class SoftwareCustomGaussianBlurImplementation
	implements PostProcessPassImplementation<SoftwareBuiltinPostProcessContext, CustomGaussianBlurOptions>
{
	public readonly id = "custom-gaussian-blur:software";

	public describeExecution() {
		return SOFTWARE_IN_PLACE_EXECUTION;
	}

	public execute(
		request: PostProcessPassRequest<CustomGaussianBlurOptions>,
		_context: SoftwareBuiltinPostProcessContext | undefined
	): PostProcessPassResult {
		const pixels = request.frameContext.attachments.pixels;
		if (!pixels || pixels.length === 0) {
			return { ran: false };
		}

		const options = {
			...DEFAULT_CUSTOM_GAUSSIAN_BLUR_OPTIONS,
			...request.options,
		};

		const radius = Math.max(1, Math.min(Math.round(options.radius), 5));
		const sigma = Math.max(0.1, options.sigma);
		const width = request.frameContext.attachments.width;
		const height = request.frameContext.attachments.height;

		const dirtyRects = resolveSoftwareDirtyRects(request.frameContext);

		// Precompute 2D Gaussian Kernel weights
		const kernelSize = radius * 2 + 1;
		const weights = new Float32Array(kernelSize * kernelSize);
		let weightSum = 0;
		let wIdx = 0;
		for (let dy = -radius; dy <= radius; dy++) {
			for (let dx = -radius; dx <= radius; dx++) {
				const weight = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
				weights[wIdx++] = weight;
				weightSum += weight;
			}
		}
		for (let i = 0; i < weights.length; i++) {
			weights[i] /= weightSum;
		}

		// Make a copy of the source pixels to sample from
		const srcPixels = new Uint8ClampedArray(pixels);

		forEachSoftwareDirtyRect(dirtyRects, (rect) => {
			for (let y = rect.minY; y <= rect.maxY; y++) {
				const rowOffset = y * width;
				for (let x = rect.minX; x <= rect.maxX; x++) {
					const pixelIdx = (rowOffset + x) << 2;

					let r = 0, g = 0, b = 0;
					let wIdx = 0;
					for (let dy = -radius; dy <= radius; dy++) {
						const py = Math.max(0, Math.min(y + dy, height - 1));
						const pyOffset = py * width;
						for (let dx = -radius; dx <= radius; dx++) {
							const px = Math.max(0, Math.min(x + dx, width - 1));
							const srcIdx = (pyOffset + px) << 2;
							const weight = weights[wIdx++];
							r += srcPixels[srcIdx] * weight;
							g += srcPixels[srcIdx + 1] * weight;
							b += srcPixels[srcIdx + 2] * weight;
						}
					}

					pixels[pixelIdx] = Math.round(r);
					pixels[pixelIdx + 1] = Math.round(g);
					pixels[pixelIdx + 2] = Math.round(b);
					// Retain alpha channel from source
					pixels[pixelIdx + 3] = srcPixels[pixelIdx + 3];
				}
			}
		});

		return { ran: true };
	}
}

// -----------------------------------------------------------------------------
// WebGPU Implementation
// -----------------------------------------------------------------------------
interface WebGPUMapResources {
	shared: WebGPUPostProcessServices;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
	paramData: Float32Array<ArrayBuffer>;
}

export class WebGPUCustomGaussianBlurImplementation
	implements PostProcessPassImplementation<WebGPURuntimePostProcessContext, CustomGaussianBlurOptions>
{
	public readonly id = "custom-gaussian-blur:webgpu";

	public describeExecution() {
		return WEBGPU_VERSIONED_EXECUTION;
	}

	private _resources = new Map<WebGPUPostProcessServices, WebGPUMapResources>();

	public async warmup(
		context: WebGPURuntimePostProcessContext | undefined
	): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		request: PostProcessPassRequest<CustomGaussianBlurOptions>,
		context: WebGPURuntimePostProcessContext | undefined
	): Promise<PostProcessPassResult> {
		if (!context?.encoder || !context.targets) {
			return { ran: false };
		}
		const ran = await this._runBlurKernel(request, context);
		return ran ? { ran: true } : { ran: false };
	}

	public invalidate(): void {
		for (const resources of this._resources.values()) {
			resources.shared.invalidateBindingsByPrefix("custom-blur-");
		}
	}

	public destroy(): void {
		for (const resources of this._resources.values()) {
			resources.shared.destroyManagedResource(
				resources.pipeline,
				"custom-blur pipeline"
			);
			resources.shared.destroyManagedResource(
				resources.module,
				"custom-blur shader module"
			);
			resources.shared.destroyManagedResource(
				resources.params,
				"custom-blur params buffer"
			);
			resources.shared.invalidateBindingsByPrefix("custom-blur-");
			resources.module = null;
			resources.pipeline = null;
			resources.params = null;
		}
		this._resources.clear();
	}

	private async _runBlurKernel(
		request: PostProcessPassRequest<CustomGaussianBlurOptions>,
		context: WebGPURuntimePostProcessContext
	): Promise<boolean> {
		const resources = await this._ensureResources(context.shared);
		if (
			!context.encoder ||
			!context.targets ||
			!resources.pipeline ||
			!resources.params
		) {
			return false;
		}

		const targets = context.targets;
		const target = resolveWebGPUTarget(context);
		const input = context.resources.color.input;
		if (!input) {
			return false;
		}

		const options = {
			...DEFAULT_CUSTOM_GAUSSIAN_BLUR_OPTIONS,
			...request.options,
		};

		// Fill uniforms
		const data = resources.paramData;
		data[0] = Math.max(1, Math.min(Math.round(options.radius), 5));
		data[1] = Math.max(0.1, options.sigma);
		data[2] = target.width;
		data[3] = target.height;

		context.shared.compute.writeBuffer(resources.params, data);

		const binding = context.shared.getCachedBindGroup(
			`custom-blur-${target === targets.postPing ? "ping" : "pong"}`,
			resources.pipeline,
			[
				{ binding: 0, resource: input },
				{ binding: 1, resource: resources.params },
				{ binding: 2, resource: target },
			],
			"WebGPUCustomBlur_Binding"
		);

		context.encoder.beginComputePass({ label: "WebGPUCustomGaussianBlur" });
		context.encoder.setComputePipeline(resources.pipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WEBGPU_2D_COMPUTE_WORKGROUP_SIZE),
			ceilDiv(target.height, WEBGPU_2D_COMPUTE_WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();

		return true;
	}

	private async _ensureResources(
		shared: WebGPUPostProcessServices
	): Promise<WebGPUMapResources> {
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

		await shared.ensureCommonResources();

		if (!resources.module) {
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUCustomGaussianBlurShader",
				code: blurShaderWGSL,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}

		if (!resources.pipeline) {
			resources.pipeline = await shared.compute.createComputePipeline({
				label: "WebGPUCustomGaussianBlurPipeline",
				compute: { module: resources.module, entryPoint: "csMain" },
			});
		}

		if (!resources.params) {
			resources.params = shared.compute.createBuffer({
				label: "WebGPUCustomGaussianBlurParams",
				size: 16, // 4 floats * 4 bytes
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}

		return resources;
	}
}

// -----------------------------------------------------------------------------
// WebGL Implementation
// -----------------------------------------------------------------------------
interface WebGLBlurProgram {
	readonly program: WebGLProgram;
	readonly uniforms: {
		readonly sourceMap: WebGLUniformLocation | null;
		readonly blurParams: WebGLUniformLocation | null;
	};
}

export class WebGLCustomGaussianBlurImplementation
	implements PostProcessPassImplementation<WebGLScreenPostProcessContext, CustomGaussianBlurOptions>
{
	public readonly id = "custom-gaussian-blur:webgl";

	public describeExecution() {
		return WEBGL_VERSIONED_EXECUTION;
	}

	private _programCompiler: WebGLProgramCompiler | null = null;
	private _programSlot: WebGLProgramSlot<WebGLBlurProgram> | null = null;

	public warmup(context: WebGLScreenPostProcessContext | undefined): void {
		if (context) {
			this._getProgramSlot(context.programCompiler).warmup();
		}
	}

	public execute(
		request: PostProcessPassRequest<CustomGaussianBlurOptions>,
		context: WebGLScreenPostProcessContext | undefined
	): PostProcessPassResult {
		const target = resolveWebGLTarget(context);
		if (!target) {
			return { ran: false };
		}

		const options = {
			...DEFAULT_CUSTOM_GAUSSIAN_BLUR_OPTIONS,
			...request.options,
		};

		const radius = Math.max(1, Math.min(Math.round(options.radius), 5));
		const sigma = Math.max(0.1, options.sigma);

		const gl = context.gl;
		const program = this._getProgramSlot(context.programCompiler).tryGet();
		if (!program) {
			return { ran: false };
		}

		bindWebGLPostTarget(context, program.program, target.texture);

		// Bind Source Texture to Unit 0
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, target.source);
		if (program.uniforms.sourceMap) {
			gl.uniform1i(program.uniforms.sourceMap, 0);
		}

		// Write Uniforms
		if (program.uniforms.blurParams) {
			gl.uniform4f(
				program.uniforms.blurParams,
				radius,
				sigma,
				context.width,
				context.height
			);
		}

		context.drawFullscreen();

		// Cleanup bindings
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
	): WebGLProgramSlot<WebGLBlurProgram> {
		if (this._programCompiler !== compiler) {
			this._programSlot?.destroy();
			this._programCompiler = compiler;
			this._programSlot = compiler.createSlot({
				label: "WebGLCustomGaussianBlurProgram",
				vertex: () => ShaderSource.get("webgl.part.presentVertex.raw"),
				fragment: () => blurShaderGLSL,
				reflect: (gl, webglProgram) => ({
					program: webglProgram,
					uniforms: {
						sourceMap: gl.getUniformLocation(webglProgram, "uSourceMap"),
						blurParams: gl.getUniformLocation(webglProgram, "uBlurParams"),
					},
				}),
			});
		}
		return this._programSlot!;
	}
}

// -----------------------------------------------------------------------------
// Custom Post Process Pass Definition
// -----------------------------------------------------------------------------
export class CustomGaussianBlurPass extends PostProcessPass<
	CustomGaussianBlurOptions,
	CustomGaussianBlurOptions
> {
	public constructor(
		config: Omit<
			PostProcessPassConfig<CustomGaussianBlurOptions>,
			| "id"
			| "builtIn"
			| "label"
			| "schedule"
			| "implementations"
		> = {}
	) {
		super({
			...config,
			id: CUSTOM_GAUSSIAN_BLUR_PASS_ID,
			schedule: { placement: "present", order: 850 },
			builtIn: false,
			label: "custom gaussian blur",
			implementations: {
				software: () => new SoftwareCustomGaussianBlurImplementation(),
				webgpu: () => new WebGPUCustomGaussianBlurImplementation(),
				webgl: () => new WebGLCustomGaussianBlurImplementation(),
			},
		});
	}

	public override normalizeOptions(): CustomGaussianBlurOptions {
		return {
			...DEFAULT_CUSTOM_GAUSSIAN_BLUR_OPTIONS,
			...this.getRawOptions(),
		};
	}
}
