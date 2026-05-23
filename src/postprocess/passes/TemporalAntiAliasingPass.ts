import {
	DEFAULT_TAA_OPTIONS,
	defineTransientKey,
	type FrameAttachments,
	type TAAOptions,
} from "../../pipeline/types";
import type { Matrix4 } from "../../maths/Matrix4";
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
import { clamp } from "../../maths/Common";
import {
	ceilDiv,
	clampHistoryToNeighborhoodYCoCg,
	reprojectHistoryUv,
} from "../../maths/Misc";
import { loadPostProcessShaderPartComposite } from "../../shaders/webgpu/shaderSource";
import { PostProcessPass, type PostProcessPassConfig } from "../PostProcessPass";
import type {
	PostProcessHistoryDescriptor,
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessHistorySlots,
	PostProcessPassRequirements,
} from "../types";

const DEFAULT_HISTORY_USAGE = ["sampled", "storage", "render-target"] as const;
const WORKGROUP_SIZE = 8;

export const TAA_HISTORY_WEIGHT_RANGE: [number, number] = [0, 0.99];
export const TAA_DEPTH_THRESHOLD_RANGE: [number, number] = [1e-4, 1];
export const TAA_MOTION_FACTOR_RANGE: [number, number] = [0, 512];
export const TAA_VARIANCE_GAMMA_RANGE: [number, number] = [0, 8];
export const TAA_SHARPEN_RANGE: [number, number] = [0, 2];
export const TAA_JITTER_SCALE_RANGE: [number, number] = [0, 8];

export type ResolvedTAAOptions = Required<
	Pick<
		TAAOptions,
		| "jitterScale"
		| "historyWeight"
		| "disocclusionDepthThreshold"
		| "motionFactor"
		| "varianceClampGamma"
		| "sharpen"
	>
>;

export interface SoftwareTAARenderState {
	readonly currentJitter: [number, number];
	readonly previousJitter: [number, number];
	readonly previousViewProjection: Matrix4 | null;
	readonly currentViewProjection: Matrix4;
	readonly previousWorldMatrices: ReadonlyMap<string, Matrix4>;
	readonly currentWorldMatrices: Map<string, Matrix4>;
}

export const SOFTWARE_TAA_RENDER_STATE_KEY =
	defineTransientKey<SoftwareTAARenderState>("postprocess:taa-software-state");

export interface SoftwareTAAContext {
	readonly attachments: FrameAttachments;
}

export interface WebGPUTAAContext {
	readonly encoder: ICommandEncoder;
	readonly targets: WebGPUFrameTargets;
	readonly shared: PostProcessSharedContext;
	publishColorTarget(texture: IRenderTexture): void;
	writeMotionHistoryFromCurrent(): void | Promise<void>;
}

export interface WebGLTAAContext {
	readonly gl: WebGL2RenderingContext;
	readonly programs: WebGLProgramLibrary;
	readonly fullscreenVao: WebGLVertexArrayObject | null;
	readonly postFramebuffer: WebGLFramebuffer | null;
	readonly sceneColorTexture: WebGLTexture | null;
	readonly sceneMotionTexture: WebGLTexture | null;
	readonly width: number;
	readonly height: number;
	readonly historyRead: WebGLTexture | null;
	readonly historyWrite: WebGLTexture | null;
	readonly motionHistoryRead: WebGLTexture | null;
	readonly motionHistoryWrite: WebGLTexture | null;
	getSourceTexture(): WebGLTexture | null;
	resolveTargetTexture(sourceTexture: WebGLTexture): WebGLTexture | null;
	publishColorTexture(texture: WebGLTexture): void;
	warn(key: string, message: string): void;
}

interface SampledColor {
	r: number;
	g: number;
	b: number;
	a: number;
}

interface WebGPUTAAResources {
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
}

function finiteClamped(
	value: unknown,
	fallback: number,
	min: number,
	max: number
): number {
	const resolved =
		typeof value === "number" && Number.isFinite(value) ? value : fallback;
	return Math.min(max, Math.max(min, resolved));
}

/**
 * Resolves TAA options with the same numeric ranges for every backend.
 *
 * @param options User-provided TAA options.
 * @returns Fully resolved option values.
 * @sideEffects None.
 */
export function resolveTAAOptions(options?: TAAOptions | null): ResolvedTAAOptions {
	return {
		jitterScale: finiteClamped(
			options?.jitterScale,
			DEFAULT_TAA_OPTIONS.jitterScale,
			TAA_JITTER_SCALE_RANGE[0],
			TAA_JITTER_SCALE_RANGE[1]
		),
		historyWeight: finiteClamped(
			options?.historyWeight,
			DEFAULT_TAA_OPTIONS.historyWeight,
			TAA_HISTORY_WEIGHT_RANGE[0],
			TAA_HISTORY_WEIGHT_RANGE[1]
		),
		disocclusionDepthThreshold: finiteClamped(
			options?.disocclusionDepthThreshold,
			DEFAULT_TAA_OPTIONS.disocclusionDepthThreshold,
			TAA_DEPTH_THRESHOLD_RANGE[0],
			TAA_DEPTH_THRESHOLD_RANGE[1]
		),
		motionFactor: finiteClamped(
			options?.motionFactor,
			DEFAULT_TAA_OPTIONS.motionFactor,
			TAA_MOTION_FACTOR_RANGE[0],
			TAA_MOTION_FACTOR_RANGE[1]
		),
		varianceClampGamma: finiteClamped(
			options?.varianceClampGamma,
			DEFAULT_TAA_OPTIONS.varianceClampGamma,
			TAA_VARIANCE_GAMMA_RANGE[0],
			TAA_VARIANCE_GAMMA_RANGE[1]
		),
		sharpen: finiteClamped(
			options?.sharpen,
			DEFAULT_TAA_OPTIONS.sharpen,
			TAA_SHARPEN_RANGE[0],
			TAA_SHARPEN_RANGE[1]
		),
	};
}

/**
 * Resolves whether TAA may sample temporal history this frame.
 *
 * @param histories Current frame history slots.
 * @returns `true` when both TAA color and motion histories are valid.
 * @sideEffects None.
 */
export function resolveTAAHistoryValid(
	histories: PostProcessHistorySlots
): boolean {
	return (histories.taa?.valid ?? false) && (histories.motion?.valid ?? false);
}

/**
 * Creates the packed GPU parameter buffer shared by WebGPU and WebGL TAA.
 *
 * @param width Target width.
 * @param height Target height.
 * @param options User-provided TAA options.
 * @param historyValid Whether temporal history can be sampled.
 * @returns Eight float parameters expected by the TAA kernels.
 * @sideEffects None.
 */
export function createTAAKernelParams(
	width: number,
	height: number,
	options: TAAOptions | ResolvedTAAOptions | undefined,
	historyValid: boolean
): Float32Array {
	const resolved = resolveTAAOptions(options);
	return new Float32Array([
		1 / Math.max(1, width),
		1 / Math.max(1, height),
		resolved.historyWeight,
		resolved.disocclusionDepthThreshold,
		resolved.motionFactor,
		resolved.varianceClampGamma,
		resolved.sharpen,
		historyValid ? 1 : 0,
	]);
}

/**
 * CPU implementation of the cross-backend temporal anti-aliasing pass.
 */
export class SoftwareTemporalAntiAliasingImplementation
	implements PostProcessPassImplementation<SoftwareTAAContext>
{
	public readonly id = "taa:software";

	public execute(
		request: PostProcessPassRequest,
		context: SoftwareTAAContext | undefined
	): PostProcessPassResult {
		if (!context) {
			return { ran: false };
		}
		const ran = this._runTAAKernel(request, context);
		return ran ? { ran: true, updatedHistoryIds: ["taa", "motion"] } : { ran: false };
	}

	private _runTAAKernel(
		request: PostProcessPassRequest,
		context: SoftwareTAAContext
	): boolean {
		const { pixels, motionBuffer, width, height } = context.attachments;
		const historyRead = request.histories.taa?.read.resource as Float32Array | null;
		const historyWrite = request.histories.taa?.write.resource as Float32Array | null;
		const motionHistoryRead = request.histories.motion?.read
			.resource as Float32Array | null;
		const motionHistoryWrite = request.histories.motion?.write
			.resource as Float32Array | null;
		if (
			!pixels ||
			!motionBuffer ||
			!historyRead ||
			!historyWrite ||
			!motionHistoryRead ||
			!motionHistoryWrite ||
			width <= 0 ||
			height <= 0
		) {
			return false;
		}

		const options = resolveTAAOptions(request.options as TAAOptions);
		const historyValid = resolveTAAHistoryValid(request.histories);
		const source = new Uint8ClampedArray(pixels);
		const invW = 1 / Math.max(1, width);
		const invH = 1 / Math.max(1, height);

		for (let y = 0; y < height; y++) {
			for (let x = 0; x < width; x++) {
				const pixelIndex = y * width + x;
				const rgbaIndex = pixelIndex << 2;
				const uv: [number, number] = [
					(x + 0.5) * invW,
					(y + 0.5) * invH,
				];
				const curr = samplePixel(source, width, height, x, y);
				const motionIndex = pixelIndex << 2;
				const motion: [number, number] = [
					motionBuffer[motionIndex] ?? 0,
					motionBuffer[motionIndex + 1] ?? 0,
				];
				const prevUv = reprojectHistoryUv(uv, motion);
				const inside =
					prevUv[0] >= 0 &&
					prevUv[0] <= 1 &&
					prevUv[1] >= 0 &&
					prevUv[1] <= 1;

				let blend = 0;
				let history = curr;
				if (historyValid && inside) {
					const sampledHistory = sampleFloatTexture(
						historyRead,
						width,
						height,
						prevUv[0],
						prevUv[1]
					);
					const neighborhood = collectNeighborhood(source, width, height, x, y);
					const clampedHistory = clampHistoryToNeighborhoodYCoCg(
						[sampledHistory.r, sampledHistory.g, sampledHistory.b],
						neighborhood,
						options.varianceClampGamma
					);
					history = {
						r: clampedHistory[0],
						g: clampedHistory[1],
						b: clampedHistory[2],
						a: sampledHistory.a,
					};

					const currDepth = minCrossDepth(motionBuffer, width, height, x, y);
					const prevDepth = minCrossDepthUv(
						motionHistoryRead,
						width,
						height,
						prevUv[0],
						prevUv[1]
					);
					const depthConfidence = computeDepthConfidence(
						currDepth,
						prevDepth,
						options.disocclusionDepthThreshold
					);
					const previousMotion = sampleFloatTexture(
						motionHistoryRead,
						width,
						height,
						prevUv[0],
						prevUv[1]
					);
					const forwardUv: [number, number] = [
						prevUv[0] + previousMotion.r * 0.5,
						prevUv[1] - previousMotion.g * 0.5,
					];
					const reprojectionErrorPx = Math.max(
						Math.abs(forwardUv[0] - uv[0]) / invW,
						Math.abs(forwardUv[1] - uv[1]) / invH
					);
					const reprojectionConfidence =
						1 - smoothstep(0.75, 3, reprojectionErrorPx);
					const currLuma = luma(curr);
					const histLuma = luma(history);
					const lumaDiff =
						Math.abs(currLuma - histLuma) /
						Math.max(Math.max(currLuma, histLuma), 1e-3);
					const colorConfidence = 1 - smoothstep(0.12, 0.7, lumaDiff);
					const historyConfidence = clamp(
						depthConfidence * reprojectionConfidence * colorConfidence,
						0,
						1
					);
					const motionMag = Math.hypot(motion[0], motion[1]);
					const adaptive = clamp(
						options.historyWeight * Math.exp(-motionMag * options.motionFactor),
						0,
						0.96
					);
					blend = clamp(adaptive * historyConfidence, 0, 0.96);
				}

				const temporal = mixColor(curr, history, blend);
				const blur = averageCross(source, width, height, x, y);
				const sharpen = Math.max(options.sharpen, 0) * (1 - blend * 0.5);
				const out = {
					r: Math.max(0, temporal.r + (temporal.r - blur.r) * sharpen),
					g: Math.max(0, temporal.g + (temporal.g - blur.g) * sharpen),
					b: Math.max(0, temporal.b + (temporal.b - blur.b) * sharpen),
					a: temporal.a,
				};
				pixels[rgbaIndex] = Math.round(clamp(out.r, 0, 1) * 255);
				pixels[rgbaIndex + 1] = Math.round(clamp(out.g, 0, 1) * 255);
				pixels[rgbaIndex + 2] = Math.round(clamp(out.b, 0, 1) * 255);
				pixels[rgbaIndex + 3] = Math.round(clamp(out.a, 0, 1) * 255);
				historyWrite[rgbaIndex] = temporal.r;
				historyWrite[rgbaIndex + 1] = temporal.g;
				historyWrite[rgbaIndex + 2] = temporal.b;
				historyWrite[rgbaIndex + 3] = temporal.a;
				motionHistoryWrite[rgbaIndex] = motionBuffer[motionIndex] ?? 0;
				motionHistoryWrite[rgbaIndex + 1] = motionBuffer[motionIndex + 1] ?? 0;
				motionHistoryWrite[rgbaIndex + 2] = motionBuffer[motionIndex + 2] ?? 0;
				motionHistoryWrite[rgbaIndex + 3] = motionBuffer[motionIndex + 3] ?? 0;
			}
		}
		return true;
	}
}

/**
 * WebGPU implementation of the cross-backend temporal anti-aliasing pass.
 */
export class WebGPUTemporalAntiAliasingImplementation
	implements PostProcessPassImplementation<WebGPUTAAContext>
{
	public readonly id = "taa:webgpu";
	private _resources = new WeakMap<PostProcessSharedContext, WebGPUTAAResources>();

	public async execute(
		request: PostProcessPassRequest,
		context: WebGPUTAAContext | undefined
	): Promise<PostProcessPassResult> {
		if (!context) {
			return { ran: false };
		}
		const ran = await this._runTAAKernel(request, context);
		if (!ran) {
			return { ran: false };
		}
		await context.writeMotionHistoryFromCurrent();
		return { ran: true, updatedHistoryIds: ["taa", "motion"] };
	}

	private async _runTAAKernel(
		request: PostProcessPassRequest,
		context: WebGPUTAAContext
	): Promise<boolean> {
		const resources = await this._ensureResources(context.shared);
		if (!context.shared.sampler || !resources.pipeline || !resources.params) {
			return false;
		}

		const targets = context.targets;
		const target =
			targets.sceneColor === targets.postPong ? targets.postPing : targets.postPong;
		const params = createTAAKernelParams(
			target.width,
			target.height,
			request.options as TAAOptions,
			resolveTAAHistoryValid(request.histories)
		);
		context.shared.compute.writeBuffer(
			resources.params,
			params as unknown as BufferSource
		);

		const binding = context.shared.getCachedBindGroup(
			`taa-${target === targets.postPing ? "ping" : "pong"}`,
			resources.pipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.historyRead },
				{ binding: 2, resource: targets.gMotionDepth },
				{ binding: 3, resource: targets.motionHistoryRead },
				{ binding: 4, resource: context.shared.sampler },
				{ binding: 5, resource: resources.params },
				{ binding: 6, resource: target },
				{ binding: 7, resource: targets.historyWrite },
			],
			"WebGPUTAA_Binding"
		);

		context.encoder.beginComputePass({ label: "WebGPUTAA" });
		context.encoder.setComputePipeline(resources.pipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WORKGROUP_SIZE),
			ceilDiv(target.height, WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();
		context.publishColorTarget(target);
		return true;
	}

	private async _ensureResources(
		shared: PostProcessSharedContext
	): Promise<WebGPUTAAResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = { module: null, pipeline: null, params: null };
			this._resources.set(shared, resources);
		}
		await shared.ensureCommonResources();
		if (!resources.module) {
			const shader = await loadPostProcessShaderPartComposite("taa");
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUTAAShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!resources.pipeline) {
			resources.pipeline = shared.compute.createComputePipeline({
				label: "WebGPUTAAPipeline",
				compute: { module: resources.module, entryPoint: "csMain" },
			});
		}
		if (!resources.params) {
			resources.params = shared.compute.createBuffer({
				label: "WebGPUTAAParams",
				size: 8 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		return resources;
	}
}

/**
 * WebGL implementation of the cross-backend temporal anti-aliasing pass.
 */
export class WebGLTemporalAntiAliasingImplementation
	implements PostProcessPassImplementation<WebGLTAAContext>
{
	public readonly id = "taa:webgl";

	public execute(
		request: PostProcessPassRequest,
		context: WebGLTAAContext | undefined
	): PostProcessPassResult {
		if (!context) {
			return { ran: false };
		}
		return this._runTAAKernel(request, context);
	}

	private _runTAAKernel(
		request: PostProcessPassRequest,
		context: WebGLTAAContext
	): PostProcessPassResult {
		const gl = context.gl;
		const maxDrawBuffers = Number(gl.getParameter(gl.MAX_DRAW_BUFFERS) ?? 4);
		if (maxDrawBuffers < 3) {
			context.warn(
				"webgl-taa-mrt-unsupported",
				"WebGL TAA requires at least three draw buffers."
			);
			return { ran: false };
		}
		if (
			!context.sceneMotionTexture ||
			!context.postFramebuffer ||
			!context.fullscreenVao ||
			!context.historyRead ||
			!context.historyWrite ||
			!context.motionHistoryRead ||
			!context.motionHistoryWrite
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

		const options = resolveTAAOptions(request.options as TAAOptions);
		const program = context.programs.getTAAProgram();
		gl.bindFramebuffer(gl.FRAMEBUFFER, context.postFramebuffer);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT0,
			gl.TEXTURE_2D,
			targetTexture,
			0
		);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT1,
			gl.TEXTURE_2D,
			context.historyWrite,
			0
		);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT2,
			gl.TEXTURE_2D,
			context.motionHistoryWrite,
			0
		);
		gl.drawBuffers([
			gl.COLOR_ATTACHMENT0,
			gl.COLOR_ATTACHMENT1,
			gl.COLOR_ATTACHMENT2,
		]);

		gl.viewport(0, 0, context.width, context.height);
		gl.useProgram(program.program);
		gl.bindVertexArray(context.fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);

		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, context.historyRead);
		gl.activeTexture(gl.TEXTURE2);
		gl.bindTexture(gl.TEXTURE_2D, context.sceneMotionTexture);
		gl.activeTexture(gl.TEXTURE3);
		gl.bindTexture(gl.TEXTURE_2D, context.motionHistoryRead);

		const uniforms = program.uniforms;
		if (uniforms.sceneColor) gl.uniform1i(uniforms.sceneColor, 0);
		if (uniforms.historyMap) gl.uniform1i(uniforms.historyMap, 1);
		if (uniforms.motionMap) gl.uniform1i(uniforms.motionMap, 2);
		if (uniforms.motionHistory) gl.uniform1i(uniforms.motionHistory, 3);
		if (uniforms.texelSize) {
			gl.uniform2f(
				uniforms.texelSize,
				1 / Math.max(1, context.width),
				1 / Math.max(1, context.height)
			);
		}
		if (uniforms.historyWeight) {
			gl.uniform1f(uniforms.historyWeight, options.historyWeight);
		}
		if (uniforms.depthThreshold) {
			gl.uniform1f(
				uniforms.depthThreshold,
				options.disocclusionDepthThreshold
			);
		}
		if (uniforms.motionFactor) {
			gl.uniform1f(uniforms.motionFactor, options.motionFactor);
		}
		if (uniforms.varianceClampGamma) {
			gl.uniform1f(
				uniforms.varianceClampGamma,
				options.varianceClampGamma
			);
		}
		if (uniforms.sharpen) {
			gl.uniform1f(uniforms.sharpen, options.sharpen);
		}
		if (uniforms.historyValid) {
			gl.uniform1f(
				uniforms.historyValid,
				resolveTAAHistoryValid(request.histories) ? 1 : 0
			);
		}

		gl.drawArrays(gl.TRIANGLES, 0, 3);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT1,
			gl.TEXTURE_2D,
			null,
			0
		);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT2,
			gl.TEXTURE_2D,
			null,
			0
		);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
		gl.bindVertexArray(null);
		context.publishColorTexture(targetTexture);
		return { ran: true, updatedHistoryIds: ["taa", "motion"] };
	}
}

export interface TemporalAntiAliasingPassConfig
	extends Omit<
		PostProcessPassConfig<TAAOptions>,
		"id" | "placement" | "implementations"
	> {}

/**
 * Stateful logical temporal anti-aliasing pass shared by all rendering backends.
 */
export class TemporalAntiAliasingPass extends PostProcessPass<
	TAAOptions,
	ResolvedTAAOptions
> {
	public constructor(config: TemporalAntiAliasingPassConfig = {}) {
		super({
			...config,
			id: "taa",
			placement: "temporal",
			implementations: {
				software: new SoftwareTemporalAntiAliasingImplementation(),
				webgpu: new WebGPUTemporalAntiAliasingImplementation(),
				webgl: new WebGLTemporalAntiAliasingImplementation(),
			},
		});
	}

	public override normalizeOptions(): ResolvedTAAOptions {
		return resolveTAAOptions(this.getRawOptions());
	}

	public override getRequirements(): PostProcessPassRequirements {
		return { gBuffer: ["motion"] };
	}

	public override getHistoryDescriptors(): readonly PostProcessHistoryDescriptor[] {
		return [
			{ id: "taa", usage: DEFAULT_HISTORY_USAGE },
			{ id: "motion", usage: ["sampled", "copy-dst", "render-target"] },
		];
	}
}

function samplePixel(
	pixels: Uint8ClampedArray,
	width: number,
	height: number,
	x: number,
	y: number
): SampledColor {
	const sx = Math.max(0, Math.min(width - 1, x));
	const sy = Math.max(0, Math.min(height - 1, y));
	const index = (sy * width + sx) << 2;
	return {
		r: pixels[index] / 255,
		g: pixels[index + 1] / 255,
		b: pixels[index + 2] / 255,
		a: pixels[index + 3] / 255,
	};
}

function sampleFloatTexture(
	data: Float32Array,
	width: number,
	height: number,
	u: number,
	v: number
): SampledColor {
	const x = clamp(u, 0, 1) * width - 0.5;
	const y = clamp(v, 0, 1) * height - 0.5;
	const x0 = Math.max(0, Math.min(width - 1, Math.floor(x)));
	const y0 = Math.max(0, Math.min(height - 1, Math.floor(y)));
	const x1 = Math.max(0, Math.min(width - 1, x0 + 1));
	const y1 = Math.max(0, Math.min(height - 1, y0 + 1));
	const tx = clamp(x - x0, 0, 1);
	const ty = clamp(y - y0, 0, 1);
	const c00 = sampleFloatPixel(data, width, x0, y0);
	const c10 = sampleFloatPixel(data, width, x1, y0);
	const c01 = sampleFloatPixel(data, width, x0, y1);
	const c11 = sampleFloatPixel(data, width, x1, y1);
	return mixColor(mixColor(c00, c10, tx), mixColor(c01, c11, tx), ty);
}

function sampleFloatPixel(
	data: Float32Array,
	width: number,
	x: number,
	y: number
): SampledColor {
	const index = (y * width + x) << 2;
	return {
		r: data[index] ?? 0,
		g: data[index + 1] ?? 0,
		b: data[index + 2] ?? 0,
		a: data[index + 3] ?? 1,
	};
}

function collectNeighborhood(
	pixels: Uint8ClampedArray,
	width: number,
	height: number,
	x: number,
	y: number
): [number, number, number][] {
	return [
		sampleRgb(pixels, width, height, x, y),
		sampleRgb(pixels, width, height, x - 1, y),
		sampleRgb(pixels, width, height, x + 1, y),
		sampleRgb(pixels, width, height, x, y - 1),
		sampleRgb(pixels, width, height, x, y + 1),
	];
}

function sampleRgb(
	pixels: Uint8ClampedArray,
	width: number,
	height: number,
	x: number,
	y: number
): [number, number, number] {
	const c = samplePixel(pixels, width, height, x, y);
	return [c.r, c.g, c.b];
}

function averageCross(
	pixels: Uint8ClampedArray,
	width: number,
	height: number,
	x: number,
	y: number
): SampledColor {
	const left = samplePixel(pixels, width, height, x - 1, y);
	const right = samplePixel(pixels, width, height, x + 1, y);
	const up = samplePixel(pixels, width, height, x, y - 1);
	const down = samplePixel(pixels, width, height, x, y + 1);
	return {
		r: (left.r + right.r + up.r + down.r) * 0.25,
		g: (left.g + right.g + up.g + down.g) * 0.25,
		b: (left.b + right.b + up.b + down.b) * 0.25,
		a: (left.a + right.a + up.a + down.a) * 0.25,
	};
}

function minCrossDepth(
	motionDepth: Float32Array,
	width: number,
	height: number,
	x: number,
	y: number
): number {
	let depth = sampleMotionDepth(motionDepth, width, height, x, y);
	depth = minPositive(depth, sampleMotionDepth(motionDepth, width, height, x + 1, y));
	depth = minPositive(depth, sampleMotionDepth(motionDepth, width, height, x - 1, y));
	depth = minPositive(depth, sampleMotionDepth(motionDepth, width, height, x, y + 1));
	depth = minPositive(depth, sampleMotionDepth(motionDepth, width, height, x, y - 1));
	return depth;
}

function minCrossDepthUv(
	motionDepth: Float32Array,
	width: number,
	height: number,
	u: number,
	v: number
): number {
	const x = Math.floor(clamp(u, 0, 1) * width);
	const y = Math.floor(clamp(v, 0, 1) * height);
	return minCrossDepth(motionDepth, width, height, x, y);
}

function sampleMotionDepth(
	motionDepth: Float32Array,
	width: number,
	height: number,
	x: number,
	y: number
): number {
	const sx = Math.max(0, Math.min(width - 1, x));
	const sy = Math.max(0, Math.min(height - 1, y));
	return motionDepth[((sy * width + sx) << 2) + 2] ?? 0;
}

function minPositive(a: number, b: number): number {
	if (a <= 0) return Math.max(0, b);
	if (b <= 0) return Math.max(0, a);
	return Math.min(a, b);
}

function computeDepthConfidence(
	currentDepth: number,
	previousDepth: number,
	threshold: number
): number {
	if (!(currentDepth > 0) || !(previousDepth > 0)) {
		return 0;
	}
	const relativeDiff =
		Math.abs(currentDepth - previousDepth) /
		Math.max(Math.max(currentDepth, previousDepth), 1e-4);
	const safeThreshold = Math.max(threshold, 1e-4);
	return 1 - smoothstep(safeThreshold * 0.5, safeThreshold * 2.5, relativeDiff);
}

function smoothstep(edge0: number, edge1: number, value: number): number {
	const t = clamp((value - edge0) / Math.max(edge1 - edge0, 1e-6), 0, 1);
	return t * t * (3 - 2 * t);
}

function luma(color: SampledColor): number {
	return color.r * 0.2126 + color.g * 0.7152 + color.b * 0.0722;
}

function mixColor(a: SampledColor, b: SampledColor, t: number): SampledColor {
	const s = clamp(t, 0, 1);
	const inv = 1 - s;
	return {
		r: a.r * inv + b.r * s,
		g: a.g * inv + b.g * s,
		b: a.b * inv + b.b * s,
		a: a.a * inv + b.a * s,
	};
}
