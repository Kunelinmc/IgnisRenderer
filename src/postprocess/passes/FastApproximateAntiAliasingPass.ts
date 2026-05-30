import type { FrameAttachments, FrameContext } from "../../pipeline/types";
import type { ICommandEncoder } from "../../renderers/ICommandEncoder";
import {
	FXAA_EDGE_THRESHOLD_MIN,
	FXAA_EDGE_THRESHOLD_MULTIPLIER,
	FXAA_QUALITY,
	FXAA_SUBPIX_QUALITY,
} from "../../renderers/constants";
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
import type { PostProcessSharedContext } from "../../renderers/webgpu/postprocess/PostProcessSharedContext";
import type { WebGLProgramLibrary } from "../../renderers/webgl/WebGLProgramLibrary";
import { clamp } from "../../maths/Common";
import { ceilDiv } from "../../maths/Misc";
import { loadPostProcessShaderPartComposite } from "../../shaders/webgpu/shaderSource";
import { PostProcessPass, type PostProcessPassConfig } from "../PostProcessPass";
import type {
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassResult,
} from "../types";

export const FAST_APPROXIMATE_ANTI_ALIASING_PASS_ID = "fxaa";

interface IncrementalDirtyRect {
	minX: number;
	minY: number;
	maxX: number;
	maxY: number;
}

interface SampledByteColor {
	r: number;
	g: number;
	b: number;
	a: number;
}

export interface SoftwareFXAAContext {
	readonly attachments: FrameAttachments;
	readonly canvasContext: CanvasRenderingContext2D | null;
}

export interface WebGPUFXAAContext {
	readonly encoder?: ICommandEncoder;
	readonly targets?: WebGPUPostProcessFrameTargets;
	readonly shared: PostProcessSharedContext;
	publishColorTarget?(texture: IRenderTexture): void;
}

export interface WebGLFXAAContext {
	readonly gl: WebGL2RenderingContext;
	readonly programs: WebGLProgramLibrary;
	readonly fullscreenVao: WebGLVertexArrayObject | null;
	readonly postFramebuffer: WebGLFramebuffer | null;
	readonly sceneColorTexture: WebGLTexture | null;
	readonly width: number;
	readonly height: number;
	getSourceTexture(): WebGLTexture | null;
	resolveTargetTexture(sourceTexture: WebGLTexture): WebGLTexture | null;
	bindColorTarget(texture: WebGLTexture): void;
	drawFullscreen(): void;
	publishColorTexture(texture: WebGLTexture): void;
}

interface WebGPUFXAAResources {
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
}

/**
 * Creates the packed WebGPU FXAA parameter buffer.
 *
 * @param width Target width.
 * @param height Target height.
 * @returns Six float parameters expected by the FXAA compute shader.
 * @sideEffects None.
 */
export function createFXAAKernelParams(width: number, height: number): Float32Array {
	return new Float32Array([
		1 / Math.max(width, 1),
		1 / Math.max(height, 1),
		FXAA_EDGE_THRESHOLD_MIN,
		FXAA_EDGE_THRESHOLD_MULTIPLIER,
		FXAA_SUBPIX_QUALITY,
		0,
	]);
}

/**
 * CPU implementation of the cross-backend FXAA pass.
 */
export class SoftwareFastApproximateAntiAliasingImplementation
	implements PostProcessPassImplementation<SoftwareFXAAContext>
{
	public readonly id = "fxaa:software";
	private _output: Uint8ClampedArray | null = null;
	private _luma: Float32Array | null = null;

	public execute(
		request: PostProcessPassRequest,
		context: SoftwareFXAAContext | undefined
	): PostProcessPassResult {
		if (!context) {
			return { ran: false };
		}
		return this._runFXAAKernel(request.frameContext, context);
	}

	private _runFXAAKernel(
		frameContext: FrameContext,
		context: SoftwareFXAAContext
	): PostProcessPassResult {
		const { width, height } = context.attachments;
		let pixels = context.attachments.pixels;
		let imageData: ImageData | null = null;
		if (!pixels) {
			if (!context.canvasContext) {
				return { ran: false };
			}
			imageData = context.canvasContext.getImageData(0, 0, width, height);
			pixels = imageData.data;
		}
		if (width <= 0 || height <= 0 || pixels.length === 0) {
			return { ran: false };
		}

		if (!this._output || this._output.length !== pixels.length) {
			this._output = new Uint8ClampedArray(pixels.length);
		}
		const output = this._output;
		output.set(pixels);
		const dirtyRects = resolveDirtyRects(frameContext);
		if (dirtyRects.length === 0) {
			return { ran: false };
		}

		const lumaSize = width * height;
		if (!this._luma || this._luma.length !== lumaSize) {
			this._luma = new Float32Array(lumaSize);
		}
		const luma = this._luma;

		for (let i = 0, len = pixels.length; i < len; i += 4) {
			const r = pixels[i] / 255;
			const g = pixels[i + 1] / 255;
			const b = pixels[i + 2] / 255;
			luma[i >> 2] = Math.sqrt(0.2126 * r + 0.7152 * g + 0.0722 * b);
		}

		const outCol: SampledByteColor = { r: 0, g: 0, b: 0, a: 0 };
		forEachDirtyRect(dirtyRects, (rect) => {
			for (let y = rect.minY; y <= rect.maxY; y++) {
				const row = y * width;
				for (let x = rect.minX; x <= rect.maxX; x++) {
					const i = row + x;
					const idx = i << 2;
					const l = luma[i];
					const ln = y > 0 ? luma[i - width] : l;
					const ls = y < height - 1 ? luma[i + width] : l;
					const le = x < width - 1 ? luma[i + 1] : l;
					const lw = x > 0 ? luma[i - 1] : l;
					const lMin = Math.min(l, ln, ls, le, lw);
					const lMax = Math.max(l, ln, ls, le, lw);
					const lRange = lMax - lMin;

					if (
						lRange <
						Math.max(FXAA_EDGE_THRESHOLD_MIN, lMax * FXAA_EDGE_THRESHOLD_MULTIPLIER)
					) {
						output[idx] = pixels[idx];
						output[idx + 1] = pixels[idx + 1];
						output[idx + 2] = pixels[idx + 2];
						output[idx + 3] = pixels[idx + 3];
						continue;
					}

					const lnw = y > 0 && x > 0 ? luma[i - width - 1] : ln;
					const lne = y > 0 && x < width - 1 ? luma[i - width + 1] : ln;
					const lsw = y < height - 1 && x > 0 ? luma[i + width - 1] : ls;
					const lse =
						y < height - 1 && x < width - 1 ? luma[i + width + 1] : ls;

					let lFiltered = 2 * (ln + ls + le + lw);
					lFiltered += lne + lnw + lse + lsw;
					lFiltered /= 12;
					const subpixOffset1 = Math.abs(lFiltered - l);
					const subpixOffset2 = clamp(subpixOffset1 / Math.max(lRange, 1e-4), 0, 1);
					const subpixOffset3 =
						(-2 * subpixOffset2 + 3) * subpixOffset2 * subpixOffset2;
					const subpixOffset = subpixOffset3 * subpixOffset3 * FXAA_SUBPIX_QUALITY;

					const edgeHorz =
						Math.abs(-2 * lw + lnw + lsw) +
						Math.abs(-2 * l + ln + ls) * 2 +
						Math.abs(-2 * le + lne + lse);
					const edgeVert =
						Math.abs(-2 * ln + lnw + lne) +
						Math.abs(-2 * l + lw + le) * 2 +
						Math.abs(-2 * ls + lsw + lse);
					const isHorz = edgeHorz >= edgeVert;

					const l1 = isHorz ? ln : lw;
					const l2 = isHorz ? ls : le;
					const gradient1 = Math.abs(l1 - l);
					const gradient2 = Math.abs(l2 - l);
					const is1Steeper = gradient1 >= gradient2;
					const gradientScaled = 0.25 * Math.max(gradient1, gradient2);
					const stepSign = is1Steeper ? -1 : 1;
					const lEdge = is1Steeper ? (l1 + l) * 0.5 : (l2 + l) * 0.5;

					let posNX = x;
					let posNY = y;
					if (isHorz) {
						posNY += stepSign * 0.5;
					} else {
						posNX += stepSign * 0.5;
					}
					let posPX = posNX;
					let posPY = posNY;
					const offX = isHorz ? 1 : 0;
					const offY = isHorz ? 0 : 1;
					let doneN = false;
					let doneP = false;
					let lEndN = 0;
					let lEndP = 0;
					for (let j = 0; j < FXAA_QUALITY.length; j++) {
						if (!doneN) {
							lEndN = sampleLumaBilinear(luma, width, height, posNX, posNY);
							doneN = Math.abs(lEndN - lEdge) >= gradientScaled;
						}
						if (!doneP) {
							lEndP = sampleLumaBilinear(luma, width, height, posPX, posPY);
							doneP = Math.abs(lEndP - lEdge) >= gradientScaled;
						}
						if (doneN && doneP) {
							break;
						}
						if (!doneN) {
							posNX -= offX * FXAA_QUALITY[j];
							posNY -= offY * FXAA_QUALITY[j];
						}
						if (!doneP) {
							posPX += offX * FXAA_QUALITY[j];
							posPY += offY * FXAA_QUALITY[j];
						}
					}

					const distN = isHorz ? x - posNX : y - posNY;
					const distP = isHorz ? posPX - x : posPY - y;
					const isNDistSmaller = distN < distP;
					const distMin = Math.min(distN, distP);
					const lEndMin = isNDistSmaller ? lEndN : lEndP;
					const isLPositive = l - lEdge >= 0;
					const isEndPositive = lEndMin - lEdge >= 0;
					const reachedProperly = isEndPositive !== isLPositive;
					let edgeOffset = -distMin / Math.max(distN + distP, 1e-4) + 0.5;
					if (!reachedProperly) {
						edgeOffset = 0;
					}

					const pixelOffset = Math.max(subpixOffset, edgeOffset);
					let finalX = x;
					let finalY = y;
					if (isHorz) {
						finalY += stepSign * pixelOffset;
					} else {
						finalX += stepSign * pixelOffset;
					}
					sampleByteBilinear(pixels, width, height, finalX, finalY, outCol);
					output[idx] = outCol.r;
					output[idx + 1] = outCol.g;
					output[idx + 2] = outCol.b;
					output[idx + 3] = outCol.a;
				}
			}
		});

		if (imageData) {
			imageData.data.set(output);
			context.canvasContext?.putImageData(imageData, 0, 0);
		} else {
			pixels.set(output);
		}
		return { ran: true };
	}
}

/**
 * WebGPU implementation of the cross-backend FXAA pass.
 */
export class WebGPUFastApproximateAntiAliasingImplementation
	implements PostProcessPassImplementation<WebGPUFXAAContext>
{
	public readonly id = "fxaa:webgpu";
	public readonly metadata = {
		context: WEBGPU_SCREEN_POST_PROCESS_CONTEXT_METADATA,
	};
	private _resources = new WeakMap<PostProcessSharedContext, WebGPUFXAAResources>();

	public async warmup(context: WebGPUFXAAContext | undefined): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		_request: PostProcessPassRequest,
		context: WebGPUFXAAContext | undefined
	): Promise<PostProcessPassResult> {
		if (!context?.encoder || !context.targets) {
			return { ran: false };
		}
		const ran = await this._runFXAAKernel(context);
		return ran ? { ran: true } : { ran: false };
	}

	private async _runFXAAKernel(context: WebGPUFXAAContext): Promise<boolean> {
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
		context.shared.compute.writeBuffer(
			resources.params,
			createFXAAKernelParams(target.width, target.height) as unknown as BufferSource
		);
		const binding = context.shared.getCachedBindGroup(
			`fxaa-${target === targets.postPing ? "ping" : "pong"}`,
			resources.pipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: context.shared.sampler },
				{ binding: 2, resource: resources.params },
				{ binding: 3, resource: target },
			],
			"WebGPUFXAA_Binding"
		);
		context.encoder.beginComputePass({ label: "WebGPUFXAA" });
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
	): Promise<WebGPUFXAAResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = { module: null, pipeline: null, params: null };
			this._resources.set(shared, resources);
		}
		await shared.ensureCommonResources();
		if (!resources.module) {
			const shader = await loadPostProcessShaderPartComposite("fxaa");
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUFXAAShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!resources.pipeline) {
			resources.pipeline = shared.compute.createComputePipeline({
				label: "WebGPUFXAAPipeline",
				compute: { module: resources.module, entryPoint: "csMain" },
			});
		}
		if (!resources.params) {
			resources.params = shared.compute.createBuffer({
				label: "WebGPUFXAAParams",
				size: 6 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		return resources;
	}
}

/**
 * WebGL implementation of the cross-backend FXAA pass.
 */
export class WebGLFastApproximateAntiAliasingImplementation
	implements PostProcessPassImplementation<WebGLFXAAContext>
{
	public readonly id = "fxaa:webgl";

	public warmup(context: WebGLFXAAContext | undefined): void {
		context?.programs.getFXAAProgram();
	}

	public execute(
		_request: PostProcessPassRequest,
		context: WebGLFXAAContext | undefined
	): PostProcessPassResult {
		if (!context) {
			return { ran: false };
		}
		if (
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

		const gl = context.gl;
		const fxaaProgram = context.programs.getFXAAProgram();
		gl.bindFramebuffer(gl.FRAMEBUFFER, context.postFramebuffer);
		context.bindColorTarget(targetTexture);
		gl.viewport(0, 0, context.width, context.height);
		gl.useProgram(fxaaProgram.program);
		gl.bindVertexArray(context.fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		if (fxaaProgram.uniforms.sourceMap) {
			gl.uniform1i(fxaaProgram.uniforms.sourceMap, 0);
		}
		if (fxaaProgram.uniforms.texelSize) {
			gl.uniform2f(
				fxaaProgram.uniforms.texelSize,
				1 / Math.max(1, context.width),
				1 / Math.max(1, context.height)
			);
		}
		context.drawFullscreen();
		gl.bindVertexArray(null);
		context.publishColorTexture(targetTexture);
		return { ran: true };
	}
}

export interface FastApproximateAntiAliasingPassConfig
	extends Omit<
		PostProcessPassConfig<Record<string, never>>,
		| "id"
		| "builtIn"
		| "warningLabel"
		| "placement"
		| "order"
		| "implementations"
	> {}

/**
 * Stateful logical fast approximate anti-aliasing pass.
 */
export class FastApproximateAntiAliasingPass extends PostProcessPass<
	Record<string, never>,
	Record<string, never>
> {
	public constructor(config: FastApproximateAntiAliasingPassConfig = {}) {
		super({
			...config,
			id: FAST_APPROXIMATE_ANTI_ALIASING_PASS_ID,
			builtIn: true,
			warningLabel: "FXAA",
			placement: "ldr",
			order: 710,
			implementations: {
				software: new SoftwareFastApproximateAntiAliasingImplementation(),
				webgpu: new WebGPUFastApproximateAntiAliasingImplementation(),
				webgl: new WebGLFastApproximateAntiAliasingImplementation(),
			},
		});
	}
}

function resolveDirtyRects(context: FrameContext): IncrementalDirtyRect[] {
	const width = Math.max(1, context.attachments.width);
	const height = Math.max(1, context.attachments.height);
	const incremental = context.incremental;
	if (
		!incremental.enabled ||
		incremental.forceFullFrame ||
		incremental.dirtyRects.length === 0
	) {
		return [{
			minX: 0,
			minY: 0,
			maxX: width - 1,
			maxY: height - 1,
		}];
	}
	const dirtyRects: IncrementalDirtyRect[] = [];
	for (const rect of incremental.dirtyRects) {
		const minX = Math.max(0, Math.floor(rect.x));
		const minY = Math.max(0, Math.floor(rect.y));
		const maxX = Math.min(width - 1, Math.ceil(rect.x + rect.width) - 1);
		const maxY = Math.min(height - 1, Math.ceil(rect.y + rect.height) - 1);
		if (minX > maxX || minY > maxY) {
			continue;
		}
		dirtyRects.push({ minX, minY, maxX, maxY });
	}
	return dirtyRects;
}

function forEachDirtyRect(
	dirtyRects: IncrementalDirtyRect[],
	callback: (rect: IncrementalDirtyRect) => void
): void {
	for (const rect of dirtyRects) {
		if (rect.minX > rect.maxX || rect.minY > rect.maxY) {
			continue;
		}
		callback(rect);
	}
}

function sampleLumaBilinear(
	luma: Float32Array,
	width: number,
	height: number,
	x: number,
	y: number
): number {
	const x0 = Math.floor(clamp(x, 0, width - 1));
	const y0 = Math.floor(clamp(y, 0, height - 1));
	const x1 = Math.min(width - 1, x0 + 1);
	const y1 = Math.min(height - 1, y0 + 1);
	const tx = clamp(x - x0, 0, 1);
	const ty = clamp(y - y0, 0, 1);
	const i1 = y0 * width + x0;
	const i2 = y0 * width + x1;
	const i3 = y1 * width + x0;
	const i4 = y1 * width + x1;
	return (
		luma[i1] * (1 - tx) * (1 - ty) +
		luma[i2] * tx * (1 - ty) +
		luma[i3] * (1 - tx) * ty +
		luma[i4] * tx * ty
	);
}

function sampleByteBilinear(
	pixels: Uint8ClampedArray,
	width: number,
	height: number,
	x: number,
	y: number,
	out: SampledByteColor
): void {
	const x0 = Math.floor(clamp(x, 0, width - 1));
	const y0 = Math.floor(clamp(y, 0, height - 1));
	const x1 = Math.min(width - 1, x0 + 1);
	const y1 = Math.min(height - 1, y0 + 1);
	const tx = clamp(x - x0, 0, 1);
	const ty = clamp(y - y0, 0, 1);
	const w1 = (1 - tx) * (1 - ty);
	const w2 = tx * (1 - ty);
	const w3 = (1 - tx) * ty;
	const w4 = tx * ty;
	const i1 = (y0 * width + x0) << 2;
	const i2 = (y0 * width + x1) << 2;
	const i3 = (y1 * width + x0) << 2;
	const i4 = (y1 * width + x1) << 2;
	out.r = pixels[i1] * w1 + pixels[i2] * w2 + pixels[i3] * w3 + pixels[i4] * w4;
	out.g =
		pixels[i1 + 1] * w1 +
		pixels[i2 + 1] * w2 +
		pixels[i3 + 1] * w3 +
		pixels[i4 + 1] * w4;
	out.b =
		pixels[i1 + 2] * w1 +
		pixels[i2 + 2] * w2 +
		pixels[i3 + 2] * w3 +
		pixels[i4 + 2] * w4;
	out.a =
		pixels[i1 + 3] * w1 +
		pixels[i2 + 3] * w2 +
		pixels[i3 + 3] * w3 +
		pixels[i4 + 3] * w4;
}
