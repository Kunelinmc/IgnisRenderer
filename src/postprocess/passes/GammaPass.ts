import { linearToSRGB, sRGBToLinear } from "../../maths/Common";
import { DEFAULT_GAMMA } from "../../renderers/constants";
import {
	WEBGPU_PRESENT_POST_PROCESS_CONTEXT_METADATA,
} from "../../renderers/webgpu/WebGPUPostProcessContracts";
import { PostProcessPass, type PostProcessPassConfig } from "../PostProcessPass";
import type { PostProcessPassMetadata } from "../ordering";
import type {
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassResult,
} from "../types";
import {
	forEachSoftwareDirtyRect,
	resolveSoftwareDirtyRects,
	type EmptyOptions,
	type SoftwareBuiltinPostProcessContext,
	type WebGLGammaContext,
	type WebGPUGammaContext,
} from "./ScreenPassShared";

export type { WebGLGammaContext, WebGPUGammaContext } from "./ScreenPassShared";

export const GAMMA_PASS_ID = "gamma";
export const GAMMA_PASS_ORDER = {
	id: GAMMA_PASS_ID,
	placement: "present",
	order: 900,
	incremental: {
		firstPass: "gamma",
		grade: "light",
		inflationRadius: 0,
	},
} as const satisfies PostProcessPassMetadata;
/** @internal Software implementation for the built-in gamma pass. */
export class SoftwareGammaImplementation
	implements PostProcessPassImplementation<SoftwareBuiltinPostProcessContext>
{
	public readonly id = "gamma:software";
	private readonly _sRGBLUT = new Uint8Array(256);
	private _lutBuilt = false;
	private _lastGamma = -1;

	public execute(
		request: PostProcessPassRequest,
		context: SoftwareBuiltinPostProcessContext | undefined
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
		const gamma = request.frameContext.postProcess.isEnabled(GAMMA_PASS_ID)
			? DEFAULT_GAMMA
			: 1;
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
			this._sRGBLUT[i] =
				isStandardSRGB ?
					Math.round(linearToSRGB(value) * 255)
				:	Math.round(Math.pow(value, invGamma) * 255);
		}
		this._lutBuilt = true;
		this._lastGamma = gamma;
	}
}
/** @internal WebGPU implementation for the built-in gamma pass. */
export class WebGPUGammaImplementation
	implements PostProcessPassImplementation<WebGPUGammaContext, EmptyOptions>
{
	public readonly id = "gamma:webgpu";
	public readonly metadata = {
		context: WEBGPU_PRESENT_POST_PROCESS_CONTEXT_METADATA,
	};

	public async warmup(context: WebGPUGammaContext | undefined): Promise<void> {
		await context?.warmupPresent?.();
	}

	public async execute(
		request: PostProcessPassRequest<EmptyOptions>,
		context: WebGPUGammaContext | undefined
	): Promise<PostProcessPassResult> {
		const source = context?.targets?.sceneColor;
		if (!source || !context?.presentToCanvas) {
			return { ran: false };
		}
		await context.presentToCanvas(
			source,
			request.frameContext.postProcess.isEnabled(GAMMA_PASS_ID)
		);
		return { ran: true };
	}
}
/** @internal WebGL implementation for the built-in gamma pass. */
export class WebGLGammaImplementation
	implements PostProcessPassImplementation<WebGLGammaContext, EmptyOptions>
{
	public readonly id = "gamma:webgl";

	public warmup(context: WebGLGammaContext | undefined): void {
		context?.programs.warmupPresentProgram();
	}

	public execute(
		request: PostProcessPassRequest<EmptyOptions>,
		context: WebGLGammaContext | undefined
	): PostProcessPassResult {
		const sourceTexture = context?.getSourceTexture();
		if (!context?.fullscreenVao || !sourceTexture) {
			return { ran: false };
		}
		const gl = context.gl;
		const program = context.programs.tryGetPresentProgram();
		if (!program) {
			return { ran: false };
		}
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		gl.viewport(0, 0, context.width, context.height);
		gl.useProgram(program.program);
		gl.bindVertexArray(context.fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		if (program.uniforms.sourceMap) {
			gl.uniform1i(program.uniforms.sourceMap, 0);
		}
		if (program.uniforms.applyGamma) {
			gl.uniform1i(
				program.uniforms.applyGamma,
				request.frameContext.postProcess.isEnabled(GAMMA_PASS_ID) ? 1 : 0
			);
		}
		context.drawFullscreen();
		gl.bindVertexArray(null);
		context.markPresented();
		return { ran: true };
	}
}
/**
 * Stateful logical gamma correction pass.
 */
export class GammaPass extends PostProcessPass<EmptyOptions, EmptyOptions> {
	public constructor(
		config: Omit<
			PostProcessPassConfig<EmptyOptions>,
			| "id"
			| "builtIn"
			| "warningLabel"
			| "placement"
			| "order"
			| "implementations"
		> = {}
	) {
		super({
			...config,
			...GAMMA_PASS_ORDER,
			incremental: config.incremental ?? GAMMA_PASS_ORDER.incremental,
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
