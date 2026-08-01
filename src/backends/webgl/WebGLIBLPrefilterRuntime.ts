import type { Texture, TextureData } from "../../core/Texture";
import { float32ToFloat16Bits } from "../../foundation/Float16";
import {
	assertIBLPrefilterSourceRevision,
	type IBLPrefilterExecutionRequest,
	type IBLPrefilterExecutorAvailability,
	type IBLPrefilterExecutorLike,
	type IBLPrefilterMipData,
	type IBLPrefilterPlan,
} from "../../lights/ibl/IBLPrefilterExecutor";

import type { WebGLProgramLibrary } from "./WebGLProgramLibrary";

const GPU_MAX_SAMPLE_COUNT = 256;
const GPU_MIN_SAMPLE_COUNT = 48;

interface WebGLIBLPrefilterRuntimeHost {
	readonly gl: WebGL2RenderingContext;
	readonly programs: WebGLProgramLibrary;
	getFullscreenVao(): WebGLVertexArrayObject | null;
}

interface WebGLPixelPackState {
	alignment: number;
	rowLength: number;
	skipPixels: number;
	skipRows: number;
	buffer: WebGLBuffer | null;
	unpackAlignment: number;
	unpackRowLength: number;
	unpackImageHeight: number;
	unpackSkipPixels: number;
	unpackSkipRows: number;
	unpackSkipImages: number;
	unpackBuffer: WebGLBuffer | null;
}

interface WebGLIBLPrefilterResources {
	sourceTexture: WebGLTexture;
	outputTexture: WebGLTexture;
	framebuffer: WebGLFramebuffer;
	sourceMipLevelCount: number;
}

/** Owns context-scoped WebGL2 fragment execution for environment prefiltering. */
export class WebGLIBLPrefilterRuntime implements IBLPrefilterExecutorLike {
	public readonly id = "webgl" as const;
	private readonly _host: WebGLIBLPrefilterRuntimeHost;
	private readonly _baseAvailability: IBLPrefilterExecutorAvailability;
	private _destroyed = false;

	public constructor(host: WebGLIBLPrefilterRuntimeHost) {
		this._host = host;
		this._baseAvailability = resolveAvailability(host.gl);
	}

	public getAvailability(): IBLPrefilterExecutorAvailability {
		if (this._destroyed) {
			return {
				state: "unsupported",
				acceptsRequests: false,
				reason: "WebGL IBL prefilter runtime has been destroyed.",
			};
		}
		if (this._host.gl.isContextLost?.()) {
			return {
				state: "temporarily-unavailable",
				acceptsRequests: false,
				reason: "WebGL IBL prefilter acceleration is unavailable while the context is lost.",
			};
		}
		return this._baseAvailability;
	}

	public async execute(
		request: IBLPrefilterExecutionRequest,
	): Promise<IBLPrefilterMipData[]> {
		const { envMap, plan: workPlan } = request;
		assertIBLPrefilterSourceRevision(envMap, request.sourceRevision);
		const availability = this.getAvailability();
		if (!availability.acceptsRequests) {
			throw new Error(
				availability.reason ??
					"WebGL IBL prefilter acceleration is unavailable.",
			);
		}
		assertNotAborted(request.signal);
		if (workPlan.mipLevels.length <= 0) {
			throw new Error("WebGL IBL prefilter work plan has no mip levels.");
		}

		const gl = this._host.gl;
		const pixelPackState = capturePixelPackState(gl);
		let resources: WebGLIBLPrefilterResources | null = null;
		try {
			preparePixelTransferState(gl);
			resources = createResources(gl, envMap, workPlan);
			return this._renderMipLevels(
				envMap,
				workPlan,
				resources,
				request,
			);
		} finally {
			if (resources) {
				destroyResources(gl, resources);
			}
			restorePixelPackState(gl, pixelPackState);
		}
	}

	public destroy(): void {
		this._destroyed = true;
	}

	private _renderMipLevels(
		envMap: Texture,
		workPlan: IBLPrefilterPlan,
		resources: WebGLIBLPrefilterResources,
		request: IBLPrefilterExecutionRequest,
	): IBLPrefilterMipData[] {
		const gl = this._host.gl;
		const vao = this._host.getFullscreenVao();
		if (!vao) {
			throw new Error("WebGL IBL prefilter fullscreen VAO is unavailable.");
		}
		const program = this._host.programs.getIBLPrefilterProgram();
		const mipData: IBLPrefilterMipData[] = [];

		gl.bindFramebuffer(gl.FRAMEBUFFER, resources.framebuffer);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
		gl.readBuffer(gl.COLOR_ATTACHMENT0);
		gl.disable(gl.SCISSOR_TEST);
		gl.disable(gl.BLEND);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.depthMask(false);
		gl.colorMask(true, true, true, true);
		gl.useProgram(program.program);
		gl.bindVertexArray(vao);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, resources.sourceTexture);
		setUniform1i(gl, program.uniforms.environmentMap, 0);
		setUniform2f(
			gl,
			program.uniforms.sourceSize,
			Math.max(1, envMap.width),
			Math.max(1, envMap.height),
		);
		setUniform1i(
			gl,
			program.uniforms.sourceIsLinear,
			envMap.colorSpace === "sRGB" ? 0 : 1,
		);
		setUniform1i(
			gl,
			program.uniforms.sourceMipLevelCount,
			resources.sourceMipLevelCount,
		);

		for (const mipPlan of workPlan.mipLevels) {
			assertNotAborted(request.signal);
			if (gl.isContextLost?.()) {
				throw new Error(
					"WebGL context was lost during IBL prefilter acceleration.",
				);
			}
			gl.framebufferTexture2D(
				gl.FRAMEBUFFER,
				gl.COLOR_ATTACHMENT0,
				gl.TEXTURE_2D,
				resources.outputTexture,
				mipPlan.level,
			);
			const framebufferStatus = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
			if (framebufferStatus !== gl.FRAMEBUFFER_COMPLETE) {
				throw new Error(
					`WebGL IBL prefilter framebuffer is incomplete at mip ${mipPlan.level} ` +
						`(status 0x${framebufferStatus.toString(16)}).`,
				);
			}

			gl.viewport(0, 0, mipPlan.width, mipPlan.height);
			setUniform2f(
				gl,
				program.uniforms.outputSize,
				mipPlan.width,
				mipPlan.height,
			);
			setUniform1f(gl, program.uniforms.roughness, mipPlan.roughness);
			setUniform1i(
				gl,
				program.uniforms.sampleCount,
				resolveSampleCount(mipPlan.roughness),
			);
			gl.drawArrays(gl.TRIANGLES, 0, 3);
			throwOnWebGLError(gl, `rendering mip ${mipPlan.level}`);

			const data = new Float32Array(
				mipPlan.width * mipPlan.height * 4,
			);
			gl.readPixels(
				0,
				0,
				mipPlan.width,
				mipPlan.height,
				gl.RGBA,
				gl.FLOAT,
				data,
			);
			throwOnWebGLError(gl, `reading mip ${mipPlan.level}`);
			for (let index = 3; index < data.length; index += 4) {
				data[index] = 1;
			}
			mipData.push({
				level: mipPlan.level,
				width: mipPlan.width,
				height: mipPlan.height,
				data,
			});
			request.onMipComplete?.(mipPlan.level);
		}

		return mipData;
	}
}

function resolveAvailability(
	gl: WebGL2RenderingContext,
): IBLPrefilterExecutorAvailability {
	const colorBufferFloat = getExtension(gl, "EXT_color_buffer_float");
	if (!colorBufferFloat) {
		return {
			state: "unsupported",
			acceptsRequests: false,
			reason:
				"WebGL IBL prefilter acceleration requires EXT_color_buffer_float.",
		};
	}
	const floatLinear =
		getExtension(gl, "OES_texture_float_linear") ??
		getExtension(gl, "OES_texture_half_float_linear");
	if (!floatLinear) {
		return {
			state: "unsupported",
			acceptsRequests: false,
			reason:
				"WebGL IBL prefilter acceleration requires " +
					"OES_texture_float_linear or OES_texture_half_float_linear.",
		};
	}
	return {
		state: "ready",
		acceptsRequests: true,
		reason: null,
	};
}

function getExtension(
	gl: WebGL2RenderingContext,
	name: string,
): unknown | null {
	try {
		return gl.getExtension(name);
	} catch {
		return null;
	}
}

function createResources(
	gl: WebGL2RenderingContext,
	envMap: Texture,
	workPlan: IBLPrefilterPlan,
): WebGLIBLPrefilterResources {
	const sourceTexture = gl.createTexture();
	const outputTexture = gl.createTexture();
	const framebuffer = gl.createFramebuffer();
	if (!sourceTexture || !outputTexture || !framebuffer) {
		if (sourceTexture) gl.deleteTexture(sourceTexture);
		if (outputTexture) gl.deleteTexture(outputTexture);
		if (framebuffer) gl.deleteFramebuffer(framebuffer);
		throw new Error("Failed to allocate WebGL IBL prefilter resources.");
	}

	try {
		const sourceMipLevelCount = uploadSourceTexture(
			gl,
			sourceTexture,
			envMap,
		);
		gl.bindTexture(gl.TEXTURE_2D, outputTexture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_BASE_LEVEL, 0);
		gl.texParameteri(
			gl.TEXTURE_2D,
			gl.TEXTURE_MAX_LEVEL,
			workPlan.mipLevels.length - 1,
		);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		for (const mipPlan of workPlan.mipLevels) {
			gl.texImage2D(
				gl.TEXTURE_2D,
				mipPlan.level,
				gl.RGBA16F,
				mipPlan.width,
				mipPlan.height,
				0,
				gl.RGBA,
				gl.HALF_FLOAT,
				null,
			);
		}
		throwOnWebGLError(gl, "allocating output mip levels");
		return {
			sourceTexture,
			outputTexture,
			framebuffer,
			sourceMipLevelCount,
		};
	} catch (error) {
		gl.deleteFramebuffer(framebuffer);
		gl.deleteTexture(outputTexture);
		gl.deleteTexture(sourceTexture);
		throw error;
	}
}

function uploadSourceTexture(
	gl: WebGL2RenderingContext,
	texture: WebGLTexture,
	envMap: Texture,
): number {
	const mipLevelCount = resolveSourceMipLevelCount(envMap);
	const uploadFloat = envMap.colorSpace !== "sRGB";
	gl.bindTexture(gl.TEXTURE_2D, texture);
	gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
	gl.texParameteri(
		gl.TEXTURE_2D,
		gl.TEXTURE_MIN_FILTER,
		mipLevelCount > 1 ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR,
	);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_BASE_LEVEL, 0);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, mipLevelCount - 1);
	for (let level = 0; level < mipLevelCount; level++) {
		const descriptor = envMap.getMipLevelDescriptor(level);
		const width = Math.max(1, descriptor?.width ?? envMap.width >> level);
		const height = Math.max(1, descriptor?.height ?? envMap.height >> level);
		const source =
			descriptor?.data ?? envMap.mipmaps[level] ?? envMap.data;
		if (!source) {
			throw new Error(
				`WebGL IBL prefilter source mip ${level} has no pixel data.`,
			);
		}
		if (uploadFloat) {
			gl.texImage2D(
				gl.TEXTURE_2D,
				level,
				gl.RGBA16F,
				width,
				height,
				0,
				gl.RGBA,
				gl.HALF_FLOAT,
				toRGBA16F(source, width, height),
			);
		} else {
			gl.texImage2D(
				gl.TEXTURE_2D,
				level,
				gl.RGBA8,
				width,
				height,
				0,
				gl.RGBA,
				gl.UNSIGNED_BYTE,
				toRGBA8(source, width, height),
			);
		}
	}
	throwOnWebGLError(gl, "uploading the source environment");
	return mipLevelCount;
}

function resolveSourceMipLevelCount(envMap: Texture): number {
	const declaredCount = Math.max(
		1,
		envMap.levels.length,
		envMap.mipmaps.length,
	);
	const naturalCount =
		Math.floor(Math.log2(Math.max(1, envMap.width, envMap.height))) + 1;
	return Math.min(declaredCount, naturalCount);
}

function toRGBA16F(
	source: TextureData,
	width: number,
	height: number,
): Uint16Array {
	const length = Math.max(1, width * height * 4);
	const result = new Uint16Array(length);
	for (let index = 0; index < length; index++) {
		const channel = index & 3;
		const fallback = channel === 3 ? 1 : 0;
		const rawValue = source[index];
		const value =
			rawValue === undefined ? fallback
			: source instanceof Float32Array ? rawValue
			: rawValue / 255;
		result[index] = float32ToFloat16Bits(value);
	}
	return result;
}

function toRGBA8(
	source: TextureData,
	width: number,
	height: number,
): Uint8Array {
	const length = Math.max(1, width * height * 4);
	const result = new Uint8Array(length);
	for (let index = 0; index < length; index++) {
		const channel = index & 3;
		const fallback = channel === 3 ? 255 : 0;
		const rawValue = source[index];
		const value =
			rawValue === undefined ? fallback
			: source instanceof Float32Array ?
				Math.round(Math.max(0, Math.min(1, rawValue)) * 255)
			: rawValue;
		result[index] = value;
	}
	return result;
}

function resolveSampleCount(roughness: number): number {
	const clampedRoughness = Math.max(0, Math.min(1, roughness));
	return Math.max(
		GPU_MIN_SAMPLE_COUNT,
		Math.min(
			GPU_MAX_SAMPLE_COUNT,
			Math.floor(
				GPU_MAX_SAMPLE_COUNT +
					(GPU_MIN_SAMPLE_COUNT - GPU_MAX_SAMPLE_COUNT) *
						clampedRoughness,
			),
		),
	);
}

function capturePixelPackState(
	gl: WebGL2RenderingContext,
): WebGLPixelPackState {
	return {
		alignment: getNumericParameter(gl, gl.PACK_ALIGNMENT, 4),
		rowLength: getNumericParameter(gl, gl.PACK_ROW_LENGTH, 0),
		skipPixels: getNumericParameter(gl, gl.PACK_SKIP_PIXELS, 0),
		skipRows: getNumericParameter(gl, gl.PACK_SKIP_ROWS, 0),
		buffer: getBufferParameter(gl, gl.PIXEL_PACK_BUFFER_BINDING),
		unpackAlignment: getNumericParameter(gl, gl.UNPACK_ALIGNMENT, 4),
		unpackRowLength: getNumericParameter(gl, gl.UNPACK_ROW_LENGTH, 0),
		unpackImageHeight: getNumericParameter(gl, gl.UNPACK_IMAGE_HEIGHT, 0),
		unpackSkipPixels: getNumericParameter(gl, gl.UNPACK_SKIP_PIXELS, 0),
		unpackSkipRows: getNumericParameter(gl, gl.UNPACK_SKIP_ROWS, 0),
		unpackSkipImages: getNumericParameter(gl, gl.UNPACK_SKIP_IMAGES, 0),
		unpackBuffer: getBufferParameter(gl, gl.PIXEL_UNPACK_BUFFER_BINDING),
	};
}

function preparePixelTransferState(gl: WebGL2RenderingContext): void {
	gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
	gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, null);
	gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
	gl.pixelStorei(gl.PACK_ROW_LENGTH, 0);
	gl.pixelStorei(gl.PACK_SKIP_PIXELS, 0);
	gl.pixelStorei(gl.PACK_SKIP_ROWS, 0);
	gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
	gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 0);
	gl.pixelStorei(gl.UNPACK_IMAGE_HEIGHT, 0);
	gl.pixelStorei(gl.UNPACK_SKIP_PIXELS, 0);
	gl.pixelStorei(gl.UNPACK_SKIP_ROWS, 0);
	gl.pixelStorei(gl.UNPACK_SKIP_IMAGES, 0);
}

function restorePixelPackState(
	gl: WebGL2RenderingContext,
	state: WebGLPixelPackState,
): void {
	gl.bindBuffer(gl.PIXEL_PACK_BUFFER, state.buffer);
	gl.pixelStorei(gl.PACK_ALIGNMENT, state.alignment);
	gl.pixelStorei(gl.PACK_ROW_LENGTH, state.rowLength);
	gl.pixelStorei(gl.PACK_SKIP_PIXELS, state.skipPixels);
	gl.pixelStorei(gl.PACK_SKIP_ROWS, state.skipRows);
	gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, state.unpackBuffer);
	gl.pixelStorei(gl.UNPACK_ALIGNMENT, state.unpackAlignment);
	gl.pixelStorei(gl.UNPACK_ROW_LENGTH, state.unpackRowLength);
	gl.pixelStorei(gl.UNPACK_IMAGE_HEIGHT, state.unpackImageHeight);
	gl.pixelStorei(gl.UNPACK_SKIP_PIXELS, state.unpackSkipPixels);
	gl.pixelStorei(gl.UNPACK_SKIP_ROWS, state.unpackSkipRows);
	gl.pixelStorei(gl.UNPACK_SKIP_IMAGES, state.unpackSkipImages);
}

function getNumericParameter(
	gl: WebGL2RenderingContext,
	parameter: number,
	fallback: number,
): number {
	try {
		const value = gl.getParameter(parameter);
		return typeof value === "number" && Number.isFinite(value) ?
			value : fallback;
	} catch {
		return fallback;
	}
}

function getBufferParameter(
	gl: WebGL2RenderingContext,
	parameter: number,
): WebGLBuffer | null {
	try {
		return gl.getParameter(parameter) as WebGLBuffer | null;
	} catch {
		return null;
	}
}

function destroyResources(
	gl: WebGL2RenderingContext,
	resources: WebGLIBLPrefilterResources,
): void {
	gl.bindFramebuffer(gl.FRAMEBUFFER, null);
	gl.bindTexture(gl.TEXTURE_2D, null);
	gl.deleteFramebuffer(resources.framebuffer);
	gl.deleteTexture(resources.outputTexture);
	gl.deleteTexture(resources.sourceTexture);
}

function setUniform1i(
	gl: WebGL2RenderingContext,
	location: WebGLUniformLocation | null,
	value: number,
): void {
	if (location !== null) gl.uniform1i(location, value);
}

function setUniform1f(
	gl: WebGL2RenderingContext,
	location: WebGLUniformLocation | null,
	value: number,
): void {
	if (location !== null) gl.uniform1f(location, value);
}

function setUniform2f(
	gl: WebGL2RenderingContext,
	location: WebGLUniformLocation | null,
	x: number,
	y: number,
): void {
	if (location !== null) gl.uniform2f(location, x, y);
}

function assertNotAborted(signal?: AbortSignal | null): void {
	if (!signal?.aborted) return;
	const error = new Error("IBL prefilter was aborted");
	error.name = "AbortError";
	throw error;
}

function throwOnWebGLError(
	gl: WebGL2RenderingContext,
	operation: string,
): void {
	const error = gl.getError();
	if (error === gl.NO_ERROR) return;
	throw new Error(
		`WebGL IBL prefilter failed while ${operation} (GL error 0x${error.toString(16)}).`,
	);
}
