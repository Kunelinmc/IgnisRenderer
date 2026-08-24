import { TextureFormat } from "../../core/TextureFormat";

export interface WebGLColorRenderTargetFormat {
	readonly internalFormat: number;
	readonly format: number;
	readonly allocationType: number;
	readonly readType: number;
	readonly channelCount: number;
	readonly bytesPerPixel: number;
	readonly repackFloat16: boolean;
}

export interface WebGLDepthRenderTargetFormat {
	readonly internalFormat: number;
	readonly format: number;
	readonly type: number;
}

export function resolveWebGLColorRenderTargetFormat(
	gl: WebGL2RenderingContext,
	format: TextureFormat,
	floatColorSupported: boolean
): WebGLColorRenderTargetFormat | null {
	const values = gl as WebGL2RenderingContext & Record<string, number>;
	switch (format) {
		case TextureFormat.R8Unorm:
			return color(values.R8, gl.RED, gl.UNSIGNED_BYTE, gl.UNSIGNED_BYTE, 1, 1);
		case TextureFormat.RG8Unorm:
			return color(values.RG8, gl.RG, gl.UNSIGNED_BYTE, gl.UNSIGNED_BYTE, 2, 2);
		case TextureFormat.RGBA8Unorm:
			return color(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.UNSIGNED_BYTE, 4, 4);
		case TextureFormat.RGBA8UnormSrgb:
			return color(
				values.SRGB8_ALPHA8,
				gl.RGBA,
				gl.UNSIGNED_BYTE,
				gl.UNSIGNED_BYTE,
				4,
				4
			);
		case TextureFormat.R16Float:
			return floatColorSupported ?
				color(values.R16F, gl.RED, gl.HALF_FLOAT, gl.FLOAT, 1, 2, true)
			:	null;
		case TextureFormat.RG16Float:
			return floatColorSupported ?
				color(values.RG16F, gl.RG, gl.HALF_FLOAT, gl.FLOAT, 2, 4, true)
			:	null;
		case TextureFormat.RGBA16Float:
			return floatColorSupported ?
				color(gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.FLOAT, 4, 8, true)
			:	null;
		case TextureFormat.R32Float:
			return floatColorSupported ?
				color(values.R32F, gl.RED, gl.FLOAT, gl.FLOAT, 1, 4)
			:	null;
		case TextureFormat.RG32Float:
			return floatColorSupported ?
				color(values.RG32F, gl.RG, gl.FLOAT, gl.FLOAT, 2, 8)
			:	null;
		case TextureFormat.RGBA32Float:
			return floatColorSupported ?
				color(gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.FLOAT, 4, 16)
			:	null;
		default:
			return null;
	}
}

export function isWebGLFloatColorRenderTargetFormat(format: TextureFormat): boolean {
	return format === TextureFormat.R16Float ||
		format === TextureFormat.RG16Float ||
		format === TextureFormat.RGBA16Float ||
		format === TextureFormat.R32Float ||
		format === TextureFormat.RG32Float ||
		format === TextureFormat.RGBA32Float;
}

export function resolveWebGLDepthRenderTargetFormat(
	gl: WebGL2RenderingContext,
	format: TextureFormat
): WebGLDepthRenderTargetFormat | null {
	const values = gl as WebGL2RenderingContext & Record<string, number>;
	switch (format) {
		case TextureFormat.Depth16Unorm:
			return {
				internalFormat: values.DEPTH_COMPONENT16,
				format: gl.DEPTH_COMPONENT,
				type: gl.UNSIGNED_SHORT,
			};
		case TextureFormat.Depth24Plus:
			return {
				internalFormat: gl.DEPTH_COMPONENT24,
				format: gl.DEPTH_COMPONENT,
				type: gl.UNSIGNED_INT,
			};
		case TextureFormat.Depth32Float:
			return {
				internalFormat: values.DEPTH_COMPONENT32F,
				format: gl.DEPTH_COMPONENT,
				type: gl.FLOAT,
			};
		default:
			return null;
	}
}

function color(
	internalFormat: number,
	format: number,
	allocationType: number,
	readType: number,
	channelCount: number,
	bytesPerPixel: number,
	repackFloat16 = false
): WebGLColorRenderTargetFormat | null {
	if (
		!Number.isFinite(internalFormat) ||
		!Number.isFinite(format) ||
		!Number.isFinite(allocationType) ||
		!Number.isFinite(readType)
	) {
		return null;
	}
	return {
		internalFormat,
		format,
		allocationType,
		readType,
		channelCount,
		bytesPerPixel,
		repackFloat16,
	};
}
