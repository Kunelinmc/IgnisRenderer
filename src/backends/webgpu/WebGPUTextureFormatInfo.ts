import { getTextureFormatInfo } from "../TextureFormatInfo";
import { TextureFormat } from "../types";

const WEBGPU_EIGHT_BYTE_RENDER_TARGET_FORMATS = new Set<TextureFormat>([
	TextureFormat.RGBA8Unorm,
	TextureFormat.RGBA8UnormSrgb,
	TextureFormat.RGBA8Snorm,
	TextureFormat.BGRA8Unorm,
	TextureFormat.BGRA8UnormSrgb,
	TextureFormat.RGB10A2Uint,
	TextureFormat.RGB10A2Unorm,
	TextureFormat.RG11B10UFloat,
]);

/**
 * Returns the WebGPU color-attachment byte cost for one pixel of a format.
 *
 * @internal Owned by the WebGPU backend for attachment-limit validation.
 */
export function getWebGPURenderTargetPixelByteCost(
	format: TextureFormat
): number {
	const info = getTextureFormatInfo(format);
	if (info.formatClass !== "color") {
		return 0;
	}
	return WEBGPU_EIGHT_BYTE_RENDER_TARGET_FORMATS.has(format)
		? 8
		: info.bytesPerBlock;
}
