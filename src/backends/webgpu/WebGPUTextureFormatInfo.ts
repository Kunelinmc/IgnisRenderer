import {
	getTextureFormatInfo,
	TextureFormat,
} from "../../core/TextureFormat";

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

const WEBGPU_STORAGE_FORMATS = new Set<TextureFormat>([
	TextureFormat.R32Uint, TextureFormat.R32Sint, TextureFormat.R32Float,
	TextureFormat.RG32Uint, TextureFormat.RG32Sint, TextureFormat.RG32Float,
	TextureFormat.RGBA8Unorm, TextureFormat.RGBA8Snorm,
	TextureFormat.RGBA8Uint, TextureFormat.RGBA8Sint,
	TextureFormat.RGBA16Uint, TextureFormat.RGBA16Sint, TextureFormat.RGBA16Float,
	TextureFormat.RGBA32Uint, TextureFormat.RGBA32Sint, TextureFormat.RGBA32Float,
]);

const WEBGPU_TIER1_STORAGE_FORMATS = new Set<TextureFormat>([
	TextureFormat.R8Unorm, TextureFormat.R8Snorm, TextureFormat.R8Uint, TextureFormat.R8Sint,
	TextureFormat.RG8Unorm, TextureFormat.RG8Snorm, TextureFormat.RG8Uint, TextureFormat.RG8Sint,
	TextureFormat.R16Uint, TextureFormat.R16Sint, TextureFormat.R16Float,
	TextureFormat.RG16Uint, TextureFormat.RG16Sint, TextureFormat.RG16Float,
	TextureFormat.RGB10A2Uint, TextureFormat.RGB10A2Unorm, TextureFormat.RG11B10UFloat,
	TextureFormat.R16Unorm, TextureFormat.R16Snorm,
	TextureFormat.RG16Unorm, TextureFormat.RG16Snorm,
	TextureFormat.RGBA16Unorm, TextureFormat.RGBA16Snorm,
]);

const WEBGPU_TIER1_RENDER_ATTACHMENT_FORMATS = new Set<TextureFormat>([
	TextureFormat.R8Snorm, TextureFormat.RG8Snorm, TextureFormat.RGBA8Snorm,
	TextureFormat.R16Unorm, TextureFormat.R16Snorm,
	TextureFormat.RG16Unorm, TextureFormat.RG16Snorm,
	TextureFormat.RGBA16Unorm, TextureFormat.RGBA16Snorm,
]);

/**
 * Checks whether source uploads can use a linear byte copy with the default aspect.
 *
 * @internal Owned by WebGPU source uploads. Applications use `ComputeRuntime`.
 */
export function supportsWebGPUTextureSourceUploadFormat(format: TextureFormat): boolean {
	const info = getTextureFormatInfo(format);
	return (!info.hasDepth && !info.hasStencil) ||
		format === TextureFormat.Depth16Unorm || format === TextureFormat.Stencil8;
}

/**
 * Checks render-attachment format availability against enabled WebGPU features.
 *
 * @internal Owned by WebGPU format validation. Applications use `ComputeRuntime`.
 */
export function supportsWebGPURenderAttachmentFormat(
	format: TextureFormat,
	features?: ReadonlySet<string>,
): boolean {
	const info = getTextureFormatInfo(format);
	if (info.isCompressed || format === TextureFormat.RGB9E5UFloat) return false;
	if (format === TextureFormat.Depth32FloatStencil8) {
		return features?.has("depth32float-stencil8") ?? false;
	}
	if (format === TextureFormat.RG11B10UFloat ||
		WEBGPU_TIER1_RENDER_ATTACHMENT_FORMATS.has(format)) {
		return !!features?.has("texture-formats-tier1") ||
			!!features?.has("texture-formats-tier2") ||
			(format === TextureFormat.RG11B10UFloat &&
				!!features?.has("rg11b10ufloat-renderable"));
	}
	return true;
}

/**
 * Checks storage usage availability without assuming shader access mode support.
 *
 * @internal Owned by WebGPU format validation. Applications use `ComputeRuntime`.
 */
export function supportsWebGPUStorageTextureFormat(
	format: TextureFormat,
	features?: ReadonlySet<string>,
): boolean {
	if (WEBGPU_STORAGE_FORMATS.has(format)) return true;
	if (format === TextureFormat.BGRA8Unorm) {
		return features?.has("bgra8unorm-storage") ?? false;
	}
	return WEBGPU_TIER1_STORAGE_FORMATS.has(format) &&
		(!!features?.has("texture-formats-tier1") || !!features?.has("texture-formats-tier2"));
}

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
