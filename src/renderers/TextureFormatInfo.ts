import { TextureFormat } from "./types";

export type TextureFormatClass =
	| "color"
	| "depth"
	| "stencil"
	| "depth-stencil"
	| "compressed";

export type TextureFormatComponentType =
	| "unorm"
	| "snorm"
	| "uint"
	| "sint"
	| "float"
	| "ufloat"
	| "depth"
	| "stencil"
	| "mixed";

export type TextureFormatSampleType =
	| "float"
	| "unfilterable-float"
	| "depth"
	| "sint"
	| "uint";

export interface TextureFormatInfo {
	readonly format: TextureFormat;
	readonly formatClass: TextureFormatClass;
	readonly componentType: TextureFormatComponentType;
	readonly channelCount: number;
	readonly bytesPerBlock: number;
	readonly blockWidth: number;
	readonly blockHeight: number;
	readonly sampleType: TextureFormatSampleType;
	readonly isSRGB: boolean;
	readonly isCompressed: boolean;
	readonly hasDepth: boolean;
	readonly hasStencil: boolean;
	readonly isFilterable: boolean;
	readonly isRenderable: boolean;
	readonly supportsStorageBinding: boolean;
	readonly requiredFeature?: string;
	readonly fallbackFormat: TextureFormat;
}

const FORMAT_INFO = new Map<TextureFormat, TextureFormatInfo>();

defineColor(TextureFormat.R8Unorm, 1, "unorm", 1);
defineColor(TextureFormat.R8Snorm, 1, "snorm", 1);
defineColor(TextureFormat.R8Uint, 1, "uint", 1);
defineColor(TextureFormat.R8Sint, 1, "sint", 1);
defineColor(TextureFormat.R16Unorm, 1, "unorm", 2);
defineColor(TextureFormat.R16Snorm, 1, "snorm", 2);
defineColor(TextureFormat.R16Uint, 1, "uint", 2);
defineColor(TextureFormat.R16Sint, 1, "sint", 2);
defineColor(TextureFormat.R16Float, 1, "float", 2, {
	sampleType: "unfilterable-float",
	fallbackFormat: TextureFormat.RGBA16Float,
});
defineColor(TextureFormat.RG8Unorm, 2, "unorm", 2);
defineColor(TextureFormat.RG8Snorm, 2, "snorm", 2);
defineColor(TextureFormat.RG8Uint, 2, "uint", 2);
defineColor(TextureFormat.RG8Sint, 2, "sint", 2);
defineColor(TextureFormat.R32Uint, 1, "uint", 4, {
	isFilterable: false,
	fallbackFormat: TextureFormat.RGBA32Uint,
});
defineColor(TextureFormat.R32Sint, 1, "sint", 4, {
	isFilterable: false,
	fallbackFormat: TextureFormat.RGBA32Sint,
});
defineColor(TextureFormat.R32Float, 1, "float", 4, {
	isFilterable: false,
	sampleType: "unfilterable-float",
	fallbackFormat: TextureFormat.RGBA32Float,
});
defineColor(TextureFormat.RG16Unorm, 2, "unorm", 4);
defineColor(TextureFormat.RG16Snorm, 2, "snorm", 4);
defineColor(TextureFormat.RG16Uint, 2, "uint", 4);
defineColor(TextureFormat.RG16Sint, 2, "sint", 4);
defineColor(TextureFormat.RG16Float, 2, "float", 4, {
	sampleType: "unfilterable-float",
	fallbackFormat: TextureFormat.RGBA16Float,
});
defineColor(TextureFormat.RGBA8Unorm, 4, "unorm", 4);
defineColor(TextureFormat.RGBA8UnormSrgb, 4, "unorm", 4, {
	isSRGB: true,
	supportsStorageBinding: false,
	fallbackFormat: TextureFormat.RGBA8Unorm,
});
defineColor(TextureFormat.RGBA8Snorm, 4, "snorm", 4);
defineColor(TextureFormat.RGBA8Uint, 4, "uint", 4);
defineColor(TextureFormat.RGBA8Sint, 4, "sint", 4);
defineColor(TextureFormat.BGRA8Unorm, 4, "unorm", 4);
defineColor(TextureFormat.BGRA8UnormSrgb, 4, "unorm", 4, {
	isSRGB: true,
	supportsStorageBinding: false,
	fallbackFormat: TextureFormat.BGRA8Unorm,
});
defineColor(TextureFormat.RGB9E5UFloat, 3, "ufloat", 4, {
	supportsStorageBinding: false,
	fallbackFormat: TextureFormat.RGBA16Float,
});
defineColor(TextureFormat.RGB10A2Uint, 4, "uint", 4);
defineColor(TextureFormat.RGB10A2Unorm, 4, "mixed", 4);
defineColor(TextureFormat.RG11B10UFloat, 3, "ufloat", 4, {
	supportsStorageBinding: false,
	fallbackFormat: TextureFormat.RGBA16Float,
});
defineColor(TextureFormat.RG32Uint, 2, "uint", 8, {
	isFilterable: false,
	fallbackFormat: TextureFormat.RGBA32Uint,
});
defineColor(TextureFormat.RG32Sint, 2, "sint", 8, {
	isFilterable: false,
	fallbackFormat: TextureFormat.RGBA32Sint,
});
defineColor(TextureFormat.RG32Float, 2, "float", 8, {
	isFilterable: false,
	sampleType: "unfilterable-float",
	fallbackFormat: TextureFormat.RGBA32Float,
});
defineColor(TextureFormat.RGBA16Unorm, 4, "unorm", 8);
defineColor(TextureFormat.RGBA16Snorm, 4, "snorm", 8);
defineColor(TextureFormat.RGBA16Uint, 4, "uint", 8);
defineColor(TextureFormat.RGBA16Sint, 4, "sint", 8);
defineColor(TextureFormat.RGBA16Float, 4, "float", 8, {
	sampleType: "unfilterable-float",
});
defineColor(TextureFormat.RGBA32Uint, 4, "uint", 16, {
	isFilterable: false,
});
defineColor(TextureFormat.RGBA32Sint, 4, "sint", 16, {
	isFilterable: false,
});
defineColor(TextureFormat.RGBA32Float, 4, "float", 16, {
	isFilterable: false,
	sampleType: "unfilterable-float",
});

defineDepth(TextureFormat.Stencil8, "stencil", 1, false, true);
defineDepth(TextureFormat.Depth16Unorm, "depth", 2, true, false);
defineDepth(TextureFormat.Depth24Plus, "depth", 4, true, false);
defineDepth(TextureFormat.Depth24PlusStencil8, "depth-stencil", 4, true, true);
defineDepth(TextureFormat.Depth32Float, "depth", 4, true, false);
defineDepth(TextureFormat.Depth32FloatStencil8, "depth-stencil", 5, true, true);

defineCompressed(TextureFormat.BC1RGBAUnorm, 4, 4, 8, "texture-compression-bc");
defineCompressed(TextureFormat.BC1RGBAUnormSrgb, 4, 4, 8, "texture-compression-bc", true);
defineCompressed(TextureFormat.BC2RGBAUnorm, 4, 4, 16, "texture-compression-bc");
defineCompressed(TextureFormat.BC2RGBAUnormSrgb, 4, 4, 16, "texture-compression-bc", true);
defineCompressed(TextureFormat.BC3RGBAUnorm, 4, 4, 16, "texture-compression-bc");
defineCompressed(TextureFormat.BC3RGBAUnormSrgb, 4, 4, 16, "texture-compression-bc", true);
defineCompressed(TextureFormat.BC4RUnorm, 4, 4, 8, "texture-compression-bc", false, 1);
defineCompressed(TextureFormat.BC4RSnorm, 4, 4, 8, "texture-compression-bc", false, 1, "snorm");
defineCompressed(TextureFormat.BC5RGUnorm, 4, 4, 16, "texture-compression-bc", false, 2);
defineCompressed(TextureFormat.BC5RGSnorm, 4, 4, 16, "texture-compression-bc", false, 2, "snorm");
defineCompressed(TextureFormat.BC6HRGBUFloat, 4, 4, 16, "texture-compression-bc", false, 3, "ufloat");
defineCompressed(TextureFormat.BC6HRGBFloat, 4, 4, 16, "texture-compression-bc", false, 3, "float");
defineCompressed(TextureFormat.BC7RGBAUnorm, 4, 4, 16, "texture-compression-bc");
defineCompressed(TextureFormat.BC7RGBAUnormSrgb, 4, 4, 16, "texture-compression-bc", true);

defineCompressed(TextureFormat.ETC2RGB8Unorm, 4, 4, 8, "texture-compression-etc2", false, 3);
defineCompressed(TextureFormat.ETC2RGB8UnormSrgb, 4, 4, 8, "texture-compression-etc2", true, 3);
defineCompressed(TextureFormat.ETC2RGB8A1Unorm, 4, 4, 8, "texture-compression-etc2");
defineCompressed(TextureFormat.ETC2RGB8A1UnormSrgb, 4, 4, 8, "texture-compression-etc2", true);
defineCompressed(TextureFormat.ETC2RGBA8Unorm, 4, 4, 16, "texture-compression-etc2");
defineCompressed(TextureFormat.ETC2RGBA8UnormSrgb, 4, 4, 16, "texture-compression-etc2", true);
defineCompressed(TextureFormat.EACR11Unorm, 4, 4, 8, "texture-compression-etc2", false, 1);
defineCompressed(TextureFormat.EACR11Snorm, 4, 4, 8, "texture-compression-etc2", false, 1, "snorm");
defineCompressed(TextureFormat.EACRG11Unorm, 4, 4, 16, "texture-compression-etc2", false, 2);
defineCompressed(TextureFormat.EACRG11Snorm, 4, 4, 16, "texture-compression-etc2", false, 2, "snorm");

defineASTC(TextureFormat.ASTC4x4Unorm, 4, 4);
defineASTC(TextureFormat.ASTC4x4UnormSrgb, 4, 4, true);
defineASTC(TextureFormat.ASTC5x4Unorm, 5, 4);
defineASTC(TextureFormat.ASTC5x4UnormSrgb, 5, 4, true);
defineASTC(TextureFormat.ASTC5x5Unorm, 5, 5);
defineASTC(TextureFormat.ASTC5x5UnormSrgb, 5, 5, true);
defineASTC(TextureFormat.ASTC6x5Unorm, 6, 5);
defineASTC(TextureFormat.ASTC6x5UnormSrgb, 6, 5, true);
defineASTC(TextureFormat.ASTC6x6Unorm, 6, 6);
defineASTC(TextureFormat.ASTC6x6UnormSrgb, 6, 6, true);
defineASTC(TextureFormat.ASTC8x5Unorm, 8, 5);
defineASTC(TextureFormat.ASTC8x5UnormSrgb, 8, 5, true);
defineASTC(TextureFormat.ASTC8x6Unorm, 8, 6);
defineASTC(TextureFormat.ASTC8x6UnormSrgb, 8, 6, true);
defineASTC(TextureFormat.ASTC8x8Unorm, 8, 8);
defineASTC(TextureFormat.ASTC8x8UnormSrgb, 8, 8, true);
defineASTC(TextureFormat.ASTC10x5Unorm, 10, 5);
defineASTC(TextureFormat.ASTC10x5UnormSrgb, 10, 5, true);
defineASTC(TextureFormat.ASTC10x6Unorm, 10, 6);
defineASTC(TextureFormat.ASTC10x6UnormSrgb, 10, 6, true);
defineASTC(TextureFormat.ASTC10x8Unorm, 10, 8);
defineASTC(TextureFormat.ASTC10x8UnormSrgb, 10, 8, true);
defineASTC(TextureFormat.ASTC10x10Unorm, 10, 10);
defineASTC(TextureFormat.ASTC10x10UnormSrgb, 10, 10, true);
defineASTC(TextureFormat.ASTC12x10Unorm, 12, 10);
defineASTC(TextureFormat.ASTC12x10UnormSrgb, 12, 10, true);
defineASTC(TextureFormat.ASTC12x12Unorm, 12, 12);
defineASTC(TextureFormat.ASTC12x12UnormSrgb, 12, 12, true);

/**
 * Returns immutable metadata for a texture format.
 */
export function getTextureFormatInfo(format: TextureFormat): TextureFormatInfo {
	const info = FORMAT_INFO.get(format);
	if (!info) {
		throw new Error(`Unsupported texture format "${String(format)}".`);
	}
	return info;
}

export function tryGetTextureFormatInfo(
	format: TextureFormat | string | undefined | null
): TextureFormatInfo | null {
	if (!format) {
		return null;
	}
	return FORMAT_INFO.get(format as TextureFormat) ?? null;
}

export function isTextureFormatSRGB(format: TextureFormat): boolean {
	return getTextureFormatInfo(format).isSRGB;
}

export function isTextureFormatCompressed(format: TextureFormat): boolean {
	return getTextureFormatInfo(format).isCompressed;
}

export function getTextureFormatBlockCount(
	format: TextureFormat,
	width: number,
	height: number
): { width: number; height: number } {
	const info = getTextureFormatInfo(format);
	return {
		width: Math.max(1, Math.ceil(Math.max(1, width) / info.blockWidth)),
		height: Math.max(1, Math.ceil(Math.max(1, height) / info.blockHeight)),
	};
}

export function getTextureFormatBytesPerRow(
	format: TextureFormat,
	width: number
): number {
	const info = getTextureFormatInfo(format);
	const blockWidth = Math.max(1, Math.ceil(Math.max(1, width) / info.blockWidth));
	return blockWidth * info.bytesPerBlock;
}

export function getTextureFormatLevelByteLength(
	format: TextureFormat,
	width: number,
	height: number,
	depthOrArrayLayers: number = 1,
	bytesPerRow?: number
): number {
	const info = getTextureFormatInfo(format);
	const blocks = getTextureFormatBlockCount(format, width, height);
	const rowBytes = Math.max(
		bytesPerRow ?? blocks.width * info.bytesPerBlock,
		blocks.width * info.bytesPerBlock
	);
	return rowBytes * blocks.height * Math.max(1, depthOrArrayLayers | 0);
}

export function getTextureFormatFallback(format: TextureFormat): TextureFormat {
	return getTextureFormatInfo(format).fallbackFormat;
}

export function textureFormatRequiresFeature(
	format: TextureFormat,
	features: GPUSupportedFeatures | ReadonlySet<string> | null | undefined
): boolean {
	const requiredFeature = getTextureFormatInfo(format).requiredFeature;
	if (!requiredFeature) {
		return false;
	}
	return !features || typeof features.has !== "function" || !features.has(requiredFeature);
}

function defineColor(
	format: TextureFormat,
	channelCount: number,
	componentType: TextureFormatComponentType,
	bytesPerBlock: number,
	options: Partial<TextureFormatInfo> = {}
): void {
	const sampleType =
		componentType === "uint" ? "uint"
		: componentType === "sint" ? "sint"
		: componentType === "float" && !options.isFilterable ? "unfilterable-float"
		: "float";
	define({
		format,
		formatClass: "color",
		componentType,
		channelCount,
		bytesPerBlock,
		blockWidth: 1,
		blockHeight: 1,
		sampleType,
		isSRGB: false,
		isCompressed: false,
		hasDepth: false,
		hasStencil: false,
		isFilterable:
			componentType !== "uint" &&
			componentType !== "sint" &&
			componentType !== "float",
		isRenderable: componentType !== "sint" && componentType !== "uint",
		supportsStorageBinding:
			componentType !== "snorm" &&
			componentType !== "ufloat" &&
			componentType !== "mixed",
		fallbackFormat:
			componentType === "float" || componentType === "ufloat"
				? TextureFormat.RGBA16Float
				: componentType === "uint"
					? TextureFormat.RGBA8Uint
					: componentType === "sint"
						? TextureFormat.RGBA8Sint
						: TextureFormat.RGBA8Unorm,
		...options,
	});
}

function defineDepth(
	format: TextureFormat,
	formatClass: "depth" | "stencil" | "depth-stencil",
	bytesPerBlock: number,
	hasDepth: boolean,
	hasStencil: boolean
): void {
	define({
		format,
		formatClass,
		componentType: hasDepth ? "depth" : "stencil",
		channelCount: hasDepth && hasStencil ? 2 : 1,
		bytesPerBlock,
		blockWidth: 1,
		blockHeight: 1,
		sampleType: hasDepth ? "depth" : "uint",
		isSRGB: false,
		isCompressed: false,
		hasDepth,
		hasStencil,
		isFilterable: hasDepth && !hasStencil,
		isRenderable: true,
		supportsStorageBinding: false,
		fallbackFormat: hasDepth ? TextureFormat.Depth32Float : TextureFormat.R8Uint,
	});
}

function defineCompressed(
	format: TextureFormat,
	blockWidth: number,
	blockHeight: number,
	bytesPerBlock: number,
	requiredFeature: string,
	isSRGB = false,
	channelCount = 4,
	componentType: TextureFormatComponentType = "unorm"
): void {
	define({
		format,
		formatClass: "compressed",
		componentType,
		channelCount,
		bytesPerBlock,
		blockWidth,
		blockHeight,
		sampleType: componentType === "float" || componentType === "ufloat"
			? "unfilterable-float"
			: "float",
		isSRGB,
		isCompressed: true,
		hasDepth: false,
		hasStencil: false,
		isFilterable: componentType !== "float" && componentType !== "ufloat",
		isRenderable: false,
		supportsStorageBinding: false,
		requiredFeature,
		fallbackFormat:
			componentType === "float" || componentType === "ufloat"
				? TextureFormat.RGBA16Float
				: isSRGB
					? TextureFormat.RGBA8UnormSrgb
					: TextureFormat.RGBA8Unorm,
	});
}

function defineASTC(
	format: TextureFormat,
	blockWidth: number,
	blockHeight: number,
	isSRGB = false
): void {
	defineCompressed(
		format,
		blockWidth,
		blockHeight,
		16,
		"texture-compression-astc",
		isSRGB
	);
}

function define(info: TextureFormatInfo): void {
	FORMAT_INFO.set(info.format, Object.freeze(info));
}
