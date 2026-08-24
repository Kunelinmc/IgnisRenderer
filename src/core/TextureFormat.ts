export enum TextureFormat {
	R8Unorm = "r8unorm",
	R8Snorm = "r8snorm",
	R8Uint = "r8uint",
	R8Sint = "r8sint",
	R16Unorm = "r16unorm",
	R16Snorm = "r16snorm",
	R16Uint = "r16uint",
	R16Sint = "r16sint",
	R16Float = "r16float",
	RG8Unorm = "rg8unorm",
	RG8Snorm = "rg8snorm",
	RG8Uint = "rg8uint",
	RG8Sint = "rg8sint",
	R32Uint = "r32uint",
	R32Sint = "r32sint",
	R32Float = "r32float",
	RG16Unorm = "rg16unorm",
	RG16Snorm = "rg16snorm",
	RG16Uint = "rg16uint",
	RG16Sint = "rg16sint",
	RG16Float = "rg16float",
	RGBA8Unorm = "rgba8unorm",
	RGBA8UnormSrgb = "rgba8unorm-srgb",
	RGBA8Snorm = "rgba8snorm",
	RGBA8Uint = "rgba8uint",
	RGBA8Sint = "rgba8sint",
	BGRA8Unorm = "bgra8unorm",
	BGRA8UnormSrgb = "bgra8unorm-srgb",
	RGB9E5UFloat = "rgb9e5ufloat",
	RGB10A2Uint = "rgb10a2uint",
	RGB10A2Unorm = "rgb10a2unorm",
	RG11B10UFloat = "rg11b10ufloat",
	RG32Uint = "rg32uint",
	RG32Sint = "rg32sint",
	RG32Float = "rg32float",
	RGBA16Unorm = "rgba16unorm",
	RGBA16Snorm = "rgba16snorm",
	RGBA16Uint = "rgba16uint",
	RGBA16Sint = "rgba16sint",
	RGBA16Float = "rgba16float",
	RGBA32Uint = "rgba32uint",
	RGBA32Sint = "rgba32sint",
	RGBA32Float = "rgba32float",
	Stencil8 = "stencil8",
	Depth16Unorm = "depth16unorm",
	Depth24Plus = "depth24plus",
	Depth32Float = "depth32float",
	Depth24PlusStencil8 = "depth24plus-stencil8",
	Depth32FloatStencil8 = "depth32float-stencil8",
	BC1RGBAUnorm = "bc1-rgba-unorm",
	BC1RGBAUnormSrgb = "bc1-rgba-unorm-srgb",
	BC2RGBAUnorm = "bc2-rgba-unorm",
	BC2RGBAUnormSrgb = "bc2-rgba-unorm-srgb",
	BC3RGBAUnorm = "bc3-rgba-unorm",
	BC3RGBAUnormSrgb = "bc3-rgba-unorm-srgb",
	BC4RUnorm = "bc4-r-unorm",
	BC4RSnorm = "bc4-r-snorm",
	BC5RGUnorm = "bc5-rg-unorm",
	BC5RGSnorm = "bc5-rg-snorm",
	BC6HRGBUFloat = "bc6h-rgb-ufloat",
	BC6HRGBFloat = "bc6h-rgb-float",
	BC7RGBAUnorm = "bc7-rgba-unorm",
	BC7RGBAUnormSrgb = "bc7-rgba-unorm-srgb",
	ETC2RGB8Unorm = "etc2-rgb8unorm",
	ETC2RGB8UnormSrgb = "etc2-rgb8unorm-srgb",
	ETC2RGB8A1Unorm = "etc2-rgb8a1unorm",
	ETC2RGB8A1UnormSrgb = "etc2-rgb8a1unorm-srgb",
	ETC2RGBA8Unorm = "etc2-rgba8unorm",
	ETC2RGBA8UnormSrgb = "etc2-rgba8unorm-srgb",
	EACR11Unorm = "eac-r11unorm",
	EACR11Snorm = "eac-r11snorm",
	EACRG11Unorm = "eac-rg11unorm",
	EACRG11Snorm = "eac-rg11snorm",
	ASTC4x4Unorm = "astc-4x4-unorm",
	ASTC4x4UnormSrgb = "astc-4x4-unorm-srgb",
	ASTC5x4Unorm = "astc-5x4-unorm",
	ASTC5x4UnormSrgb = "astc-5x4-unorm-srgb",
	ASTC5x5Unorm = "astc-5x5-unorm",
	ASTC5x5UnormSrgb = "astc-5x5-unorm-srgb",
	ASTC6x5Unorm = "astc-6x5-unorm",
	ASTC6x5UnormSrgb = "astc-6x5-unorm-srgb",
	ASTC6x6Unorm = "astc-6x6-unorm",
	ASTC6x6UnormSrgb = "astc-6x6-unorm-srgb",
	ASTC8x5Unorm = "astc-8x5-unorm",
	ASTC8x5UnormSrgb = "astc-8x5-unorm-srgb",
	ASTC8x6Unorm = "astc-8x6-unorm",
	ASTC8x6UnormSrgb = "astc-8x6-unorm-srgb",
	ASTC8x8Unorm = "astc-8x8-unorm",
	ASTC8x8UnormSrgb = "astc-8x8-unorm-srgb",
	ASTC10x5Unorm = "astc-10x5-unorm",
	ASTC10x5UnormSrgb = "astc-10x5-unorm-srgb",
	ASTC10x6Unorm = "astc-10x6-unorm",
	ASTC10x6UnormSrgb = "astc-10x6-unorm-srgb",
	ASTC10x8Unorm = "astc-10x8-unorm",
	ASTC10x8UnormSrgb = "astc-10x8-unorm-srgb",
	ASTC10x10Unorm = "astc-10x10-unorm",
	ASTC10x10UnormSrgb = "astc-10x10-unorm-srgb",
	ASTC12x10Unorm = "astc-12x10-unorm",
	ASTC12x10UnormSrgb = "astc-12x10-unorm-srgb",
	ASTC12x12Unorm = "astc-12x12-unorm",
	ASTC12x12UnormSrgb = "astc-12x12-unorm-srgb",
}

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
