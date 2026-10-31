import { clamp } from "../../maths/Common";
import { resolveTextureSamplingFormat, type Texture, type TextureData } from "../../core/Texture";
import { float16BitsToFloat32, float32ToFloat16Bits } from "../../foundation/Float16";
import {
	getTextureFormatBytesPerRow,
	getTextureFormatInfo,
	getTextureFormatLevelByteLength,
	TextureFormat,
	type TextureFormatInfo,
} from "../../core/TextureFormat";

/**
 * @internal WebGPU resource uploads own this queue-write layout. Applications
 * should supply texture data through `Texture` instead.
 */
export interface WebGPUTextureUploadLevel {
	data: Uint8Array;
	bytesPerRow: number;
	rowsPerImage: number;
	/** Logical mip dimensions, before rounding compressed copy extents. */
	width: number;
	height: number;
	copyWidth: number;
	copyHeight: number;
	mipLevel: number;
	format: TextureFormat;
}

/**
 * @internal WebGPU resource uploads prepare authored mip layouts here;
 * applications should use `ComputeRuntime.createTexture({ source })`.
 */
export function createTextureMipUploadLevels(
	texture: Texture,
	format: TextureFormat = resolveTextureSamplingFormat(texture)
): WebGPUTextureUploadLevel[] {
	const mipCount = Math.max(
		1,
		texture.levels.length,
		texture.mipmaps.length,
		1
	);
	const levels: WebGPUTextureUploadLevel[] = [];
	for (let mipLevel = 0; mipLevel < mipCount; mipLevel++) {
		levels.push(createTextureMipUploadData(texture, mipLevel, format));
	}
	return levels;
}

/**
 * Prepares one 2D mip for queue upload without buffer-copy row padding.
 *
 * @internal Owned by WebGPU resource uploads; applications should use `Texture`.
 * @param texture - Source texture, including optional native raw row metadata.
 * @param mipLevel - Mip level to prepare.
 * @param format - Native upload format, including resolved sampling encoding.
 * @returns Source bytes and their matching queue-write layout and copy extent.
 * @throws When an explicit raw layout is invalid or needs an unsupported conversion.
 */
export function createTextureMipUploadData(
	texture: Texture,
	mipLevel: number,
	format: TextureFormat = resolveTextureSamplingFormat(texture)
): WebGPUTextureUploadLevel {
	const level = Math.max(0, mipLevel | 0);
	const levelDescriptor = texture.getMipLevelDescriptor(level);
	const width = Math.max(1, Math.floor(levelDescriptor?.width ?? texture.width >> level));
	const height = Math.max(
		1,
		Math.floor(levelDescriptor?.height ?? texture.height >> level)
	);
	const sourceData =
		levelDescriptor?.data ?? texture.readPixelData(level);
	const info = getTextureFormatInfo(format);
	const blockRows = Math.ceil(height / info.blockHeight);
	const minimumBytesPerRow = getTextureFormatBytesPerRow(format, width);
	const storageCompatible = texture.format === format ||
		texture.format.replace(/-srgb$/, "") === format.replace(/-srgb$/, "");
	const explicitLayout = levelDescriptor?.bytesPerRow !== undefined ||
		levelDescriptor?.rowsPerImage !== undefined;
	let bytesPerRow = minimumBytesPerRow;
	let rowsPerImage = blockRows;
	let pixelData: Uint8Array;

	if (explicitLayout) {
		if (!(sourceData instanceof Uint8Array)) {
			throw new TypeError("WebGPU explicit texture layouts require Uint8Array raw bytes.");
		}
		if (!storageCompatible) {
			throw new Error("WebGPU raw texture uploads require a storage-compatible format.");
		}
		bytesPerRow = levelDescriptor?.bytesPerRow ?? minimumBytesPerRow;
		rowsPerImage = levelDescriptor?.rowsPerImage ?? blockRows;
		if (!Number.isSafeInteger(bytesPerRow) || bytesPerRow < minimumBytesPerRow) {
			throw new RangeError("WebGPU texture bytesPerRow must cover a complete block row.");
		}
		if (!Number.isSafeInteger(rowsPerImage) || rowsPerImage < blockRows) {
			throw new RangeError("WebGPU texture rowsPerImage must cover all block rows.");
		}
		// A single 2D image needs no padding after its final block row.
		const requiredBytes = bytesPerRow * (blockRows - 1) + minimumBytesPerRow;
		if (!Number.isSafeInteger(requiredBytes) || sourceData.byteLength < requiredBytes) {
			throw new RangeError("WebGPU raw texture payload is too short for its row layout.");
		}
		pixelData = sourceData.byteLength === requiredBytes ?
			sourceData : sourceData.subarray(0, requiredBytes);
	} else {
		const nativeSource = sourceData instanceof Uint8Array && storageCompatible &&
			sourceData.byteLength === minimumBytesPerRow * blockRows;
		if (nativeSource) {
			pixelData = sourceData;
		} else if (sourceData instanceof Uint8Array && !storageCompatible &&
			sourceData.byteLength === getTextureFormatLevelByteLength(texture.format, width, height)) {
			pixelData = convertRawFloatTextureData(sourceData, width * height,
				getTextureFormatInfo(texture.format), info);
		} else {
			pixelData = createTextureFormatUploadData(sourceData, width, height, format);
		}
	}

	return {
		data: pixelData,
		bytesPerRow,
		rowsPerImage,
		width,
		height,
		copyWidth: Math.ceil(width / info.blockWidth) * info.blockWidth,
		copyHeight: blockRows * info.blockHeight,
		mipLevel: level,
		format,
	};
}

export function alignTo(value: number, alignment: number): number {
	return Math.ceil(value / alignment) * alignment;
}

function convertRawFloatTextureData(
	data: Uint8Array,
	pixelCount: number,
	sourceInfo: TextureFormatInfo,
	targetInfo: TextureFormatInfo,
): Uint8Array {
	if (sourceInfo.formatClass !== "color" || sourceInfo.componentType !== "float" ||
		targetInfo.formatClass !== "color" || targetInfo.componentType !== "float") {
		throw new Error(
			`Unsupported WebGPU raw texture conversion: ${sourceInfo.format} -> ${targetInfo.format}.`
		);
	}
	const sourceComponentBytes = sourceInfo.bytesPerBlock / sourceInfo.channelCount;
	const targetComponentBytes = targetInfo.bytesPerBlock / targetInfo.channelCount;
	const sourceView = new DataView(data.buffer, data.byteOffset, data.byteLength);
	const output = new Uint8Array(pixelCount * targetInfo.bytesPerBlock);
	const targetView = new DataView(output.buffer);
	for (let pixel = 0; pixel < pixelCount; pixel++) {
		for (let channel = 0; channel < targetInfo.channelCount; channel++) {
			let value = defaultChannelValue(channel);
			if (channel < sourceInfo.channelCount) {
				// Raw floating-point bytes encode components, not normalized byte channels.
				const offset = (pixel * sourceInfo.channelCount + channel) * sourceComponentBytes;
				value = sourceComponentBytes === 2 ?
					float16BitsToFloat32(sourceView.getUint16(offset, true)) :
					sourceView.getFloat32(offset, true);
			}
			const offset = (pixel * targetInfo.channelCount + channel) * targetComponentBytes;
			if (targetComponentBytes === 2) {
				// The numerical-source encoder sanitizes non-finite values; raw data preserves them.
				const bits = Number.isFinite(value) ? float32ToFloat16Bits(value) :
					Number.isNaN(value) ? 0x7e00 : value < 0 ? 0xfc00 : 0x7c00;
				targetView.setUint16(offset, bits, true);
			} else {
				targetView.setFloat32(offset, value, true);
			}
		}
	}
	return output;
}

function createTextureFormatUploadData(
	data: TextureData | null,
	width: number,
	height: number,
	format: TextureFormat
): Uint8Array {
	const info = getTextureFormatInfo(format);
	if (info.isCompressed || info.hasDepth || info.hasStencil) {
		return toRawTextureData(data, width, height, format);
	}
	switch (format) {
		case TextureFormat.RGBA8Unorm:
		case TextureFormat.RGBA8UnormSrgb:
		case TextureFormat.BGRA8Unorm:
		case TextureFormat.BGRA8UnormSrgb:
			return toUnorm8TextureData(data, width, height, info);
		case TextureFormat.R8Unorm:
		case TextureFormat.RG8Unorm:
			return toUnorm8TextureData(data, width, height, info);
		case TextureFormat.R16Float:
		case TextureFormat.RG16Float:
		case TextureFormat.RGBA16Float:
			return toFloat16TextureData(data, width, height, info);
		case TextureFormat.R32Float:
		case TextureFormat.RG32Float:
		case TextureFormat.RGBA32Float:
			return toFloat32TextureData(data, width, height, info);
		default:
			return toRawOrConvertedTextureData(data, width, height, format, info);
	}
}

function toUnorm8TextureData(
	data: TextureData | null,
	width: number,
	height: number,
	info: TextureFormatInfo
): Uint8Array {
	if (!data) {
		return new Uint8Array(width * height * info.channelCount);
	}

	const expectedLength = width * height * info.channelCount;

	if (data instanceof Uint8Array && !(data instanceof Uint8ClampedArray)) {
		if (data.length === expectedLength) return data;
		if (data.length < expectedLength) {
			const copy = new Uint8Array(expectedLength);
			copy.set(data.subarray(0, expectedLength));
			return copy;
		}
	}

	if (data instanceof Uint8ClampedArray) {
		const copy = new Uint8Array(
			data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)
		);
		if (copy.length === expectedLength) return copy;
		if (copy.length < expectedLength) {
			const sized = new Uint8Array(expectedLength);
			sized.set(copy.subarray(0, expectedLength));
			return sized;
		}
	}

	const converted = new Uint8Array(expectedLength);
	const pixelCount = width * height;
	const sourceChannels = inferSourceChannelCount(data, pixelCount, info.channelCount);
	for (let pixel = 0; pixel < pixelCount; pixel++) {
		for (let channel = 0; channel < info.channelCount; channel++) {
			const srcIndex = pixel * sourceChannels + Math.min(channel, sourceChannels - 1);
			converted[pixel * info.channelCount + channel] = toUnorm8Value(
				data[srcIndex] ?? defaultChannelValue(channel)
			);
		}
	}

	return converted;
}

function toFloat16TextureData(
	data: TextureData | null,
	width: number,
	height: number,
	info: TextureFormatInfo
): Uint8Array {
	const expectedLength = width * height * info.channelCount;
	const output = new Uint8Array(expectedLength * 2);
	if (!data) {
		return output;
	}

	const view = new DataView(output.buffer);
	const pixelCount = width * height;
	const sourceChannels = inferSourceChannelCount(data, pixelCount, info.channelCount);
	for (let pixel = 0; pixel < pixelCount; pixel++) {
		for (let channel = 0; channel < info.channelCount; channel++) {
			const srcIndex = pixel * sourceChannels + Math.min(channel, sourceChannels - 1);
			const dstIndex = pixel * info.channelCount + channel;
			view.setUint16(
				dstIndex * 2,
				float32ToFloat16Bits(toFloatValue(data, srcIndex, channel)),
				true
			);
		}
	}
	return output;
}

function toFloat32TextureData(
	data: TextureData | null,
	width: number,
	height: number,
	info: TextureFormatInfo
): Uint8Array {
	const expectedLength = width * height * info.channelCount;
	const output = new Float32Array(expectedLength);
	if (!data) {
		return new Uint8Array(output.buffer);
	}
	const pixelCount = width * height;
	const sourceChannels = inferSourceChannelCount(data, pixelCount, info.channelCount);
	for (let pixel = 0; pixel < pixelCount; pixel++) {
		for (let channel = 0; channel < info.channelCount; channel++) {
			const srcIndex = pixel * sourceChannels + Math.min(channel, sourceChannels - 1);
			output[pixel * info.channelCount + channel] = toFloatValue(
				data,
				srcIndex,
				channel
			);
		}
	}
	return new Uint8Array(output.buffer);
}

function toRawOrConvertedTextureData(
	data: TextureData | null,
	width: number,
	height: number,
	format: TextureFormat,
	info: TextureFormatInfo
): Uint8Array {
	if (
		info.componentType === "uint" ||
		info.componentType === "sint" ||
		info.componentType === "snorm" ||
		info.componentType === "unorm"
	) {
		return toIntegerComponentTextureData(data, width, height, info);
	}
	return toRawTextureData(data, width, height, format);
}

function toIntegerComponentTextureData(
	data: TextureData | null,
	width: number,
	height: number,
	info: TextureFormatInfo
): Uint8Array {
	const expectedByteLength = getTextureFormatLevelByteLength(
		info.format,
		width,
		height
	);
	if (!data) {
		return new Uint8Array(expectedByteLength);
	}
	const bytesPerComponent = info.bytesPerBlock / info.channelCount;
	if (!Number.isInteger(bytesPerComponent) || bytesPerComponent < 1) {
		return toRawTextureData(data, width, height, info.format);
	}
	const raw = tryUseRawData(data, expectedByteLength);
	if (raw) {
		return raw;
	}
	const pixelCount = width * height;
	const sourceChannels = inferSourceChannelCount(data, pixelCount, info.channelCount);
	const output = new Uint8Array(expectedByteLength);
	const view = new DataView(output.buffer);
	for (let pixel = 0; pixel < pixelCount; pixel++) {
		for (let channel = 0; channel < info.channelCount; channel++) {
			const srcIndex = pixel * sourceChannels + Math.min(channel, sourceChannels - 1);
			const dstOffset = (pixel * info.channelCount + channel) * bytesPerComponent;
			writeIntegerComponent(
				view,
				dstOffset,
				bytesPerComponent,
				info.componentType,
				data[srcIndex] ?? defaultChannelValue(channel)
			);
		}
	}
	return output;
}

function toRawTextureData(
	data: TextureData | null,
	width: number,
	height: number,
	format: TextureFormat
): Uint8Array {
	const expectedByteLength = getTextureFormatLevelByteLength(format, width, height);
	if (!data) {
		return new Uint8Array(expectedByteLength);
	}
	const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
	if (bytes.byteLength === expectedByteLength) {
		return bytes;
	}
	const resized = new Uint8Array(expectedByteLength);
	resized.set(bytes.subarray(0, expectedByteLength));
	return resized;
}

function tryUseRawData(
	data: TextureData,
	expectedByteLength: number
): Uint8Array | null {
	if (data.byteLength !== expectedByteLength) {
		return null;
	}
	return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

function inferSourceChannelCount(
	data: TextureData,
	pixelCount: number,
	targetChannels: number
): number {
	if (data.length >= pixelCount * 4) {
		return 4;
	}
	if (data.length >= pixelCount * targetChannels) {
		return targetChannels;
	}
	if (data.length >= pixelCount * 2) {
		return 2;
	}
	return 1;
}

function defaultChannelValue(channel: number): number {
	return channel === 3 ? 1 : 0;
}

function toUnorm8Value(value: number): number {
	if (Number.isInteger(value) && value >= 0 && value <= 255) {
		return value;
	}
	return clamp(Math.round(value * 255), 0, 255);
}

function toFloatValue(data: TextureData, index: number, channel: number): number {
	const value = data[index] ?? defaultChannelValue(channel);
	if (data instanceof Float32Array) {
		return value;
	}
	return value / 255;
}

function writeIntegerComponent(
	view: DataView,
	offset: number,
	bytesPerComponent: number,
	componentType: TextureFormatInfo["componentType"],
	value: number
): void {
	const normalized =
		componentType === "unorm" || componentType === "snorm" ?
			clamp(value, componentType === "snorm" ? -1 : 0, 1)
		:	value;
	switch (bytesPerComponent) {
		case 1:
			if (componentType === "sint" || componentType === "snorm") {
				view.setInt8(offset, Math.round(normalized * 127));
			} else {
				view.setUint8(offset, Math.round(normalized * 255));
			}
			break;
		case 2:
			if (componentType === "sint" || componentType === "snorm") {
				view.setInt16(offset, Math.round(normalized * 32767), true);
			} else {
				view.setUint16(offset, Math.round(normalized * 65535), true);
			}
			break;
		case 4:
			if (componentType === "sint") {
				view.setInt32(offset, Math.round(value), true);
			} else {
				view.setUint32(offset, Math.round(value), true);
			}
			break;
		default:
			break;
	}
}
