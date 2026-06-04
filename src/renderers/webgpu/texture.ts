import { clamp } from "../../maths/Common";
import type { Texture, TextureData } from "../../core/Texture";
import { float32ToFloat16Bits } from "../../foundation/Float16";
import { TextureFormat } from "../types";
import {
	getTextureFormatBytesPerRow,
	getTextureFormatInfo,
	getTextureFormatLevelByteLength,
	type TextureFormatInfo,
} from "../TextureFormatInfo";

export interface WebGPUTextureUploadLevel {
	data: Uint8Array;
	bytesPerRow: number;
	width: number;
	height: number;
	mipLevel: number;
	format: TextureFormat;
}

export function createTextureUploadData(texture: Texture): {
	data: Uint8Array;
	bytesPerRow: number;
	width: number;
	height: number;
} {
	const baseLevel = createTextureMipUploadData(texture, 0);
	return {
		data: baseLevel.data,
		bytesPerRow: baseLevel.bytesPerRow,
		width: baseLevel.width,
		height: baseLevel.height,
	};
}

/**
 * Resolves the WebGPU upload texture format needed to preserve source data.
 *
 * @param texture - Texture metadata and mip data to upload.
 * @returns Explicit texture format metadata, falling back to legacy inference.
 */
export function resolveWebGPUTextureUploadFormat(texture: Texture): TextureFormat {
	return texture.format ?? resolveLegacyTextureUploadFormat(texture);
}

export function createTextureMipUploadLevels(
	texture: Texture,
	format: TextureFormat = resolveWebGPUTextureUploadFormat(texture)
): WebGPUTextureUploadLevel[] {
	const mipCount = Math.max(
		1,
		texture.levels.length || texture.mipmaps.length || 1
	);
	const levels: WebGPUTextureUploadLevel[] = [];
	for (let mipLevel = 0; mipLevel < mipCount; mipLevel++) {
		levels.push(createTextureMipUploadData(texture, mipLevel, format));
	}
	return levels;
}

export function createTextureMipUploadData(
	texture: Texture,
	mipLevel: number,
	format: TextureFormat = resolveWebGPUTextureUploadFormat(texture)
): WebGPUTextureUploadLevel {
	const level = Math.max(0, mipLevel | 0);
	const levelDescriptor = texture.getMipLevelDescriptor(level);
	const width = Math.max(1, Math.floor(levelDescriptor?.width ?? texture.width >> level));
	const height = Math.max(
		1,
		Math.floor(levelDescriptor?.height ?? texture.height >> level)
	);
	const sourceData = levelDescriptor?.data ?? null;
	const pixelData = createTextureFormatUploadData(sourceData, width, height, format);
	const unalignedBytesPerRow =
		levelDescriptor?.bytesPerRow ?? getTextureFormatBytesPerRow(format, width);
	const bytesPerRow = alignTo(unalignedBytesPerRow, 256);

	if (bytesPerRow === unalignedBytesPerRow) {
		return {
			data: pixelData,
			bytesPerRow,
			width,
			height,
			mipLevel: level,
			format,
		};
	}

	const padded = new Uint8Array(bytesPerRow * height);
	for (let y = 0; y < height; y++) {
		const srcStart = y * unalignedBytesPerRow;
		const dstStart = y * bytesPerRow;
		padded.set(
			pixelData.subarray(srcStart, srcStart + unalignedBytesPerRow),
			dstStart
		);
	}

	return {
		data: padded,
		bytesPerRow,
		width,
		height,
		mipLevel: level,
		format,
	};
}

export function alignTo(value: number, alignment: number): number {
	return Math.ceil(value / alignment) * alignment;
}

export function createTextureFormatUploadData(
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

function resolveLegacyTextureUploadFormat(texture: Texture): TextureFormat {
	if (texture.data instanceof Float32Array) {
		return TextureFormat.RGBA16Float;
	}
	for (const mip of texture.mipmaps) {
		if (mip instanceof Float32Array) {
			return TextureFormat.RGBA16Float;
		}
	}
	return TextureFormat.RGBA8Unorm;
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
