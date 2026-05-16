import { clamp } from "../../maths/Common";
import type { Texture } from "../../core/Texture";
import { float32ToFloat16Bits } from "../../foundation/Float16";
import { TextureFormat } from "../types";

export interface WebGPUTextureUploadLevel {
	data: Uint8Array;
	bytesPerRow: number;
	width: number;
	height: number;
	mipLevel: number;
	format: TextureFormat.RGBA8Unorm | TextureFormat.RGBA16Float;
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
 * @returns `rgba16float` for Float32-backed textures, otherwise `rgba8unorm`.
 */
export function resolveWebGPUTextureUploadFormat(
	texture: Texture
): TextureFormat.RGBA8Unorm | TextureFormat.RGBA16Float {
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

export function createTextureMipUploadLevels(
	texture: Texture,
	format: TextureFormat.RGBA8Unorm | TextureFormat.RGBA16Float =
		resolveWebGPUTextureUploadFormat(texture)
): WebGPUTextureUploadLevel[] {
	const mipCount = Math.max(1, texture.mipmaps.length || 1);
	const levels: WebGPUTextureUploadLevel[] = [];
	for (let mipLevel = 0; mipLevel < mipCount; mipLevel++) {
		levels.push(createTextureMipUploadData(texture, mipLevel, format));
	}
	return levels;
}

export function createTextureMipUploadData(
	texture: Texture,
	mipLevel: number,
	format: TextureFormat.RGBA8Unorm | TextureFormat.RGBA16Float =
		resolveWebGPUTextureUploadFormat(texture)
): WebGPUTextureUploadLevel {
	const level = Math.max(0, mipLevel | 0);
	const width = Math.max(1, texture.width >> level);
	const height = Math.max(1, texture.height >> level);
	const sourceData =
		texture.mipmaps[level] ??
		(level === 0 ? texture.data : null) ??
		texture.mipmaps[0] ??
		null;
	const pixelData =
		format === TextureFormat.RGBA16Float ?
			toRGBA16FloatTextureData(sourceData, width, height)
		:	toRGBA8TextureData(sourceData, width, height);
	const bytesPerPixel = format === TextureFormat.RGBA16Float ? 8 : 4;
	const unalignedBytesPerRow = width * bytesPerPixel;
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

function toRGBA8TextureData(
	data: Texture["data"] | null,
	width: number,
	height: number
): Uint8Array {
	if (!data) {
		return new Uint8Array(width * height * 4);
	}

	const expectedLength = width * height * 4;

	if (data instanceof Uint8Array && !(data instanceof Uint8ClampedArray)) {
		if (data.length === expectedLength) return data;
		const copy = new Uint8Array(expectedLength);
		copy.set(data.subarray(0, expectedLength));
		return copy;
	}

	if (data instanceof Uint8ClampedArray) {
		const copy = new Uint8Array(
			data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)
		);
		if (copy.length === expectedLength) return copy;
		const sized = new Uint8Array(expectedLength);
		sized.set(copy.subarray(0, expectedLength));
		return sized;
	}

	const converted = new Uint8Array(expectedLength);
	for (let i = 0; i < expectedLength; i++) {
		converted[i] = clamp(Math.round((data[i] ?? 0) * 255), 0, 255);
	}

	return converted;
}

function toRGBA16FloatTextureData(
	data: Texture["data"] | null,
	width: number,
	height: number
): Uint8Array {
	const expectedLength = width * height * 4;
	const output = new Uint8Array(expectedLength * 2);
	if (!data) {
		return output;
	}

	const view = new DataView(output.buffer);
	for (let i = 0; i < expectedLength; i++) {
		const value =
			data instanceof Float32Array ?
				data[i] ?? 0
			:	(data[i] ?? 0) / 255;
		view.setUint16(i * 2, float32ToFloat16Bits(value), true);
	}
	return output;
}
