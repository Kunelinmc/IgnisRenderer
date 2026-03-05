import { clamp } from '../../../maths/Common'
import type { Texture } from '../../Texture'

export interface WebGPUTextureUploadLevel {
	data: Uint8Array
	bytesPerRow: number
	width: number
	height: number
	mipLevel: number
}

export function createTextureUploadData(texture: Texture): {
	data: Uint8Array
	bytesPerRow: number
	width: number
	height: number
} {
	const baseLevel = createTextureMipUploadData(texture, 0)
	return {
		data: baseLevel.data,
		bytesPerRow: baseLevel.bytesPerRow,
		width: baseLevel.width,
		height: baseLevel.height,
	}
}

export function createTextureMipUploadLevels(
	texture: Texture
): WebGPUTextureUploadLevel[] {
	const mipCount = Math.max(1, texture.mipmaps.length || 1)
	const levels: WebGPUTextureUploadLevel[] = []
	for (let mipLevel = 0; mipLevel < mipCount; mipLevel++) {
		levels.push(createTextureMipUploadData(texture, mipLevel))
	}
	return levels
}

export function createTextureMipUploadData(
	texture: Texture,
	mipLevel: number
): WebGPUTextureUploadLevel {
	const level = Math.max(0, mipLevel | 0)
	const width = Math.max(1, texture.width >> level)
	const height = Math.max(1, texture.height >> level)
	const sourceData =
		texture.mipmaps[level] ??
		(level === 0 ? texture.data : null) ??
		texture.mipmaps[0] ??
		null
	const pixelData = toUint8TextureData(sourceData, width, height)
	const bytesPerPixel = 4
	const unalignedBytesPerRow = width * bytesPerPixel
	const bytesPerRow = alignTo(unalignedBytesPerRow, 256)

	if (bytesPerRow === unalignedBytesPerRow) {
		return {
			data: pixelData,
			bytesPerRow,
			width,
			height,
			mipLevel: level,
		}
	}

	const padded = new Uint8Array(bytesPerRow * height)
	for (let y = 0; y < height; y++) {
		const srcStart = y * unalignedBytesPerRow
		const dstStart = y * bytesPerRow
		padded.set(
			pixelData.subarray(srcStart, srcStart + unalignedBytesPerRow),
			dstStart
		)
	}

	return {
		data: padded,
		bytesPerRow,
		width,
		height,
		mipLevel: level,
	}
}

export function alignTo(value: number, alignment: number): number {
	return Math.ceil(value / alignment) * alignment
}

function toUint8TextureData(
	data: Texture['data'] | null,
	width: number,
	height: number
): Uint8Array {
	if (!data) {
		return new Uint8Array(width * height * 4)
	}

	const expectedLength = width * height * 4

	if (data instanceof Uint8Array && !(data instanceof Uint8ClampedArray)) {
		if (data.length === expectedLength) return data
		const copy = new Uint8Array(expectedLength)
		copy.set(data.subarray(0, expectedLength))
		return copy
	}

	if (data instanceof Uint8ClampedArray) {
		const copy = new Uint8Array(
			data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)
		)
		if (copy.length === expectedLength) return copy
		const sized = new Uint8Array(expectedLength)
		sized.set(copy.subarray(0, expectedLength))
		return sized
	}

	const converted = new Uint8Array(expectedLength)
	for (let i = 0; i < expectedLength; i++) {
		converted[i] = clamp(Math.round((data[i] ?? 0) * 255), 0, 255)
	}

	return converted
}
