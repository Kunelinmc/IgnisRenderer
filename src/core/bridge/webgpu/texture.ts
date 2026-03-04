import { clamp } from '../../../maths/Common'
import type { Texture } from '../../Texture'

export function createTextureUploadData(texture: Texture): {
	data: Uint8Array
	bytesPerRow: number
	width: number
	height: number
} {
	const pixelData = toUint8TextureData(texture)
	const bytesPerPixel = 4
	const unalignedBytesPerRow = texture.width * bytesPerPixel
	const bytesPerRow = alignTo(unalignedBytesPerRow, 256)

	if (bytesPerRow === unalignedBytesPerRow) {
		return {
			data: pixelData,
			bytesPerRow,
			width: texture.width,
			height: texture.height,
		}
	}

	const padded = new Uint8Array(bytesPerRow * texture.height)
	for (let y = 0; y < texture.height; y++) {
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
		width: texture.width,
		height: texture.height,
	}
}

export function alignTo(value: number, alignment: number): number {
	return Math.ceil(value / alignment) * alignment
}

function toUint8TextureData(texture: Texture): Uint8Array {
	const data = texture.data

	if (!data) {
		return new Uint8Array(texture.width * texture.height * 4)
	}

	if (data instanceof Uint8Array && !(data instanceof Uint8ClampedArray)) {
		return data
	}

	if (data instanceof Uint8ClampedArray) {
		return new Uint8Array(
			data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)
		)
	}

	const converted = new Uint8Array(texture.width * texture.height * 4)
	for (let i = 0; i < converted.length; i++) {
		converted[i] = clamp(Math.round(data[i] * 255), 0, 255)
	}

	return converted
}
