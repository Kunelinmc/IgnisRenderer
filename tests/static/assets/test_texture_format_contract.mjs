import assert from "node:assert/strict";

import { Texture, resolveTextureSamplingFormat } from "../../../src/core/Texture.ts";
import { CubeTexture } from "../../../src/core/CubeTexture.ts";
import {
	float16BitsToFloat32,
	float32ToFloat16Bits,
} from "../../../src/foundation/Float16.ts";
import { TextureUsage } from "../../../src/backends/types.ts";
import {
	getTextureFormatBlockCount,
	getTextureFormatBytesPerRow,
	getTextureFormatInfo,
	TextureFormat,
	textureFormatRequiresFeature,
} from "../../../src/core/TextureFormat.ts";
import {
	createTextureMipUploadData,
} from "../../../src/backends/webgpu/texture.ts";
import { ComputeRuntime } from "../../../src/backends/webgpu/ComputeRuntime.ts";
import { createWebGPUComputeFacade } from "../../../src/backends/webgpu/ComputeFacade.ts";
import { FakeWebGPUBackend } from "../../helpers/fakes.mjs";

function nearlyEqual(actual, expected, epsilon = 1e-4) {
	assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} != ${expected}`);
}

function testTextureDefaultsPreserveSamplingEncoding() {
	const pixels = new Uint8Array([128, 64, 32, 127]);
	const srgb = new Texture({ data: pixels, width: 1, height: 1 });
	const linear = new Texture({ data: pixels, width: 1, height: 1, colorSpace: "Linear" });
	const hdr = new Texture({ data: new Float32Array([0.5, 2, 4, 1]) });
	assert.equal(srgb.format, TextureFormat.RGBA8UnormSrgb);
	assert.equal(linear.format, TextureFormat.RGBA8Unorm);
	assert.equal(hdr.format, TextureFormat.RGBA16Float);
	assert.equal(hdr.colorSpace, "HDR");
	assert.equal(srgb.data, pixels);
}

function testExplicitFormatInfersSourceEncoding() {
	for (const [format, colorSpace] of [
		[TextureFormat.RGBA8UnormSrgb, "sRGB"],
		[TextureFormat.BC7RGBAUnormSrgb, "sRGB"],
		[TextureFormat.R8Unorm, "Linear"],
		[TextureFormat.RGBA8Unorm, "Linear"],
		[TextureFormat.RGBA16Float, "HDR"],
	]) {
		assert.equal(new Texture({ format }).colorSpace, colorSpace);
	}
}

function testFormatMutationInvalidatesSamplingRevision() {
	const texture = new Texture();
	const revision = texture.samplingRevision;
	texture.format = TextureFormat.RGBA16Float;
	assert.ok(texture.samplingRevision > revision);
}

function testColorAndDataSamplingResolveCompatibleFormats() {
	for (const [linear, srgb] of [
		[TextureFormat.RGBA8Unorm, TextureFormat.RGBA8UnormSrgb],
		[TextureFormat.BGRA8Unorm, TextureFormat.BGRA8UnormSrgb],
		[TextureFormat.BC1RGBAUnorm, TextureFormat.BC1RGBAUnormSrgb],
		[TextureFormat.ETC2RGBA8Unorm, TextureFormat.ETC2RGBA8UnormSrgb],
		[TextureFormat.ASTC5x4Unorm, TextureFormat.ASTC5x4UnormSrgb],
	]) {
		const texture = new Texture({ format: linear, colorSpace: "sRGB" });
		assert.equal(resolveTextureSamplingFormat(texture), srgb);
		assert.equal(resolveTextureSamplingFormat(texture, false), linear);
		texture.format = srgb;
		texture.colorSpace = "Linear";
		assert.equal(resolveTextureSamplingFormat(texture), linear);
		assert.equal(resolveTextureSamplingFormat(texture, false), linear);
	}
}

function testUnsupportedSrgbEncodingIsRejectedBeforeUpload() {
	for (const format of [TextureFormat.R8Unorm, TextureFormat.RGBA16Float]) {
		const texture = new Texture({ format, colorSpace: "sRGB" });
		assert.throws(() => resolveTextureSamplingFormat(texture), /sRGB sampling format/);
		assert.equal(resolveTextureSamplingFormat(texture, false), format);
	}
}

function testCubeReplacementPreservesSamplingEncoding() {
	const faces = Array.from({ length: 6 }, () => new Uint8Array([128, 64, 32, 255]));
	const cube = new CubeTexture({ faces });
	cube.replaceFaces({ faces });
	assert.equal(cube.format, TextureFormat.RGBA8UnormSrgb);
	cube.replaceFaces({ faces, colorSpace: "Linear" });
	assert.equal(cube.format, TextureFormat.RGBA8Unorm);
	const hdrFaces = Array.from({ length: 6 }, () => new Float32Array([2, 4, 1, 1]));
	const hdr = new CubeTexture({ faces: hdrFaces });
	assert.equal(hdr.colorSpace, "HDR");
	assert.equal(hdr.format, TextureFormat.RGBA16Float);
}

function testTextureParamsKeepMipLevelMetadata() {
	const texture = new Texture({
		width: 2,
		height: 1,
		format: TextureFormat.R8Unorm,
		colorSpace: "Linear",
		levels: [
			{
				data: new Uint8Array([32, 200]),
				width: 2,
				height: 1,
			},
		],
	});

	assert.equal(texture.sourceKind, "static");
	assert.equal(texture.format, TextureFormat.R8Unorm);
	assert.equal(texture.data?.[0], 32);
	assert.equal(texture.mipmaps.length, 1);
	assert.equal(texture.getMipLevelDescriptor(0)?.width, 2);
	assert.equal(texture.getMipLevelDescriptor(0)?.data?.[1], 200);
}

function testTextureRejectsLegacyPositionalInitialization() {
	assert.throws(
		() => new Texture(new Uint8Array([255, 255, 255, 255])),
		/Texture requires a parameter object/
	);
}

function testTextureFormatRegistryHandlesCompressedBlocks() {
	const bc1 = getTextureFormatInfo(TextureFormat.BC1RGBAUnorm);
	assert.equal(bc1.isCompressed, true);
	assert.equal(bc1.bytesPerBlock, 8);
	assert.equal(bc1.blockWidth, 4);
	assert.equal(getTextureFormatBytesPerRow(TextureFormat.BC1RGBAUnorm, 8), 16);
	assert.deepEqual(
		getTextureFormatBlockCount(TextureFormat.ASTC5x4Unorm, 11, 9),
		{ width: 3, height: 3 }
	);
	assert.equal(
		textureFormatRequiresFeature(TextureFormat.BC1RGBAUnorm, new Set()),
		true
	);
	assert.equal(
		textureFormatRequiresFeature(
			TextureFormat.BC1RGBAUnorm,
			new Set(["texture-compression-bc"])
		),
		false
	);
}

function testWebGPUUploadExtractsNarrowUnormChannels() {
	const texture = new Texture({
		width: 2,
		height: 1,
		format: TextureFormat.R8Unorm,
		data: new Uint8Array([10, 20, 30, 40, 50, 60, 70, 80]),
	});

	const upload = createTextureMipUploadData(texture, 0);

	assert.equal(upload.format, TextureFormat.R8Unorm);
	assert.equal(upload.bytesPerRow, 2);
	assert.equal(upload.data.byteLength, 2);
	assert.equal(upload.data[0], 10);
	assert.equal(upload.data[1], 50);
}

function testWebGPUUploadRespectsExplicitMipDimensions() {
	const texture = new Texture({
		width: 8,
		height: 8,
		format: TextureFormat.R8Unorm,
		levels: [
			{
				data: new Uint8Array(64),
				width: 8,
				height: 8,
			},
			{
				data: new Uint8Array(9),
				width: 3,
				height: 3,
			},
		],
	});

	const upload = createTextureMipUploadData(texture, 1);

	assert.equal(upload.width, 3);
	assert.equal(upload.height, 3);
	assert.equal(upload.bytesPerRow, 3);
	assert.equal(upload.data.byteLength, 9);
}

function testWebGPUUploadConvertsFloat16Formats() {
	const texture = new Texture({
		width: 1,
		height: 1,
		format: TextureFormat.RG16Float,
		colorSpace: "HDR",
		data: new Float32Array([0.25, 2, 0, 1]),
	});

	const upload = createTextureMipUploadData(texture, 0);
	const view = new DataView(
		upload.data.buffer,
		upload.data.byteOffset,
		upload.data.byteLength
	);

	assert.equal(upload.format, TextureFormat.RG16Float);
	assert.equal(upload.bytesPerRow, 4);
	assert.equal(upload.data.byteLength, 4);
	nearlyEqual(float16BitsToFloat32(view.getUint16(0, true)), 0.25);
	nearlyEqual(float16BitsToFloat32(view.getUint16(2, true)), 2);
}

function testWebGPUUploadReusesTightlyPackedSourceViews() {
	for (const { width, height, length, stride } of [
		{ width: 17, height: 1024, length: 69632, stride: 68 },
		{ width: 64, height: 2, length: 512, stride: 256 },
		{ width: 1, height: 1, length: 4, stride: 4 },
	]) {
		const backing = new Uint8Array(length + 11).fill(255);
		const data = backing.subarray(7, 7 + length);
		for (let i = 0; i < data.length; i++) data[i] = i % 251;
		const texture = new Texture({
			width, height, data, format: TextureFormat.RGBA8Unorm,
		});
		const upload = createTextureMipUploadData(texture, 0);
		assert.equal(upload.bytesPerRow, stride);
		assert.equal(upload.rowsPerImage, height);
		assert.equal(upload.data.buffer, data.buffer, "native uploads must not copy payloads");
		assert.equal(upload.data.byteOffset, 7);
		assert.equal(upload.data.byteLength, length);
		assert.deepEqual(upload.data, data);
		assert.equal(backing[0], 255);
		assert.equal(backing[backing.length - 1], 255);
	}
}

function testWebGPUUploadPreservesExplicitRawRows() {
	for (const { stride, length } of [
		{ stride: 12, length: 20 },
		{ stride: 256, length: 264 },
		{ stride: 9, length: 17 },
	]) {
		const backing = new Uint8Array(length + 5).fill(199);
		const data = backing.subarray(3, 3 + length);
		data.set([1, 2, 3, 4, 5, 6, 7, 8]);
		data.set([21, 22, 23, 24, 25, 26, 27, 28], stride);
		const before = data.slice();
		const texture = new Texture({
			format: TextureFormat.RGBA8UnormSrgb,
			levels: [{ width: 2, height: 2, data, bytesPerRow: stride, rowsPerImage: 5 }],
		});
		const upload = createTextureMipUploadData(texture, 0, TextureFormat.RGBA8Unorm);
		assert.equal(upload.bytesPerRow, stride);
		assert.equal(upload.rowsPerImage, 5);
		assert.equal(upload.data.buffer, data.buffer);
		assert.equal(upload.data.byteOffset, 3);
		assert.equal(upload.data.byteLength, length);
		assert.deepEqual(Array.from(upload.data.subarray(stride, stride + 8)),
			[21, 22, 23, 24, 25, 26, 27, 28]);
		assert.deepEqual(data, before, "upload preparation must not mutate raw bytes");
	}
}

function testWebGPUUploadKeepsNativeFloat16Bytes() {
	const data = new Uint8Array([0, 52, 0, 64]);
	const texture = new Texture({
		width: 1, height: 1, format: TextureFormat.RG16Float, data,
	});
	const upload = createTextureMipUploadData(texture, 0);
	assert.equal(upload.data.buffer, data.buffer);
	assert.deepEqual(Array.from(upload.data), [0, 52, 0, 64]);
}

function testWebGPUUploadConvertsNativeFloatBytesBySourceFormat() {
	const formats = [TextureFormat.R16Float, TextureFormat.RG16Float,
		TextureFormat.RGBA16Float, TextureFormat.R32Float, TextureFormat.RG32Float,
		TextureFormat.RGBA32Float];
	const values = [4, -2, 0.25, 0.5, Infinity, -Infinity, NaN, -0];
	for (const sourceFormat of formats) {
		const sourceInfo = getTextureFormatInfo(sourceFormat);
		const componentBytes = sourceInfo.bytesPerBlock / sourceInfo.channelCount;
		const backing = new Uint8Array(4 * sourceInfo.bytesPerBlock + 9).fill(199);
		const data = backing.subarray(3, backing.length - 6);
		const sourceView = new DataView(data.buffer, data.byteOffset, data.byteLength);
		for (let i = 0; i < 4 * sourceInfo.channelCount; i++) {
			const value = values[i % values.length];
			if (componentBytes === 2) {
				const bits = Number.isFinite(value) ? float32ToFloat16Bits(value) :
					Number.isNaN(value) ? 0x7e00 : value < 0 ? 0xfc00 : 0x7c00;
				sourceView.setUint16(i * 2, bits, true);
			} else {
				sourceView.setFloat32(i * 4, value, true);
			}
		}
		const before = backing.slice();
		const source = new Texture({ width: 2, height: 2, format: sourceFormat, data });
		for (const targetFormat of formats) {
			const targetInfo = getTextureFormatInfo(targetFormat);
			const targetComponentBytes = targetInfo.bytesPerBlock / targetInfo.channelCount;
			const upload = createTextureMipUploadData(source, 0, targetFormat);
			assert.equal(upload.bytesPerRow, 2 * targetInfo.bytesPerBlock);
			assert.equal(upload.rowsPerImage, 2);
			assert.equal(upload.data.byteLength, 4 * targetInfo.bytesPerBlock);
			if (sourceFormat === targetFormat) {
				assert.equal(upload.data.buffer, data.buffer);
				assert.equal(upload.data.byteOffset, 3);
			}
			const view = new DataView(upload.data.buffer, upload.data.byteOffset,
				upload.data.byteLength);
			for (let pixel = 0; pixel < 4; pixel++) {
				for (let channel = 0; channel < targetInfo.channelCount; channel++) {
					const expected = channel < sourceInfo.channelCount ?
						values[(pixel * sourceInfo.channelCount + channel) % values.length] :
						channel === 3 ? 1 : 0;
					const offset = (pixel * targetInfo.channelCount + channel) * targetComponentBytes;
					const actual = targetComponentBytes === 2 ?
						float16BitsToFloat32(view.getUint16(offset, true)) : view.getFloat32(offset, true);
					assert.equal(actual, expected, `${sourceFormat} -> ${targetFormat}`);
				}
			}
			assert.deepEqual(backing, before, "conversion must preserve the full source buffer");
		}
	}
}

function testWebGPUUploadRejectsUnsupportedNativeRawConversions() {
	for (const [sourceFormat, targetFormat] of [
		[TextureFormat.RGBA8Unorm, TextureFormat.RGBA16Float],
		[TextureFormat.RGBA32Uint, TextureFormat.RGBA32Float],
		[TextureFormat.RGBA16Float, TextureFormat.R8Unorm],
	]) {
		const source = new Texture({ width: 1, height: 1, format: sourceFormat,
			data: new Uint8Array(getTextureFormatInfo(sourceFormat).bytesPerBlock) });
		assert.throws(() => createTextureMipUploadData(source, 0, targetFormat),
			/raw.*conversion|conversion.*raw/i);
	}
	const stridedFloat = new Texture({ format: TextureFormat.RGBA32Float,
		levels: [{ width: 1, height: 1, data: new Uint8Array(16), bytesPerRow: 16 }] });
	assert.throws(() => createTextureMipUploadData(stridedFloat, 0, TextureFormat.RGBA16Float),
		/raw.*format|format.*compatible/i);
}

function testWebGPUUploadCountsCompressedBlockRows() {
	const texture = new Texture({
		format: TextureFormat.BC1RGBAUnorm,
		levels: [
			{ width: 8, height: 8, data: new Uint8Array(32).fill(17) },
			{ width: 4, height: 4, data: new Uint8Array(8).fill(29) },
			{ width: 2, height: 2, data: new Uint8Array(8).fill(41) },
			{ width: 1, height: 1, data: new Uint8Array(8).fill(53) },
		],
	});
	for (const [mip, width, height, stride, rows, length, copyWidth, copyHeight] of [
		[0, 8, 8, 16, 2, 32, 8, 8],
		[1, 4, 4, 8, 1, 8, 4, 4],
		[2, 2, 2, 8, 1, 8, 4, 4],
		[3, 1, 1, 8, 1, 8, 4, 4],
	]) {
		const upload = createTextureMipUploadData(texture, mip);
		assert.deepEqual([
			upload.width, upload.height, upload.bytesPerRow, upload.rowsPerImage,
			upload.data.byteLength, upload.copyWidth, upload.copyHeight,
		], [width, height, stride, rows, length, copyWidth, copyHeight]);
		assert.equal(upload.data.buffer, texture.levels[mip].data.buffer);
	}
	const data = new Uint8Array(36).fill(71);
	const strided = new Texture({
		format: TextureFormat.BC1RGBAUnorm,
		levels: [{ width: 8, height: 8, data, bytesPerRow: 20 }],
	});
	const upload = createTextureMipUploadData(strided, 0);
	assert.equal(upload.bytesPerRow, 20);
	assert.equal(upload.rowsPerImage, 2);
	assert.equal(upload.data.byteLength, 36);
	assert.equal(upload.data.buffer, data.buffer);
}

function testWebGPUUploadRejectsInvalidExplicitRawLayouts() {
	for (const layout of [
		{ bytesPerRow: 0 }, { bytesPerRow: -1 }, { bytesPerRow: 7 },
		{ bytesPerRow: 8.5 }, { bytesPerRow: NaN }, { bytesPerRow: Infinity },
		{ rowsPerImage: 0 }, { rowsPerImage: 1 }, { rowsPerImage: 2.5 },
		{ rowsPerImage: NaN }, { rowsPerImage: Infinity },
	]) {
		const texture = new Texture({
			format: TextureFormat.RGBA8Unorm,
			levels: [{ width: 2, height: 2, data: new Uint8Array(32), ...layout }],
		});
		assert.throws(() => createTextureMipUploadData(texture, 0), RangeError);
	}
	const short = new Texture({
		format: TextureFormat.RGBA8Unorm,
		levels: [{ width: 2, height: 2, data: new Uint8Array(19), bytesPerRow: 12 }],
	});
	assert.throws(() => createTextureMipUploadData(short, 0), /payload|data.*short/i);
	const converted = new Texture({
		format: TextureFormat.RGBA8Unorm,
		levels: [{ width: 2, height: 2, data: new Uint8Array(20), bytesPerRow: 12 }],
	});
	assert.throws(() => createTextureMipUploadData(converted, 0, TextureFormat.R8Unorm),
		/raw.*format|format.*compatible/i);
	const nonRaw = new Texture({
		format: TextureFormat.RGBA16Float,
		levels: [{ width: 1, height: 1, data: new Float32Array(4), bytesPerRow: 8 }],
	});
	assert.throws(() => createTextureMipUploadData(nonRaw, 0), /Uint8Array|raw bytes/i);
}

async function testComputeRuntimeReadsNarrowFormatsAsRGBA() {
	const backend = new FakeWebGPUBackend();
	const runtime = new ComputeRuntime(createWebGPUComputeFacade(backend));
	const texture = runtime.createTexture({
		width: 2,
		height: 1,
		format: TextureFormat.R8Unorm,
		usage:
			TextureUsage.CopySrc |
			TextureUsage.CopyDst |
			TextureUsage.TextureBinding,
		label: "R8Readback",
	});
	const data = new Uint8Array(256);
	data[0] = 64;
	data[1] = 128;
	runtime.writeTexture(
		texture,
		data,
		{ bytesPerRow: 256, rowsPerImage: 1 },
		{ width: 2, height: 1 }
	);

	const readback = await runtime.readTexture({
		texture,
		width: 2,
		height: 1,
		format: TextureFormat.R8Unorm,
	});
	const rgba = readback.toRGBAFloat32();

	nearlyEqual(rgba[0], 64 / 255);
	assert.equal(rgba[1], 0);
	assert.equal(rgba[2], 0);
	assert.equal(rgba[3], 1);
	nearlyEqual(rgba[4], 128 / 255);
	assert.equal(rgba[7], 1);
	runtime.destroy();
}

async function testComputeRuntimeRejectsPackedRGBAReadback() {
	const backend = new FakeWebGPUBackend();
	const runtime = new ComputeRuntime(createWebGPUComputeFacade(backend));
	const texture = runtime.createTexture({
		width: 1,
		height: 1,
		format: TextureFormat.RGB10A2Unorm,
		usage:
			TextureUsage.CopySrc |
			TextureUsage.CopyDst |
			TextureUsage.TextureBinding,
		label: "PackedReadback",
	});
	const data = new Uint8Array(256);
	data.set([255, 0, 0, 255]);
	runtime.writeTexture(
		texture,
		data,
		{ bytesPerRow: 256, rowsPerImage: 1 },
		{ width: 1, height: 1 }
	);

	const readback = await runtime.readTexture({
		texture,
		width: 1,
		height: 1,
		format: TextureFormat.RGB10A2Unorm,
	});
	assert.throws(
		() => readback.toRGBAFloat32(),
		/does not support texture format "rgb10a2unorm"/
	);
	runtime.destroy();
}

async function run() {
	testTextureDefaultsPreserveSamplingEncoding();
	testExplicitFormatInfersSourceEncoding();
	testFormatMutationInvalidatesSamplingRevision();
	testColorAndDataSamplingResolveCompatibleFormats();
	testUnsupportedSrgbEncodingIsRejectedBeforeUpload();
	testCubeReplacementPreservesSamplingEncoding();
	testTextureParamsKeepMipLevelMetadata();
	testTextureRejectsLegacyPositionalInitialization();
	testTextureFormatRegistryHandlesCompressedBlocks();
	testWebGPUUploadReusesTightlyPackedSourceViews();
	testWebGPUUploadPreservesExplicitRawRows();
	testWebGPUUploadKeepsNativeFloat16Bytes();
	testWebGPUUploadConvertsNativeFloatBytesBySourceFormat();
	testWebGPUUploadRejectsUnsupportedNativeRawConversions();
	testWebGPUUploadCountsCompressedBlockRows();
	testWebGPUUploadRejectsInvalidExplicitRawLayouts();
	testWebGPUUploadExtractsNarrowUnormChannels();
	testWebGPUUploadRespectsExplicitMipDimensions();
	testWebGPUUploadConvertsFloat16Formats();
	await testComputeRuntimeReadsNarrowFormatsAsRGBA();
	await testComputeRuntimeRejectsPackedRGBAReadback();
	console.log("Texture format contract tests passed");
}

run().catch((error) => {
	console.error(error);
	process.exit(1);
});
