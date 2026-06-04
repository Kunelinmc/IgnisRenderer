import assert from "node:assert/strict";

import { Texture } from "../src/core/Texture.ts";
import { float16BitsToFloat32 } from "../src/foundation/Float16.ts";
import {
	TextureFormat,
	TextureUsage,
} from "../src/renderers/types.ts";
import {
	getTextureFormatBlockCount,
	getTextureFormatBytesPerRow,
	getTextureFormatInfo,
	textureFormatRequiresFeature,
} from "../src/renderers/TextureFormatInfo.ts";
import {
	createTextureMipUploadData,
} from "../src/renderers/webgpu/texture.ts";
import { ComputeRuntime } from "../src/renderers/webgpu/ComputeRuntime.ts";
import { FakeWebGPUBackend } from "./helpers/test_fakes.mjs";

function nearlyEqual(actual, expected, epsilon = 1e-4) {
	assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} != ${expected}`);
}

function testTextureDescriptorKeepsMipLevelMetadata() {
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

	assert.equal(texture.format, TextureFormat.R8Unorm);
	assert.equal(texture.data?.[0], 32);
	assert.equal(texture.mipmaps.length, 1);
	assert.equal(texture.getMipLevelDescriptor(0)?.width, 2);
	assert.equal(texture.getMipLevelDescriptor(0)?.data?.[1], 200);
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
	assert.equal(upload.bytesPerRow, 256);
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
	assert.equal(upload.bytesPerRow, 256);
	assert.equal(upload.data.byteLength, 256 * 3);
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
	nearlyEqual(float16BitsToFloat32(view.getUint16(0, true)), 0.25);
	nearlyEqual(float16BitsToFloat32(view.getUint16(2, true)), 2);
}

async function testComputeRuntimeReadsNarrowFormatsAsRGBA() {
	const backend = new FakeWebGPUBackend();
	const runtime = new ComputeRuntime(backend);
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
	const runtime = new ComputeRuntime(backend);
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
	testTextureDescriptorKeepsMipLevelMetadata();
	testTextureFormatRegistryHandlesCompressedBlocks();
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
