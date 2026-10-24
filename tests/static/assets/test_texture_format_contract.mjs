import assert from "node:assert/strict";

import { Texture, resolveTextureSamplingFormat } from "../../../src/core/Texture.ts";
import { CubeTexture } from "../../../src/core/CubeTexture.ts";
import { float16BitsToFloat32 } from "../../../src/foundation/Float16.ts";
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
