import assert from "node:assert/strict";

import {
	getTextureFormatInfo,
	TextureFormat,
} from "../../../src/core/TextureFormat.ts";
import {
	getWebGPURenderTargetPixelByteCost,
	supportsWebGPURenderAttachmentFormat,
	supportsWebGPUStorageTextureFormat,
} from "../../../src/backends/webgpu/WebGPUTextureFormatInfo.ts";

function testWebGPURenderTargetCostStaysBackendOwned() {
	assert.equal(getTextureFormatInfo(TextureFormat.RGBA8Unorm).bytesPerBlock, 4);
	assert.equal(
		getWebGPURenderTargetPixelByteCost(TextureFormat.RGBA8Unorm),
		8
	);
	assert.equal(
		getWebGPURenderTargetPixelByteCost(TextureFormat.RGBA16Float),
		8
	);
	assert.equal(
		getWebGPURenderTargetPixelByteCost(TextureFormat.RGB10A2Unorm),
		8
	);
	assert.equal(
		getWebGPURenderTargetPixelByteCost(TextureFormat.RGBA8Uint),
		4
	);
	assert.equal(
		getWebGPURenderTargetPixelByteCost(TextureFormat.Depth32Float),
		0
	);
	assert.equal(
		getWebGPURenderTargetPixelByteCost(TextureFormat.BC1RGBAUnorm),
		0
	);
}

function testStorageFormatSupportUsesWebGPUFeatures() {
	const tier1 = new Set(["texture-formats-tier1"]);
	const tier2 = new Set(["texture-formats-tier2"]);
	const bgra = new Set(["bgra8unorm-storage"]);
	assert.equal(supportsWebGPUStorageTextureFormat(TextureFormat.RGBA8Snorm), true);
	assert.equal(supportsWebGPUStorageTextureFormat(TextureFormat.RGBA16Float), true);
	for (const format of [TextureFormat.RG11B10UFloat, TextureFormat.R8Snorm,
		TextureFormat.RGB10A2Unorm, TextureFormat.RGBA16Snorm]) {
		assert.equal(supportsWebGPUStorageTextureFormat(format), false);
		assert.equal(supportsWebGPUStorageTextureFormat(format, tier1), true);
		assert.equal(supportsWebGPUStorageTextureFormat(format, tier2), true);
	}
	assert.equal(supportsWebGPUStorageTextureFormat(TextureFormat.BGRA8Unorm, tier1), false);
	assert.equal(supportsWebGPUStorageTextureFormat(TextureFormat.BGRA8Unorm, bgra), true);
	for (const format of [TextureFormat.RGB9E5UFloat, TextureFormat.RGBA8UnormSrgb,
		TextureFormat.Depth32Float, TextureFormat.BC1RGBAUnorm]) {
		assert.equal(supportsWebGPUStorageTextureFormat(format, tier1), false);
	}
}

function testRenderAttachmentSupportUsesEnabledWebGPUFeatures() {
	const tier1 = new Set(["texture-formats-tier1"]);
	const tier2 = new Set(["texture-formats-tier2"]);
	const renderable = new Set(["rg11b10ufloat-renderable"]);
	for (const format of [TextureFormat.RGBA8Unorm, TextureFormat.RGBA8Uint,
		TextureFormat.R32Sint, TextureFormat.RGBA16Float, TextureFormat.RGBA32Float,
		TextureFormat.Stencil8, TextureFormat.Depth16Unorm, TextureFormat.Depth32Float]) {
		assert.equal(supportsWebGPURenderAttachmentFormat(format), true, format);
	}
	for (const format of [TextureFormat.R8Snorm, TextureFormat.RG8Snorm,
		TextureFormat.RGBA8Snorm, TextureFormat.R16Unorm, TextureFormat.R16Snorm,
		TextureFormat.RG16Unorm, TextureFormat.RG16Snorm, TextureFormat.RGBA16Unorm,
		TextureFormat.RGBA16Snorm, TextureFormat.RG11B10UFloat]) {
		assert.equal(supportsWebGPURenderAttachmentFormat(format), false, format);
		assert.equal(supportsWebGPURenderAttachmentFormat(format, tier1), true, format);
		assert.equal(supportsWebGPURenderAttachmentFormat(format, tier2), true, format);
		assert.equal(supportsWebGPURenderAttachmentFormat(format, renderable),
			format === TextureFormat.RG11B10UFloat, format);
	}
	for (const format of [TextureFormat.RGB9E5UFloat, TextureFormat.BC1RGBAUnorm,
		TextureFormat.ETC2RGBA8Unorm, TextureFormat.ASTC4x4Unorm]) {
		assert.equal(supportsWebGPURenderAttachmentFormat(format, tier2), false, format);
	}
	assert.equal(supportsWebGPURenderAttachmentFormat(TextureFormat.Depth32FloatStencil8), false);
	assert.equal(supportsWebGPURenderAttachmentFormat(TextureFormat.Depth32FloatStencil8,
		new Set(["depth32float-stencil8"])), true);
}

testWebGPURenderTargetCostStaysBackendOwned();
testStorageFormatSupportUsesWebGPUFeatures();
testRenderAttachmentSupportUsesEnabledWebGPUFeatures();
console.log("WebGPU texture format info tests passed");
