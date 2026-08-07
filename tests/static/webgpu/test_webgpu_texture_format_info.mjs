import assert from "node:assert/strict";

import { TextureFormat } from "../../../src/backends/types.ts";
import { getTextureFormatInfo } from "../../../src/backends/TextureFormatInfo.ts";
import {
	getWebGPURenderTargetPixelByteCost,
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

testWebGPURenderTargetCostStaysBackendOwned();
console.log("WebGPU texture format info tests passed");
