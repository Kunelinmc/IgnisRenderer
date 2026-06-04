import assert from "node:assert/strict";

import { Texture } from "../src/core/Texture.ts";
import { TextureFormat, TextureUsage } from "../src/renderers/types.ts";
import { WEBGPU_TEXTURE_SLOT } from "../src/renderers/webgpu/constants.ts";
import { WebGPUMipmapGenerator } from "../src/renderers/webgpu/WebGPUMipmapGenerator.ts";
import { WebGPUTextureRegistry } from "../src/renderers/webgpu/WebGPUTextureRegistry.ts";

import { FakeWebGPUBackend } from "./helpers/test_fakes.mjs";

function testRegistryAutoGeneratesMipmapChainForMipmapMinFilter() {
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend);
	const texture = new Texture(
		new Uint8ClampedArray(8 * 4 * 4).fill(255),
		8,
		4,
		"sRGB"
	);
	texture.minFilter = "LinearMipmapLinear";

	registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);

	assert.equal(backend.createTextureCalls[0].mipLevelCount, 4);
	assert.ok(
		(backend.createTextureCalls[0].usage & TextureUsage.RenderAttachment) !==
			0
	);
	assert.equal(backend.textureWrites.length, 1);
	assert.equal(backend.recordedRenderPasses.length, 3);
	assert.deepEqual(
		backend.textureViews.map((view) => view.desc.baseMipLevel),
		[0, 1, 2, 3]
	);
	assert.equal(backend.submits, 1);

	registry.destroy();
}

function testRegistryKeepsExplicitMipmapsAuthoritative() {
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend);
	const texture = new Texture(
		new Uint8ClampedArray(8 * 4 * 4).fill(255),
		8,
		4,
		"sRGB"
	);
	texture.mipmaps = [
		texture.data,
		new Uint8ClampedArray(4 * 2 * 4).fill(128),
		new Uint8ClampedArray(2 * 1 * 4).fill(64),
	];
	texture.minFilter = "LinearMipmapLinear";

	registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);

	assert.equal(backend.createTextureCalls[0].mipLevelCount, 3);
	assert.equal(
		(backend.createTextureCalls[0].usage & TextureUsage.RenderAttachment),
		0
	);
	assert.equal(backend.textureWrites.length, 3);
	assert.equal(backend.recordedRenderPasses.length, 0);

	registry.destroy();
}

function testRegistrySkipsUnsupportedMipmapFormat() {
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend);
	const texture = new Texture({
		data: new Float32Array(4 * 4 * 4).fill(1),
		width: 4,
		height: 4,
		format: TextureFormat.RGBA32Float,
		colorSpace: "HDR",
	});
	texture.minFilter = "LinearMipmapLinear";

	registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);

	assert.equal(backend.createTextureCalls[0].mipLevelCount, 1);
	assert.equal(
		(backend.createTextureCalls[0].usage & TextureUsage.RenderAttachment),
		0
	);
	assert.equal(backend.textureWrites.length, 1);
	assert.equal(backend.recordedRenderPasses.length, 0);

	registry.destroy();
}

function testGeneratorCachesPipelinePerFormat() {
	const backend = new FakeWebGPUBackend();
	const generator = new WebGPUMipmapGenerator(backend);
	const usage =
		TextureUsage.TextureBinding |
		TextureUsage.CopyDst |
		TextureUsage.RenderAttachment;
	const textureA = backend.createTexture({
		width: 8,
		height: 8,
		format: TextureFormat.RGBA8Unorm,
		usage,
		mipLevelCount: 4,
		label: "MipA",
	});
	const textureB = backend.createTexture({
		width: 4,
		height: 4,
		format: TextureFormat.RGBA8Unorm,
		usage,
		mipLevelCount: 3,
		label: "MipB",
	});

	assert.equal(
		generator.generate(textureA, TextureFormat.RGBA8Unorm, 4),
		true
	);
	assert.equal(backend.renderPipelines.length, 1);
	assert.equal(
		generator.generate(textureB, TextureFormat.RGBA8Unorm, 3),
		true
	);
	assert.equal(backend.renderPipelines.length, 1);
	assert.equal(backend.recordedRenderPasses.length, 5);

	generator.destroy();
}

function run() {
	testRegistryAutoGeneratesMipmapChainForMipmapMinFilter();
	testRegistryKeepsExplicitMipmapsAuthoritative();
	testRegistrySkipsUnsupportedMipmapFormat();
	testGeneratorCachesPipelinePerFormat();
	console.log("WebGPU mipmap generator tests passed");
}

run();
