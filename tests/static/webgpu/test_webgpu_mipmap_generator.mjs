import assert from "node:assert/strict";

import { Texture } from "../../../src/core/Texture.ts";
import { TextureFormat, TextureUsage } from "../../../src/backends/types.ts";
import { WEBGPU_TEXTURE_SLOT } from "../../../src/backends/webgpu/constants.ts";
import { WebGPUMipmapGenerator } from "../../../src/backends/webgpu/WebGPUMipmapGenerator.ts";
import { WebGPUTextureRegistry } from "../../../src/backends/webgpu/WebGPUTextureRegistry.ts";

import { FakeWebGPUBackend } from "../../helpers/fakes.mjs";

async function waitForCondition(predicate, message, count = 32) {
	for (let i = 0; i < count; i++) {
		if (predicate()) {
			return;
		}
		await Promise.resolve();
		await new Promise((resolve) => setTimeout(resolve, 0));
	}
	assert.ok(predicate(), message);
}

async function testRegistryAutoGeneratesMipmapChainForMipmapMinFilter() {
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend, backend);
	const texture = new Texture({
		data: new Uint8ClampedArray(8 * 4 * 4).fill(255),
		width: 8,
		height: 4,
		colorSpace: "sRGB",
	});
	texture.minFilter = "LinearMipmapLinear";

	registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
	await waitForCondition(
		() => backend.recordedRenderPasses.length === 3,
		"Expected async mipmap generation to record all mip passes"
	);

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
	assert.equal(backend.bindingGroupDestroyCalls, 3);

	registry.destroy();
}

function testRegistryKeepsExplicitMipmapsAuthoritative() {
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend, backend);
	const texture = new Texture({
		data: new Uint8ClampedArray(8 * 4 * 4).fill(255),
		width: 8,
		height: 4,
		colorSpace: "sRGB",
	});
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
	const registry = new WebGPUTextureRegistry(backend, backend);
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

async function testGeneratorCachesPipelinePerFormat() {
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
		await generator.generate(textureA, TextureFormat.RGBA8Unorm, 4),
		true
	);
	assert.equal(backend.renderPipelines.length, 1);
	assert.equal(
		await generator.generate(textureB, TextureFormat.RGBA8Unorm, 3),
		true
	);
	assert.equal(backend.renderPipelines.length, 1);
	assert.equal(backend.recordedRenderPasses.length, 5);
	assert.equal(backend.bindingGroupDestroyCalls, 5);

	generator.destroy();
}

async function testGeneratorCoalescesConcurrentPipelineCreation() {
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
		label: "ConcurrentMipA",
	});
	const textureB = backend.createTexture({
		width: 4,
		height: 4,
		format: TextureFormat.RGBA8Unorm,
		usage,
		mipLevelCount: 3,
		label: "ConcurrentMipB",
	});

	const results = await Promise.all([
		generator.generate(textureA, TextureFormat.RGBA8Unorm, 4),
		generator.generate(textureB, TextureFormat.RGBA8Unorm, 3),
	]);

	assert.deepEqual(results, [true, true]);
	assert.equal(backend.shaderModules.length, 1);
	assert.equal(backend.renderPipelines.length, 1);
	assert.equal(backend.recordedRenderPasses.length, 5);
	assert.equal(backend.bindingGroupDestroyCalls, 5);

	generator.destroy();
}

async function run() {
	await testRegistryAutoGeneratesMipmapChainForMipmapMinFilter();
	testRegistryKeepsExplicitMipmapsAuthoritative();
	testRegistrySkipsUnsupportedMipmapFormat();
	await testGeneratorCachesPipelinePerFormat();
	await testGeneratorCoalescesConcurrentPipelineCreation();
	console.log("WebGPU mipmap generator tests passed");
}

run();
