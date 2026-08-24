import assert from "node:assert/strict";
import { CanvasTexture } from "../../../src/core/CanvasTexture.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { WebGPUTextureRegistry } from "../../../src/backends/webgpu/WebGPUTextureRegistry.ts";
import { WEBGPU_TEXTURE_SLOT } from "../../../src/backends/webgpu/constants.ts";
import { TextureFormat } from "../../../src/core/TextureFormat.ts";
import { TextureUsage } from "../../../src/backends/types.ts";
import { sampleSoftwareTextureMap } from "../../../src/shaders/software/textureSampling.ts";

import { FakeCanvasContext2D, FakeWebGPUBackend, FakeCanvas } from "../../helpers/fakes.mjs";

function createFakeContext(width = 2, height = 1) {
	const canvas = new FakeCanvas(width, height);
	const context = canvas.getContext("2d");
	return { canvas, context };
}

function testCanvasTextureRejectsLegacyPositionalInitialization() {
	const { context } = createFakeContext();
	assert.throws(
		() => new CanvasTexture(context),
		/CanvasTexture requires a parameter object/
	);
}

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

function testCanvasTextureTracksContextMutations() {
	const { context } = createFakeContext();
	const texture = new CanvasTexture({ context });
	try {
		assert.equal(texture.data, null);
		assert.equal(texture.sourceKind, "dynamic");
		assert.deepEqual(texture.levels, []);
		assert.deepEqual(texture.mipmaps, []);
		assert.equal(texture.getUploadSource(), texture.canvas);
		assert.equal(context.getImageDataCalls, 0);
		assert.equal(texture.update(1), false);
		assert.equal(context.getImageDataCalls, 0);

		context.fillRect(0, 0, 1, 1);
		assert.equal(texture.update(16), false);
		assert.equal(context.getImageDataCalls, 0);

		const sampled = texture.sample(0.1, 0.5);
		assert.equal(sampled.r, 24);
		assert.equal(sampled.g, 24);
		assert.equal(sampled.b, 24);
		assert.equal(sampled.a, 255);
	} finally {
		texture.dispose();
	}
}

function testCanvasTextureRespectsUpdateInterval() {
	const { context } = createFakeContext();
	const texture = new CanvasTexture({
		context,
		autoUpdate: false,
		minUpdateIntervalMs: 20,
	});
	try {
		assert.equal(context.getImageDataCalls, 0);
		context.fillRect(0, 0, 1, 1);
		assert.equal(texture.update(5), false);
		assert.equal(context.getImageDataCalls, 0);

		assert.equal(texture.update(24), true);
		assert.equal(context.getImageDataCalls, 0);
	} finally {
		texture.dispose();
	}
}

function testCanvasTextureDynamicUpdateIntegration() {
	const { context } = createFakeContext();
	const texture = new CanvasTexture({ context });
	try {
		const initialRevision = Texture.contentRevision;
		context.drawImage(null, 0, 0);
		assert.ok(Texture.contentRevision > initialRevision);
		assert.equal(texture.data, null);
	} finally {
		texture.dispose();
	}
}

function testSoftwareSamplingCachesExternalPixelsByVersion() {
	const { context } = createFakeContext();
	const texture = new CanvasTexture({ context });
	try {
		const first = sampleSoftwareTextureMap(texture, 0.1, 0.5);
		const second = sampleSoftwareTextureMap(texture, 0.1, 0.5);
		assert.equal(first?.r, 8);
		assert.equal(second?.r, 8);
		assert.equal(context.getImageDataCalls, 1);

		context.fillRect(0, 0, 1, 1);
		const updated = sampleSoftwareTextureMap(texture, 0.1, 0.5);
		assert.equal(updated?.r, 24);
		assert.equal(context.getImageDataCalls, 2);
		assert.equal(texture.data, null);
	} finally {
		texture.dispose();
	}
}

function testWebGPURegistryUsesExternalCanvasUploadPath() {
	const { context } = createFakeContext();
	const texture = new CanvasTexture({ context });
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend, backend);

	try {
		registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
		assert.equal(backend.createTextureCalls.length, 1);
		assert.equal(backend.createTextureCalls[0].format, TextureFormat.RGBA8Unorm);
		assert.ok(
			(backend.createTextureCalls[0].usage & TextureUsage.CopyDst) !== 0
		);
		assert.ok(
			(backend.createTextureCalls[0].usage & TextureUsage.RenderAttachment) !==
				0
		);
		assert.equal(backend.copyCalls.length, 1);
		assert.equal(backend.writeCalls.length, 0);
		assert.equal(context.getImageDataCalls, 0);

		context.fillRect(0, 0, 1, 1);
		registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
		assert.equal(backend.copyCalls.length, 2);
		assert.equal(backend.writeCalls.length, 0);
		assert.equal(context.getImageDataCalls, 0);
	} finally {
		texture.dispose();
	}
}

async function testWebGPURegistryGeneratesMipmapsAfterCanvasUpload() {
	const { context } = createFakeContext();
	const texture = new CanvasTexture({ context });
	texture.minFilter = "LinearMipmapLinear";
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend, backend);

	try {
		registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
		assert.equal(backend.createTextureCalls[0].mipLevelCount, 2);
		assert.equal(backend.copyCalls.length, 1);
		assert.equal(backend.writeCalls.length, 0);
		await waitForCondition(
			() => backend.recordedRenderPasses.length === 1,
			"Expected canvas upload mipmap pass to be recorded"
		);
		assert.equal(backend.recordedRenderPasses.length, 1);

		context.fillRect(0, 0, 1, 1);
		registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
		assert.equal(backend.copyCalls.length, 2);
		await waitForCondition(
			() => backend.recordedRenderPasses.length === 2,
			"Expected updated canvas upload mipmap pass to be recorded"
		);
		assert.equal(backend.recordedRenderPasses.length, 2);
	} finally {
		texture.dispose();
		registry.destroy();
	}
}

async function run() {
	testCanvasTextureRejectsLegacyPositionalInitialization();
	testCanvasTextureTracksContextMutations();
	testCanvasTextureRespectsUpdateInterval();
	testCanvasTextureDynamicUpdateIntegration();
	testSoftwareSamplingCachesExternalPixelsByVersion();
	testWebGPURegistryUsesExternalCanvasUploadPath();
	await testWebGPURegistryGeneratesMipmapsAfterCanvasUpload();
	console.log("Canvas texture tests passed");
}

run();
