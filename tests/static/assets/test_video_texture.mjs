import assert from "node:assert/strict";
import { VideoTexture } from "../../../src/core/VideoTexture.ts";
import { WebGPUTextureRegistry } from "../../../src/backends/webgpu/WebGPUTextureRegistry.ts";
import { WEBGPU_TEXTURE_SLOT } from "../../../src/backends/webgpu/constants.ts";
import { TextureFormat, TextureUsage } from "../../../src/backends/types.ts";

import { FakeVideo, FakeCanvas, FakeWebGPUBackend, FakeCanvasContext2D } from "../../helpers/fakes.mjs";

function installCanvasMock(frameProvider) {
	const originalDocument = globalThis.document;
	const originalOffscreenCanvas = globalThis.OffscreenCanvas;
	const contexts = [];

	globalThis.OffscreenCanvas = undefined;
	globalThis.document = {
		createElement(tag) {
			if (tag !== "canvas") {
				throw new Error(`Unsupported element creation: ${tag}`);
			}
			const canvas = new FakeCanvas();
			const context = new FakeCanvasContext2D(canvas, frameProvider);
			canvas._context = context;
			contexts.push(context);
			return canvas;
		},
	};

	return {
		contexts,
		restore() {
			globalThis.document = originalDocument;
			globalThis.OffscreenCanvas = originalOffscreenCanvas;
		},
	};
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

function testVideoTextureUsesRequestVideoFrameCallback() {
	const { contexts, restore } = installCanvasMock(
		(width, height, callIndex) => {
			const value = 10 * callIndex;
			const data = new Uint8ClampedArray(width * height * 4);
			data[0] = value;
			data[1] = value;
			data[2] = value;
			data[3] = 255;
			return data;
		}
	);

	const originalHTMLMediaElement = globalThis.HTMLMediaElement;
	globalThis.HTMLMediaElement = { HAVE_CURRENT_DATA: 2 };

	try {
		const video = new FakeVideo({ supportsRVFC: true });
		const texture = new VideoTexture({ video });
		assert.equal(contexts.length, 0);
		assert.equal(texture.data, null);
		assert.equal(texture.sourceKind, "dynamic");
		assert.deepEqual(texture.levels, []);
		assert.deepEqual(texture.mipmaps, []);
		assert.equal(texture.getUploadSource(), video);

		const noFrameUpdate = texture.update();
		assert.equal(noFrameUpdate, false);

		const versionBeforeFrame = texture.version;
		video.presentFrame(1 / 30);
		const hasFrameUpdate = texture.update();
		assert.equal(hasFrameUpdate, false);
		assert.ok(texture.version > versionBeforeFrame);
		assert.equal(contexts.length, 0);

		const sampled = texture.sample(0.1, 0.5);
		const context = contexts[0];
		assert.ok(context);
		assert.equal(context.getImageDataCalls, 1);
		assert.equal(sampled.r, 10);
		assert.equal(sampled.g, 10);
		assert.equal(sampled.b, 10);
		assert.equal(sampled.a, 255);
		assert.equal(texture.data, null);

		texture.dispose();
		assert.ok(video.cancelCalls >= 1);
	} finally {
		restore();
		globalThis.HTMLMediaElement = originalHTMLMediaElement;
	}
}

function testVideoTextureRejectsLegacyPositionalInitialization() {
	const video = new FakeVideo({ supportsRVFC: false });
	assert.throws(
		() => new VideoTexture(video),
		/VideoTexture requires a parameter object/
	);
}

function testVideoTextureFallsBackWithoutRVFC() {
	const { contexts, restore } = installCanvasMock(
		(width, height, callIndex) => {
			const data = new Uint8ClampedArray(width * height * 4);
			data[0] = 30 * callIndex;
			data[1] = 0;
			data[2] = 0;
			data[3] = 255;
			return data;
		}
	);

	const originalHTMLMediaElement = globalThis.HTMLMediaElement;
	globalThis.HTMLMediaElement = { HAVE_CURRENT_DATA: 2 };

	try {
		const video = new FakeVideo({ supportsRVFC: false });
		const texture = new VideoTexture({ video });
		assert.equal(contexts.length, 0);
		assert.equal(texture.data, null);

		const noAdvance = texture.update();
		assert.equal(noAdvance, false);

		video.currentTime = 0.2;
		const advanced = texture.update();
		assert.equal(advanced, true);
		assert.equal(contexts.length, 0);
		assert.equal(texture.data, null);

		texture.dispose();
	} finally {
		restore();
		globalThis.HTMLMediaElement = originalHTMLMediaElement;
	}
}

function testWebGPURegistryUsesExternalVideoUploadPath() {
	const { restore } = installCanvasMock((width, height, callIndex) => {
		const data = new Uint8ClampedArray(width * height * 4);
		data[0] = callIndex;
		data[1] = 0;
		data[2] = 0;
		data[3] = 255;
		return data;
	});

	const originalHTMLMediaElement = globalThis.HTMLMediaElement;
	globalThis.HTMLMediaElement = { HAVE_CURRENT_DATA: 2 };

	try {
		const video = new FakeVideo({ supportsRVFC: true });
		const texture = new VideoTexture({ video });
		const backend = new FakeWebGPUBackend();
		const registry = new WebGPUTextureRegistry(backend, backend);

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

		video.presentFrame(1 / 24);
		registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
		assert.equal(backend.copyCalls.length, 2);
		assert.equal(backend.writeCalls.length, 0);

		texture.dispose();
	} finally {
		restore();
		globalThis.HTMLMediaElement = originalHTMLMediaElement;
	}
}

async function testWebGPURegistryGeneratesMipmapsAfterVideoUpload() {
	const { restore } = installCanvasMock((width, height, callIndex) => {
		const data = new Uint8ClampedArray(width * height * 4);
		data[0] = callIndex;
		data[1] = 0;
		data[2] = 0;
		data[3] = 255;
		return data;
	});

	const originalHTMLMediaElement = globalThis.HTMLMediaElement;
	globalThis.HTMLMediaElement = { HAVE_CURRENT_DATA: 2 };

	try {
		const video = new FakeVideo({ supportsRVFC: true });
		const texture = new VideoTexture({ video });
		texture.minFilter = "LinearMipmapLinear";
		const backend = new FakeWebGPUBackend();
		const registry = new WebGPUTextureRegistry(backend, backend);

		registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
		assert.equal(backend.createTextureCalls[0].mipLevelCount, 2);
		assert.equal(backend.copyCalls.length, 1);
		assert.equal(backend.writeCalls.length, 0);
		await waitForCondition(
			() => backend.recordedRenderPasses.length === 1,
			"Expected video upload mipmap pass to be recorded"
		);
		assert.equal(backend.recordedRenderPasses.length, 1);

		video.presentFrame(1 / 24);
		registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
		assert.equal(backend.copyCalls.length, 2);
		await waitForCondition(
			() => backend.recordedRenderPasses.length === 2,
			"Expected updated video upload mipmap pass to be recorded"
		);
		assert.equal(backend.recordedRenderPasses.length, 2);

		texture.dispose();
		registry.destroy();
	} finally {
		restore();
		globalThis.HTMLMediaElement = originalHTMLMediaElement;
	}
}

async function run() {
	testVideoTextureRejectsLegacyPositionalInitialization();
	testVideoTextureUsesRequestVideoFrameCallback();
	testVideoTextureFallsBackWithoutRVFC();
	testWebGPURegistryUsesExternalVideoUploadPath();
	await testWebGPURegistryGeneratesMipmapsAfterVideoUpload();
	console.log("Video texture tests passed");
}

run();
