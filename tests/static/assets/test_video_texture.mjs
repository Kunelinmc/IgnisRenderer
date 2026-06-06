import assert from "node:assert/strict";
import { VideoTexture } from "../../../src/core/VideoTexture.ts";
import { WebGPUTextureRegistry } from "../../../src/renderers/webgpu/WebGPUTextureRegistry.ts";
import { WEBGPU_TEXTURE_SLOT } from "../../../src/renderers/webgpu/constants.ts";
import { TextureFormat, TextureUsage } from "../../../src/renderers/types.ts";

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
		const texture = new VideoTexture(video);
		const context = contexts[0];
		assert.ok(context);
		assert.equal(context.getImageDataCalls, 1);

		const noFrameUpdate = texture.update();
		assert.equal(noFrameUpdate, false);
		assert.equal(context.getImageDataCalls, 1);

		video.presentFrame(1 / 30);
		const hasFrameUpdate = texture.update();
		assert.equal(hasFrameUpdate, true);
		assert.equal(context.getImageDataCalls, 2);

		const sampled = texture.sample(0.1, 0.5);
		assert.equal(sampled.r, 20);
		assert.equal(sampled.g, 20);
		assert.equal(sampled.b, 20);
		assert.equal(sampled.a, 255);

		texture.dispose();
		assert.ok(video.cancelCalls >= 1);
	} finally {
		restore();
		globalThis.HTMLMediaElement = originalHTMLMediaElement;
	}
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
		const texture = new VideoTexture(video);
		const context = contexts[0];
		assert.ok(context);
		assert.equal(context.getImageDataCalls, 1);

		const noAdvance = texture.update();
		assert.equal(noAdvance, false);
		assert.equal(context.getImageDataCalls, 1);

		video.currentTime = 0.2;
		const advanced = texture.update();
		assert.equal(advanced, true);
		assert.equal(context.getImageDataCalls, 2);

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
		const texture = new VideoTexture(video);
		const backend = new FakeWebGPUBackend();
		const registry = new WebGPUTextureRegistry(backend);

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
		texture.update();
		registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
		assert.equal(backend.copyCalls.length, 2);
		assert.equal(backend.writeCalls.length, 0);

		texture.dispose();
	} finally {
		restore();
		globalThis.HTMLMediaElement = originalHTMLMediaElement;
	}
}

function testWebGPURegistryGeneratesMipmapsAfterVideoUpload() {
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
		const texture = new VideoTexture(video);
		texture.minFilter = "LinearMipmapLinear";
		const backend = new FakeWebGPUBackend();
		const registry = new WebGPUTextureRegistry(backend);

		registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
		assert.equal(backend.createTextureCalls[0].mipLevelCount, 2);
		assert.equal(backend.copyCalls.length, 1);
		assert.equal(backend.writeCalls.length, 0);
		assert.equal(backend.recordedRenderPasses.length, 1);

		video.presentFrame(1 / 24);
		texture.update();
		registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
		assert.equal(backend.copyCalls.length, 2);
		assert.equal(backend.recordedRenderPasses.length, 2);

		texture.dispose();
		registry.destroy();
	} finally {
		restore();
		globalThis.HTMLMediaElement = originalHTMLMediaElement;
	}
}

function run() {
	testVideoTextureUsesRequestVideoFrameCallback();
	testVideoTextureFallsBackWithoutRVFC();
	testWebGPURegistryUsesExternalVideoUploadPath();
	testWebGPURegistryGeneratesMipmapsAfterVideoUpload();
	console.log("Video texture tests passed");
}

run();
