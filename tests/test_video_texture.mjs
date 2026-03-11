import assert from "node:assert/strict";
import { VideoTexture } from "../src/core/VideoTexture.ts";
import { WebGPUTextureRegistry } from "../src/renderers/webgpu/WebGPUTextureRegistry.ts";
import { WEBGPU_TEXTURE_SLOT } from "../src/renderers/webgpu/constants.ts";

class FakeCanvas2DContext {
	constructor(frameProvider) {
		this._frameProvider = frameProvider;
		this.drawImageCalls = 0;
		this.getImageDataCalls = 0;
	}

	drawImage() {
		this.drawImageCalls++;
	}

	getImageData(_x, _y, width, height) {
		this.getImageDataCalls++;
		return {
			data: this._frameProvider(width, height, this.getImageDataCalls),
		};
	}
}

class FakeCanvas {
	constructor(context) {
		this.width = 1;
		this.height = 1;
		this._context = context;
	}

	getContext(type) {
		if (type !== "2d") {
			return null;
		}
		return this._context;
	}
}

class FakeVideo {
	constructor({ supportsRVFC }) {
		this.readyState = 2;
		this.videoWidth = 2;
		this.videoHeight = 1;
		this.currentTime = 0;
		this._listeners = new Map();
		this._rvfcCallbacks = new Map();
		this._nextRVFCId = 1;
		this.cancelCalls = 0;

		if (supportsRVFC) {
			this.requestVideoFrameCallback = (callback) => {
				const id = this._nextRVFCId++;
				this._rvfcCallbacks.set(id, callback);
				return id;
			};
			this.cancelVideoFrameCallback = (id) => {
				this.cancelCalls++;
				this._rvfcCallbacks.delete(id);
			};
		}
	}

	addEventListener(eventName, callback) {
		const list = this._listeners.get(eventName) ?? [];
		list.push(callback);
		this._listeners.set(eventName, list);
	}

	removeEventListener(eventName, callback) {
		const list = this._listeners.get(eventName) ?? [];
		this._listeners.set(
			eventName,
			list.filter((entry) => entry !== callback)
		);
	}

	emit(eventName) {
		const list = this._listeners.get(eventName) ?? [];
		for (const callback of list) {
			callback();
		}
	}

	presentFrame(currentTime) {
		this.currentTime = currentTime;
		const callbacks = Array.from(this._rvfcCallbacks.values());
		this._rvfcCallbacks.clear();
		for (const callback of callbacks) {
			callback(0, {});
		}
	}
}

class FakeWebGPUBackend {
	constructor() {
		this.copyCalls = [];
		this.writeCalls = [];
		this.queue = {
			copyExternalImageToTexture: (...args) => {
				this.copyCalls.push(args);
			},
		};
	}

	createTexture(desc) {
		return {
			width: desc.width,
			height: desc.height,
			_gpuTexture: {},
			destroy() {},
		};
	}

	writeTexture(...args) {
		this.writeCalls.push(args);
	}

	createSampler(desc) {
		return { desc };
	}
}

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
			const context = new FakeCanvas2DContext(frameProvider);
			contexts.push(context);
			return new FakeCanvas(context);
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

function run() {
	testVideoTextureUsesRequestVideoFrameCallback();
	testVideoTextureFallsBackWithoutRVFC();
	testWebGPURegistryUsesExternalVideoUploadPath();
	console.log("Video texture tests passed");
}

run();
