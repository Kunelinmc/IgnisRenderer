import assert from "node:assert/strict";
import { CanvasTexture } from "../src/core/CanvasTexture.ts";
import { Texture } from "../src/core/Texture.ts";
import { WebGPUTextureRegistry } from "../src/renderers/webgpu/WebGPUTextureRegistry.ts";
import { WEBGPU_TEXTURE_SLOT } from "../src/renderers/webgpu/constants.ts";
import { TextureUsage } from "../src/renderers/types.ts";

class FakeCanvasContext2D {
	constructor(canvas) {
		this.canvas = canvas;
		this.getImageDataCalls = 0;
		this.fillRectCalls = 0;
		this.drawImageCalls = 0;
		this._frameValue = 8;
	}

	fillRect() {
		this.fillRectCalls++;
		this._frameValue = (this._frameValue + 16) & 0xff;
	}

	drawImage() {
		this.drawImageCalls++;
		this._frameValue = (this._frameValue + 8) & 0xff;
	}

	getImageData(_x, _y, width, height) {
		this.getImageDataCalls++;
		const data = new Uint8ClampedArray(width * height * 4);
		for (let i = 0; i < data.length; i += 4) {
			data[i] = this._frameValue;
			data[i + 1] = this._frameValue;
			data[i + 2] = this._frameValue;
			data[i + 3] = 255;
		}
		return { data };
	}
}

class FakeWebGPUBackend {
	constructor() {
		this.copyCalls = [];
		this.writeCalls = [];
		this.createTextureCalls = [];
		this.queue = {
			copyExternalImageToTexture: (...args) => {
				this.copyCalls.push(args);
			},
		};
	}

	createTexture(desc) {
		this.createTextureCalls.push(desc);
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

function createFakeContext(width = 2, height = 1) {
	const canvas = {
		width,
		height,
	};
	const context = new FakeCanvasContext2D(canvas);
	return { canvas, context };
}

function testCanvasTextureTracksContextMutations() {
	const { context } = createFakeContext();
	const texture = new CanvasTexture(context);
	try {
		assert.equal(context.getImageDataCalls, 1);
		assert.equal(texture.update(1), false);
		assert.equal(context.getImageDataCalls, 1);

		context.fillRect(0, 0, 1, 1);
		assert.equal(texture.update(16), true);
		assert.equal(context.getImageDataCalls, 2);

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
	const texture = new CanvasTexture(context, {
		minUpdateIntervalMs: 20,
	});
	try {
		assert.equal(context.getImageDataCalls, 1);
		context.fillRect(0, 0, 1, 1);
		assert.equal(texture.update(5), false);
		assert.equal(context.getImageDataCalls, 1);

		assert.equal(texture.update(24), true);
		assert.equal(context.getImageDataCalls, 2);
	} finally {
		texture.dispose();
	}
}

function testCanvasTextureDynamicUpdateIntegration() {
	const { context } = createFakeContext();
	const texture = new CanvasTexture(context);
	try {
		assert.equal(Texture.updateDynamicTextures(0), false);
		context.drawImage(null, 0, 0);
		assert.equal(Texture.updateDynamicTextures(16), true);
		assert.equal(Texture.updateDynamicTextures(32), false);
	} finally {
		texture.dispose();
	}
}

function testWebGPURegistryUsesExternalCanvasUploadPath() {
	const { context } = createFakeContext();
	const texture = new CanvasTexture(context);
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend);

	try {
		registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
		assert.equal(backend.createTextureCalls.length, 1);
		assert.ok(
			(backend.createTextureCalls[0].usage & TextureUsage.CopyDst) !== 0
		);
		assert.ok(
			(backend.createTextureCalls[0].usage & TextureUsage.RenderAttachment) !==
				0
		);
		assert.equal(backend.copyCalls.length, 1);
		assert.equal(backend.writeCalls.length, 0);

		context.fillRect(0, 0, 1, 1);
		texture.update(16);
		registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
		assert.equal(backend.copyCalls.length, 2);
		assert.equal(backend.writeCalls.length, 0);
	} finally {
		texture.dispose();
	}
}

function run() {
	testCanvasTextureTracksContextMutations();
	testCanvasTextureRespectsUpdateInterval();
	testCanvasTextureDynamicUpdateIntegration();
	testWebGPURegistryUsesExternalCanvasUploadPath();
	console.log("Canvas texture tests passed");
}

run();

