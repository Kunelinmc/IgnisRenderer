import assert from "node:assert/strict";
import { SoftwareBackend } from "../src/renderers/SoftwareBackend.ts";
import { WebGPUBackend } from "../src/renderers/WebGPUBackend.ts";
import { WebGLBackend } from "../src/renderers/WebGLBackend.ts";
import { FakeImageData as MockImageData } from "./helpers/test_fakes.mjs";

function run() {
	const software = new SoftwareBackend();
	const webgpu = new WebGPUBackend();
	const webgl = new WebGLBackend();

	assert.deepEqual(software.capabilities, {
		sh: true,
		shadows: true,
		reflection: true,
		skybox: true,
		ssao: true,
		ssgi: false,
		taa: false,
		ssr: false,
		volumetric: true,
		motionBlur: false,
		dof: false,
		bloom: false,
		clusteredLighting: false,
	});

	assert.deepEqual(webgpu.capabilities, {
		sh: true,
		shadows: true,
		reflection: false,
		skybox: true,
		ssao: true,
		ssgi: true,
		taa: true,
		ssr: true,
		volumetric: true,
		motionBlur: true,
		dof: true,
		bloom: true,
		clusteredLighting: true,
	});

	assert.deepEqual(webgl.capabilities, {
		sh: false,
		shadows: true,
		reflection: false,
		skybox: true,
		ssao: true,
		ssgi: false,
		taa: true,
		ssr: false,
		volumetric: false,
		motionBlur: true,
		dof: true,
		bloom: true,
		clusteredLighting: false,
	});

	assert.equal(software.passExecutors["particle-sim"], "backend");
	assert.equal(webgpu.passExecutors["particle-sim"], "backend");
	assert.equal(webgl.passExecutors["particle-sim"], "backend");
	assert.equal(software.frameScheduling, "on-demand");
	assert.equal(webgpu.frameScheduling, "on-demand");
	assert.equal(webgl.frameScheduling, "on-demand");

	testSoftwareBackendReusesFrameImageData();
	testSoftwareBackendHandlesResizeDuringFrame();

	console.log("Backend capability tests passed");
}

function testSoftwareBackendReusesFrameImageData() {
	const OriginalImageData = globalThis.ImageData;
	MockImageData.instances = [];
	globalThis.ImageData = MockImageData;

	try {
		const backend = new SoftwareBackend();
		const pixels = new Uint8ClampedArray(16);
		pixels[0] = 7;
		const putCalls = [];

		backend._renderer = {
			pixels,
			canvas: {
				width: 2,
				height: 2,
			},
		};
		backend._ctx = {
			putImageData(imageData, x, y) {
				putCalls.push({ imageData, x, y });
			},
		};

		backend.endFrame();
		pixels[0] = 21;
		backend.endFrame();

		assert.equal(MockImageData.instances.length, 1);
		assert.equal(putCalls.length, 2);
		assert.equal(putCalls[0].x, 0);
		assert.equal(putCalls[0].y, 0);
		assert.strictEqual(putCalls[0].imageData, putCalls[1].imageData);
		assert.strictEqual(putCalls[0].imageData.data, pixels);
		assert.equal(putCalls[1].imageData.data[0], 21);
	} finally {
		globalThis.ImageData = OriginalImageData;
	}
}

function testSoftwareBackendHandlesResizeDuringFrame() {
	const OriginalImageData = globalThis.ImageData;

	globalThis.ImageData = MockImageData;

	try {
		const backend = new SoftwareBackend();
		const attachments = backend.getAttachments(2, 2);
		const putCalls = [];

		attachments.pixels[0] = 99;
		backend._renderer = {
			canvas: {
				width: 1,
				height: 1,
			},
		};
		backend._ctx = {
			putImageData(imageData, x, y) {
				putCalls.push({ imageData, x, y });
			},
		};

		assert.doesNotThrow(() => backend.endFrame());
		assert.equal(putCalls.length, 1);
		assert.equal(putCalls[0].imageData.width, 2);
		assert.equal(putCalls[0].imageData.height, 2);
		assert.equal(putCalls[0].imageData.data[0], 99);
	} finally {
		globalThis.ImageData = OriginalImageData;
	}
}

run();
