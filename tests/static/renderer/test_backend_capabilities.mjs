import assert from "node:assert/strict";
import { SoftwareBackend } from "../../../src/renderers/SoftwareBackend.ts";
import { WebGPUBackend } from "../../../src/renderers/WebGPUBackend.ts";
import { WebGLBackend } from "../../../src/renderers/WebGLBackend.ts";
import { FakeImageData as MockImageData } from "../../helpers/fakes.mjs";
import { createBackendSession } from "../../helpers/TestRenderBackend.mjs";

function run() {
	const softwareProvider = new SoftwareBackend();
	const webgpuProvider = new WebGPUBackend();
	const webglProvider = new WebGLBackend();
	const software = createBackendSession(softwareProvider);
	const webgpu = createBackendSession(webgpuProvider);
	const webgl = createBackendSession(webglProvider);

	assert.equal(softwareProvider.id, "software");
	assert.equal(webgpuProvider.id, "webgpu");
	assert.equal(webglProvider.id, "webgl");
	assert.equal(software.profile.id, softwareProvider.id);
	assert.equal(webgpu.profile.id, webgpuProvider.id);
	assert.equal(webgl.profile.id, webglProvider.id);

	assert.deepEqual(software.profile.capabilities, {
		sh: true,
		shadows: true,
		reflection: true,
		environment: true,
		clusteredLighting: false,
		oit: false,
		occlusionCulling: false,
		postProcess: true,
	});
	assert.equal("postProcessCapabilities" in software, false);

	assert.deepEqual(webgpu.profile.capabilities, {
		sh: true,
		shadows: true,
		reflection: true,
		environment: true,
		clusteredLighting: true,
		oit: true,
		occlusionCulling: true,
		postProcess: true,
	});
	assert.equal(
		createBackendSession(
			new WebGPUBackend({ enableOcclusionCulling: false })
		).profile.capabilities
			.occlusionCulling,
		false
	);
	assert.equal("postProcessCapabilities" in webgpu, false);

	assert.deepEqual(webgl.profile.capabilities, {
		sh: true,
		shadows: true,
		reflection: false,
		environment: true,
		clusteredLighting: true,
		oit: true,
		occlusionCulling: false,
		postProcess: true,
	});
	assert.equal("postProcessCapabilities" in webgl, false);

	assert.equal("passExecutors" in software, false);
	assert.equal("passExecutors" in webgpu, false);
	assert.equal("passExecutors" in webgl, false);
	assert.equal(software.profile.frameScheduling, "on-demand");
	assert.equal(webgpu.profile.frameScheduling, "on-demand");
	assert.equal(webgl.profile.frameScheduling, "on-demand");

	testSoftwareBackendReusesFrameImageData();
	testSoftwareBackendHandlesResizeDuringFrame();

	console.log("Backend capability tests passed");
}

function testSoftwareBackendReusesFrameImageData() {
	const OriginalImageData = globalThis.ImageData;
	MockImageData.instances = [];
	globalThis.ImageData = MockImageData;

	try {
		const backend = createBackendSession(
			new SoftwareBackend(),
			{ width: 2, height: 2 }
		);
		const { pixels } = backend.getAttachments({ width: 2, height: 2 });
		pixels[0] = 7;
		const putCalls = [];

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
		const canvas = { width: 1, height: 1 };
		const backend = createBackendSession(new SoftwareBackend(), canvas);
		const attachments = backend.getAttachments({ width: 2, height: 2 });
		const putCalls = [];

		attachments.pixels[0] = 99;
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
