import assert from "node:assert/strict";
import { SoftwareBackend } from "../../../src/backends/software/SoftwareBackend.ts";
import { WebGPUBackend } from "../../../src/backends/webgpu/WebGPUBackend.ts";
import { WebGLBackend } from "../../../src/backends/webgl/WebGLBackend.ts";
import { FakeImageData as MockImageData } from "../../helpers/fakes.mjs";
import { attachBackend } from "../../helpers/TestRenderBackend.mjs";

function run() {
	const softwareProvider = new SoftwareBackend();
	const webgpuProvider = new WebGPUBackend();
	const webglProvider = new WebGLBackend();
	const software = attachBackend(softwareProvider);
	const webgpu = attachBackend(webgpuProvider);
	const webgl = attachBackend(webglProvider);

	assert.equal(softwareProvider.profile.id, "software");
	assert.equal(webgpuProvider.profile.id, "webgpu");
	assert.equal(webglProvider.profile.id, "webgl");
	assert.equal(software.profile.id, "software");
	assert.equal(webgpu.profile.id, "webgpu");
	assert.equal(webgl.profile.id, "webgl");

	assert.deepEqual(software.profile.capabilities, {
		displayHDR: false,
		sh: true,
		shadows: true,
		reflection: true,
		environment: true,
		clusteredLighting: false,
		oit: false,
		occlusionCulling: false,
		postProcess: true,
		customRenderTargets: false,
		customRenderPasses: false,
		renderTargetReadback: false,
	});
	assert.equal("postProcessCapabilities" in software, false);

	assert.deepEqual(webgpu.profile.capabilities, {
		displayHDR: true,
		sh: true,
		shadows: true,
		reflection: true,
		environment: true,
		clusteredLighting: true,
		oit: true,
		occlusionCulling: true,
		postProcess: true,
		customRenderTargets: true,
		customRenderPasses: true,
		renderTargetReadback: true,
	});
	assert.equal(
		attachBackend(
			new WebGPUBackend({ enableOcclusionCulling: false })
		).profile.capabilities
			.occlusionCulling,
		false
	);
	assert.equal("postProcessCapabilities" in webgpu, false);

	assert.deepEqual(webgl.profile.capabilities, {
		displayHDR: false,
		sh: true,
		shadows: true,
		reflection: false,
		environment: true,
		clusteredLighting: true,
		oit: true,
		occlusionCulling: false,
		postProcess: true,
		customRenderTargets: true,
		customRenderPasses: true,
		renderTargetReadback: true,
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
		const backend = attachBackend(
			new SoftwareBackend(),
			{ width: 2, height: 2 }
		);
		const { pixels } = backend.getAttachments({ width: 2, height: 2 });
		pixels[0] = 7;
		const putCalls = [];

		backend._surface._ctx = {
			putImageData(imageData, x, y) {
				putCalls.push({ imageData, x, y });
			},
		};

		backend._surface.present();
		pixels[0] = 21;
		backend._surface.present();

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
		const backend = attachBackend(new SoftwareBackend(), canvas);
		const attachments = backend.getAttachments({ width: 2, height: 2 });
		const putCalls = [];

		attachments.pixels[0] = 99;
		backend._surface._ctx = {
			putImageData(imageData, x, y) {
				putCalls.push({ imageData, x, y });
			},
		};

		assert.doesNotThrow(() => backend._surface.present());
		assert.equal(putCalls.length, 1);
		assert.equal(putCalls[0].imageData.width, 2);
		assert.equal(putCalls[0].imageData.height, 2);
		assert.equal(putCalls[0].imageData.data[0], 99);
	} finally {
		globalThis.ImageData = OriginalImageData;
	}
}

run();
