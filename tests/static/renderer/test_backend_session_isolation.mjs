import assert from "node:assert/strict";

import { Renderer } from "../../../src/renderers/Renderer.ts";
import { SoftwareBackend } from "../../../src/renderers/SoftwareBackend.ts";
import { WebGLBackend } from "../../../src/renderers/WebGLBackend.ts";
import { WebGPUBackend } from "../../../src/renderers/WebGPUBackend.ts";
import { WEBGPU_COMPUTE_EXTENSION } from "../../../src/renderers/BackendExtensions.ts";

function attachBackend(backend, canvas = {}, events = []) {
	backend.attach({
		surface: { canvas },
		events: { emit: (event) => events.push(event) },
	});
	return backend;
}

function assertRuntimeSurface(backend) {
	assert.equal(typeof backend.id, "string");
	assert.equal(typeof backend.attach, "function");
	assert.equal(typeof backend.initialize, "function");
	assert.equal(typeof backend.restore, "function");
	assert.equal(typeof backend.resize, "function");
	assert.equal(typeof backend.getAttachments, "function");
	assert.equal(typeof backend.beginFrame, "function");
	assert.equal(typeof backend.executePass, "function");
	assert.equal(typeof backend.endFrame, "function");
	assert.equal(typeof backend.abortFrame, "function");
	assert.equal(typeof backend.destroy, "function");
	assert.ok(backend.profile);
	assert.ok(backend.extensions);
	assert.equal("createSession" in backend, false);
}

function assertProfileOnly(backend) {
	for (const key of ["type", "capabilities", "frameScheduling"]) {
		assert.equal(key in backend, false, `backend must use profile instead of ${key}`);
	}
}

function assertSecondAttachThrows(backend) {
	attachBackend(backend, {});
	assert.throws(
		() => attachBackend(backend, {}),
		/is already attached to a renderer/
	);
}

function testBackendsExposeRuntimeSurface() {
	for (const backend of [
		new SoftwareBackend(),
		new WebGLBackend(),
		new WebGPUBackend(),
	]) {
		assertRuntimeSurface(backend);
		assertProfileOnly(backend);
	}
}

function testBackendsRejectSecondAttach() {
	assertSecondAttachThrows(new SoftwareBackend());
	assertSecondAttachThrows(new WebGLBackend());
	assertSecondAttachThrows(new WebGPUBackend());
}

function testSoftwareAttachmentsReflectAttachedRuntime() {
	const backend = attachBackend(new SoftwareBackend(), { width: 4, height: 4 });
	const attachments = backend.getAttachments({ width: 4, height: 4 });

	assert.equal(attachments.pixels.length, 4 * 4 * 4);
}

function testWebGLStateStaysRuntimeScoped() {
	const events = [];
	const backend = attachBackend(new WebGLBackend(), {}, events);

	backend.onDeviceLost({ reason: "test-loss" });

	assert.equal(backend._contextLost, true);
	assert.equal(events.length, 0);
}

function testWebGPUExtensionsStayStable() {
	const backend = attachBackend(new WebGPUBackend(), {});
	const firstCompute = backend.extensions.requireBackendExtension(
		WEBGPU_COMPUTE_EXTENSION
	);
	const secondCompute = backend.extensions.requireBackendExtension(
		WEBGPU_COMPUTE_EXTENSION
	);

	assert.strictEqual(firstCompute, secondCompute);
	assert.strictEqual(backend.extensions, backend.extensions);
}

function testRendererRejectsReusedBackend() {
	const backend = new SoftwareBackend();
	const originalWindow = globalThis.window;
	if (!originalWindow) {
		globalThis.window = { devicePixelRatio: 1 };
	}
	const canvasA = {
		width: 1,
		height: 1,
		getBoundingClientRect: () => ({ width: 1, height: 1 }),
		getContext: () => null,
	};
	const canvasB = {
		width: 1,
		height: 1,
		getBoundingClientRect: () => ({ width: 1, height: 1 }),
		getContext: () => null,
	};

	try {
		new Renderer({ backend, canvas: canvasA });
		assert.throws(
			() => new Renderer({ backend, canvas: canvasB }),
			/SoftwareBackend is already attached to a renderer/
		);
	} finally {
		if (!originalWindow) {
			delete globalThis.window;
		}
	}
}

testBackendsExposeRuntimeSurface();
testBackendsRejectSecondAttach();
testSoftwareAttachmentsReflectAttachedRuntime();
testWebGLStateStaysRuntimeScoped();
testWebGPUExtensionsStayStable();
testRendererRejectsReusedBackend();
console.log("Backend one-to-one runtime tests passed");
