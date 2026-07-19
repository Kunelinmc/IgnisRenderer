import assert from "node:assert/strict";

import { Renderer } from "../../../src/rendering/Renderer.ts";
import { SoftwareBackend } from "../../../src/backends/software/SoftwareBackend.ts";
import { WebGLBackend } from "../../../src/backends/webgl/WebGLBackend.ts";
import { WebGPUBackend } from "../../../src/backends/webgpu/WebGPUBackend.ts";
import { WEBGPU_COMPUTE_EXTENSION } from "../../../src/backends/BackendExtensions.ts";

function attachBackend(backend, canvas = {}, events = []) {
	backend.attach({
		surface: { canvas },
		events: { emit: (event) => events.push(event) },
	});
	return backend;
}

function assertRuntimeSurface(backend) {
	assert.equal(typeof backend.profile.id, "string");
	assert.equal(typeof backend.attach, "function");
	assert.equal(typeof backend.initialize, "function");
	assert.equal(typeof backend.restore, "function");
	assert.equal(typeof backend.resize, "function");
	assert.equal(typeof backend.getAttachments, "function");
	assert.equal(typeof backend.beginFrame, "function");
	assert.equal(typeof backend.executePass, "function");
	assert.equal(typeof backend.endFrame, "function");
	assert.equal(typeof backend.abortFrame, "function");
	assert.equal(typeof backend.getDebugInfo, "function");
	assert.equal(typeof backend.destroy, "function");
	assert.ok(backend.profile);
	assert.ok(backend.extensions);
	assert.equal("createSession" in backend, false);
}

function assertProfileOnly(backend) {
	for (const key of ["id", "type", "capabilities", "frameScheduling"]) {
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
		const debugInfo = backend.getDebugInfo();
		assert.equal(debugInfo.backend, backend.profile.id);
		assert.equal(debugInfo.available, false);
		assert.equal(typeof debugInfo.unavailableReason, "string");
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

function testWebGPUDebugInfoSnapshotFromMocks() {
	const backend = new WebGPUBackend();
	const adapterInfo = {
		vendor: "nvidia",
		architecture: "ada",
		device: "2684",
		description: "NVIDIA test adapter",
		isFallbackAdapter: false,
	};
	const adapter = {
		info: adapterInfo,
		limits: {
			maxTextureDimension2D: 8192,
			maxTextureArrayLayers: 128,
			maxBindGroups: 4,
		},
	};
	const device = {
		adapterInfo,
		limits: {
			maxTextureDimension2D: 4096,
			maxTextureArrayLayers: 256,
			maxBindGroups: 8,
			maxBindingsPerBindGroup: 640,
			maxBufferSize: 1024,
			maxStorageBufferBindingSize: 2048,
			maxUniformBufferBindingSize: 4096,
			maxSampledTexturesPerShaderStage: 16,
			maxSamplersPerShaderStage: 8,
			maxStorageBuffersPerShaderStage: 12,
			maxStorageTexturesPerShaderStage: 4,
			maxColorAttachments: 8,
			maxColorAttachmentBytesPerSample: 64,
		},
		features: new Set(["timestamp-query", "indirect-first-instance"]),
	};

	const debugInfo = backend._createDebugInfo(adapter, device);
	assert.equal(debugInfo.available, true);
	assert.equal(debugInfo.device.vendor, "nvidia");
	assert.equal(debugInfo.device.architecture, "ada");
	assert.equal(debugInfo.device.device, "2684");
	assert.equal(debugInfo.device.description, "NVIDIA test adapter");
	assert.equal(debugInfo.device.isFallbackAdapter, false);
	assert.equal(debugInfo.device.raw.vendor, "nvidia");
	assert.equal(debugInfo.limits.maxTextureDimension2D, 4096);
	assert.equal(debugInfo.limits.maxTextureArrayLayers, 256);
	assert.equal(debugInfo.limits.maxBindGroups, 8);
	assert.deepEqual(debugInfo.features, [
		"indirect-first-instance",
		"timestamp-query",
	]);
}

function testWebGPUDebugInfoHandlesMissingAdapterInfo() {
	const backend = new WebGPUBackend();
	const debugInfo = backend._createDebugInfo(
		{
			limits: {
				maxTextureDimension2D: 2048,
			},
		},
		{
			limits: {},
			features: null,
		}
	);

	assert.equal(debugInfo.available, true);
	assert.equal(debugInfo.device, undefined);
	assert.equal(debugInfo.limits.maxTextureDimension2D, 2048);
	assert.deepEqual(debugInfo.features, []);
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

function testRendererDelegatesBackendDebugInfo() {
	const backend = new SoftwareBackend();
	const originalWindow = globalThis.window;
	if (!originalWindow) {
		globalThis.window = { devicePixelRatio: 1 };
	}
	const canvas = {
		width: 1,
		height: 1,
		getBoundingClientRect: () => ({ width: 1, height: 1 }),
		getContext: () => null,
	};

	try {
		const renderer = new Renderer({ backend, canvas });
		assert.deepEqual(renderer.getBackendDebugInfo(), backend.getDebugInfo());
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
testWebGPUDebugInfoSnapshotFromMocks();
testWebGPUDebugInfoHandlesMissingAdapterInfo();
testRendererRejectsReusedBackend();
testRendererDelegatesBackendDebugInfo();
console.log("Backend one-to-one runtime tests passed");
