import assert from "node:assert/strict";
import {
	WebGPUMSAAController,
} from "../../../src/backends/webgpu/WebGPUMSAAController.ts";
import { TextureFormat } from "../../../src/backends/types.ts";

if (!globalThis.GPUTextureUsage) {
	globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1 << 4 };
}

function createDevice(supportedSampleCounts = [1, 2, 4]) {
	let createTextureCalls = 0;
	return {
		limits: {
			maxColorAttachments: 8,
			maxColorAttachmentBytesPerSample: 64,
		},
		createTexture(desc) {
			createTextureCalls++;
			if (!supportedSampleCounts.includes(desc.sampleCount ?? 1)) {
				throw new Error("unsupported sample count");
			}
			return { destroy() {} };
		},
		get createTextureCalls() {
			return createTextureCalls;
		},
	};
}

function createController({ device, sampleCount } = {}) {
	let runtimeFallbacks = 0;
	const controller = new WebGPUMSAAController(
		{
			device: device ?? createDevice(),
			canvasFormat: "bgra8unorm",
			canvasDepthFormat: TextureFormat.Depth24Plus,
			objectIdentity: { getCacheToken: () => "device" },
			onRuntimeFallback() {
				runtimeFallbacks++;
			},
		},
		sampleCount
	);
	return { controller, get runtimeFallbacks() { return runtimeFallbacks; } };
}

function testDefaultsAndNormalization() {
	const defaultController = createController();
	defaultController.controller.activateDevice();
	assert.equal(defaultController.controller.sampleCount, 1);

	const configuredController = createController({ sampleCount: 3 });
	configuredController.controller.activateDevice();
	assert.equal(configuredController.controller.sampleCount, 2);

	const disabledController = createController({ sampleCount: 1 });
	disabledController.controller.activateDevice();
	assert.equal(disabledController.controller.sampleCount, 1);
	assert.throws(
		() => createController({ sampleCount: Number.POSITIVE_INFINITY }),
		/finite number/
	);
}

function testCapabilityCacheAndReset() {
	const device = createDevice();
	const { controller } = createController({ device, sampleCount: 4 });
	controller.activateDevice();
	const callsAfterActivation = device.createTextureCalls;
	assert.equal(controller.resolveSupportedSampleCount(4), 4);
	assert.equal(device.createTextureCalls, callsAfterActivation);
	controller.clearCapabilityCache();
	assert.equal(controller.resolveSupportedSampleCount(4), 4);
	assert.ok(device.createTextureCalls > callsAfterActivation);
}

function testFallbackPersistsUntilDeviceReset() {
	const state = createController({ sampleCount: 4 });
	state.controller.activateDevice();
	assert.equal(state.controller.fallbackToSingleSample(), true);
	assert.equal(state.controller.sampleCount, 1);
	assert.equal(state.runtimeFallbacks, 1);
	assert.equal(state.controller.fallbackToSingleSample(), false);
	assert.equal(state.runtimeFallbacks, 1);

	state.controller.resetDevice();
	state.controller.activateDevice();
	assert.equal(state.controller.sampleCount, 4);
}

function run() {
	testDefaultsAndNormalization();
	testCapabilityCacheAndReset();
	testFallbackPersistsUntilDeviceReset();
	console.log("WebGPU MSAA controller tests passed.");
}

run();
