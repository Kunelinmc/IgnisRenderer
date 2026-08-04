import assert from "node:assert/strict";

import {
	WebGPUSampleCountResolver,
} from "../../../src/backends/webgpu/WebGPUSampleCountResolver.ts";

if (!globalThis.GPUTextureUsage) {
	globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1 << 4 };
}

function createDevice(supportedByFormat = new Map()) {
	let createTextureCalls = 0;
	return {
		limits: {
			maxColorAttachments: 8,
			maxColorAttachmentBytesPerSample: 64,
		},
		createTexture(desc) {
			createTextureCalls++;
			const supported = supportedByFormat.get(desc.format) ?? [1, 2, 4];
			if (!supported.includes(desc.sampleCount ?? 1)) {
				throw new Error("unsupported sample count");
			}
			return { destroy() {} };
		},
		get createTextureCalls() {
			return createTextureCalls;
		},
	};
}

function createResolver(device = createDevice()) {
	return {
		device,
		resolver: new WebGPUSampleCountResolver({
			device,
			objectIdentity: { getCacheToken: () => "device" },
		}),
	};
}

function testNormalizationAndFormatIntersection() {
	const { resolver } = createResolver(createDevice(new Map([
		["rgba16float", [1, 2, 4]],
		["rgba8unorm", [1, 2]],
	])));
	assert.equal(resolver.normalizeRequestedSampleCount(undefined), 1);
	assert.equal(resolver.normalizeRequestedSampleCount(3.9), 3);
	assert.equal(resolver.normalizeRequestedSampleCount(0), 1);
	assert.throws(
		() => resolver.normalizeRequestedSampleCount(Number.NaN),
		/finite number/,
	);
	assert.equal(
		resolver.resolveSupportedSampleCount(4, ["rgba16float", "rgba8unorm"]),
		2,
	);
}

function testCapabilityCacheAndDomainFallback() {
	const { device, resolver } = createResolver();
	const formats = ["rgba16float", "depth32float"];
	const first = resolver.resolveDomainSampleCount("main-scene", 4, formats);
	const callsAfterFirst = device.createTextureCalls;
	const reordered = resolver.resolveDomainSampleCount(
		"main-scene",
		4,
		["depth32float", "rgba16float", "rgba16float"],
	);
	assert.equal(reordered.signature, first.signature);
	assert.equal(device.createTextureCalls, callsAfterFirst);
	assert.equal(resolver.fallbackToSingleSample(first.signature), true);
	assert.equal(
		resolver.resolveDomainSampleCount("main-scene", 4, formats).sampleCount,
		1,
	);
	assert.equal(
		resolver.resolveDomainSampleCount("custom-target:a", 4, formats).sampleCount,
		4,
	);

	resolver.clearCapabilityCache();
	assert.equal(
		resolver.resolveDomainSampleCount("main-scene", 4, formats).sampleCount,
		1,
		"capability invalidation must preserve runtime fallback",
	);
	resolver.resetDevice();
	assert.equal(
		resolver.resolveDomainSampleCount("main-scene", 4, formats).sampleCount,
		4,
	);
}

function testAttachmentLimitsAndDeviceIdentityParticipateInCaching() {
	const firstDevice = createDevice();
	firstDevice.token = "first";
	firstDevice.limits.maxColorAttachments = 2;
	firstDevice.limits.maxColorAttachmentBytesPerSample = 16;
	const host = {
		device: firstDevice,
		objectIdentity: { getCacheToken: (device) => device?.token ?? "none" },
	};
	const resolver = new WebGPUSampleCountResolver(host);
	assert.equal(
		resolver.resolveSupportedSampleCount(4, ["rgba8unorm"], {
			colorAttachmentCount: 2,
			colorAttachmentBytesPerSample: 16,
		}),
		4,
	);
	assert.equal(
		resolver.resolveSupportedSampleCount(4, ["rgba8unorm"], {
			colorAttachmentCount: 3,
			colorAttachmentBytesPerSample: 16,
		}),
		1,
	);

	const secondDevice = createDevice(new Map([["rgba8unorm", [1, 2]]]));
	secondDevice.token = "second";
	host.device = secondDevice;
	assert.equal(
		resolver.resolveSupportedSampleCount(4, ["rgba8unorm"], {
			colorAttachmentCount: 2,
			colorAttachmentBytesPerSample: 16,
		}),
		2,
	);
	const callsBeforeClear = secondDevice.createTextureCalls;
	resolver.clearCapabilityCache();
	resolver.resolveSupportedSampleCount(4, ["rgba8unorm"], {
		colorAttachmentCount: 2,
		colorAttachmentBytesPerSample: 16,
	});
	assert.ok(secondDevice.createTextureCalls > callsBeforeClear);
}

function run() {
	testNormalizationAndFormatIntersection();
	testCapabilityCacheAndDomainFallback();
	testAttachmentLimitsAndDeviceIdentityParticipateInCaching();
	console.log("WebGPU sample-count resolver tests passed.");
}

run();
