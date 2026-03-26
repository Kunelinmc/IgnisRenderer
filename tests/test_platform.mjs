import assert from "node:assert/strict";
import { Platform } from "../src/foundation/Platform.ts";

function testNodeRuntimeDetection() {
	assert.equal(Platform.isNodeRuntime(), true);
	assert.equal(Platform.resolveRuntime(), "node");
}

function testSyntheticBrowserScopeDetection() {
	const fakeScope = {
		window: {},
		document: {
			createElement(tagName) {
				if (tagName !== "canvas") return {};
				return {
					getContext(contextId) {
						if (contextId === "webgl2") {
							return { version: "webgl2" };
						}
						return null;
					},
				};
			},
		},
		navigator: {
			hardwareConcurrency: 12,
			gpu: {
				requestAdapter() {
					return Promise.resolve(null);
				},
			},
		},
		Worker() {},
		SharedArrayBuffer() {},
		crossOriginIsolated: true,
	};

	assert.equal(Platform.isBrowserRuntime(fakeScope), true);
	assert.equal(Platform.hasWebGPU(fakeScope), true);
	assert.equal(Platform.hasWebGL2(fakeScope), true);
	assert.equal(Platform.hasWorker(fakeScope), true);
	assert.equal(Platform.hasSharedArrayBuffer(fakeScope), true);
	assert.equal(Platform.supportsSharedArrayBufferTransport(fakeScope), true);
	assert.equal(Platform.getHardwareConcurrency(4, fakeScope), 12);

	const summary = Platform.detect(fakeScope);
	assert.equal(summary.runtime, "browser");
	assert.equal(summary.hasWebGPU, true);
	assert.equal(summary.hasWebGL2, true);
	assert.equal(summary.crossOriginIsolated, true);
}

function testSyntheticWorkerScopeDetection() {
	const fakeScope = {
		process: undefined,
		importScripts() {},
		SharedArrayBuffer() {},
		crossOriginIsolated: undefined,
	};
	fakeScope.self = fakeScope;

	assert.equal(Platform.isWorkerRuntime(fakeScope), true);
	assert.equal(Platform.resolveRuntime(fakeScope), "worker");
	assert.equal(Platform.isCrossOriginIsolated(fakeScope), false);
	assert.equal(Platform.supportsSharedArrayBufferTransport(fakeScope), false);
	assert.equal(
		Platform.supportsSharedArrayBufferTransport(fakeScope, true),
		true
	);
}

function testFallbackHardwareConcurrency() {
	const fakeScope = {
		navigator: {
			hardwareConcurrency: Number.NaN,
		},
	};
	assert.equal(Platform.getHardwareConcurrency(8, fakeScope), 8);
	assert.equal(Platform.getHardwareConcurrency(0, fakeScope), 1);
}

function testTouchAndMobileDetection() {
	const touchMobileScope = {
		window: {
			ontouchstart: null,
		},
		navigator: {
			maxTouchPoints: 5,
			userAgent:
				"Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) " +
				"AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 " +
				"Mobile/15E148 Safari/604.1",
		},
	};

	const desktopScope = {
		window: {},
		navigator: {
			maxTouchPoints: 0,
			userAgent:
				"Mozilla/5.0 (Windows NT 10.0; Win64; x64) " +
				"AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 " +
				"Safari/537.36",
		},
	};

	const mobileFromUserAgentDataScope = {
		navigator: {
			userAgentData: {
				mobile: true,
			},
		},
	};

	assert.equal(Platform.isTouchDevice(touchMobileScope), true);
	assert.equal(Platform.isMobileDevice(touchMobileScope), true);
	assert.equal(Platform.isTouchDevice(desktopScope), false);
	assert.equal(Platform.isMobileDevice(desktopScope), false);
	assert.equal(Platform.isMobileDevice(mobileFromUserAgentDataScope), true);
}

function run() {
	testNodeRuntimeDetection();
	testSyntheticBrowserScopeDetection();
	testSyntheticWorkerScopeDetection();
	testFallbackHardwareConcurrency();
	testTouchAndMobileDetection();
	console.log("Platform detection tests passed");
}

run();
