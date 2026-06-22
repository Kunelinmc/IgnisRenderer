import assert from "node:assert/strict";

import { SoftwareBackend } from "../../../src/renderers/SoftwareBackend.ts";
import { WebGLBackend } from "../../../src/renderers/WebGLBackend.ts";
import { WebGPUBackend } from "../../../src/renderers/WebGPUBackend.ts";
import { WEBGPU_COMPUTE_EXTENSION } from "../../../src/renderers/BackendExtensions.ts";

const LEGACY_RUNTIME_KEYS = [
	"initialize",
	"restore",
	"resize",
	"getAttachments",
	"beginFrame",
	"executePass",
	"endFrame",
	"abortFrame",
	"destroy",
	"profile",
	"extensions",
];

function createSession(provider, canvas, events) {
	return provider.createSession({
		surface: { canvas },
		events: { emit: (event) => events.push(event) },
	});
}

function assertProviderOnly(provider) {
	assert.equal(typeof provider.id, "string");
	assert.equal(typeof provider.createSession, "function");
	for (const key of LEGACY_RUNTIME_KEYS) {
		assert.equal(key in provider, false, `provider must not expose ${key}`);
	}
}

function assertSessionProfileOnly(session) {
	for (const key of ["type", "capabilities", "frameScheduling"]) {
		assert.equal(key in session, false, `session must use profile instead of ${key}`);
	}
}

function testSoftwareSessionsOwnIndependentAttachments() {
	const provider = new SoftwareBackend();
	assertProviderOnly(provider);

	const first = createSession(provider, { width: 4, height: 4 }, []);
	const second = createSession(provider, { width: 8, height: 8 }, []);
	assertSessionProfileOnly(first);
	assertSessionProfileOnly(second);
	const firstAttachments = first.getAttachments({ width: 4, height: 4 });
	const secondAttachments = second.getAttachments({ width: 8, height: 8 });

	assert.notStrictEqual(first, second);
	assert.notStrictEqual(firstAttachments.pixels, secondAttachments.pixels);
	assert.equal(firstAttachments.pixels.length, 4 * 4 * 4);
	assert.equal(secondAttachments.pixels.length, 8 * 8 * 4);
}

function testWebGLSessionStateStaysScoped() {
	const provider = new WebGLBackend();
	assertProviderOnly(provider);

	const firstEvents = [];
	const secondEvents = [];
	const first = createSession(provider, {}, firstEvents);
	const second = createSession(provider, {}, secondEvents);
	assertSessionProfileOnly(first);
	assertSessionProfileOnly(second);

	first.onDeviceLost({ reason: "test-loss" });

	assert.notStrictEqual(first, second);
	assert.equal(first._contextLost, true);
	assert.equal(second._contextLost, false);
	assert.equal(secondEvents.length, 0);
}

function testWebGPUSessionExtensionsStayScoped() {
	const provider = new WebGPUBackend();
	assertProviderOnly(provider);

	const first = createSession(provider, {}, []);
	const second = createSession(provider, {}, []);
	assertSessionProfileOnly(first);
	assertSessionProfileOnly(second);
	const firstCompute = first.extensions.requireBackendExtension(
		WEBGPU_COMPUTE_EXTENSION
	);
	const secondCompute = second.extensions.requireBackendExtension(
		WEBGPU_COMPUTE_EXTENSION
	);

	assert.notStrictEqual(first, second);
	assert.notStrictEqual(first.extensions, second.extensions);
	assert.notStrictEqual(firstCompute, secondCompute);
}

testSoftwareSessionsOwnIndependentAttachments();
testWebGLSessionStateStaysScoped();
testWebGPUSessionExtensionsStayScoped();
console.log("Backend session isolation tests passed");
