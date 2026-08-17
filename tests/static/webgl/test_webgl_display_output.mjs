import assert from "node:assert/strict";

import {
	resolveDisplayOutputOptions,
} from "../../../src/rendering/DisplayOutput.ts";
import {
	WebGLDisplayOutputManager,
} from "../../../src/backends/webgl/WebGLDisplayOutputManager.ts";

function installDynamicRange(matches) {
	const listeners = new Set();
	globalThis.matchMedia = () => ({
		matches,
		addEventListener: (_type, listener) => listeners.add(listener),
		removeEventListener: (_type, listener) => listeners.delete(listener),
	});
	return listeners;
}

function createContext(options = {}) {
	const storageCalls = [];
	let error = 0;
	const gl = {
		NO_ERROR: 0,
		INVALID_OPERATION: 0x0502,
		RGBA8: 0x8058,
		RGBA16F: 0x881a,
		drawingBufferFormat: 0x8058,
		drawingBufferColorSpace: "srgb",
		getExtension(name) {
			return name === "EXT_color_buffer_float" && options.floatExtension !== false ? {} : null;
		},
		getError() {
			const current = error;
			error = 0;
			return current;
		},
		drawingBufferStorage(format, width, height) {
			storageCalls.push({ format, width, height });
			if (format === this.RGBA16F && options.throwOnHDR) {
				throw new Error("HDR rejected");
			}
			if (format === this.RGBA8 && options.throwOnSDR) {
				throw new Error("SDR rejected");
			}
			if (format === this.RGBA16F && options.errorOnHDR) {
				error = this.INVALID_OPERATION;
				return;
			}
			if (format === this.RGBA16F && options.ignoreHDR) return;
			this.drawingBufferFormat = format;
		},
	};
	if (options.omitStorage) delete gl.drawingBufferStorage;
	if (options.omitFormat) delete gl.drawingBufferFormat;
	if (options.omitColorSpace) delete gl.drawingBufferColorSpace;
	return { gl, storageCalls };
}

function createManager(options) {
	return new WebGLDisplayOutputManager(resolveDisplayOutputOptions(options));
}

function testDefaultSDR() {
	installDynamicRange(true);
	const manager = createManager({ mode: "sdr" });
	const { gl, storageCalls } = createContext();
	const state = manager.configure(gl, 640, 360);
	assert.equal(state.activeDynamicRange, "sdr");
	assert.equal(state.colorSpace, "srgb");
	assert.equal(state.fallbackReason, undefined);
	assert.deepEqual(storageCalls.at(-1), {
		format: gl.RGBA8,
		width: 640,
		height: 360,
	});
}

function testAutoHDRSuccessAndSDRSwitch() {
	installDynamicRange(true);
	const manager = createManager({ mode: "auto", hdrHeadroom: 6 });
	manager.observeDynamicRange(() => {});
	const { gl, storageCalls } = createContext();
	let state = manager.configure(gl, 1280, 720);
	assert.equal(state.activeDynamicRange, "hdr");
	assert.equal(state.colorSpace, "display-p3");
	assert.equal(gl.drawingBufferFormat, gl.RGBA16F);
	assert.equal(gl.drawingBufferColorSpace, "display-p3");

	manager.setRequested({ mode: "sdr" });
	state = manager.configure(gl, 800, 600);
	assert.equal(state.activeDynamicRange, "sdr");
	assert.equal(gl.drawingBufferFormat, gl.RGBA8);
	assert.equal(gl.drawingBufferColorSpace, "srgb");
	assert.deepEqual(storageCalls.at(-1), {
		format: gl.RGBA8,
		width: 800,
		height: 600,
	});
}

function testFallbackReasons() {
	installDynamicRange(false);
	let manager = createManager({ mode: "auto" });
	manager.observeDynamicRange(() => {});
	let state = manager.configure(createContext().gl, 16, 16);
	assert.equal(state.fallbackReason, undefined);

	manager = createManager({ mode: "hdr" });
	manager.observeDynamicRange(() => {});
	state = manager.configure(createContext().gl, 16, 16);
	assert.equal(state.fallbackReason, "display-not-hdr-capable");

	installDynamicRange(true);
	for (const options of [
		{ omitStorage: true },
		{ omitFormat: true },
		{ omitColorSpace: true },
		{ floatExtension: false },
	]) {
		manager = createManager({ mode: "hdr" });
		manager.observeDynamicRange(() => {});
		state = manager.configure(createContext(options).gl, 16, 16);
		assert.equal(state.fallbackReason, "canvas-hdr-output-unsupported");
	}

	for (const options of [
		{ throwOnHDR: true },
		{ errorOnHDR: true },
		{ ignoreHDR: true },
	]) {
		manager = createManager({ mode: "hdr" });
		manager.observeDynamicRange(() => {});
		const { gl } = createContext(options);
		state = manager.configure(gl, 16, 16);
		assert.equal(state.fallbackReason, "hdr-context-configuration-failed");
		assert.equal(gl.drawingBufferFormat, gl.RGBA8);
		assert.equal(gl.drawingBufferColorSpace, "srgb");
	}
}

function testDynamicRangeObservationLifecycle() {
	const listeners = installDynamicRange(true);
	const manager = createManager({ mode: "auto" });
	let changes = 0;
	manager.observeDynamicRange(() => changes++);
	assert.equal(listeners.size, 1);
	for (const listener of listeners) listener();
	assert.equal(changes, 1);
	manager.destroy();
	assert.equal(listeners.size, 0);
}

function testFatalSDRRestoreFailure() {
	installDynamicRange(true);
	const manager = createManager({ mode: "hdr" });
	manager.observeDynamicRange(() => {});
	assert.throws(
		() => manager.configure(
			createContext({ ignoreHDR: true, throwOnSDR: true }).gl,
			16,
			16,
		),
		/WebGL failed to restore SDR presentation/,
	);
}

function run() {
	testDefaultSDR();
	testAutoHDRSuccessAndSDRSwitch();
	testFallbackReasons();
	testDynamicRangeObservationLifecycle();
	testFatalSDRRestoreFailure();
	console.log("WebGL display-output tests passed");
}

run();
