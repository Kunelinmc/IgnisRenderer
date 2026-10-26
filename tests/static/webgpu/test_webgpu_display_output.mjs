import assert from "node:assert/strict";

import {
	resolveDisplayOutputOptions,
} from "../../../src/rendering/DisplayOutput.ts";
import {
	WebGPUDisplayOutputManager,
} from "../../../src/backends/webgpu/WebGPUDisplayOutputManager.ts";
import { TextureFormat } from "../../../src/core/TextureFormat.ts";

globalThis.GPUTextureUsage = {
	RENDER_ATTACHMENT: 1,
	COPY_SRC: 2,
};

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
	let active = null;
	const configurations = [];
	const context = {
		configure(configuration) {
			if (options.throwOnHDR && configuration.format === "rgba16float") {
				throw new Error("HDR rejected");
			}
			active = configuration;
			configurations.push(configuration);
		},
		unconfigure() {
			active = null;
		},
	};
	if (!options.omitGetConfiguration) {
		context.getConfiguration = () => {
			if (!active) return null;
			if (options.ignoreHDR && active.format === "rgba16float") {
				return {
					...active,
					format: "bgra8unorm",
					colorSpace: "srgb",
					toneMapping: { mode: "standard" },
				};
			}
			return active;
		};
	}
	return { context, configurations };
}

function createManager(options) {
	return new WebGPUDisplayOutputManager(
		resolveDisplayOutputOptions(options),
	);
}

function testDefaultSDR() {
	installDynamicRange(true);
	const manager = createManager({ mode: "sdr" });
	const { context, configurations } = createContext();
	const resolved = manager.configure(
		context,
		{},
		TextureFormat.BGRA8Unorm,
	);
	assert.equal(resolved.state.activeDynamicRange, "sdr");
	assert.equal(resolved.state.colorSpace, "srgb");
	assert.equal(resolved.state.fallbackReason, undefined);
	assert.equal(configurations.at(-1).format, "bgra8unorm");
	assert.equal(configurations.at(-1).toneMapping.mode, "standard");
}

function testAutoHDRSuccess() {
	installDynamicRange(true);
	const manager = createManager({ mode: "auto" });
	manager.observeDynamicRange(() => {});
	const { context } = createContext();
	const resolved = manager.configure(
		context,
		{},
		TextureFormat.BGRA8Unorm,
	);
	assert.equal(resolved.state.activeDynamicRange, "hdr");
	assert.equal(resolved.state.colorSpace, "display-p3");
	assert.equal(resolved.format, "rgba16float");
	assert.equal(resolved.canvas.toneMapping.mode, "extended");
}

function testAutoOnSDRDisplayIsSilent() {
	installDynamicRange(false);
	const manager = createManager({ mode: "auto" });
	manager.observeDynamicRange(() => {});
	const { context } = createContext();
	const resolved = manager.configure(
		context,
		{},
		TextureFormat.BGRA8Unorm,
	);
	assert.equal(resolved.state.activeDynamicRange, "sdr");
	assert.equal(resolved.state.fallbackReason, undefined);
}

function testExplicitHDRFallbacks() {
	installDynamicRange(false);
	let manager = createManager({ mode: "hdr" });
	manager.observeDynamicRange(() => {});
	let harness = createContext();
	let resolved = manager.configure(
		harness.context,
		{},
		TextureFormat.BGRA8Unorm,
	);
	assert.equal(resolved.state.fallbackReason, "display-not-hdr-capable");

	installDynamicRange(true);
	manager = createManager({ mode: "hdr" });
	manager.observeDynamicRange(() => {});
	harness = createContext({ omitGetConfiguration: true });
	resolved = manager.configure(
		harness.context,
		{},
		TextureFormat.BGRA8Unorm,
	);
	assert.equal(
		resolved.state.fallbackReason,
		"canvas-tone-mapping-unsupported",
	);

	harness = createContext({ ignoreHDR: true });
	resolved = manager.configure(
		harness.context,
		{},
		TextureFormat.BGRA8Unorm,
	);
	assert.equal(
		resolved.state.fallbackReason,
		"hdr-context-configuration-failed",
	);

	harness = createContext({ throwOnHDR: true });
	resolved = manager.configure(
		harness.context,
		{},
		TextureFormat.BGRA8Unorm,
	);
	assert.equal(
		resolved.state.fallbackReason,
		"hdr-context-configuration-failed",
	);
}

function run() {
	testDefaultSDR();
	testAutoHDRSuccess();
	testAutoOnSDRDisplayIsSilent();
	testExplicitHDRFallbacks();
	console.log("WebGPU display-output tests passed");
}

run();
