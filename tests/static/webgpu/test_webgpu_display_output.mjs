import assert from "node:assert/strict";

import {
	DEFAULT_DISPLAY_OUTPUT_OPTIONS,
	applyHDRSoftShoulder,
	linearSrgbToDisplayP3,
	resolveDisplayOutputOptions,
} from "../../../src/rendering/DisplayOutput.ts";
import { linearToSRGB } from "../../../src/maths/Common.ts";
import {
	WebGPUDisplayOutputManager,
} from "../../../src/backends/webgpu/WebGPUDisplayOutputManager.ts";
import { TextureFormat } from "../../../src/backends/types.ts";

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

function testOptionValidation() {
	assert.deepEqual(
		resolveDisplayOutputOptions(),
		DEFAULT_DISPLAY_OUTPUT_OPTIONS,
	);
	assert.throws(
		() => resolveDisplayOutputOptions({ exposure: -1 }),
		RangeError,
	);
	assert.throws(
		() => resolveDisplayOutputOptions({ hdrHeadroom: 17 }),
		RangeError,
	);
	assert.throws(
		() => resolveDisplayOutputOptions({ exposure: Number.NaN }),
		RangeError,
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

function testDisplayMappingNumerics() {
	assert.ok(Math.abs(linearToSRGB(0.0031308) - 0.040449936) < 1e-7);
	const atWhite = applyHDRSoftShoulder([1, 0.5, 0.25], 1, 4);
	assert.deepEqual(atWhite, [1, 0.5, 0.25]);
	const aboveWhite = applyHDRSoftShoulder([1.000001, 0.5000005, 0.25], 1, 4);
	assert.ok(Math.abs(aboveWhite[0] - 1) < 0.000002);

	let previous = 1;
	for (const peak of [1.25, 2, 4, 8, 32]) {
		const mapped = applyHDRSoftShoulder([peak, peak / 2, peak / 4], 1, 4);
		assert.ok(mapped[0] >= previous);
		assert.ok(mapped[0] <= 4);
		assert.ok(Math.abs(mapped[1] / mapped[0] - 0.5) < 1e-8);
		assert.ok(Math.abs(mapped[2] / mapped[0] - 0.25) < 1e-8);
		previous = mapped[0];
	}

	const red = linearSrgbToDisplayP3([1, 0, 0]);
	assert.deepEqual(red, [0.82259287, 0.03319951, 0.01708535]);
	const white = linearSrgbToDisplayP3([1, 1, 1]);
	assert.ok(white.every((value) => Math.abs(value - 1) < 0.00025));
}

function run() {
	testOptionValidation();
	testDefaultSDR();
	testAutoHDRSuccess();
	testAutoOnSDRDisplayIsSilent();
	testExplicitHDRFallbacks();
	testDisplayMappingNumerics();
	console.log("WebGPU display-output tests passed");
}

run();
