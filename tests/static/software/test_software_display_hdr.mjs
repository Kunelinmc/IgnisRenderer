import assert from "node:assert/strict";

import { SoftwareBackend } from "../../../src/backends/software/SoftwareBackend.ts";
import { SoftwareDisplayOutputManager } from "../../../src/backends/software/SoftwareDisplayOutputManager.ts";
import { SoftwareSurfaceRuntime } from "../../../src/backends/software/SoftwareSurfaceRuntime.ts";
import {
	DEFAULT_DISPLAY_OUTPUT_OPTIONS,
} from "../../../src/rendering/DisplayOutput.ts";
import {
	encodeLinearSRGB,
} from "../../../src/postprocess/passes/GammaPass.ts";
import {
	applyACESToneMapping,
} from "../../../src/postprocess/passes/ToneMappingPass.ts";

const originalImageData = globalThis.ImageData;
const originalMatchMedia = globalThis.matchMedia;

class FakeImageData {
	constructor(dataOrWidth, width, height) {
		if (ArrayBuffer.isView(dataOrWidth)) {
			this.data = dataOrWidth;
			this.width = width;
			this.height = height;
		} else {
			this.width = dataOrWidth;
			this.height = width;
			this.data = new Uint8ClampedArray(this.width * this.height * 4);
		}
	}
}

function createMediaQuery(matches = true) {
	return {
		matches,
		listener: null,
		addEventListener(_type, listener) {
			this.listener = listener;
		},
		removeEventListener(_type, listener) {
			if (this.listener === listener) this.listener = null;
		},
	};
}

function createHarness(options = {}) {
	const probeAttributes = options.probeAttributes ?? {
		colorSpace: "display-p3",
		colorType: "float16",
	};
	const visibleAttributes = options.visibleAttributes ?? probeAttributes;
	const captured = {
		image: null,
		probeSettings: null,
		visibleSettings: null,
	};
	const makeContext = (attributes, readback = false) => ({
		getContextAttributes: () => attributes,
		putImageData(image) {
			captured.image = image;
			if (options.presentationThrows && !readback) {
				throw new Error("presentation failed");
			}
		},
		getImageData() {
			const source = captured.image?.data;
			const red = options.preserveHDR === false ? 1 : source?.[0] ?? 0;
			return new FakeImageData(new Float16Array([red, 0.25, 0.5, 1]), 1, 1);
		},
	});
	const probeContext = makeContext(probeAttributes, true);
	const visibleContext = makeContext(visibleAttributes, false);
	const probeCanvas = {
		width: 1,
		height: 1,
		getContext(_kind, settings) {
			captured.probeSettings = settings ?? null;
			if (options.float16Unavailable && settings?.colorType === "float16") {
				return null;
			}
			return probeContext;
		},
	};
	const canvas = {
		width: 1,
		height: 1,
		ownerDocument: { createElement: () => probeCanvas },
		getContext(_kind, settings) {
			captured.visibleSettings = settings ?? null;
			if (options.visibleThrows && settings?.colorType === "float16") {
				throw new Error("visible HDR context rejected");
			}
			return visibleContext;
		},
	};
	return { canvas, captured };
}

function createManager(mode = "hdr") {
	return new SoftwareDisplayOutputManager({
		...DEFAULT_DISPLAY_OUTPUT_OPTIONS,
		mode,
	});
}

function configureManager(manager, harness) {
	const probe = manager.detect(harness.canvas);
	const context = harness.canvas.getContext("2d", probe.contextSettings);
	return {
		context,
		state: manager.configure(context),
	};
}

function initializeManager(manager, harness) {
	return configureManager(manager, harness).state;
}

function testCapabilityMatrix() {
	const detection = createManager();
	const detectionHarness = createHarness();
	const probe = detection.detect(detectionHarness.canvas);
	assert.equal(probe.hdrCanvasSupported, true);
	assert.equal(probe.contextSettings.colorType, "float16");
	assert.equal(detectionHarness.captured.visibleSettings, null);
	detection.destroy();

	const success = createManager();
	const successHarness = createHarness();
	assert.deepEqual(initializeManager(success, successHarness), {
		requested: { mode: "hdr", exposure: 1, hdrHeadroom: 4 },
		activeDynamicRange: "hdr",
		colorSpace: "display-p3",
	});
	assert.equal(successHarness.captured.probeSettings.willReadFrequently, true);
	assert.equal(successHarness.captured.visibleSettings.willReadFrequently, true);
	const switchingHarness = createHarness();
	const switching = createManager("sdr");
	initializeManager(switching, switchingHarness);
	assert.equal(switchingHarness.captured.visibleSettings.colorType, "float16");
	assert.equal(switching.setRequested({ mode: "hdr" }).activeDynamicRange, "hdr");

	const missingFloat16Harness = createHarness({ float16Unavailable: true });
	const missingFloat16 = createManager();
	assert.equal(
		initializeManager(missingFloat16, missingFloat16Harness)
			.fallbackReason,
		"canvas-hdr-output-unsupported",
	);
	assert.equal(
		missingFloat16Harness.captured.visibleSettings.willReadFrequently,
		true,
	);

	const ignoredP3 = createManager();
	assert.equal(
		initializeManager(ignoredP3, createHarness({
			probeAttributes: { colorSpace: "srgb", colorType: "float16" },
		})).fallbackReason,
		"canvas-hdr-output-unsupported",
	);

	const clippedReadback = createManager();
	assert.equal(
		initializeManager(clippedReadback, createHarness({ preserveHDR: false }))
			.fallbackReason,
		"canvas-hdr-output-unsupported",
	);

	const configurationFailed = createManager();
	assert.equal(
		initializeManager(configurationFailed, createHarness({
			visibleAttributes: { colorSpace: "srgb", colorType: "uint8" },
		})).fallbackReason,
		"hdr-context-configuration-failed",
	);

	const mediaQuery = globalThis.matchMedia();
	mediaQuery.matches = false;
	const nonHDRDisplay = createManager();
	assert.equal(
		initializeManager(nonHDRDisplay, createHarness()).fallbackReason,
		"display-not-hdr-capable",
	);
	mediaQuery.matches = true;
}

function testPresentationAndTargetLifecycle() {
	const manager = createManager("hdr");
	const harness = createHarness();
	const surface = new SoftwareSurfaceRuntime(manager);
	const configured = configureManager(manager, harness);
	surface.initialize(configured.context);
	const attachments = surface.getAttachments({ width: 1, height: 1 });
	const sceneColor = surface.getSceneColorTarget();
	sceneColor.set([4, 2, 1, 1]);
	const frame = {
		attachments: { ...attachments, color: sceneColor },
		clipRegions: [{ minX: 0, minY: 0, maxXExclusive: 1, maxYExclusive: 1 }],
	};
	surface.present(frame, "scene-linear-hdr");
	assert.ok(harness.captured.image.data instanceof Float16Array);
	assert.ok(harness.captured.image.data[0] > 1);
	assert.equal(attachments.pixels[0], 255);
	assert.equal(attachments.pixels[1], 255);
	assert.equal(attachments.pixels[3], 255);

	const sameTarget = surface.getSceneColorTarget();
	surface.getAttachments({ width: 1, height: 1 });
	assert.equal(surface.getSceneColorTarget(), sameTarget);
	surface.resize({ width: 2, height: 1 });
	surface.getAttachments({ width: 2, height: 1 });
	assert.notEqual(surface.getSceneColorTarget(), sameTarget);
	surface.resize({ width: 2, height: 2 });
	const resizedAttachments = surface.getAttachments({ width: 2, height: 2 });
	const resizedColor = surface.getSceneColorTarget();
	resizedColor.fill(1);
	surface.present({
		attachments: { ...resizedAttachments, color: resizedColor },
		clipRegions: [{ minX: 0, minY: 0, maxXExclusive: 2, maxYExclusive: 2 }],
	}, "scene-linear-hdr");
	assert.equal(harness.captured.image.width, 2);
	assert.equal(harness.captured.image.height, 2);
	surface.destroy();

	const sdrManager = createManager("sdr");
	const sdrHarness = createHarness();
	const sdrSurface = new SoftwareSurfaceRuntime(sdrManager);
	const sdrConfigured = configureManager(sdrManager, sdrHarness);
	sdrSurface.initialize(sdrConfigured.context);
	const sdrAttachments = sdrSurface.getAttachments({ width: 1, height: 1 });
	const sdrColor = sdrSurface.getSceneColorTarget();
	sdrColor.set([4, 0.5, 0.25, 1]);
	sdrSurface.present({
		attachments: { ...sdrAttachments, color: sdrColor },
		clipRegions: [{ minX: 0, minY: 0, maxXExclusive: 1, maxYExclusive: 1 }],
	}, "scene-linear-hdr");
	const mapped = applyACESToneMapping([4, 0.5, 0.25], 1);
	assert.equal(sdrAttachments.pixels[0], Math.round(encodeLinearSRGB(mapped[0]) * 255));
	assert.ok(sdrHarness.captured.image.data instanceof Uint8ClampedArray);
}

async function testMediaQueryEventsAndPresentationFailure() {
	const mediaQuery = globalThis.matchMedia();
	const events = [];
	const backendHarness = createHarness();
	const backend = new SoftwareBackend();
	backend.attach({
		surface: {
			canvas: backendHarness.canvas,
			displayOutput: { mode: "auto", exposure: 1, hdrHeadroom: 4 },
		},
		events: { emit: (event) => events.push(event) },
	});
	await backend.initialize();
	assert.equal(backendHarness.captured.visibleSettings.colorType, "float16");
	assert.equal(backend.getDisplayOutputState().activeDynamicRange, "hdr");
	mediaQuery.matches = false;
	mediaQuery.listener();
	assert.equal(backend.getDisplayOutputState().activeDynamicRange, "sdr");
	assert.ok(events.some((event) => event.type === "display-output-change"));
	assert.ok(events.some((event) =>
		event.type === "render-invalidated" && event.reason === "display-output"));
	backend.destroy();

	const fallbackHarness = createHarness({
		visibleThrows: true,
		visibleAttributes: { colorSpace: "srgb", colorType: "uint8" },
	});
	const fallbackBackend = new SoftwareBackend();
	fallbackBackend.attach({
		surface: {
			canvas: fallbackHarness.canvas,
			displayOutput: { mode: "hdr", exposure: 1, hdrHeadroom: 4 },
		},
		events: { emit: () => {} },
	});
	await fallbackBackend.initialize();
	assert.equal(fallbackBackend.getDisplayOutputState().activeDynamicRange, "sdr");
	assert.equal(
		fallbackBackend.getDisplayOutputState().fallbackReason,
		"hdr-context-configuration-failed",
	);
	assert.equal(fallbackHarness.captured.visibleSettings.colorType, undefined);
	fallbackBackend.destroy();

	const failingManager = createManager("sdr");
	const failingSurface = new SoftwareSurfaceRuntime(failingManager);
	const failingHarness = createHarness({ presentationThrows: true });
	const failingConfigured = configureManager(failingManager, failingHarness);
	failingSurface.initialize(failingConfigured.context);
	const attachments = failingSurface.getAttachments({ width: 1, height: 1 });
	const color = failingSurface.getSceneColorTarget();
	color.set([1, 1, 1, 1]);
	assert.throws(() => failingSurface.present({
		attachments: { ...attachments, color },
		clipRegions: [{ minX: 0, minY: 0, maxXExclusive: 1, maxYExclusive: 1 }],
	}, "scene-linear-hdr"), /presentation failed/);
}

async function run() {
	const mediaQuery = createMediaQuery(true);
	globalThis.ImageData = FakeImageData;
	globalThis.matchMedia = () => mediaQuery;
	testCapabilityMatrix();
	testPresentationAndTargetLifecycle();
	await testMediaQueryEventsAndPresentationFailure();
	console.log("Software display HDR tests passed");
}

try {
	await run();
} finally {
	globalThis.ImageData = originalImageData;
	globalThis.matchMedia = originalMatchMedia;
}
