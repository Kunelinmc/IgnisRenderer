import assert from "node:assert/strict";

import { SoftwareBackend } from "../../../src/backends/software/SoftwareBackend.ts";
import { createSoftwareFrameView } from "../../../src/backends/software/SoftwareFrameView.ts";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

function createBackend(events = []) {
	const backend = new SoftwareBackend();
	backend.attach({
		surface: { canvas: { width: 4, height: 4 } },
		events: { emit: (event) => events.push(event) },
	});
	return backend;
}

function createContext(backend, options = {}) {
	const camera = new Camera();
	camera.position.set(0, 0, 4);
	camera.aspectRatio = 1;
	camera.updateMatrices();
	const zeroSH = Array.from({ length: 9 }, () => ({ r: 0, g: 0, b: 0 }));
	return {
		viewCamera: camera,
		attachments: backend.getAttachments(options.size ?? { width: 4, height: 4 }),
		features: {
			enableLighting: false,
			enableSH: false,
			enableShadows: false,
			enableReflection: false,
			enableEnvironment: false,
			warnings: [],
		},
		postProcess: createResolvedPostProcess({
			taa: { enabled: options.taa ?? false },
			gamma: { enabled: false },
		}),
		shadowMaps: new Map(),
		scene: {
			sceneBounds: { center: { x: 0, y: 0, z: 0 }, radius: 1 },
			lights: [],
			particleSystems: [],
			hasActiveAnimations: false,
			camera,
			environment: null,
			meshInstances: [],
			shadowMaps: new Map(),
			opaquePackets: [],
			transparentPackets: [],
			shadowCasterSubmissions: [],
			shadowTransmitterSubmissions: [],
			reflectivePackets: [],
			decalPackets: [],
			spatialIndex: null,
		},
		shCoeffs: zeroSH,
		shAmbientCoeffs: zeroSH,
		worldMatrix: Matrix4.identity(),
		incremental: {
			enabled: false,
			forceFullFrame: true,
			dirtyRects: [],
			dirtyTileSize: 0,
			dirtyTileColumns: 0,
			dirtyTileRows: 0,
			dirtyTiles: [],
			dirtyAreaRatio: 1,
			firstPass: null,
			reasonMask: 0,
			temporalHistoryReset: options.temporalHistoryReset ?? false,
		},
		transient: new Map(),
	};
}

async function testStrictLifecycleAndDeferredResize() {
	const backend = createBackend();
	assert.throws(() => backend.beginFrame({}), /ready state/);
	await backend.initialize();
	await assert.rejects(() => backend.initialize(), /attached state/);

	const context = createContext(backend);
	backend.beginFrame(context);
	assert.throws(() => backend.beginFrame(context), /ready state/);
	await assert.rejects(
		() => backend.executePass({ stage: "main-opaque" }, {}),
		/foreign frame context/
	);
	backend.resize({ width: 8, height: 8 });
	assert.equal(context.attachments.width, 4);
	backend.endFrame();
	assert.equal(backend.getAttachments({ width: 8, height: 8 }).width, 8);

	backend.destroy();
	backend.destroy();
	await assert.rejects(() => backend.initialize(), /attached state/);
	assert.throws(() => backend.attach({}), /already attached/);
}

async function testAbortIsIdempotentAndRollsBackTemporalState() {
	const backend = createBackend();
	await backend.initialize();
	const first = createContext(backend, { taa: true });
	backend.beginFrame(first);
	const firstState = backend.getTemporalDebugState();
	await backend.abortFrame();
	await backend.abortFrame();

	const second = createContext(backend, { taa: true });
	backend.beginFrame(second);
	const secondState = backend.getTemporalDebugState();
	assert.deepEqual(secondState.jitter, firstState.jitter);
	assert.equal(secondState.previousViewProjection, null);
	await backend.abortFrame();
}

async function testCommitAndPresentationFailureDoNotAdvanceTemporalHistory() {
	const backend = createBackend();
	await backend.initialize();
	const committed = createContext(backend, { taa: true });
	backend.beginFrame(committed);
	backend.endFrame();

	const beforeFailure = createContext(backend, { taa: true });
	backend.beginFrame(beforeFailure);
	const expectedHistory = backend.getTemporalDebugState().previousViewProjection;
	const originalPresent = backend._surface.present;
	backend._surface.present = () => {
		throw new Error("present failed");
	};
	assert.throws(() => backend.endFrame(), /present failed/);
	backend._surface.present = originalPresent;
	await backend.abortFrame();

	const afterFailure = createContext(backend, { taa: true });
	backend.beginFrame(afterFailure);
	assert.equal(
		backend.getTemporalDebugState().previousViewProjection,
		expectedHistory
	);
	await backend.abortFrame();
}

async function testRestoreReportsSuccessAndPreservesRuntimeOnFailure() {
	const events = [];
	const backend = createBackend(events);
	await backend.initialize();
	await backend.restore();
	assert.deepEqual(events, [{ type: "device-restored" }]);

	const originalInitialize = backend._surface.initialize;
	backend._surface.initialize = () => {
		throw new Error("restore failed");
	};
	await assert.rejects(() => backend.restore(), /restore failed/);
	backend._surface.initialize = originalInitialize;
	const context = createContext(backend);
	backend.beginFrame(context);
	await backend.abortFrame();
}

async function testFrameViewValidatesAttachmentsAndNormalizesRegions() {
	const backend = createBackend();
	await backend.initialize();
	const context = createContext(backend);
	context.incremental.enabled = true;
	context.incremental.forceFullFrame = false;
	context.incremental.dirtyRects = [
		{ x: -1.25, y: 0.25, width: 2.5, height: 2.25 },
		{ x: 8, y: 8, width: 1, height: 1 },
	];
	const temporal = {
		currentJitter: [0, 0],
		previousJitter: [0, 0],
		previousViewProjection: null,
		currentViewProjection: context.viewCamera.viewProjectionMatrix,
		previousWorldMatrices: new Map(),
		currentWorldMatrices: new Map(),
	};
	const frame = createSoftwareFrameView(context, temporal);
	assert.deepEqual(frame.clipRegions, [
		{ minX: 0, minY: 0, maxXExclusive: 2, maxYExclusive: 3 },
	]);
	const cameraPosition = { ...frame.camera.position };
	context.viewCamera.position.set(7, 8, 9);
	assert.deepEqual(frame.camera.position, cameraPosition);

	const malformed = createContext(backend);
	malformed.attachments.pixels = new Uint8ClampedArray(3);
	assert.throws(
		() => createSoftwareFrameView(malformed, temporal),
		/^Error: Software frame attachments invalid:/,
	);
	const invalidTypes = createContext(backend);
	invalidTypes.attachments.pixels = new Uint8Array(4 * 4 * 4);
	assert.throws(
		() => createSoftwareFrameView(invalidTypes, temporal),
		/^Error: Software frame attachments invalid:/,
	);
	const invalidDepth = createContext(backend);
	invalidDepth.attachments.depthBuffer = new Float64Array(4 * 4);
	assert.throws(
		() => createSoftwareFrameView(invalidDepth, temporal),
		/^Error: Software frame attachments invalid:/,
	);
	backend.destroy();
}

async function testAbortPreservesLastCompletedCoverage() {
	const backend = createBackend();
	await backend.initialize();
	const partial = createContext(backend, { size: { width: 8, height: 8 } });
	partial.incremental.enabled = true;
	partial.incremental.forceFullFrame = false;
	partial.incremental.dirtyRects = [{ x: 0, y: 0, width: 1, height: 1 }];
	partial.incremental.dirtyTileSize = 4;
	partial.incremental.dirtyTileColumns = 2;
	partial.incremental.dirtyTileRows = 2;
	partial.incremental.dirtyTiles = [0];
	partial.incremental.temporalHistoryReset = false;
	backend.beginFrame(partial);
	backend.endFrame();
	assert.equal(backend.getCompletedFrameCoverage(), "dirty-tiles");

	const aborted = createContext(backend, { size: { width: 8, height: 8 } });
	aborted.incremental.enabled = true;
	aborted.incremental.forceFullFrame = false;
	aborted.incremental.dirtyRects = [{ x: 0, y: 0, width: 1, height: 1 }];
	aborted.incremental.dirtyTileSize = 4;
	aborted.incremental.dirtyTileColumns = 2;
	aborted.incremental.dirtyTileRows = 2;
	aborted.incremental.dirtyTiles = [0];
	aborted.incremental.temporalHistoryReset = false;
	backend.beginFrame(aborted);
	await backend.abortFrame(new Error("expected abort"));
	assert.equal(backend.getCompletedFrameCoverage(), "dirty-tiles");
	backend.destroy();
}

await testStrictLifecycleAndDeferredResize();
await testAbortIsIdempotentAndRollsBackTemporalState();
await testCommitAndPresentationFailureDoNotAdvanceTemporalHistory();
await testRestoreReportsSuccessAndPreservesRuntimeOnFailure();
await testFrameViewValidatesAttachmentsAndNormalizesRegions();
await testAbortPreservesLastCompletedCoverage();
console.log("Software backend lifecycle tests passed");
