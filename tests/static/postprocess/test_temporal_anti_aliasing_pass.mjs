import assert from "node:assert/strict";

import {
	TemporalAntiAliasingPass,
} from "../../../src/postprocess/index.ts";
import { BackendPostProcessRuntime } from "../../../src/postprocess/BackendPostProcessRuntime.ts";
import {
	createResolvedPostProcess,
} from "../../helpers/postprocess.mjs";

function createSoftwareExecutor() {
	return {
		backend: "software",
		created: [],
		fallbackCalls: [],
		createResource(desc) {
			const handle = {
				id: desc.id,
				backend: "software",
				width: desc.width,
				height: desc.height,
				format: desc.format,
				resource: new Float32Array(desc.width * desc.height * 4),
			};
			this.created.push(handle);
			return handle;
		},
		destroyResource() {},
		createGBufferBridge(context) {
			return createGBuffer(context);
		},
		createPassExecutionContext(request) {
			if (request.passId !== "taa") {
				return undefined;
			}
			return {
				attachments: request.frameContext.attachments,
				resources: {
					color: {
						input: request.frameContext.sceneColor,
						output: request.frameContext.sceneColor,
					},
				},
			};
		},
		executePass(passId) {
			this.fallbackCalls.push(passId);
			return { ran: true };
		},
	};
}

function createFrameContext() {
	const width = 2;
	const height = 2;
	const pixels = new Uint8ClampedArray([
		64, 32, 16, 255,
		64, 32, 16, 255,
		64, 32, 16, 255,
		64, 32, 16, 255,
	]);
	const motionBuffer = new Float32Array(width * height * 4);
	const sceneColor = new Float32Array(width * height * 4);
	for (let i = 0; i < width * height; i++) {
		sceneColor[i << 2] = 4;
		sceneColor[(i << 2) + 1] = 0.5;
		sceneColor[(i << 2) + 2] = 0.25;
		sceneColor[(i << 2) + 3] = 1;
	}
	for (let i = 0; i < width * height; i++) {
		motionBuffer[(i << 2) + 2] = 1;
	}
	return {
		sceneColor,
		viewCamera: {
			type: "perspective",
			fov: 60,
			aspectRatio: 1,
			near: 0.1,
			far: 100,
		},
		attachments: {
			width,
			height,
			pixels,
			depthBuffer: new Float32Array(width * height),
			normalBuffer: new Float32Array(width * height * 3),
			motionBuffer,
		},
		features: {},
		postProcess: createResolvedPostProcess(
			{
				taa: { enabled: true, options: { sharpen: 0 } },
				gamma: { enabled: false },
			},
			"software"
		),
		shadowMaps: new Map(),
		scene: {},
		shCoeffs: [],
		shAmbientCoeffs: [],
		worldMatrix: null,
		incremental: {
			enabled: false,
			forceFullFrame: true,
			dirtyRects: [{ x: 0, y: 0, width, height }],
			dirtyTileSize: 64,
			dirtyTileColumns: 1,
			dirtyTileRows: 1,
			dirtyTiles: [0],
			dirtyAreaRatio: 1,
			firstPass: null,
			postProcessStartPass: null,
			reasonMask: 0,
			temporalHistoryReset: false,
		},
		transient: new Map(),
	};
}

function createGBuffer(context) {
	return {
		width: context.attachments.width,
		height: context.attachments.height,
		normalSpace: "view",
		depthEncoding: "linear-view-z",
		motionEncoding: "ndc-delta",
		channels: {
			motion: {
				semantic: "motion",
				width: context.attachments.width,
				height: context.attachments.height,
				handle: {
					backend: "software",
					data: context.attachments.motionBuffer,
					stride: 4,
				},
			},
		},
		worldPosition: {
			source: "derived",
			available: true,
		},
	};
}

async function testTAADescriptorAndImplementationRoute() {
	const pass = new TemporalAntiAliasingPass({ enabled: true });
	assert.equal(pass.id, "taa");
	const declaration = pass.getImplementation("software").describeExecution({});
	assert.deepEqual(declaration.gBuffer.map((entry) => entry.semantic), ["motion"]);
	assert.deepEqual(
		declaration.histories.map((history) => history.descriptor.id),
		["taa", "motion"]
	);
	assert.deepEqual(declaration.frameRequirements, {
		cameraJitter: { sequence: "halton-2-3", scale: 1 },
	});
	assert.equal(
		typeof pass.getImplementation("software").execute,
		"function"
	);

	const executor = createSoftwareExecutor();
	const frameContext = createFrameContext();
	const runtime = new BackendPostProcessRuntime({
		executor,
		backend: { type: "software" },
	});
	await runtime.execute(frameContext);

	assert.deepEqual(executor.fallbackCalls, []);
	assert.deepEqual(
		executor.created.map((handle) => handle.id),
		["taa:read", "taa:write", "motion:read", "motion:write"]
	);
	assert.equal(frameContext.sceneColor[0], 4);
	assert.equal(
		executor.created.find((handle) => handle.id === "taa:write").resource[0],
		4,
	);
}

await testTAADescriptorAndImplementationRoute();
console.log("TemporalAntiAliasingPass tests passed");
