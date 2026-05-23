import assert from "node:assert/strict";

import {
	PostProcessPipeline,
	TemporalAntiAliasingPass,
} from "../src/postprocess/index.ts";
import {
	ALL_POST_PROCESS_CAPABILITIES,
	createResolvedPostProcess,
} from "./helpers/postprocess.mjs";

const TAA_ONLY_CAPABILITIES = Object.fromEntries(
	Object.keys(ALL_POST_PROCESS_CAPABILITIES).map((key) => [key, key === "taa"])
);

function createSoftwareExecutor() {
	return {
		backend: "software",
		capabilities: TAA_ONLY_CAPABILITIES,
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
		getPassExecutionContext(passId, request) {
			if (passId !== "taa") {
				return undefined;
			}
			return { attachments: request.frameContext.attachments };
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
	for (let i = 0; i < width * height; i++) {
		motionBuffer[(i << 2) + 2] = 1;
	}
	return {
		camera: {
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
			TAA_ONLY_CAPABILITIES,
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
	assert.deepEqual(pass.getRequirements({}).gBuffer, ["motion"]);
	assert.deepEqual(
		pass.getHistoryDescriptors({}).map((history) => history.id),
		["taa", "motion"]
	);
	assert.equal(
		typeof pass.getImplementation("software").execute,
		"function"
	);

	const pipeline = new PostProcessPipeline();
	const executor = createSoftwareExecutor();
	const frameContext = createFrameContext();
	const result = await pipeline.execute({
		frameContext,
		executor,
		gBuffer: createGBuffer(frameContext),
	});

	assert.deepEqual(result.executedPassIds, ["taa"]);
	assert.deepEqual(executor.fallbackCalls, []);
	assert.deepEqual(
		executor.created.map((handle) => handle.id),
		["taa:read", "taa:write", "motion:read", "motion:write"]
	);
	assert.equal(frameContext.attachments.pixels[0], 64);
}

await testTAADescriptorAndImplementationRoute();
console.log("TemporalAntiAliasingPass tests passed");
