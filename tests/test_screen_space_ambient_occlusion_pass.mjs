import assert from "node:assert/strict";

import {
	PostProcessPipeline,
	ScreenSpaceAmbientOcclusionPass,
	SoftwareScreenSpaceAmbientOcclusionImplementation,
	createSSAOKernelParams,
	resolveSSAODownsample,
	resolveSSAOOptions,
} from "../src/postprocess/index.ts";
import { CameraType } from "../src/cameras/Camera.ts";
import {
	createResolvedPostProcess,
} from "./helpers/postprocess.mjs";

function createOrthographicCamera() {
	const identity = [
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1],
	];
	return {
		type: CameraType.Orthographic,
		fov: 60,
		aspectRatio: 1,
		near: 0.1,
		far: 100,
		projectionMatrix: { elements: identity },
		viewMatrix: { elements: identity },
		getWorldPosition() {
			return { x: 0, y: 0, z: 0 };
		},
		getBounds() {
			return { left: -1, right: 1, bottom: -1, top: 1 };
		},
	};
}

function createIncremental(width, height) {
	return {
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
	};
}

function createSoftwareFrameContext(options = {}) {
	const width = 4;
	const height = 4;
	const pixels = new Uint8ClampedArray(width * height * 4);
	for (let i = 0; i < pixels.length; i += 4) {
		pixels[i] = 100;
		pixels[i + 1] = 100;
		pixels[i + 2] = 100;
		pixels[i + 3] = 255;
	}
	const depthBuffer = new Float32Array(width * height).fill(1);
	depthBuffer[1 * width + 1] = 5;
	const normalBuffer = new Float32Array(width * height * 3);
	for (let i = 0; i < width * height; i++) {
		normalBuffer[i * 3 + 2] = 1;
	}
	return {
		camera: createOrthographicCamera(),
		attachments: {
			width,
			height,
			pixels,
			depthBuffer,
			normalBuffer,
		},
		features: {},
		postProcess: createResolvedPostProcess(
			{
				ssao: { enabled: true, options },
				gamma: { enabled: false },
			},
			"software"
		),
		shadowMaps: new Map(),
		scene: {},
		shCoeffs: [],
		shAmbientCoeffs: [],
		worldMatrix: null,
		incremental: createIncremental(width, height),
		transient: new Map(),
	};
}

function createGBuffer(context) {
	return {
		width: context.attachments.width,
		height: context.attachments.height,
		normalSpace: "view",
		depthEncoding: "linear-view-z",
		channels: {
			depth: {
				semantic: "depth",
				width: context.attachments.width,
				height: context.attachments.height,
				handle: {
					backend: "software",
					data: context.attachments.depthBuffer,
					stride: 1,
				},
			},
			normal: {
				semantic: "normal",
				width: context.attachments.width,
				height: context.attachments.height,
				handle: {
					backend: "software",
					data: context.attachments.normalBuffer,
					stride: 3,
				},
			},
		},
		worldPosition: {
			source: "derived",
			available: true,
		},
	};
}

function createSoftwareExecutor() {
	return {
		backend: "software",
		fallbackCalls: [],
		createResource(desc) {
			return {
				id: desc.id,
				backend: "software",
				width: desc.width,
				height: desc.height,
				format: desc.format,
				resource: new Float32Array(desc.width * desc.height * 4),
			};
		},
		destroyResource() {},
		getPassExecutionContext(request) {
			if (request.passId !== "ssao") {
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

function createPassRequest(frameContext, pass) {
	return {
		frameContext,
		postProcess: frameContext.postProcess,
		gBuffer: createGBuffer(frameContext),
		histories: {},
		pass,
		passId: "ssao",
		options: frameContext.postProcess.getOptions("ssao"),
		startPassId: null,
	};
}

async function testSSAOPipelineUsesPassOwnedImplementation() {
	const pass = new ScreenSpaceAmbientOcclusionPass({ enabled: true });
	assert.equal(pass.id, "ssao");
	assert.deepEqual(
		pass.getRequirements({}).gBuffer,
		["depth", "normal"]
	);
	assert.equal(
		typeof pass.getImplementation("software").execute,
		"function"
	);
	assert.equal(
		typeof pass.getImplementation("webgpu").execute,
		"function"
	);
	assert.equal(
		typeof pass.getImplementation("webgl").execute,
		"function"
	);

	const pipeline = new PostProcessPipeline();
	const executor = createSoftwareExecutor();
	const frameContext = createSoftwareFrameContext({ samples: 4, radius: 1 });
	const result = await pipeline.execute({
		frameContext,
		executor,
		gBuffer: createGBuffer(frameContext),
	});

	assert.deepEqual(result.executedPassIds, ["ssao"]);
	assert.deepEqual(executor.fallbackCalls, []);
}

function testSoftwareSSAOModifiesPixelsAndSkipsMissingBuffers() {
	const pass = new ScreenSpaceAmbientOcclusionPass({ enabled: true });
	const implementation = new SoftwareScreenSpaceAmbientOcclusionImplementation();
	implementation._kernel = new Array(4).fill({ x: 0, y: 0, z: -1 });
	implementation._noise = new Array(16).fill({ x: 1, y: 0, z: 0 });

	const frameContext = createSoftwareFrameContext({
		samples: 4,
		radius: 1,
		bias: 0.01,
		intensity: 1,
		blurRadius: 1,
	});
	const centerIndex = ((1 * frameContext.attachments.width + 1) << 2);
	const before = frameContext.attachments.pixels[centerIndex];
	const result = implementation.execute(
		createPassRequest(frameContext, pass),
		{ attachments: frameContext.attachments }
	);

	assert.equal(result.ran, true);
	assert.ok(frameContext.attachments.pixels[centerIndex] < before);

	const missing = createSoftwareFrameContext();
	missing.attachments.normalBuffer = null;
	const skipped = implementation.execute(
		createPassRequest(missing, pass),
		{ attachments: missing.attachments }
	);
	assert.deepEqual(skipped, { ran: false });
}

function testSSAOOptionHelpersClampAndPackParams() {
	const options = resolveSSAOOptions({
		samples: 99,
		radius: -10,
		bias: 0,
		intensity: -1,
		downsample: 99,
		blurRadius: 99,
		blurSharpness: 0,
	});
	assert.equal(options.samples, 48);
	assert.equal(options.radius, 1);
	assert.equal(options.bias, 1e-4);
	assert.equal(options.intensity, 0);
	assert.equal(options.downsample, 8);
	assert.equal(options.blurRadius, 4);
	assert.equal(options.blurSharpness, 1e-3);
	assert.equal(resolveSSAODownsample(Number.NaN), 2);

	const params = createSSAOKernelParams(
		64,
		32,
		32,
		16,
		options,
		createOrthographicCamera(),
		0,
		1,
		0.25
	);
	assert.equal(params.length, 16);
	assert.equal(params[0], 1 / 64);
	assert.equal(params[1], 1 / 32);
	assert.equal(params[2], 1 / 32);
	assert.equal(params[3], 1 / 16);
	assert.equal(params[7], 48);
	assert.equal(params[12], 0);
	assert.equal(params[13], 1);
	assert.equal(params[14], 1);
	assert.equal(params[15], 0.25);
}

await testSSAOPipelineUsesPassOwnedImplementation();
testSoftwareSSAOModifiesPixelsAndSkipsMissingBuffers();
testSSAOOptionHelpersClampAndPackParams();
console.log("ScreenSpaceAmbientOcclusionPass tests passed");
