import assert from "node:assert/strict";

import {
	SCREEN_SPACE_REFRACTIONS_PASS_ID,
	ScreenSpaceRefractionsPass,
	createSSRefractionTraceParams,
	resolveSSRefractionOptions,
	PostProcessGraphCompiler,
} from "../../../src/postprocess/index.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

function createFrameContext(transmissionFactor = 1) {
	return {
		viewCamera: {
			type: "perspective",
			fov: 60,
			aspectRatio: 1,
			near: 0.1,
			far: 100,
		},
		attachments: {
			width: 64,
			height: 32,
		},
		features: {},
		postProcess: createResolvedPostProcess(
			{ ssrefraction: { enabled: true } },
			"webgpu"
		),
		scene: {
			transparentPackets:
				transmissionFactor > 0 ?
					[{ id: "glass", material: { transmissionFactor } }]
				:	[{ id: "alpha", material: { transmissionFactor: 0 } }],
		},
		incremental: {
			enabled: false,
			forceFullFrame: true,
			dirtyRects: [],
			temporalHistoryReset: false,
		},
		transient: new Map(),
	};
}

function createGBuffer(includeTransmission = true, includeNormal = true) {
	const channels = {
		depth: {},
		motion: {},
	};
	if (includeNormal) {
		channels.normal = {};
	}
	if (includeTransmission) {
		channels.transmission = {};
	}
	return {
		width: 64,
		height: 32,
		normalSpace: "world",
		depthEncoding: "linear-view-z",
		motionEncoding: "ndc-delta",
		channels,
		worldPosition: {
			source: "derived",
			available: true,
		},
	};
}

function testDescriptorAndOptions() {
	const pass = new ScreenSpaceRefractionsPass({ enabled: true });
	assert.equal(pass.id, SCREEN_SPACE_REFRACTIONS_PASS_ID);
	assert.equal(pass.placement, "temporal");
	assert.equal(pass.order, 215);
	assert.deepEqual(pass.getRequirements({}).gBuffer, [
		"depth",
		"motion",
		"normal",
		"transmission",
	]);
	assert.equal(typeof pass.getImplementation("webgpu").execute, "function");
	assert.equal(pass.getImplementation("software"), null);
	assert.equal(pass.getImplementation("webgl"), null);

	const options = resolveSSRefractionOptions({
		downsample: 99,
		maxSteps: 12,
		binarySearchSteps: 3,
		planeRefinementSteps: 99,
		maxDistance: 9,
		thickness: 0.5,
		stride: 2,
		intensity: 0.75,
		edgeFade: 0.2,
		roughnessMipScale: 6,
	});
	assert.equal(options.downsample, 8);
	assert.equal(options.maxSteps, 12);
	assert.equal(options.binarySearchSteps, 3);
	assert.equal(options.planeRefinementSteps, 8);
	assert.equal(options.maxDistance, 9);
	assert.equal(options.thickness, 0.5);
	assert.equal(options.stride, 2);
	assert.equal(options.intensity, 0.75);
	assert.equal(options.edgeFade, 0.2);
	assert.equal(options.roughnessMipScale, 6);

	const params = createSSRefractionTraceParams(64, 32, options, 5);
	assert.equal(params.length, 16);
	assert.equal(params[0], 1 / 64);
	assert.equal(params[1], 1 / 32);
	assert.equal(params[2], 9);
	assert.equal(params[9], 5);
	assert.equal(params[10], 6);
	assert.equal(params[11], 8);

	const defaultOptions = resolveSSRefractionOptions();
	assert.equal(defaultOptions.planeRefinementSteps, 3);
	const disabledPlaneRefinement = resolveSSRefractionOptions({
		planeRefinementSteps: 0,
	});
	assert.equal(disabledPlaneRefinement.planeRefinementSteps, 0);

	pass.destroy();
}

function testShouldExecuteRequiresTransmissionPackets() {
	const pass = new ScreenSpaceRefractionsPass({ enabled: true });
	assert.equal(
		pass.shouldExecute({
			frameContext: createFrameContext(1),
			backend: "webgpu",
			options: pass.normalizeOptions({}),
		}),
		true
	);
	assert.equal(
		pass.shouldExecute({
			frameContext: createFrameContext(0),
			backend: "webgpu",
			options: pass.normalizeOptions({}),
		}),
		false
	);
	assert.equal(pass.shouldExecute({ backend: "webgpu" }), false);
	pass.destroy();
}

function testUnsupportedBackendsDisableWithoutBuiltInWarning() {
	const software = createResolvedPostProcess(
		{ ssrefraction: { enabled: true } },
		"software"
	);
	assert.equal(software.isEnabled("ssrefraction"), false);
	assert.deepEqual(software.getWarnings(), []);
}

async function testPipelineRequiresTransmissionChannel() {
	const pass = new ScreenSpaceRefractionsPass({ enabled: true });
	const frameContext = createFrameContext(1);
	const warnings = [];
	const graph = new PostProcessGraphCompiler().compile({
		frameContext,
		backend: "webgpu",
		postProcess: frameContext.postProcess,
		gBuffer: createGBuffer(false),
		warn(key, message) {
			warnings.push({ key, message });
		},
	});
	assert.deepEqual(graph.passes, []);
	assert.deepEqual(
		warnings.map((warning) => warning.key),
		["postprocess-requirement-missing-ssrefraction"]
	);
	pass.destroy();
}

async function testPipelineRequiresNormalChannel() {
	const pass = new ScreenSpaceRefractionsPass({ enabled: true });
	const frameContext = createFrameContext(1);
	const warnings = [];
	const graph = new PostProcessGraphCompiler().compile({
		frameContext,
		backend: "webgpu",
		postProcess: frameContext.postProcess,
		gBuffer: createGBuffer(true, false),
		warn(key, message) {
			warnings.push({ key, message });
		},
	});
	assert.deepEqual(graph.passes, []);
	assert.deepEqual(
		warnings.map((warning) => warning.key),
		["postprocess-requirement-missing-ssrefraction"]
	);
	pass.destroy();
}

function testTransientDescriptors() {
	const pass = new ScreenSpaceRefractionsPass({
		enabled: true,
		options: { downsample: 2 },
	});
	const descriptors = pass.getTransientResourceDescriptors({
		backend: "webgpu",
		width: 64,
		height: 32,
		options: pass.normalizeOptions({}),
	});
	assert.deepEqual(
		descriptors.map((descriptor) => descriptor.id),
		["ssrefraction:raw"]
	);
	assert.equal(descriptors[0].widthScale, 0.5);
	assert.equal(descriptors[0].heightScale, 0.5);
	assert.deepEqual(
		pass.getTransientResourceDescriptors({ backend: "software" }),
		[]
	);
	pass.destroy();
}

testDescriptorAndOptions();
testShouldExecuteRequiresTransmissionPackets();
testUnsupportedBackendsDisableWithoutBuiltInWarning();
await testPipelineRequiresTransmissionChannel();
await testPipelineRequiresNormalChannel();
testTransientDescriptors();
console.log("ScreenSpaceRefractionsPass tests passed");
