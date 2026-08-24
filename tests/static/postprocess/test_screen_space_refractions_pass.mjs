import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import {
	SCREEN_SPACE_REFRACTIONS_PASS_ID,
	ScreenSpaceRefractionsPass,
	createSSRefractionTraceParams,
	resolveSSRefractionOptions,
	PostProcessPlanner,
} from "../../../src/postprocess/index.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";

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
					[createTestDrawPacket({
						id: "glass",
						material: { transmissionFactor },
					})]
				:	[createTestDrawPacket({
						id: "alpha",
						material: { transmissionFactor: 0 },
					})],
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
		normalSpace: "view",
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
	assert.equal(pass.schedule.placement, "temporal");
	assert.equal(pass.schedule.order, 215);
	const declaration = pass.getImplementation("webgpu").describeExecution({
		options: pass.normalizeOptions({}),
	});
	assert.deepEqual(declaration.gBuffer.map((entry) => entry.semantic), [
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
	const clampedPhysicalOptions = resolveSSRefractionOptions({
		maxSteps: 999,
		binarySearchSteps: -2,
		maxDistance: -1,
		thickness: 0,
		stride: -5,
		intensity: 4,
		edgeFade: 2,
		roughnessMipScale: -1,
	});
	assert.deepEqual(clampedPhysicalOptions, {
		downsample: 1,
		maxSteps: 256,
		binarySearchSteps: 0,
		planeRefinementSteps: 3,
		maxDistance: 0.01,
		thickness: 0.001,
		stride: 0.01,
		intensity: 1,
		edgeFade: 0.5,
		roughnessMipScale: 0,
	});

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
	const graph = new PostProcessPlanner().plan({
		frameContext,
		backend: "webgpu",
		postProcess: frameContext.postProcess,
		gBuffer: createGBuffer(false),
		resolveImplementation: (descriptor) => descriptor.getImplementation("webgpu"),
		isSharedResourceAvailable: () => true,
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
	const graph = new PostProcessPlanner().plan({
		frameContext,
		backend: "webgpu",
		postProcess: frameContext.postProcess,
		gBuffer: createGBuffer(true, false),
		resolveImplementation: (descriptor) => descriptor.getImplementation("webgpu"),
		isSharedResourceAvailable: () => true,
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
	const declaration = pass.getImplementation("webgpu").describeExecution({
		backend: "webgpu",
		width: 64,
		height: 32,
		options: pass.normalizeOptions({}),
	});
	const descriptors = declaration.transients.map((entry) => entry.descriptor);
	assert.deepEqual(
		descriptors.map((descriptor) => descriptor.id),
		["ssrefraction:raw", "ssrefraction:denoise-scratch"]
	);
	assert.equal(descriptors[0].widthScale, 0.5);
	assert.equal(descriptors[0].heightScale, 0.5);
	assert.equal(descriptors[1].widthScale, 0.5);
	assert.equal(descriptors[1].heightScale, 0.5);
	assert.equal(pass.getImplementation("software"), null);
	pass.destroy();
}

async function testShaderPreservesTransmissionEnergyContract() {
	const refractionShader = await readFile(new URL(
		"../../../src/shaders/webgpu/postprocess/ssrf.wgsl",
		import.meta.url,
	), "utf8");
	const captureShader = await readFile(new URL(
		"../../../src/shaders/webgpu/scene/fragmentSingleTarget.wgsl",
		import.meta.url,
	), "utf8");

	assert.match(
		captureShader,
		/let transmission =\s*clamp\([\s\S]*?\*\s*\(1\.0 - metalness\);/,
		"transmission capture must suppress the metallic portion",
	);
	assert.match(
		captureShader,
		/let coverage = alpha;/,
		"geometric coverage must stay independent from Fresnel",
	);
	assert.match(
		captureShader,
		/transmissionPathLength = thickness \/ max\(cosThetaT, PBR_MIN_NDOTV\);/,
		"Beer-Lambert attenuation must use the refracted path length",
	);
	assert.match(
		captureShader,
		/encodeNormalForGBuffer\(pbrNormal\)/,
		"transmission capture must use the camera-facing PBR normal",
	);
	assert.match(
		refractionShader,
		/pathLength = materialThickness \/ insideCos;/,
		"non-zero volume thickness must move the ray to a parallel exit interface",
	);
	assert.match(
		refractionShader,
		/totalInternalReflection = !entryRefraction\.valid;/,
		"invalid material-to-air refraction must enter the TIR path",
	);
	assert.match(
		refractionShader,
		/transmission \* \(1\.0 - fresnelR\) \* traceParams\.intensity/,
		"the transmitted background must apply Fresnel transmittance once",
	);
	assert.match(
		refractionShader,
		/scene\.rgb \* \(1\.0 - coverage\) \+ lighting\.rgb \+ raw\.rgb \* coverage/,
		"composition must separate uncovered scene, surface lighting, and transmission",
	);
	assert.match(
		refractionShader,
		/ROUGH_TRANSMISSION_OFFSETS/,
		"rough transmission must widen its base-level sampling footprint",
	);
	assert.doesNotMatch(
		refractionShader,
		/textureSampleLevel\(backgroundColor,[\s\S]{0,80},\s*mip\s*\)/,
		"the single-level transmission background must not be sampled at unavailable mips",
	);
}

testDescriptorAndOptions();
testShouldExecuteRequiresTransmissionPackets();
testUnsupportedBackendsDisableWithoutBuiltInWarning();
await testPipelineRequiresTransmissionChannel();
await testPipelineRequiresNormalChannel();
testTransientDescriptors();
await testShaderPreservesTransmissionEnergyContract();
console.log("ScreenSpaceRefractionsPass tests passed");
