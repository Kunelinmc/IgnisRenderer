import assert from "node:assert/strict";
import {
	computePostProcessInflationRadius,
	getDefaultIncrementalRegistry,
	resolvePostProcessGrade,
	scaleFullFrameFallbackAreaRatioForPostProcess,
} from "../src/pipeline/incremental.ts";
import { resolvePostProcessState } from "../src/pipeline/PostProcessController.ts";

const capabilities = {
	ssao: true,
	ssgi: true,
	taa: true,
	ssr: true,
	volumetric: true,
	fog: true,
	"motion-blur": true,
	dof: true,
	bloom: true,
	tonemap: true,
	"color-filter": true,
	fxaa: true,
	"interaction-outline": true,
	gamma: true,
};

function createPostProcess(overrides = {}) {
	return resolvePostProcessState(
		{
			tonemap: { enabled: false },
			"interaction-outline": { enabled: false },
			gamma: { enabled: false },
			...overrides,
		},
		capabilities,
		"test"
	);
}

function enable(id, options) {
	return {
		[id]: {
			enabled: true,
			options,
		},
	};
}

function testResolvePostProcessGrade() {
	assert.equal(resolvePostProcessGrade(createPostProcess()), "none");
	assert.equal(
		resolvePostProcessGrade(createPostProcess(enable("fxaa"))),
		"light"
	);
	assert.equal(
		resolvePostProcessGrade(createPostProcess(enable("tonemap"))),
		"light"
	);
	assert.equal(
		resolvePostProcessGrade(createPostProcess(enable("color-filter"))),
		"light"
	);
	assert.equal(
		resolvePostProcessGrade(createPostProcess(enable("ssgi"))),
		"standard"
	);
	assert.equal(
		resolvePostProcessGrade(createPostProcess(enable("dof"))),
		"cinematic"
	);
}

function testComputePostProcessInflationRadius() {
	assert.equal(computePostProcessInflationRadius(createPostProcess()), 0);
	assert.equal(
		computePostProcessInflationRadius(createPostProcess(enable("gamma"))),
		2
	);
	assert.equal(
		computePostProcessInflationRadius(
			createPostProcess(enable("tonemap"))
		),
		2
	);
	assert.equal(
		computePostProcessInflationRadius(createPostProcess(enable("ssgi"))),
		12
	);
	assert.equal(
		computePostProcessInflationRadius(
			createPostProcess({
				...enable("taa"),
				...enable("dof"),
			})
		),
		32
	);
}

function testScaleFullFrameFallbackAreaRatioForPostProcess() {
	const baseRatio = 0.3;
	const noneRatio = scaleFullFrameFallbackAreaRatioForPostProcess(
		baseRatio,
		createPostProcess()
	);
	const standardRatio = scaleFullFrameFallbackAreaRatioForPostProcess(
		baseRatio,
		createPostProcess(enable("bloom"))
	);
	const cinematicRatio = scaleFullFrameFallbackAreaRatioForPostProcess(
		baseRatio,
		createPostProcess(enable("taa"))
	);
	assert.equal(noneRatio, 0.3);
	assert.equal(standardRatio, 0.27);
	assert.equal(cinematicRatio, 0.24);
}

function testCustomPostProcessDefaultIncrementalMetadata() {
	const postProcess = createPostProcess(enable("custom-edge"));
	assert.equal(resolvePostProcessGrade(postProcess), "light");
	assert.equal(computePostProcessInflationRadius(postProcess), 2);
}

function testCustomPostProcessIncrementalMetadataOverride() {
	const registry = getDefaultIncrementalRegistry();
	registry.registerPostProcessPass("custom-cinematic", {
		firstPass: "dof",
		grade: "cinematic",
		inflationRadius: 40,
		fallbackScale: 0.5,
	});
	try {
		const postProcess = createPostProcess(enable("custom-cinematic"));
		assert.equal(resolvePostProcessGrade(postProcess), "cinematic");
		assert.equal(
			registry.resolveFirstEnabledPostProcessStage(postProcess),
			"dof"
		);
		assert.equal(computePostProcessInflationRadius(postProcess), 40);
		assert.equal(
			scaleFullFrameFallbackAreaRatioForPostProcess(0.3, postProcess),
			0.15
		);
	} finally {
		registry.unregisterPostProcessPass("custom-cinematic");
	}
}

function run() {
	testResolvePostProcessGrade();
	testComputePostProcessInflationRadius();
	testScaleFullFrameFallbackAreaRatioForPostProcess();
	testCustomPostProcessDefaultIncrementalMetadata();
	testCustomPostProcessIncrementalMetadataOverride();
	console.log("Incremental postfx grading tests passed");
}

run();
