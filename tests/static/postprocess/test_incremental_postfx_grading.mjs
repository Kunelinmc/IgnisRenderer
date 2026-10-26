import assert from "node:assert/strict";
import {
	getDefaultIncrementalRegistry,
} from "../../../src/pipeline/incremental.ts";
import { PostProcessPass } from "../../../src/postprocess/PostProcessPass.ts";
import { createPostProcessRegistryFromRequest } from "../../helpers/postprocess.mjs";

const incremental = getDefaultIncrementalRegistry();

class CustomCinematicPass extends PostProcessPass {
	constructor() {
		super({
			id: "custom-cinematic",
			schedule: {
				incremental: {
					firstPass: "dof",
					grade: "cinematic",
					inflationRadius: 40,
					fallbackScale: 0.5,
				},
			},
		});
	}
}

function createPostProcess(overrides = {}) {
	const registry = createPostProcessRegistryFromRequest(
		{
			tonemap: { enabled: false },
			gamma: { enabled: false },
			...overrides,
		},
		"test"
	);
	for (const pass of registry.getPasses()) {
		incremental.registerPostProcessPass(pass);
	}
	return registry.createSnapshot("test");
}

function enable(id, options) {
	return {
		[id]: {
			enabled: true,
			options,
		},
	};
}

function testDefaultRegistryUsesExplicitDisplayPassRegistration() {
	const registry = createPostProcessRegistryFromRequest({
		tonemap: { enabled: true },
		gamma: { enabled: true },
	});
	const postProcess = registry.createSnapshot("test");
	assert.equal(incremental.resolvePostProcessGrade(postProcess), "none");
	assert.equal(incremental.computePostProcessInflationRadius(postProcess), 0);
	assert.equal(
		incremental.scaleFullFrameFallbackAreaRatioForPostProcess(0.3, postProcess),
		0.3
	);

	for (const pass of registry.getPasses()) {
		incremental.registerPostProcessPass(pass);
	}
	assert.equal(incremental.resolvePostProcessGrade(postProcess), "light");
	assert.equal(incremental.computePostProcessInflationRadius(postProcess), 2);
	assert.equal(
		incremental.scaleFullFrameFallbackAreaRatioForPostProcess(0.3, postProcess),
		0.3
	);
}

function testResolvePostProcessGrade() {
	assert.equal(incremental.resolvePostProcessGrade(createPostProcess()), "none");
	assert.equal(
		incremental.resolvePostProcessGrade(createPostProcess(enable("fxaa"))),
		"light"
	);
	assert.equal(
		incremental.resolvePostProcessGrade(createPostProcess(enable("tonemap"))),
		"light"
	);
	assert.equal(
		incremental.resolvePostProcessGrade(createPostProcess(enable("color-filter"))),
		"light"
	);
	assert.equal(
		incremental.resolvePostProcessGrade(createPostProcess(enable("ssgi"))),
		"cinematic"
	);
	assert.equal(
		incremental.resolvePostProcessGrade(createPostProcess(enable("dof"))),
		"cinematic"
	);
}

function testComputePostProcessInflationRadius() {
	assert.equal(incremental.computePostProcessInflationRadius(createPostProcess()), 0);
	assert.equal(
		incremental.computePostProcessInflationRadius(createPostProcess(enable("gamma"))),
		2
	);
	assert.equal(
		incremental.computePostProcessInflationRadius(
			createPostProcess(enable("tonemap"))
		),
		2
	);
	assert.equal(
		incremental.computePostProcessInflationRadius(createPostProcess(enable("ssgi"))),
		24
	);
	assert.equal(
		incremental.computePostProcessInflationRadius(
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
	const noneRatio = incremental.scaleFullFrameFallbackAreaRatioForPostProcess(
		baseRatio,
		createPostProcess()
	);
	const standardRatio = incremental.scaleFullFrameFallbackAreaRatioForPostProcess(
		baseRatio,
		createPostProcess(enable("bloom"))
	);
	const cinematicRatio = incremental.scaleFullFrameFallbackAreaRatioForPostProcess(
		baseRatio,
		createPostProcess(enable("taa"))
	);
	assert.equal(noneRatio, 0.3);
	assert.equal(standardRatio, 0.27);
	assert.equal(cinematicRatio, 0.24);
}

function testScreenSpaceRefractionUsesRegisteredIncrementalMetadata() {
	const registry = getDefaultIncrementalRegistry();
	const postProcess = createPostProcess(enable("ssrefraction"));
	assert.equal(incremental.resolvePostProcessGrade(postProcess), "cinematic");
	assert.equal(
		registry.resolveFirstEnabledPostProcessStage(postProcess),
		"ssrefraction"
	);
	assert.equal(incremental.computePostProcessInflationRadius(postProcess), 24);
}

function testCustomPostProcessDefaultIncrementalMetadata() {
	const postProcess = createPostProcess(enable("custom-edge"));
	assert.equal(incremental.resolvePostProcessGrade(postProcess), "light");
	assert.equal(incremental.computePostProcessInflationRadius(postProcess), 2);
}

function testCustomPostProcessIncrementalMetadataOverride() {
	const registry = getDefaultIncrementalRegistry();
	registry.registerPostProcessPass(new CustomCinematicPass());
	try {
		const postProcess = createPostProcess(enable("custom-cinematic"));
		assert.equal(incremental.resolvePostProcessGrade(postProcess), "cinematic");
		assert.equal(
			registry.resolveFirstEnabledPostProcessStage(postProcess),
			"dof"
		);
		assert.equal(incremental.computePostProcessInflationRadius(postProcess), 40);
		assert.equal(
			incremental.scaleFullFrameFallbackAreaRatioForPostProcess(0.3, postProcess),
			0.15
		);
	} finally {
		registry.unregisterPostProcessPass("custom-cinematic");
	}
}

function run() {
	testDefaultRegistryUsesExplicitDisplayPassRegistration();
	testResolvePostProcessGrade();
	testComputePostProcessInflationRadius();
	testScaleFullFrameFallbackAreaRatioForPostProcess();
	testScreenSpaceRefractionUsesRegisteredIncrementalMetadata();
	testCustomPostProcessDefaultIncrementalMetadata();
	testCustomPostProcessIncrementalMetadataOverride();
	console.log("Incremental postfx grading tests passed");
}

run();
