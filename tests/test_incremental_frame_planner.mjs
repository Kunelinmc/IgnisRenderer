import assert from "node:assert/strict";
import {
	IncrementalFramePlanner,
	renderDirtyReasonToMask,
} from "../src/pipeline/incremental.ts";
import { resolvePostProcessState } from "../src/pipeline/PostProcessController.ts";

function createFeatures(overrides = {}) {
	return {
		enableLighting: true,
		enableSH: false,
		enableShadows: true,
		enableReflection: false,
		enableEnvironment: false,
		enableClusteredLighting: false,
		warnings: [],
		...overrides,
	};
}

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
	return resolvePostProcessState(overrides, capabilities, "test");
}

function testNoDirtyReasonsReturnsNoPass() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: 0,
		features: createFeatures(),
		postProcess: createPostProcess(),
	});
	assert.equal(plan.firstPass, null);
	assert.equal(plan.forceFullFrame, false);
	assert.equal(plan.temporalHistoryReset, false);
}

function testInteractionStartsAtInteractionOutline() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("interaction"),
		features: createFeatures(),
		postProcess: createPostProcess(),
	});
	assert.equal(plan.firstPass, "interaction-outline");
	assert.equal(plan.forceFullFrame, false);
}

function testParticlesStartAtParticleSim() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("particles"),
		features: createFeatures(),
		postProcess: createPostProcess(),
	});
	assert.equal(plan.firstPass, "particle-sim");
	assert.equal(plan.forceFullFrame, false);
}

function testPostFxStartsAtFirstEnabledPostStage() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("postfx"),
		features: createFeatures(),
		postProcess: createPostProcess({
			bloom: { enabled: true },
			fxaa: { enabled: true },
		}),
	});
	assert.equal(plan.firstPass, "bloom");
}

function testPostFxStandardReasonStartsAtEarliestEnabledPostStage() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("postfx-standard"),
		features: createFeatures(),
		postProcess: createPostProcess({
			ssao: { enabled: true },
			bloom: { enabled: true },
			fxaa: { enabled: true },
		}),
	});
	assert.equal(plan.firstPass, "ssao");
}

function testPostFxStartsAtFogWhenOnlyFogPostProcessEnabled() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("postfx"),
		features: createFeatures(),
		postProcess: createPostProcess({
			fog: {
				enabled: true,
				options: {
					application: "postprocess",
				},
			},
			fxaa: { enabled: true },
		}),
	});
	assert.equal(plan.firstPass, "fog");
}

function testPostFxSkipsFogInSceneMode() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("postfx"),
		features: createFeatures(),
		postProcess: createPostProcess({
			fog: {
				enabled: true,
				options: {
					application: "scene",
				},
			},
			fxaa: { enabled: true },
		}),
	});
	assert.equal(plan.firstPass, "tonemap");
}

function testPostFxSkipsToneMappingWhenDisabled() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("postfx"),
		features: createFeatures(),
		postProcess: createPostProcess({
			fog: {
				enabled: true,
				options: {
					application: "scene",
				},
			},
			tonemap: { enabled: false },
			fxaa: { enabled: true },
		}),
	});
	assert.equal(plan.firstPass, "fxaa");
}

function testPostFxCinematicReasonResetsTemporalHistory() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("postfx-cinematic"),
		features: createFeatures(),
		postProcess: createPostProcess({
			taa: { enabled: true },
			fxaa: { enabled: true },
		}),
	});
	assert.equal(plan.firstPass, "taa");
	assert.equal(plan.temporalHistoryReset, true);
}

function testCameraForcesFullAndResetsTemporal() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("camera"),
		features: createFeatures({ enableShadows: true }),
		postProcess: createPostProcess(),
	});
	assert.equal(plan.firstPass, "shadow");
	assert.equal(plan.forceFullFrame, true);
	assert.equal(plan.temporalHistoryReset, true);
}

function testGeometryFallsBackToMainWhenShadowsDisabled() {
	const mask =
		renderDirtyReasonToMask("transform") |
		renderDirtyReasonToMask("material") |
		renderDirtyReasonToMask("texture");
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: mask,
		features: createFeatures({ enableShadows: false }),
		postProcess: createPostProcess(),
	});
	assert.equal(plan.firstPass, "main-opaque");
	assert.equal(plan.forceFullFrame, false);
	assert.equal(plan.temporalHistoryReset, true);
}

function testDisabledIncrementalAlwaysFullFrame() {
	const plan = IncrementalFramePlanner.plan({
		enabled: false,
		reasonMask: renderDirtyReasonToMask("interaction"),
		features: createFeatures(),
		postProcess: createPostProcess(),
	});
	assert.equal(plan.firstPass, null);
	assert.equal(plan.forceFullFrame, true);
	assert.equal(plan.temporalHistoryReset, true);
}

function testEnvironmentIBLForcesFullFrameWithoutTemporalReset() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("environment-ibl"),
		features: createFeatures(),
		postProcess: createPostProcess(),
	});
	assert.equal(plan.forceFullFrame, true);
	assert.equal(plan.temporalHistoryReset, false);
	assert.equal(plan.firstPass, "main-opaque");
}

function testEnvironmentIBLCompleteResetsTemporalHistory() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("environment-ibl-complete"),
		features: createFeatures(),
		postProcess: createPostProcess(),
	});
	assert.equal(plan.forceFullFrame, true);
	assert.equal(plan.temporalHistoryReset, true);
	assert.equal(plan.firstPass, "main-opaque");
}

function run() {
	testNoDirtyReasonsReturnsNoPass();
	testInteractionStartsAtInteractionOutline();
	testParticlesStartAtParticleSim();
	testPostFxStartsAtFirstEnabledPostStage();
	testPostFxStandardReasonStartsAtEarliestEnabledPostStage();
	testPostFxStartsAtFogWhenOnlyFogPostProcessEnabled();
	testPostFxSkipsFogInSceneMode();
	testPostFxSkipsToneMappingWhenDisabled();
	testPostFxCinematicReasonResetsTemporalHistory();
	testCameraForcesFullAndResetsTemporal();
	testGeometryFallsBackToMainWhenShadowsDisabled();
	testDisabledIncrementalAlwaysFullFrame();
	testEnvironmentIBLForcesFullFrameWithoutTemporalReset();
	testEnvironmentIBLCompleteResetsTemporalHistory();
	console.log("Incremental frame planner tests passed");
}

run();
