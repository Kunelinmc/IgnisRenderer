import assert from "node:assert/strict";
import {
	IncrementalFramePlanner,
	renderDirtyReasonToMask,
} from "../src/pipeline/incremental.ts";

function createFeatures(overrides = {}) {
	return {
		enableLighting: true,
		enableGamma: true,
		enableSH: false,
		enableShadows: true,
		enableReflection: false,
		enableSkybox: false,
		enableSSAO: false,
		enableSSGI: false,
		enableTAA: false,
		enableSSR: false,
		enableVolumetric: false,
		enableMotionBlur: false,
		enableDOF: false,
		enableBloom: false,
		enableFXAA: true,
		enableClusteredLighting: false,
		warnings: [],
		...overrides,
	};
}

function testNoDirtyReasonsReturnsNoPass() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: 0,
		features: createFeatures(),
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
	});
	assert.equal(plan.firstPass, "interaction-outline");
	assert.equal(plan.forceFullFrame, false);
}

function testParticlesStartAtParticleSim() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("particles"),
		features: createFeatures(),
	});
	assert.equal(plan.firstPass, "particle-sim");
	assert.equal(plan.forceFullFrame, false);
}

function testPostFxStartsAtFirstEnabledPostStage() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("postfx"),
		features: createFeatures({
			enableSSAO: false,
			enableSSGI: false,
			enableTAA: false,
			enableSSR: false,
			enableVolumetric: false,
			enableMotionBlur: false,
			enableDOF: false,
			enableBloom: true,
			enableFXAA: true,
		}),
	});
	assert.equal(plan.firstPass, "bloom");
}

function testPostFxStandardReasonStartsAtEarliestEnabledPostStage() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("postfx-standard"),
		features: createFeatures({
			enableSSAO: true,
			enableSSGI: false,
			enableTAA: false,
			enableSSR: false,
			enableVolumetric: false,
			enableMotionBlur: false,
			enableDOF: false,
			enableBloom: true,
			enableFXAA: true,
		}),
	});
	assert.equal(plan.firstPass, "ssao");
}

function testPostFxCinematicReasonResetsTemporalHistory() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("postfx-cinematic"),
		features: createFeatures({
			enableTAA: true,
			enableFXAA: true,
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
	});
	assert.equal(plan.firstPass, null);
	assert.equal(plan.forceFullFrame, true);
	assert.equal(plan.temporalHistoryReset, true);
}

function run() {
	testNoDirtyReasonsReturnsNoPass();
	testInteractionStartsAtInteractionOutline();
	testParticlesStartAtParticleSim();
	testPostFxStartsAtFirstEnabledPostStage();
	testPostFxStandardReasonStartsAtEarliestEnabledPostStage();
	testPostFxCinematicReasonResetsTemporalHistory();
	testCameraForcesFullAndResetsTemporal();
	testGeometryFallsBackToMainWhenShadowsDisabled();
	testDisabledIncrementalAlwaysFullFrame();
	console.log("Incremental frame planner tests passed");
}

run();
