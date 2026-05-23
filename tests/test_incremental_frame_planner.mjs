import assert from "node:assert/strict";
import {
	IncrementalFramePlanner,
	getDefaultIncrementalRegistry,
	renderDirtyReasonToMask,
} from "../src/pipeline/incremental.ts";
import { createResolvedPostProcess } from "./helpers/postprocess.mjs";

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
	return createResolvedPostProcess(overrides, capabilities, "test");
}

function testNoDirtyReasonsReturnsNoPass() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: 0,
		features: createFeatures(),
		postProcess: createPostProcess(),
	});
	assert.equal(plan.firstPass, null);
	assert.equal(plan.postProcessStartPass, null);
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
	assert.equal(plan.firstPass, "postprocess");
	assert.equal(plan.postProcessStartPass, "interaction-outline");
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
	assert.equal(plan.postProcessStartPass, null);
	assert.equal(plan.forceFullFrame, true);
	assert.equal(plan.temporalHistoryReset, true);
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
	assert.equal(plan.firstPass, "postprocess");
	assert.equal(plan.postProcessStartPass, "bloom");
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
	assert.equal(plan.firstPass, "postprocess");
	assert.equal(plan.postProcessStartPass, "ssao");
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
	assert.equal(plan.firstPass, "postprocess");
	assert.equal(plan.postProcessStartPass, "fog");
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
	assert.equal(plan.firstPass, "postprocess");
	assert.equal(plan.postProcessStartPass, "fxaa");
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
	assert.equal(plan.firstPass, "postprocess");
	assert.equal(plan.postProcessStartPass, "fxaa");
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
	assert.equal(plan.firstPass, "postprocess");
	assert.equal(plan.postProcessStartPass, "taa");
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
	assert.equal(plan.postProcessStartPass, null);
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
	assert.equal(plan.postProcessStartPass, null);
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
	assert.equal(plan.postProcessStartPass, null);
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
	assert.equal(plan.postProcessStartPass, null);
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
	assert.equal(plan.postProcessStartPass, null);
}

function testCustomDirtyReasonAllocatesMaskAndPlansFirstPass() {
	const registry = getDefaultIncrementalRegistry();
	const passId = "custom-incremental-pass-test";
	const reasonId = "custom-incremental-reason-test";
	registry.registerFramePass({ id: passId, order: 3.5 });
	const mask = registry.registerDirtyReason({
		id: reasonId,
		firstPass: passId,
		temporalHistoryReset: true,
	});
	try {
		assert.equal(renderDirtyReasonToMask(reasonId), mask);
		assert.notEqual(mask, renderDirtyReasonToMask("unknown"));
		const plan = IncrementalFramePlanner.plan({
			enabled: true,
			reasonMask: mask,
			features: createFeatures(),
			postProcess: createPostProcess(),
			registry,
		});
		assert.equal(plan.firstPass, passId);
		assert.equal(plan.postProcessStartPass, null);
		assert.equal(plan.forceFullFrame, false);
		assert.equal(plan.temporalHistoryReset, true);
	} finally {
		registry.unregisterDirtyReason(reasonId);
		registry.unregisterFramePass(passId);
	}
}

function testCustomDirtyReasonUsesGroups() {
	const registry = getDefaultIncrementalRegistry();
	const reasonId = "custom-geometry-reason-test";
	const mask = registry.registerDirtyReason({
		id: reasonId,
		groups: ["geometry"],
		forceFullFrame: true,
	});
	try {
		const plan = IncrementalFramePlanner.plan({
			enabled: true,
			reasonMask: mask,
			features: createFeatures({ enableShadows: false }),
			postProcess: createPostProcess(),
			registry,
		});
		assert.equal(plan.firstPass, "main-opaque");
		assert.equal(plan.postProcessStartPass, null);
		assert.equal(plan.forceFullFrame, true);
	} finally {
		registry.unregisterDirtyReason(reasonId);
	}
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
	testCustomDirtyReasonAllocatesMaskAndPlansFirstPass();
	testCustomDirtyReasonUsesGroups();
	console.log("Incremental frame planner tests passed");
}

run();
