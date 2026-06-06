import assert from "node:assert/strict";
import {
	IncrementalFramePlanner,
	IncrementalRegistry,
	getDefaultIncrementalRegistry,
	renderDirtyReasonToMask,
} from "../../../src/pipeline/incremental.ts";
import {
	PostProcessPass,
	PostProcessPassRegistry,
} from "../../../src/postprocess/index.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

class TestBuiltInPostProcessPass extends PostProcessPass {
	constructor() {
		super({
			id: "test-built-in",
			builtIn: true,
			enabled: true,
			implementations: {
				test: {},
			},
		});
	}
}

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

function createPostProcess(overrides = {}) {
	return createResolvedPostProcess(overrides, "test");
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
		postProcess: createPostProcess({
			"interaction-outline": { enabled: true },
		}),
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

function testBuiltInPostProcessStageUsesPassMetadata() {
	const registry = new IncrementalRegistry();
	const reasonMask = registry.registerDirtyReason({
		id: "test-built-in-dirty",
		firstPass: "test-built-in",
	});
	const postProcessRegistry = new PostProcessPassRegistry();
	postProcessRegistry.registerPass(new TestBuiltInPostProcessPass());

	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask,
		features: createFeatures(),
		postProcess: postProcessRegistry.createSnapshot("test"),
		registry,
	});

	assert.equal(plan.firstPass, "postprocess");
	assert.equal(plan.postProcessStartPass, "test-built-in");
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

function testDecalStartsAtMainOpaqueWithoutFullFrame() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("decal"),
		features: createFeatures({ enableShadows: true }),
		postProcess: createPostProcess(),
	});
	assert.equal(plan.firstPass, "main-opaque");
	assert.equal(plan.postProcessStartPass, null);
	assert.equal(plan.forceFullFrame, false);
	assert.equal(plan.temporalHistoryReset, true);
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
	testBuiltInPostProcessStageUsesPassMetadata();
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
	testDecalStartsAtMainOpaqueWithoutFullFrame();
	testCustomDirtyReasonAllocatesMaskAndPlansFirstPass();
	testCustomDirtyReasonUsesGroups();
	console.log("Incremental frame planner tests passed");
}

run();
