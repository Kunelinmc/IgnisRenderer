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
import {
	createPostProcessRegistryFromRequest,
	createResolvedPostProcess,
} from "../../helpers/postprocess.mjs";

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

function createRegisteredPostProcess(overrides = {}) {
	const postProcessRegistry = createPostProcessRegistryFromRequest(
		overrides,
		"test"
	);
	const registry = new IncrementalRegistry();
	for (const pass of postProcessRegistry.getPasses()) {
		registry.registerPostProcessPass(pass);
	}
	return {
		postProcess: postProcessRegistry.createSnapshot("test"),
		registry,
	};
}

function testDisplayPassMetadataRequiresRegistration() {
	for (const registry of [new IncrementalRegistry(), getDefaultIncrementalRegistry()]) {
		const postProcessRegistry = createPostProcessRegistryFromRequest({
			tonemap: { enabled: true },
			gamma: { enabled: true },
		});
		const postProcess = postProcessRegistry.createSnapshot("test");
		assert.equal(registry.resolveFirstEnabledPostProcessStage(postProcess), null);
		assert.equal(registry.resolvePostProcessGrade(postProcess), "none");
		assert.equal(registry.computePostProcessInflationRadius(postProcess), 0);
		assert.equal(
			registry.scaleFullFrameFallbackAreaRatioForPostProcess(0.3, postProcess),
			0.3
		);

		for (const pass of postProcessRegistry.getPasses()) {
			registry.registerPostProcessPass(pass);
		}
		const plan = IncrementalFramePlanner.plan({
			enabled: true,
			reasonMask: registry.getDirtyReasonMask("postfx"),
			features: createFeatures(),
			postProcess,
			registry,
		});
		assert.equal(plan.firstPass, "postprocess");
		assert.equal(plan.postProcessStartPass, "tonemap");
		assert.equal(registry.resolvePostProcessGrade(postProcess), "light");
		assert.equal(registry.computePostProcessInflationRadius(postProcess), 2);

		postProcessRegistry.getPass("tonemap").disable();
		const gammaOnly = postProcessRegistry.createSnapshot("test");
		assert.equal(registry.resolveFirstEnabledPostProcessStage(gammaOnly), "gamma");
		postProcessRegistry.getPass("gamma").disable();
		const disabled = postProcessRegistry.createSnapshot("test");
		assert.equal(registry.resolveFirstEnabledPostProcessStage(disabled), null);
		assert.equal(registry.resolvePostProcessGrade(disabled), "none");
		assert.equal(registry.computePostProcessInflationRadius(disabled), 0);
	}
}

function testUnregisteredGammaDoesNotChoosePostProcessStart() {
	const registry = new IncrementalRegistry();
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: registry.getDirtyReasonMask("postfx"),
		features: createFeatures(),
		postProcess: createPostProcess({ gamma: { enabled: true } }),
		registry,
	});
	assert.equal(plan.firstPass, "main-opaque");
	assert.equal(plan.postProcessStartPass, null);
}

function testDisplayPassPlanningUsesRegisteredSchedules() {
	const registered = createRegisteredPostProcess({
		tonemap: { enabled: true },
		gamma: {
			enabled: true,
			schedule: {
				order: 500,
				incremental: {
					firstPass: "gamma",
					grade: "cinematic",
					inflationRadius: 40,
					fallbackScale: 0.5,
				},
			},
		},
	});
	const { registry, postProcess } = registered;
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: registry.getDirtyReasonMask("postfx"),
		features: createFeatures(),
		postProcess,
		registry,
	});
	assert.equal(plan.firstPass, "postprocess");
	assert.equal(plan.postProcessStartPass, "gamma");
	assert.equal(registry.resolvePostProcessGrade(postProcess), "cinematic");
	assert.equal(registry.computePostProcessInflationRadius(postProcess), 40);
	assert.equal(
		registry.scaleFullFrameFallbackAreaRatioForPostProcess(0.3, postProcess),
		0.15
	);
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
	const registered = createRegisteredPostProcess({
		bloom: { enabled: true },
		fxaa: { enabled: true },
	});
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("postfx"),
		features: createFeatures(),
		postProcess: registered.postProcess,
		registry: registered.registry,
	});
	assert.equal(plan.firstPass, "postprocess");
	assert.equal(plan.postProcessStartPass, "bloom");
	assert.equal(plan.forceFullFrame, true, "global postprocess changes require scene replay");
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
	const registered = createRegisteredPostProcess({
		ssao: { enabled: true },
		bloom: { enabled: true },
		fxaa: { enabled: true },
	});
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("postfx-standard"),
		features: createFeatures(),
		postProcess: registered.postProcess,
		registry: registered.registry,
	});
	assert.equal(plan.firstPass, "postprocess");
	assert.equal(plan.postProcessStartPass, "ssao");
}

function testPostFxStartsAtFogWhenOnlyFogPostProcessEnabled() {
	const registered = createRegisteredPostProcess({
		fog: { enabled: true },
		fxaa: { enabled: true },
	});
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("postfx"),
		features: createFeatures(),
		postProcess: registered.postProcess,
		registry: registered.registry,
	});
	assert.equal(plan.firstPass, "postprocess");
	assert.equal(plan.postProcessStartPass, "fog");
}

function testLegacyFogApplicationStillStartsAtFog() {
	const registered = createRegisteredPostProcess({
		fog: {
			enabled: true,
			options: {
				application: "scene",
			},
		},
		fxaa: { enabled: true },
	});
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("postfx"),
		features: createFeatures(),
		postProcess: registered.postProcess,
		registry: registered.registry,
	});
	assert.equal(plan.firstPass, "postprocess");
	assert.equal(plan.postProcessStartPass, "fog");
}

function testPostFxSkipsToneMappingWhenDisabled() {
	const registered = createRegisteredPostProcess({
		fog: { enabled: false },
		tonemap: { enabled: false },
		fxaa: { enabled: true },
	});
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("postfx"),
		features: createFeatures(),
		postProcess: registered.postProcess,
		registry: registered.registry,
	});
	assert.equal(plan.firstPass, "postprocess");
	assert.equal(plan.postProcessStartPass, "fxaa");
}

function testPostFxCinematicReasonResetsTemporalHistory() {
	const registered = createRegisteredPostProcess({
		taa: { enabled: true },
		fxaa: { enabled: true },
	});
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("postfx-cinematic"),
		features: createFeatures(),
		postProcess: registered.postProcess,
		registry: registered.registry,
	});
	assert.equal(plan.firstPass, "postprocess");
	assert.equal(plan.postProcessStartPass, "taa");
	assert.equal(plan.temporalHistoryReset, true);
}

function testCameraForcesFullAndResetsTemporal() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("camera"),
		features: createFeatures(),
		hasEnabledShadows: true,
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
		features: createFeatures(),
		hasEnabledShadows: false,
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

function testDecalWithShadowsRequiresFullReceiverReplay() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("decal"),
		features: createFeatures(),
		hasEnabledShadows: true,
		postProcess: createPostProcess(),
	});
	assert.equal(plan.firstPass, "main-opaque");
	assert.equal(plan.postProcessStartPass, null);
	assert.equal(plan.forceFullFrame, true);
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

function testProbeCaptureForcesFullFrameAndResetsTemporal() {
	for (const hasEnabledShadows of [false, true]) {
		const plan = IncrementalFramePlanner.plan({
			enabled: true,
			reasonMask: renderDirtyReasonToMask("probe-capture"),
			features: createFeatures(),
			hasEnabledShadows,
			postProcess: createPostProcess(),
		});
		assert.equal(plan.firstPass, hasEnabledShadows ? "shadow" : "main-opaque");
		assert.equal(plan.postProcessStartPass, null);
		assert.equal(plan.forceFullFrame, true);
		assert.equal(plan.temporalHistoryReset, true);
	}
}

function testRemovedReflectionProbeReasonUsesUnknownFallback() {
	const registry = new IncrementalRegistry();
	assert.equal(
		registry.getDirtyReasonMask("reflection-probe"),
		registry.getDirtyReasonMask("unknown")
	);
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
			features: createFeatures(),
			hasEnabledShadows: false,
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
	testDisplayPassMetadataRequiresRegistration();
	testUnregisteredGammaDoesNotChoosePostProcessStart();
	testDisplayPassPlanningUsesRegisteredSchedules();
	testNoDirtyReasonsReturnsNoPass();
	testParticlesStartAtParticleSim();
	testPostFxStartsAtFirstEnabledPostStage();
	testBuiltInPostProcessStageUsesPassMetadata();
	testPostFxStandardReasonStartsAtEarliestEnabledPostStage();
	testPostFxStartsAtFogWhenOnlyFogPostProcessEnabled();
	testLegacyFogApplicationStillStartsAtFog();
	testPostFxSkipsToneMappingWhenDisabled();
	testPostFxCinematicReasonResetsTemporalHistory();
	testCameraForcesFullAndResetsTemporal();
	testGeometryFallsBackToMainWhenShadowsDisabled();
	testDisabledIncrementalAlwaysFullFrame();
	testDecalWithShadowsRequiresFullReceiverReplay();
	testProbeCaptureForcesFullFrameAndResetsTemporal();
	testRemovedReflectionProbeReasonUsesUnknownFallback();
	testCustomDirtyReasonAllocatesMaskAndPlansFirstPass();
	testCustomDirtyReasonUsesGroups();
	console.log("Incremental frame planner tests passed");
}

run();
