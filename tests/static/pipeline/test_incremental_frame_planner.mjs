import assert from "node:assert/strict";
import {
	IncrementalFramePlanner,
	IncrementalRegistry,
	getDefaultIncrementalRegistry,
	renderDirtyReasonToMask,
} from "../../../src/pipeline/incremental.ts";
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

function testPostFxAlwaysReplaysScene() {
	const registry = new IncrementalRegistry();
	for (const sceneReason of [0, registry.getDirtyReasonMask("transform")]) {
		const plan = IncrementalFramePlanner.plan({
			enabled: true,
			reasonMask: registry.getDirtyReasonMask("postfx") | sceneReason,
			features: createFeatures(),
			registry,
		});
		assert.equal(plan.firstPass, "main-opaque");
		assert.equal(plan.forceFullFrame, true);
		assert.equal(plan.temporalHistoryReset, sceneReason !== 0);
	}
}

function testCustomPostFxGroupRequiresSceneReplay() {
	const registry = new IncrementalRegistry();
	const reasonMask = registry.registerDirtyReason({ id: "custom-postfx", groups: ["postfx"] });
	const plan = IncrementalFramePlanner.plan({
		enabled: true, reasonMask, features: createFeatures(), registry,
	});
	assert.equal(plan.firstPass, "main-opaque");
	assert.equal(plan.forceFullFrame, true);
	assert.equal(plan.temporalHistoryReset, false);
}

function testNoDirtyReasonsReturnsNoPass() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true, reasonMask: 0, features: createFeatures(),
	});
	assert.equal(plan.firstPass, null);
	assert.equal(plan.forceFullFrame, false);
	assert.equal(plan.temporalHistoryReset, false);
}

function testParticlesStartAtParticleSim() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("particles"),
		features: createFeatures(),
	});
	assert.equal(plan.firstPass, "particle-sim");
	assert.equal(plan.forceFullFrame, true);
	assert.equal(plan.temporalHistoryReset, true);
}

function testRemovedPostFxReasonsUseConservativeUnknownFallback() {
	const registry = new IncrementalRegistry();
	for (const reason of [
		"postfx-light",
		"postfx-standard",
		"postfx-cinematic",
	]) {
		const reasonMask = registry.getDirtyReasonMask(reason);
		assert.equal(reasonMask, registry.getDirtyReasonMask("unknown"), reason);
		const plan = IncrementalFramePlanner.plan({
			enabled: true,
			reasonMask,
			features: createFeatures(),
			registry,
		});
		assert.deepEqual(plan, {
			firstPass: "main-opaque",
			forceFullFrame: true,
			temporalHistoryReset: true,
			reasonMask,
		}, `${reason} must trigger conservative replay`);
	}
}

function testCompactDirtyMasksRetainPlanningSemantics() {
	const registry = new IncrementalRegistry();
	for (const [reasonMask, hasEnabledShadows, firstPass, forceFullFrame] of [
		[1 << 12, false, "main-opaque", false],
		[1 << 13, true, "shadow", true],
		[1 << 14, true, "shadow", true],
	]) {
		const plan = IncrementalFramePlanner.plan({
			enabled: true,
			reasonMask,
			features: createFeatures(),
			hasEnabledShadows,
			registry,
		});
		assert.deepEqual(plan, {
			firstPass,
			forceFullFrame,
			temporalHistoryReset: true,
			reasonMask,
		});
	}
}

function testCustomDirtyReasonCapacityRetainsIndependentPlanning() {
	const registry = new IncrementalRegistry();
	const passId = "custom-capacity-pass";
	registry.registerFramePass({ id: passId, order: 3.5 });
	for (let index = 0; index < 16; index++) {
		const reasonMask = registry.registerDirtyReason({
			id: `custom-capacity-${index}`,
			firstPass: passId,
		});
		const plan = IncrementalFramePlanner.plan({
			enabled: true,
			reasonMask,
			features: createFeatures(),
			registry,
		});
		assert.equal(plan.firstPass, passId);
		assert.equal(plan.forceFullFrame, false);
		assert.equal(plan.temporalHistoryReset, false);
	}
	assert.throws(() => registry.registerDirtyReason({ id: "custom-over-capacity" }));
}

function testCameraForcesFullAndResetsTemporal() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("camera"),
		features: createFeatures(),
		hasEnabledShadows: true,
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
		features: createFeatures(),
		hasEnabledShadows: false,
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

function testDecalWithShadowsRequiresFullReceiverReplay() {
	const plan = IncrementalFramePlanner.plan({
		enabled: true,
		reasonMask: renderDirtyReasonToMask("decal"),
		features: createFeatures(),
		hasEnabledShadows: true,
	});
	assert.equal(plan.firstPass, "main-opaque");
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
			registry,
		});
		assert.equal(plan.firstPass, passId);
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
		});
		assert.equal(plan.firstPass, hasEnabledShadows ? "shadow" : "main-opaque");
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
			registry,
		});
		assert.equal(plan.firstPass, "main-opaque");
		assert.equal(plan.forceFullFrame, true);
	} finally {
		registry.unregisterDirtyReason(reasonId);
	}
}

function run() {
	testPostFxAlwaysReplaysScene();
	testCustomPostFxGroupRequiresSceneReplay();
	testNoDirtyReasonsReturnsNoPass();
	testParticlesStartAtParticleSim();
	testRemovedPostFxReasonsUseConservativeUnknownFallback();
	testCompactDirtyMasksRetainPlanningSemantics();
	testCustomDirtyReasonCapacityRetainsIndependentPlanning();
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
